#!/usr/bin/env python3

import numpy as np
import rclpy
from geometry_msgs.msg import Point
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import Bool, ColorRGBA, Float32, Float32MultiArray
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker

from ..asv_path.asv_path import ParametrizedPath
from ..utils.data_conversion import DataConverter


class ASVEnvNode(Node):
    def __init__(self):
        super().__init__('asv_env_node')

        self.declare_parameter('num_agents', 2)
        self.num_agents = self.get_parameter('num_agents').value
        self.declare_parameter('training_mode', 'fast')
        self.training_mode = str(self.get_parameter('training_mode').value)
        self.debug_logging = self.training_mode == 'debug'
        self.declare_parameter('reward_mode', 'path_following')
        self.reward_mode = str(self.get_parameter('reward_mode').value)
        self.declare_parameter('max_episode_steps', 1000)
        self.max_episode_steps = int(self.get_parameter('max_episode_steps').value)
        self.declare_parameter('goal_progress_delta', 12.0)
        self.goal_progress_delta = float(self.get_parameter('goal_progress_delta').value)
        self.declare_parameter('reset_offset_range', 2.0)
        self.reset_offset_range = float(self.get_parameter('reset_offset_range').value)
        self.declare_parameter('min_reset_separation', 2.0)
        self.min_reset_separation = float(self.get_parameter('min_reset_separation').value)
        self.declare_parameter('reset_yaw_mode', 'random')
        self.reset_yaw_mode = str(self.get_parameter('reset_yaw_mode').value).strip().lower()
        self.declare_parameter('reset_yaw_noise', float(np.pi))
        self.reset_yaw_noise = max(0.0, float(self.get_parameter('reset_yaw_noise').value))
        self.declare_parameter('path_target_speed', 0.08)
        self.path_target_speed = max(1e-6, float(self.get_parameter('path_target_speed').value))
        self.declare_parameter('path_low_speed_penalty_scale', 0.8)
        self.path_low_speed_penalty_scale = max(
            0.0,
            float(self.get_parameter('path_low_speed_penalty_scale').value)
        )
        self.declare_parameter('formation_distance', 3.0)
        self.formation_distance = max(0.1, float(self.get_parameter('formation_distance').value))
        self.declare_parameter('formation_penalty_scale', 0.35)
        self.formation_penalty_scale = max(
            0.0,
            float(self.get_parameter('formation_penalty_scale').value)
        )
        self.declare_parameter('formation_bonus_scale', 0.0)
        self.formation_bonus_scale = max(
            0.0,
            float(self.get_parameter('formation_bonus_scale').value)
        )
        self.declare_parameter('formation_bonus_width', 1.0)
        self.formation_bonus_width = max(
            1e-6,
            float(self.get_parameter('formation_bonus_width').value)
        )
        self.declare_parameter('visualization_enabled', True)
        self.visualization_enabled = self._as_bool(self.get_parameter('visualization_enabled').value)

        if self.reset_yaw_mode not in ('random', 'path_aligned'):
            self.get_logger().warn(
                f"Unknown reset_yaw_mode '{self.reset_yaw_mode}', falling back to 'random'."
            )
            self.reset_yaw_mode = 'random'

        # Debug/logging controls
        self.declare_parameter('debug_log_interval_sec', 5.0)
        self.debug_log_interval = float(self.get_parameter('debug_log_interval_sec').value)
        self.spin_yaw_rate_thresh = 0.3
        self.spin_speed_thresh = 0.5
        self.spin_window = 30
        self.spin_counter = 0
        self.spin_triggered = False
        self._last_debug_metrics = {}

        self.agent_states = [None] * self.num_agents
        self.received_updates_this_step = [False] * self.num_agents
        self.loop_started = False
        self.episode_step = 0
        self.last_termination_reason = 'running'
        self._last_progress_scalar = None
        self._initial_progress_scalar = None
        self._first_publish_timer = None
        self._reference_positions = None
        self._reference_update_time = 0.0

        # Create a parametrized path for formation calculations
        self.param_path = ParametrizedPath(path_no=0)

        # Publishers for visualization
        qos_profile = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE, history=QoSHistoryPolicy.KEEP_LAST, depth=1)
        self.path_marker_pub = self.create_publisher(Marker, '/asv_env/path_marker', qos_profile)
        self.centroid_marker_pub = self.create_publisher(Marker, '/asv_env/centroid_marker', qos_profile)

        # Boundary visualization publishers
        self.core_boundary_pub = self.create_publisher(Marker, '/asv_env/core_boundary', qos_profile)
        self.safety_boundary_pub = self.create_publisher(Marker, '/asv_env/safety_boundary', qos_profile)
        self.warning_boundary_pub = self.create_publisher(Marker, '/asv_env/warning_boundary', qos_profile)

        # Timer to periodically publish visualizations. Headless training can
        # skip these marker updates and spend the budget on simulation/PPO.
        self.viz_timer = None
        if self.visualization_enabled:
            self.viz_timer = self.create_timer(0.1, self.publish_visualization)

        # Create a parametrized path for formation calculations
        self.param_path = ParametrizedPath()

        # Pre-generate the parametrized path points (performance optimization)
        self._cached_path_points = self._generate_path_points()
        self._path_cache_valid = True

        # Pre-generate boundary markers (performance optimization)
        self._cached_boundary_markers = self._generate_boundary_markers()
        self._boundary_cache_valid = True
        self.param_path.theta = 10.0  # Initial parameter value near origin

        # Assign formation angles to each agent (distributed around the circle)
        self.agent_betas = [2 * np.pi * i / self.num_agents for i in range(self.num_agents)]

        # Cache for expensive calculations (performance optimization)
        self._cached_virtual_leader_pos = None
        self._cached_virtual_leader_deriv = None
        self._cached_expected_positions = None
        self._cache_timestamp = 0.0

        # Initialize empty publisher lists first
        self.agent_action_pubs = []
        self.agent_reset_pubs = []

        # Publishers
        self.state_pub = self.create_publisher(Float32MultiArray, '/environment/state', 10)
        self.reward_pub = self.create_publisher(Float32, '/environment/reward', 10)
        self.done_pub = self.create_publisher(Bool, '/environment/done', 10)
        self.reset_service = self.create_service(Trigger, '/environment/reset', self.reset_callback)

        # Create a callback for each agent
        for i in range(self.num_agents):
            self.agent_action_pubs.append(self.create_publisher(Float32MultiArray, f'/agent_{i}/action', 10))
            self.agent_reset_pubs.append(self.create_publisher(Float32MultiArray, f'/agent_{i}/reset', 10))

        # Subscribers
        self.action_sub = self.create_subscription(
            Float32MultiArray, '/ppo/action', self.action_callback, 10
        )
        self.agent_state_subs = []
        for i in range(self.num_agents):
            self.agent_state_subs.append(
                self.create_subscription(
                    Float32MultiArray,
                    f'/agent_{i}/state_update',
                    self.create_state_callback(i),
                    10
                )
            )

        # Initial state timer
        self.initial_timer = self.create_timer(1.0, self.initial_state_publish_callback)

        # Keepalive timer (SLOWED DOWN for easier debugging/visualization)
        self.keepalive_timer = self.create_timer(5.0, self._keepalive_publish)  # 2 seconds (was 1.0s)

        # Periodic debug logger
        self.debug_timer = self.create_timer(self.debug_log_interval, self._log_debug_stats)

        self.get_logger().info(
            f'ASV Environment Node started with {self.num_agents} agents '
            f'(mode={self.training_mode}, reward_mode={self.reward_mode}, '
            f'max_episode_steps={self.max_episode_steps}, '
            f'reset_yaw_mode={self.reset_yaw_mode}, '
            f'reset_yaw_noise={self.reset_yaw_noise:.3f}, '
            f'path_target_speed={self.path_target_speed:.3f}, '
            f'formation_distance={self.formation_distance:.3f}, '
            f'formation_penalty_scale={self.formation_penalty_scale:.3f}, '
            f'formation_bonus_scale={self.formation_bonus_scale:.3f}, '
            f'visualization_enabled={self.visualization_enabled})'
        )

        # Create QoS profile for visualization (reliable, keep last 10)
        viz_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Visualization timer (update every 0.2 seconds for smoother visualization)
        # self.viz_timer = self.create_timer(0.2, self.publish_visualization)  # Already created above

    @staticmethod
    def _as_bool(value):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in ('1', 'true', 'yes', 'on')
        return bool(value)

    @staticmethod
    def _wrap_angle(angle):
        return float((angle + np.pi) % (2 * np.pi) - np.pi)

    def _sample_reset_yaw(self, path_heading):
        if self.reset_yaw_mode == 'path_aligned':
            noise = np.random.uniform(-self.reset_yaw_noise, self.reset_yaw_noise)
            return self._wrap_angle(path_heading + noise)
        return float(np.random.uniform(-np.pi, np.pi))

    def create_state_callback(self, agent_id):
        def callback(msg):
            try:
                state = np.array(msg.data, dtype=np.float32)
                if state.size != 6:
                    self.get_logger().warn(f'Agent {agent_id} state has wrong size: {state.size}')
                    return

                self.agent_states[agent_id] = state
                self.received_updates_this_step[agent_id] = True

                if self.debug_logging:
                    self.get_logger().info(f'Received state from agent {agent_id}: {state.tolist()}', throttle_duration_sec=5.0)

                # Debug current state of agent updates
                if self.debug_logging:
                    self.get_logger().info(f'Agent updates: {self.received_updates_this_step}, loop_started: {self.loop_started}', throttle_duration_sec=5.0)

                # If all agents have reported, update the environment state
                if all(self.received_updates_this_step) and self.loop_started:
                    self.publish_environment_state()
                    self.calculate_reward()
                    self.received_updates_this_step = [False] * self.num_agents
                elif self.debug_logging:
                    self.get_logger().info(f'Not publishing yet: all_reported={all(self.received_updates_this_step)}, loop_started={self.loop_started}', throttle_duration_sec=5.0)
            except Exception as e:
                self.get_logger().error(f'Error in state callback for agent {agent_id}: {str(e)}')

        return callback

    def initial_state_publish_callback(self):
        # Reset all agents and publish initial state
        self.trigger_reset()

        # IMPORTANT: Force the loop to start and publish first state
        self.loop_started = True

        # Wait a short time for agents to report their initial states
        self._first_publish_timer = self.create_timer(0.5, self._force_first_publish)

        self.initial_timer.cancel()  # Only run once

    def action_callback(self, msg):
        # Convert ROS message to NumPy array
        actions = DataConverter.ros_to_numpy(msg)

        if actions.size != 2 * self.num_agents:
            self.get_logger().error(f'Action message has wrong size: {actions.size}, expected {2 * self.num_agents}')
            return

        self.loop_started = True

        for i in range(self.num_agents):
            action_slice = actions[i*2:(i+1)*2]
            action_msg = DataConverter.numpy_to_ros(action_slice)
            self.agent_action_pubs[i].publish(action_msg)
            if self.debug_logging:
                self.get_logger().info(f'Published action to agent {i}: {action_slice.tolist()}', throttle_duration_sec=5.0)

    def publish_environment_state(self):
        try:
            # Only publish if all agents have reported at least once
            if any(s is None for s in self.agent_states):
                self.get_logger().warn('Attempted to publish state, but not all agents have reported.')
                return

            # Basic agent states array (original data)
            agent_states_array = np.array(self.agent_states)
            if self.debug_logging:
                self.get_logger().info(f'Agent states for publishing: {[s.tolist() for s in self.agent_states]}', throttle_duration_sec=5.0)

            # Calculate virtual leader position and path derivative
            pos_v, deriv = self.param_path.path(self.param_path.theta, True)
            if self.debug_logging:
                self.get_logger().info(f'Virtual leader position: {pos_v.tolist()}, derivative: {deriv}', throttle_duration_sec=5.0)

            # Calculate formation centroid from mean of agent positions
            centroid = np.mean(np.array([state[:2] for state in self.agent_states]), axis=0)

            # Calculate desired positions for each agent
            desired_positions = []
            formation_error = 0.0

            for i, state in enumerate(self.agent_states):
                # Calculate expected position based on virtual leader and formation angle
                expected_pos = pos_v.flatten() + self.formation_distance * np.array([
                    np.cos(deriv.item() + self.agent_betas[i]),
                    np.sin(deriv.item() + self.agent_betas[i])
                ])
                desired_positions.append(expected_pos)

                # Calculate formation error for this agent
                agent_error = np.linalg.norm(expected_pos - state[:2])
                formation_error += agent_error

            # Calculate average formation error
            formation_error = formation_error / self.num_agents

            # Calculate cross-track error from centroid
            cross_track_error = self.param_path.cross_track_error(centroid)

            # Create extended state array:
            # 1. Original agent states: [x, y, yaw, vx, vy, vyaw] for each agent
            # 2. Desired positions: [desired_x, desired_y] for each agent
            # 3. Formation info: [centroid_x, centroid_y, formation_error, cross_track_error]
            # 4. Virtual leader: [pos_v_x, pos_v_y]

            # Flatten original agent states
            global_state = agent_states_array.flatten()

            # Add desired positions for each agent
            for pos in desired_positions:
                global_state = np.append(global_state, pos)

            # Add formation information
            global_state = np.append(global_state, [
                centroid[0],           # Formation centroid x
                centroid[1],           # Formation centroid y
                formation_error,       # Average formation error
                cross_track_error,     # Cross track error
                pos_v.item(0),         # Virtual leader x
                pos_v.item(1)          # Virtual leader y
            ])

            # Publish the extended state
            state_msg = Float32MultiArray(data=global_state.tolist())
            self.state_pub.publish(state_msg)

            if self.debug_logging:
                self.get_logger().info(
                    f'Published extended environment state with {len(global_state)} elements',
                    throttle_duration_sec=5.0
                )
        except Exception as e:
            self.get_logger().error(f'Error in publish_environment_state: {str(e)}')

    def _keepalive_publish(self):
        # Ensure environment state is published regularly, but avoid duplicate publications
        if all(s is not None for s in self.agent_states) and self.loop_started:
            # Only publish if no recent publication from agent callbacks
            if not any(self.received_updates_this_step):
                self.publish_environment_state()
                self.calculate_reward()

    def trigger_reset(self):
        # Reset the path parameter - this sets where the virtual leader will be
        # Change to be within origin-centered bounds (-15 to 35)
        self.param_path.theta = np.random.uniform(5, 15)

        # Get position and derivative at this parameter
        pos_v, deriv = self.param_path.path(self.param_path.theta, True)
        path_heading = float(deriv.item())
        self.episode_step = 0
        self.last_termination_reason = 'running'
        self.done_pub.publish(Bool(data=False))
        self.spin_counter = 0
        self.spin_triggered = False
        self._reference_positions = None
        self._last_progress_scalar = None
        self.agent_states = [None] * self.num_agents
        self.received_updates_this_step = [False] * self.num_agents
        reset_positions = []
        self._cached_reward = None
        self._last_reward_time = 0.0
        self._cached_virtual_leader_pos = None
        self._cached_virtual_leader_deriv = None
        self._cached_expected_positions = None
        self._cache_timestamp = 0.0

        # Reset all agents to initial positions
        for i in range(self.num_agents):
            # Calculate expected position with SMALLER random offset
            expected_pos = pos_v.flatten() + self.formation_distance * np.array([
                np.cos(deriv.item() + self.agent_betas[i]),
                np.sin(deriv.item() + self.agent_betas[i])
            ])

            for attempt in range(20):
                offset = np.random.uniform(-self.reset_offset_range, self.reset_offset_range, size=2)
                x_pos = np.clip(expected_pos[0] + offset[0], -5.0, 25.0)
                y_pos = np.clip(expected_pos[1] + offset[1], -5.0, 25.0)
                candidate = np.array([x_pos, y_pos], dtype=np.float32)
                if all(np.linalg.norm(candidate - pos) >= self.min_reset_separation for pos in reset_positions):
                    break
            else:
                self.get_logger().warn(
                    f'Could not find well-separated reset pose for agent {i}; using last candidate.'
                )

            reset_positions.append(candidate)

            reset_state = np.array([
                candidate[0],   # x - clipped to bounds
                candidate[1],   # y - clipped to bounds
                self._sample_reset_yaw(path_heading),  # yaw
                0.0,                           # vx - start at zero
                0.0,                           # vy
                0.0                            # vyaw
            ])

            # Use pre-initialized publishers
            reset_msg = Float32MultiArray(data=reset_state.tolist())
            self.agent_reset_pubs[i].publish(reset_msg)

        if reset_positions:
            reset_positions_array = np.array(reset_positions, dtype=np.float32)
            self._reference_positions = reset_positions_array.copy()
            self._reference_update_time = self.get_clock().now().nanoseconds / 1e9
            reset_centroid = np.mean(reset_positions_array, axis=0)
            self._initial_progress_scalar = self._progress_scalar(reset_centroid)

        self.get_logger().info('Reset all agents to initial positions')

    def _progress_scalar(self, position):
        """Return scalar progress along the current straight path."""
        projected = self.param_path.projection(np.asarray(position).reshape(2))
        return float(np.mean(projected))

    def calculate_reward(self):
        # Only calculate if all agents have reported
        if any(s is None for s in self.agent_states):
            return

        current_time = self.get_clock().now().nanoseconds / 1e9

        # Use cache if available and fresh (within 50ms for training efficiency)
        if (hasattr(self, '_last_reward_time') and
            current_time - self._last_reward_time < 0.05 and
            hasattr(self, '_cached_reward') and
            self._cached_reward is not None):
            self.reward_pub.publish(Float32(data=float(self._cached_reward)))
            done = bool(self.check_done())
            if self._last_debug_metrics:
                self._last_debug_metrics["termination_reason"] = self.last_termination_reason
            self.done_pub.publish(Bool(data=done))
            return

        # Calculate virtual leader position and path derivative (cache when possible)
        cache_valid = (hasattr(self, '_cache_timestamp') and
                      abs(current_time - self._cache_timestamp) < 0.1)  # 100ms cache window

        if cache_valid and self._cached_virtual_leader_pos is not None:
            pos_v = self._cached_virtual_leader_pos
            deriv = self._cached_virtual_leader_deriv
            expected_positions = self._cached_expected_positions
        else:
            pos_v, deriv = self.param_path.path(self.param_path.theta, True)

            # Pre-calculate all expected positions (vectorized operation)
            cos_angles = np.cos(deriv.item() + np.array(self.agent_betas))
            sin_angles = np.sin(deriv.item() + np.array(self.agent_betas))

            expected_positions = pos_v.flatten()[np.newaxis, :] + self.formation_distance * np.column_stack([cos_angles, sin_angles])

            # Cache the calculations
            self._cached_virtual_leader_pos = pos_v
            self._cached_virtual_leader_deriv = deriv
            self._cached_expected_positions = expected_positions
            self._cache_timestamp = current_time

        # Vectorized reward calculation for better performance
        agent_positions = np.array([state[:2] for state in self.agent_states])
        agent_orientations = np.array([state[2] for state in self.agent_states])
        agent_velocities = np.array([[state[3], state[4], state[5]] for state in self.agent_states])
        out_of_bounds_now = self._positions_out_of_bounds(agent_positions)

        # Calculate position errors
        position_errors = expected_positions - agent_positions
        raw_angles_to_target = np.arctan2(position_errors[:, 1], position_errors[:, 0]) - agent_orientations
        angles_to_target = np.arctan2(np.sin(raw_angles_to_target), np.cos(raw_angles_to_target))

        centroid = np.mean(agent_positions, axis=0)
        cross_track_error = self.param_path.cross_track_error(centroid)
        along_track_error = self.param_path.along_track_error(centroid)
        progress_scalar = self._progress_scalar(centroid)
        progress_delta = 0.0 if self._last_progress_scalar is None else progress_scalar - self._last_progress_scalar
        self._last_progress_scalar = progress_scalar

        path_heading = float(deriv.item())
        path_tangent = np.array([np.cos(path_heading), np.sin(path_heading)])
        path_normal = np.array([-path_tangent[1], path_tangent[0]])
        centroid_velocity = np.mean(agent_velocities[:, :2], axis=0)
        path_speed = float(np.dot(centroid_velocity, path_tangent))
        lateral_speed = float(abs(np.dot(centroid_velocity, path_normal)))

        velocity_magnitudes = np.sqrt(
            agent_velocities[:, 0]**2 +
            agent_velocities[:, 1]**2 +
            agent_velocities[:, 2]**2
        )
        mean_yaw_rate = float(np.mean(agent_velocities[:, 2]))
        mean_abs_yaw_rate = float(np.mean(np.abs(agent_velocities[:, 2])))
        mean_speed = float(np.mean(np.sqrt(agent_velocities[:, 0]**2 + agent_velocities[:, 1]**2)))

        spin_condition = mean_abs_yaw_rate > self.spin_yaw_rate_thresh and mean_speed < self.spin_speed_thresh
        if spin_condition:
            self.spin_counter += 1
        else:
            self.spin_counter = 0

        if self.spin_counter >= self.spin_window:
            self.spin_triggered = True

        avg_stillness = 0.0
        avg_velocity_penalty = 0.0
        avg_drift_penalty = 0.0
        mean_stillness_drift = 0.0
        formation_penalty = 0.0
        formation_bonus = 0.0

        if self.reward_mode == 'stillness':
            k_stillness = 5.0
            stillness_components = k_stillness * np.exp(-velocity_magnitudes)

            k_velocity_penalty = 3.0
            velocity_penalty_components = -k_velocity_penalty * velocity_magnitudes

            if self._reference_positions is None:
                self._reference_positions = agent_positions.copy()
                self._reference_update_time = current_time

            k_drift = 2.0
            drift_distances = np.linalg.norm(agent_positions - self._reference_positions, axis=1)
            mean_stillness_drift = float(np.mean(drift_distances))
            drift_penalty_components = -k_drift * drift_distances

            avg_stillness = float(np.mean(stillness_components))
            avg_velocity_penalty = float(np.mean(velocity_penalty_components))
            avg_drift_penalty = float(np.mean(drift_penalty_components))
            reward = avg_stillness + avg_velocity_penalty + avg_drift_penalty
        else:
            formation_errors = np.linalg.norm(position_errors, axis=1)
            mean_formation_error = float(np.mean(formation_errors))
            forward_alignment = np.cos(angles_to_target)
            heading_to_path = np.arctan2(
                np.sin(agent_orientations - path_heading),
                np.cos(agent_orientations - path_heading)
            )
            path_heading_alignment = np.cos(heading_to_path)

            forward_progress_reward = 20.0 * float(np.clip(progress_delta, -0.5, 0.5))
            path_speed_reward = 6.0 * max(path_speed, 0.0)
            backward_penalty = -8.0 * max(-path_speed, 0.0)
            low_speed_ratio = np.clip(
                (self.path_target_speed - path_speed) / self.path_target_speed,
                0.0,
                1.5
            )
            low_speed_penalty = -self.path_low_speed_penalty_scale * float(low_speed_ratio)
            formation_penalty = -self.formation_penalty_scale * mean_formation_error
            formation_bonus = self.formation_bonus_scale * float(
                np.exp(-mean_formation_error / self.formation_bonus_width)
            )
            cross_track_penalty = -0.55 * float(cross_track_error)
            target_heading_reward = 0.4 * float(np.mean(forward_alignment))
            path_heading_reward = 0.6 * float(np.mean(path_heading_alignment))
            lateral_penalty = -0.8 * lateral_speed
            yaw_penalty = -1.5 * mean_abs_yaw_rate
            spin_penalty = -6.0 if self.spin_triggered else 0.0
            boundary_penalty = -25.0 if out_of_bounds_now else 0.0

            reward = (
                forward_progress_reward
                + path_speed_reward
                + backward_penalty
                + low_speed_penalty
                + formation_penalty
                + formation_bonus
                + cross_track_penalty
                + target_heading_reward
                + path_heading_reward
                + lateral_penalty
                + yaw_penalty
                + spin_penalty
                + boundary_penalty
            )

        self.episode_step += 1

        # Reset reference positions if the mode was switched dynamically.
        if self.reward_mode != 'stillness':
            self._reference_positions = agent_positions.copy()

        # --- Debug metrics ---
        formation_error = float(np.mean(np.linalg.norm(position_errors, axis=1)))
        heading_error = float(np.mean(np.abs(angles_to_target)))

        self._last_debug_metrics = {
            "reward": float(reward),
            "avg_stillness": float(avg_stillness),
            "avg_velocity_penalty": float(avg_velocity_penalty),
            "avg_drift_penalty": float(avg_drift_penalty),
            "mean_stillness_drift": float(mean_stillness_drift),
            "cross_track_error": float(cross_track_error),
            "along_track_error": float(along_track_error),
            "formation_error": formation_error,
            "heading_error": heading_error,
            "progress_scalar": float(progress_scalar),
            "progress_delta": float(progress_delta),
            "path_speed": path_speed,
            "low_speed_penalty": float(low_speed_penalty) if self.reward_mode != 'stillness' else 0.0,
            "formation_penalty": float(formation_penalty) if self.reward_mode != 'stillness' else 0.0,
            "formation_bonus": float(formation_bonus) if self.reward_mode != 'stillness' else 0.0,
            "lateral_speed": lateral_speed,
            "mean_speed": mean_speed,
            "mean_yaw_rate": mean_yaw_rate,
            "mean_abs_yaw_rate": mean_abs_yaw_rate,
            "out_of_bounds": out_of_bounds_now,
            "spin_counter": self.spin_counter,
            "spin_triggered": self.spin_triggered,
            "episode_step": self.episode_step,
            "termination_reason": self.last_termination_reason,
            "reward_mode": self.reward_mode
        }

        # Cache the reward
        self._cached_reward = reward
        self._last_reward_time = current_time

        # Publish reward
        self.reward_pub.publish(Float32(data=float(reward)))

        # Check if done and publish - use explicit bool conversion
        done = bool(self.check_done())
        self._last_debug_metrics["termination_reason"] = self.last_termination_reason
        self.done_pub.publish(Bool(data=done))

    def check_done(self):
        self.last_termination_reason = 'running'

        if self.check_out_of_bounds():
            self.last_termination_reason = 'out_of_bounds'
            return True

        if self.check_collision():
            self.last_termination_reason = 'collision'
            return True

        if self.episode_step >= self.max_episode_steps:
            self.last_termination_reason = 'max_episode_steps'
            return True

        centroid = np.mean(np.array([state[:2] for state in self.agent_states]), axis=0)
        progress_from_start = 0.0
        if self._initial_progress_scalar is not None:
            progress_from_start = self._progress_scalar(centroid) - self._initial_progress_scalar

        if self.reward_mode != 'stillness' and progress_from_start >= self.goal_progress_delta:
            self.last_termination_reason = 'goal_progress'
            return True

        return False

    def _positions_out_of_bounds(self, positions):
        min_x, max_x = -15.0, 35.0
        min_y, max_y = -15.0, 35.0
        for x_pos, y_pos in positions:
            if x_pos < min_x or x_pos > max_x or y_pos < min_y or y_pos > max_y:
                return True
        return False

    def check_out_of_bounds(self):
        if any(s is None for s in self.agent_states):
            return False

        positions = np.array([state[:2] for state in self.agent_states])
        return self._positions_out_of_bounds(positions)

    def check_collision(self):
        if self.num_agents < 2:
            return False

        # Check distances between all pairs of agents
        for i in range(self.num_agents):
            for j in range(i+1, self.num_agents):
                pos_i = self.agent_states[i][:2]
                pos_j = self.agent_states[j][:2]
                distance = np.linalg.norm(pos_i - pos_j)

                # Collision threshold - increased for formation flying
                if distance < 1.0:  # Smaller threshold since agents are in formation
                    return True

        return False

    def _force_first_publish(self):
        if self.debug_logging:
            self.get_logger().info(f'AGENT STATES: {[s is not None for s in self.agent_states]}')
        if all(s is not None for s in self.agent_states):
            if self.debug_logging:
                self.get_logger().info('First state publish forced to break action-state deadlock')
            self.publish_environment_state()
            if self._first_publish_timer is not None:
                self._first_publish_timer.cancel()
                self._first_publish_timer = None
            return True
        else:
            if self.debug_logging:
                self.get_logger().info('Waiting for all agents to report before forcing first state')
            return False

    def reset_callback(self, request, response):
        """Service to reset all agents to initial positions."""
        try:
            # Reset all agents to their initial positions
            self.reset_all_agents()

            response.success = True
            response.message = "All agents reset successfully"
            return response
        except Exception as e:
            self.get_logger().error(f"Failed to reset agents: {e}")
            response.success = False
            response.message = f"Failed to reset: {str(e)}"
            return response

    def reset_all_agents(self):
        """Reset all agents to initial positions using path-based positioning."""
        # Use the same logic as trigger_reset but for service calls
        self.trigger_reset()

    def reset_agent(self, agent_id, position):
        """
        Resets a specific agent to a given position.
        
        Args:
            agent_id: The ID of the agent to reset
            position: Array containing [x, y, yaw, vx, vy, vyaw]
        """
        if agent_id < 0 or agent_id >= self.num_agents:
            self.get_logger().error(f"Invalid agent ID for reset: {agent_id}")
            return

        # Convert position array to Float32MultiArray message
        reset_msg = Float32MultiArray(data=position)

        # Publish reset message to this agent
        self.agent_reset_pubs[agent_id].publish(reset_msg)
        self.get_logger().info(f"Reset agent {agent_id} to position {position}")

        # Clear any cached state for this agent
        self.agent_states[agent_id] = None
        self.received_updates_this_step[agent_id] = False

    # In order to see the parametrized path
    def publish_visualization(self):
        # Single timestamp for all markers (performance optimization)
        current_time = self.get_clock().now().to_msg()

        # 1. Publish the parametrized path (using cached points)
        path_marker = Marker()
        path_marker.header.frame_id = "map"
        path_marker.header.stamp = current_time
        path_marker.ns = "parametrized_path"
        path_marker.id = 0
        path_marker.type = Marker.LINE_STRIP
        path_marker.action = Marker.ADD
        path_marker.scale.x = 0.1  # Line width
        path_marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0) # Green
        path_marker.points = self._cached_path_points  # Use cached points

        self.path_marker_pub.publish(path_marker)

        # 2. Publish the centroid
        if not all(s is not None for s in self.agent_states):
            return # Wait until all agent states are available

        centroid_marker = Marker()
        centroid_marker.header.frame_id = "map"
        centroid_marker.header.stamp = current_time
        centroid_marker.ns = "formation_centroid"
        centroid_marker.id = 1
        centroid_marker.type = Marker.SPHERE
        centroid_marker.action = Marker.ADD
        centroid_marker.scale.x = 0.8
        centroid_marker.scale.y = 0.8
        centroid_marker.scale.z = 0.8
        centroid_marker.color = ColorRGBA(r=0.0, g=0.0, b=1.0, a=0.8) # Blue

        # Calculate centroid position
        positions = np.array([state[:2] for state in self.agent_states])
        centroid = np.mean(positions, axis=0)
        # Cast NumPy floats to native Python floats
        centroid_marker.pose.position = Point(x=float(centroid[0]), y=float(centroid[1]), z=0.0)

        self.centroid_marker_pub.publish(centroid_marker)

        # 3. Publish boundary visualizations (using cached markers)
        self._publish_boundaries_optimized(current_time)

    def _publish_boundaries_optimized(self, current_time):
        """Optimized boundary publishing using cached markers"""
        # Update timestamps on cached markers and publish
        for key, marker in self._cached_boundary_markers.items():
            marker.header.stamp = current_time

            if key == 'core':
                self.core_boundary_pub.publish(marker)
            elif key == 'safety':
                self.safety_boundary_pub.publish(marker)
            elif key == 'warning':
                self.warning_boundary_pub.publish(marker)

    def _create_boundary_marker(self, marker_id, namespace, bounds, color, line_width, timestamp):
        """Create a rectangular boundary marker"""
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = timestamp if timestamp is not None else self.get_clock().now().to_msg()
        marker.ns = namespace
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = line_width
        marker.color = color
        marker.pose.orientation.w = 1.0  # No rotation

        # Extract bounds
        x_min, x_max, y_min, y_max = bounds

        # Create rectangle points (closed loop)
        points = [
            Point(x=float(x_min), y=float(y_min), z=0.0),  # Bottom-left
            Point(x=float(x_max), y=float(y_min), z=0.0),  # Bottom-right
            Point(x=float(x_max), y=float(y_max), z=0.0),  # Top-right
            Point(x=float(x_min), y=float(y_max), z=0.0),  # Top-left
            Point(x=float(x_min), y=float(y_min), z=0.0),  # Back to start (close rectangle)
        ]

        marker.points = points
        return marker

    def _generate_path_points(self):
        """Pre-generate path points for performance (called once)"""
        points = []
        for theta in np.linspace(0, 150, 500):  # Adjust range as needed
            p = self.param_path.path(theta)
            # Cast NumPy floats to native Python floats
            points.append(Point(x=float(p[0]), y=float(p[1]), z=0.0))
        return points

    def _generate_boundary_markers(self):
        """Pre-generate boundary marker templates (called once)"""
        markers = {}

        # Core Operating Area (-5, 25) - Red dashed rectangle
        markers['core'] = self._create_boundary_marker(
            marker_id=10,
            namespace="core_boundary",
            bounds=(-5.0, 25.0, -5.0, 25.0),
            color=ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.8),  # Red
            line_width=0.15,
            timestamp=None  # Will be updated per publish
        )

        # Safety Buffer (-10, 30) - Yellow dashed rectangle
        markers['safety'] = self._create_boundary_marker(
            marker_id=11,
            namespace="safety_boundary",
            bounds=(-10.0, 30.0, -10.0, 30.0),
            color=ColorRGBA(r=1.0, g=1.0, b=0.0, a=0.6),  # Yellow
            line_width=0.12,
            timestamp=None  # Will be updated per publish
        )

        # Warning Zone (-15, 35) - Orange dashed rectangle
        markers['warning'] = self._create_boundary_marker(
            marker_id=12,
            namespace="warning_boundary",
            bounds=(-15.0, 35.0, -15.0, 35.0),
            color=ColorRGBA(r=1.0, g=0.5, b=0.0, a=0.4),  # Orange
            line_width=0.10,
            timestamp=None  # Will be updated per publish
        )

        return markers

    def _log_debug_stats(self):
        """Periodically log concise metrics to spot spinning or drift."""
        if not self._last_debug_metrics:
            return

        m = self._last_debug_metrics
        summary = (
            f"mode={m.get('reward_mode')} | step={m.get('episode_step')} | "
            f"reward={m.get('reward'):.3f} | cte={m.get('cross_track_error'):.3f} | "
            f"along_err={m.get('along_track_error'):.3f} | form_err={m.get('formation_error'):.3f} | "
            f"progress={m.get('progress_scalar'):.3f} d={m.get('progress_delta'):.3f} | "
            f"path_speed={m.get('path_speed'):.3f} | low_speed_pen={m.get('low_speed_penalty'):.3f} | "
            f"form_pen={m.get('formation_penalty'):.3f} | form_bonus={m.get('formation_bonus'):.3f} | "
            f"lateral={m.get('lateral_speed'):.3f} | "
            f"heading_err={m.get('heading_error'):.3f} | speed={m.get('mean_speed'):.3f} | "
            f"yaw_rate={m.get('mean_yaw_rate'):.3f} | spin_cnt={m.get('spin_counter')} | "
            f"spin_triggered={m.get('spin_triggered')} | oob={m.get('out_of_bounds')} | "
            f"done={m.get('termination_reason')}"
        )
        if m.get('reward_mode') == 'stillness':
            summary += (
                f" | still={m.get('avg_stillness'):.3f}"
                f" | vel_pen={m.get('avg_velocity_penalty'):.3f}"
                f" | drift={m.get('mean_stillness_drift'):.3f}"
                f" | drift_pen={m.get('avg_drift_penalty'):.3f}"
            )
        self.get_logger().info(f"[env_debug] {summary}")


def main(args=None):
    rclpy.init(args=args)
    node = ASVEnvNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except KeyboardInterrupt:
            pass
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
