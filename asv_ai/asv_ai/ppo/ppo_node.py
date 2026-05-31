#!/usr/bin/env python3
import json
import os
import sys
import time

import numpy as np
import rclpy
import torch as th
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from rclpy.node import Node
# from stable_baselines3 import PPO  # Replaced with CustomPPO
# from stable_baselines3.common.buffers import RolloutBuffer  # Replaced with CustomRolloutBuffer
from std_msgs.msg import Bool, Float32, Float32MultiArray
from std_srvs.srv import Trigger

from ..utils.data_conversion import DataConverter
from .networks import CustomActorCritic
from .algorithms import CustomRolloutBuffer, RolloutBufferSamples, CustomPPO


class PPONode(Node):  # Renamed from ASVPPONode for generalization (Point 8)
    @staticmethod
    def _as_bool(value):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in ('1', 'true', 'yes', 'on')
        return bool(value)

    def __init__(self):
        super().__init__('ppo_node')  # Renamed from asv_ppo_node for generalization

        # Parameters - Environment Configuration
        self.declare_parameter('num_agents', 2)
        self.declare_parameter('model_path', '')
        self.declare_parameter('rollout_dir', os.path.expanduser('~/Desktop/PPO_Rollouts'))
        self.declare_parameter('rollout_save_every', 500)
        self.declare_parameter('rollout_collection_enabled', True)
        self.declare_parameter('autoload_model', False)
        self.declare_parameter('deterministic_inference', True)
        self.declare_parameter('training_mode', 'fast')
        self.declare_parameter('use_extended_observation', True)
        self.declare_parameter('normalize_observation', True)
        self.declare_parameter('reset_grace_steps', 10)
        self.declare_parameter('min_episode_transitions', 50)
        self.declare_parameter('constant_action_enabled', False)
        self.declare_parameter('constant_yaw_action', 0.0)
        self.declare_parameter('constant_surge_action', 0.0)
        # Debug/logging cadence
        self.declare_parameter('log_interval_sec', 5.0)
        
        # Parameters - State/Action Space Dimensions (generalizable for any robot/environment)
        self.declare_parameter('obs_dim_per_agent', 6)  # Default: [x, y, yaw, vx, vy, vyaw]
        self.declare_parameter('action_dim_per_agent', 2)  # Default: [vyaw_rate, acceleration]

        # Resolve parameters
        self.num_agents = self.get_parameter('num_agents').value
        self.model_path = self.get_parameter('model_path').value
        self.rollout_dir = os.path.expanduser(self.get_parameter('rollout_dir').value)
        self.rollout_save_every = int(self.get_parameter('rollout_save_every').value)
        self.rollout_collection_enabled = self._as_bool(self.get_parameter('rollout_collection_enabled').value)
        self.autoload_model = self._as_bool(self.get_parameter('autoload_model').value)
        self.deterministic_inference = self._as_bool(self.get_parameter('deterministic_inference').value)
        self.training_mode = str(self.get_parameter('training_mode').value)
        self.use_extended_observation = self._as_bool(self.get_parameter('use_extended_observation').value)
        self.normalize_observation = self._as_bool(self.get_parameter('normalize_observation').value)
        self.reset_grace_steps = int(self.get_parameter('reset_grace_steps').value)
        self.min_episode_transitions = int(self.get_parameter('min_episode_transitions').value)
        self.constant_action_enabled = self._as_bool(self.get_parameter('constant_action_enabled').value)
        self.constant_yaw_action = float(self.get_parameter('constant_yaw_action').value)
        self.constant_surge_action = float(self.get_parameter('constant_surge_action').value)
        self.debug_logging = self.training_mode == 'debug'
        self.log_interval_sec = float(self.get_parameter('log_interval_sec').value)
        
        # Resolve space dimensions (generalizable)
        self.obs_dim_per_agent = self.get_parameter('obs_dim_per_agent').value
        self.action_dim_per_agent = self.get_parameter('action_dim_per_agent').value
        
        os.makedirs(self.rollout_dir, exist_ok=True)

        # Runtime buffers/state - Temporal sequence for RL training
        # For RL we need transition tuples: (s_t-1, a_t-1, r_t, s_t)
        self.rollout_data = []  # Collected rollout data for logging
        self.last_state = None  # Full raw state from previous step (for rollout logging)
        self.prev_obs = None    # Processed observation from PREVIOUS step (s_t-1) - where action was taken
        self.last_action = None # Action taken at previous step (a_t-1)
        self.last_reward = None # Reward received at current step (r_t)
        self.last_values = None # Value estimate from previous step (V(s_t-1))
        self.last_log_probs = None # Log probability of action from previous step
        self.current_obs = None # CURRENT observation (s_t) - result after taking action
        self.episode_start = np.ones(1, dtype=bool)  # Start of first episode
        self.model_ready = False
        self._pending_state = None  # buffer a state if it arrives before model is ready
        self.episode_id = 0
        self.step_idx = 0
        self.reset_in_progress = False
        self.awaiting_reset_state = False
        self.training_in_progress = False
        self.done_latched = False
        self.reset_grace_remaining = 0
        self.dones = np.zeros(1, dtype=bool)
        self._shutting_down = False

        # Training statistics
        self.training_stats = {
            'total_training_sessions': 0,
            'total_transitions_trained': 0,
            'average_buffer_size_at_training': 0,
            'memory_resets': 0
        }
        self.episode_rewards = []
        self.current_episode_reward = 0.0

        # Add training parameters
        self.declare_parameter('training_enabled', False)
        self.declare_parameter('train_frequency', 1000)  # How many steps before training
        self.declare_parameter('min_training_size', 0)  # 0 means use train_frequency
        self.declare_parameter('train_interval_sec', 0.0)  # 0 disables timer-based partial updates
        self.declare_parameter('n_epochs', 10)  # PPO training epochs
        self.declare_parameter('batch_size', 64)  # PPO batch size
        self.declare_parameter('learning_rate', 3e-4)
        self.declare_parameter('vf_coef', 0.1)
        self.declare_parameter('initial_log_std', -1.5)
        self.declare_parameter('initial_surge_action_bias', 0.0)
        self.declare_parameter('target_kl', 0.05)
        self.declare_parameter('gamma', 0.99)  # Discount factor
        self.declare_parameter('gae_lambda', 0.95)  # GAE lambda
        self.declare_parameter('max_episodes', 100000)
        
        # Model saving parameters
        self.declare_parameter('save_frequency', 500)  # Save every N episodes
        self.declare_parameter('auto_save_enabled', True)  # Enable periodic saving
        self.declare_parameter('save_on_improvement', True)  # Save when reward improves

        # Get training parameters
        self.training_enabled = self._as_bool(self.get_parameter('training_enabled').value)
        if self.constant_action_enabled and self.training_enabled:
            self.get_logger().warn('constant_action_enabled=true disables PPO training for this run')
            self.training_enabled = False
        self.train_frequency = int(self.get_parameter('train_frequency').value)
        requested_min_training_size = int(self.get_parameter('min_training_size').value)
        self.min_training_size = self.train_frequency if requested_min_training_size <= 0 else requested_min_training_size
        self.train_interval_sec = float(self.get_parameter('train_interval_sec').value)
        self.n_epochs = int(self.get_parameter('n_epochs').value)
        self.batch_size = int(self.get_parameter('batch_size').value)
        self.learning_rate = float(self.get_parameter('learning_rate').value)
        self.vf_coef = float(self.get_parameter('vf_coef').value)
        self.initial_log_std = float(self.get_parameter('initial_log_std').value)
        self.initial_surge_action_bias = float(self.get_parameter('initial_surge_action_bias').value)
        target_kl_param = float(self.get_parameter('target_kl').value)
        self.target_kl = None if target_kl_param <= 0.0 else target_kl_param
        self.gamma = float(self.get_parameter('gamma').value)
        self.gae_lambda = float(self.get_parameter('gae_lambda').value)
        self.max_episodes = int(self.get_parameter('max_episodes').value)
        
        # Get model saving parameters
        self.save_frequency = self.get_parameter('save_frequency').value
        self.auto_save_enabled = self._as_bool(self.get_parameter('auto_save_enabled').value)
        self.save_on_improvement = self._as_bool(self.get_parameter('save_on_improvement').value)
        
        # Model saving state
        self.best_episode_reward = float('-inf')
        self.last_save_episode = 0

        # Publishers and Subscribers FIRST to avoid missing early messages
        self.state_sub = self.create_subscription(Float32MultiArray, '/environment/state', self.state_callback, 10)
        self.reward_sub = self.create_subscription(Float32, '/environment/reward', self.reward_callback, 10)
        self.done_sub = self.create_subscription(Bool, '/environment/done', self.done_callback, 10)

        self.action_pub = self.create_publisher(Float32MultiArray, '/ppo/action', 10)

        self.reset_client = self.create_client(Trigger, '/environment/reset')

        # Periodic debug stats
        self.log_timer = self.create_timer(self.log_interval_sec, self._log_debug_stats)

        # Define observation and action spaces for PPO model (generalizable via parameters)
        # These stay in PPO node as they are model-specific, not environment-specific
        obs_dim = self._expected_obs_dim()
        action_dim = self.num_agents * self.action_dim_per_agent
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1, high=1,
            shape=(action_dim,),
            dtype=np.float32
        )

        # Smart model loading: explicit model_path first, then rollout_dir checkpoints.
        self.model = None
        self.loaded_model_path = self._find_and_load_best_model()
        if self.loaded_model_path:
            self.get_logger().info(f'Successfully loaded PPO model from {self.loaded_model_path}')
        else:
            self.model = CustomPPO(
                obs_dim=obs_dim,
                action_dim=action_dim,
                device='cpu',
                learning_rate=self.learning_rate,
                clip_range=0.2,
                ent_coef=0.0,
                vf_coef=self.vf_coef,
                max_grad_norm=0.5,
                target_kl=self.target_kl,
                initial_log_std=self.initial_log_std,
                initial_surge_action_bias=self.initial_surge_action_bias
            )
            self.get_logger().info('No compatible model found, starting with new CustomPPO model.')
        self.model.learning_rate = self.learning_rate
        self.model.lr_schedule = self.learning_rate
        for param_group in self.model.optimizer.param_groups:
            param_group['lr'] = self.learning_rate
        self.model.target_kl = self.target_kl
        self.model.vf_coef = self.vf_coef

        constant_pair = np.array(
            [self.constant_yaw_action, self.constant_surge_action],
            dtype=np.float32
        )
        constant_pair = np.clip(constant_pair, -1.0, 1.0)
        self.constant_action = np.tile(constant_pair, self.num_agents).reshape(1, -1)
        if self.constant_action_enabled:
            self.get_logger().warn(
                f'Using constant action for diagnostics: '
                f'yaw={constant_pair[0]:.3f}, surge={constant_pair[1]:.3f}'
            )

        self.model_ready = True
        # If we received a state while loading, process it now
        if self._pending_state is not None:
            self.get_logger().info('Processing buffered environment state after model init.')
            self._predict_and_publish(self._pending_state)
            self._pending_state = None

        # Add after model initialization:
        if self.training_enabled:
            # observation_space and action_space already defined above
            
            # Create training buffer with optimized configuration
            self.buffer_size = max(self.min_training_size, int(self.train_frequency))
            self.memory_limit = max(self.buffer_size * 2, 500)

            self.training_buffer = CustomRolloutBuffer(
                buffer_size=self.buffer_size,
                obs_dim=obs_dim,
                action_dim=action_dim,
                device=self.model.device,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda
            )

            # Training state variables (note: current_obs already defined above)
            self.values = []  # Store value estimates
            self.log_probs = []  # Store log probabilities
            # self.current_obs already initialized above in Runtime buffers section
            self.dones = np.zeros(1, dtype=bool)  # Episode termination flags
            self.episode_start = np.ones(1, dtype=bool)  # First collected transition starts an episode

            if self.train_interval_sec > 0.0:
                self.training_timer = self.create_timer(self.train_interval_sec, self.train_model)
                self.get_logger().warn(
                    f"Timer-based PPO updates enabled every {self.train_interval_sec:.1f}s. "
                    "Use train_interval_sec:=0.0 for strict on-policy buffer updates."
                )

            self.reset_timer = self.create_timer(30.0, self.check_and_reset)  # 30 seconds (was 20s)

            self.get_logger().info(
                "PPO training mode enabled "
                f"(obs_dim={self.obs_dim}, action_dim={self.action_dim}, "
                f"extended_observation={self.use_extended_observation}, "
                f"normalize_observation={self.normalize_observation}, "
                f"buffer_size={self.buffer_size}, min_training_size={self.min_training_size}, "
                f"target_kl={self.target_kl})"
            )

            self.get_logger().info(f'PPO Node started with {self.num_agents} agents')

        # New - create session timestamp at startup for consistent file naming
        self.session_start_time = self.get_clock().now()
        self.formatted_timestamp = self.session_start_time.to_msg()
        self.session_id = time.strftime(
            "%Y-%m-%d_%H-%M-%S",
            time.localtime(self.formatted_timestamp.sec)
        )

        # Create session folder for this training run
        self.session_dir = os.path.join(self.rollout_dir, f"session_{self.session_id}")
        os.makedirs(self.session_dir, exist_ok=True)

        # Initialize metadata file for the session
        self._initialize_session_metadata()
        
        # Setup signal handling for graceful shutdown and model saving
        import signal
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _expected_obs_dim(self) -> int:
        agent_obs_dim = self.num_agents * self.obs_dim_per_agent
        if not self.use_extended_observation:
            return agent_obs_dim

        # Environment state format:
        # agents [N * 6] + desired positions [N * 2] + formation/path summary [6]
        return agent_obs_dim + (self.num_agents * 2) + 6

    def _reset_transition_state(self):
        """Clear transition context so no pre-reset action is tied to post-reset state."""
        self.last_state = None
        self.prev_obs = None
        self.current_obs = None
        self.last_action = None
        self.last_reward = None
        self.last_values = None
        self.last_log_probs = None
        self.episode_start = np.ones(1, dtype=bool)

    def _begin_reset_wait(self):
        self.reset_in_progress = True
        self.awaiting_reset_state = True
        self.reset_grace_remaining = self.reset_grace_steps
        self._reset_transition_state()

    def _state_positions(self, flat_state: np.ndarray) -> list[list[float]]:
        positions = []
        for i in range(self.num_agents):
            agent_x = flat_state[i * self.obs_dim_per_agent]
            agent_y = flat_state[i * self.obs_dim_per_agent + 1]
            positions.append([float(agent_x), float(agent_y)])
        return positions

    def _state_to_observation(self, flat_state: np.ndarray) -> np.ndarray | None:
        if self.use_extended_observation:
            if flat_state.size < self.obs_dim:
                self.get_logger().warn(
                    f"State has {flat_state.size} values, expected at least {self.obs_dim}; skipping",
                    throttle_duration_sec=2.0
                )
                return None
            obs = flat_state[:self.obs_dim].astype(np.float32).flatten()
        else:
            obs = DataConverter.state_to_ppo_input(
                flat_state,
                self.num_agents,
                self.obs_dim_per_agent
            ).astype(np.float32).flatten()

        if obs.size != self.obs_dim:
            self.get_logger().warn(
                f"Observation has {obs.size} values, expected {self.obs_dim}; skipping",
                throttle_duration_sec=2.0
            )
            return None

        if self.normalize_observation:
            obs = self._normalize_observation(obs)

        return obs

    def _normalize_observation(self, obs: np.ndarray) -> np.ndarray:
        normalized = obs.astype(np.float32, copy=True)
        for i in range(self.num_agents):
            start = i * self.obs_dim_per_agent
            normalized[start + 0] /= 25.0  # x
            normalized[start + 1] /= 25.0  # y
            normalized[start + 2] /= np.pi  # yaw
            normalized[start + 3] /= 2.0  # vx
            normalized[start + 4] /= 2.0  # vy
            normalized[start + 5] /= 1.0  # yaw rate

        if self.use_extended_observation:
            desired_start = self.num_agents * self.obs_dim_per_agent
            desired_end = desired_start + self.num_agents * 2
            normalized[desired_start:desired_end] /= 25.0

            summary_start = desired_end
            normalized[summary_start + 0] /= 25.0  # centroid x
            normalized[summary_start + 1] /= 25.0  # centroid y
            normalized[summary_start + 2] /= 10.0  # formation error
            normalized[summary_start + 3] /= 10.0  # cross-track error
            normalized[summary_start + 4] /= 25.0  # virtual leader x
            normalized[summary_start + 5] /= 25.0  # virtual leader y

        return np.clip(normalized, -10.0, 10.0)

    def state_callback(self, msg):
        try:
            # Convert ROS message to NumPy array
            flat_state = DataConverter.ros_to_numpy(msg)
            if self.debug_logging:
                self.get_logger().info(f'PPO received state with shape {flat_state.shape}', throttle_duration_sec=1)

            if self.reset_in_progress:
                self.get_logger().info('Ignoring state while reset service is in progress.', throttle_duration_sec=2.0)
                return

            if self.awaiting_reset_state:
                if self._agents_out_of_bounds(flat_state):
                    self.get_logger().warn(
                        f"Ignoring stale out-of-bounds state after reset: {self._state_positions(flat_state)}",
                        throttle_duration_sec=2.0
                    )
                    return

                self.awaiting_reset_state = False
                self.reset_grace_remaining = self.reset_grace_steps
                self.done_latched = False
                self.dones = np.zeros(1, dtype=bool)
                self.get_logger().info('Accepted first clean state after reset.')

            # If the rollout buffer is full, train before collecting more data.
            if hasattr(self, 'training_buffer') and self.training_buffer.full:
                self.get_logger().info("Training buffer full, training before collecting more data")
                self.train_model()
                if self.training_buffer.full:
                    self.get_logger().warn("Training did not clear the full buffer; resetting it to resume collection")
                    self.training_buffer.reset()

            skip_out_of_bounds_check = self.reset_grace_remaining > 0
            if skip_out_of_bounds_check:
                self.reset_grace_remaining -= 1

            if not skip_out_of_bounds_check and self._agents_out_of_bounds(flat_state):
                problematic_positions = self._state_positions(flat_state)
                self.get_logger().warn(f"Agents detected out of bounds at positions: {problematic_positions}")
                self.dones = np.ones(1, dtype=bool)
                self._finalize_episode(reason='out_of_bounds')
                self.current_episode_reward = 0.0
                self._begin_reset_wait()
                if not self.reset_environment():
                    self.reset_in_progress = False
                    self.awaiting_reset_state = False
                return

            if not self.model_ready:
                # Buffer the latest state until model is ready
                self._pending_state = flat_state
                self.get_logger().info('Model not ready yet; buffering environment state.', throttle_duration_sec=1)
                return

            self._predict_and_publish(flat_state)
        except Exception as e:
            self.get_logger().error(f'Error in state_callback: {str(e)}')

    def _predict_and_publish(self, flat_state: np.ndarray):
        """Process environment state, predict actions, and publish to ROS topics."""
        try:
            # Format state for the model (returns 1D flat array for homogeneity)
            agent_states = self._state_to_observation(flat_state)
            if agent_states is None:
                return

            self.current_obs = agent_states  # Already flat, no need to reshape

            # Get actions from the policy. Training samples stochastically;
            # evaluation uses deterministic actions by default for repeatable RViz runs.
            with th.no_grad():
                obs_tensor = DataConverter.numpy_to_tensor(self.current_obs)
                obs_tensor = obs_tensor.reshape(1, -1)

                if self.constant_action_enabled:
                    actions_np = self.constant_action.copy()
                    values = None
                    log_probs = None
                elif self.training_enabled:
                    actions, values, log_probs = self.model.policy(obs_tensor)
                    actions_np = actions.cpu().numpy()
                else:
                    actions_np, _ = self.model.predict(
                        self.current_obs,
                        deterministic=self.deterministic_inference
                    )
                    actions_np = np.asarray(actions_np, dtype=np.float32).reshape(1, -1)
                    values = None
                    log_probs = None

            # If training is enabled, add to buffer with smart memory management
            # We create transition (s_t-1, a_t-1, r_t, s_t) using prev_obs from last step
            should_train_after_add = False
            if self.training_enabled and self.prev_obs is not None and self.last_action is not None:
                try:
                    buffer_size = len(self.training_buffer)

                    # Check if we need to reset buffer due to memory limit
                    if buffer_size >= self.memory_limit:
                        self.training_stats['memory_resets'] += 1
                        self.get_logger().info(f"Buffer reached memory limit ({buffer_size}/{self.memory_limit}), resetting for memory management - Reset #{self.training_stats['memory_resets']}")
                        self.training_buffer.reset()
                        buffer_size = 0

                    # Only add to buffer if there's space
                    if buffer_size < self.training_buffer.buffer_size:
                        with th.no_grad():
                            # Add transition to rollout buffer using prev_obs (s_t-1)
                            self.training_buffer.add(
                                obs=self.prev_obs.reshape(1, -1),
                                actions=self.last_action.reshape(1, -1),
                                rewards=np.array([self.last_reward or 0.0]),
                                episode_starts=self.episode_start,
                                values=self.last_values,
                                log_probs=self.last_log_probs
                            )
                            self.episode_start = np.zeros(1, dtype=bool)
                            if len(self.training_buffer) >= self.training_buffer.buffer_size:
                                should_train_after_add = True
                except Exception as e:
                    self.get_logger().error(f"Error adding to training buffer: {e}")

            if should_train_after_add:
                self.get_logger().info("Training buffer reached capacity; triggering PPO update")
                self.train_model()

            # Store current values for next iteration (shift current -> previous)
            self.prev_obs = self.current_obs  # Current becomes previous for next step
            self.last_action = actions_np
            self.last_values = values
            self.last_log_probs = log_probs

            # Add to rollout data if collection is enabled
            if self.rollout_collection_enabled:
                self._append_rollout_step(flat_state, actions_np)

            # Publish action to environment
            action_flat = DataConverter.ppo_output_to_actions(actions_np, self.num_agents)
            action_msg = Float32MultiArray(data=action_flat.tolist())
            self.action_pub.publish(action_msg)

            if self.debug_logging:
                self.get_logger().info(f"Published PPO action: {actions_np.tolist()}", throttle_duration_sec=5.0)

        except Exception as e:
            self.get_logger().error(f"Error in _predict_and_publish: {str(e)}")

    def _append_transition(self, next_state: np.ndarray):
        """Append a (state, action, reward, next_state) transition to the rollout buffer.
        Uses defaults if some fields are not yet available (e.g., reward)."""
        if self.last_state is None or self.last_action is None:
            return  # Need at least previous state and action to form a transition

        reward = float(self.last_reward) if self.last_reward is not None else 0.0

        # Format the state and next_state arrays into structured dictionaries
        structured_state = self._format_state_for_logging(self.last_state)
        structured_next_state = self._format_state_for_logging(next_state)

        # Format actions into a structured dictionary
        structured_action = self._format_action_for_logging(self.last_action)

        transition = {
            "episode": int(self.episode_id),
            "step": int(self.step_idx),
            "state": structured_state,
            "action": structured_action,
            "reward": reward,
            "next_state": structured_next_state,
        }
        self.rollout_data.append(transition)
        self.step_idx += 1

    def _format_state_for_logging(self, state_array):
        """Convert flat state array to structured dictionary."""
        try:
            # Calculate indices for different parts of the state
            # Use parameterized observation dimensions instead of hardcoded values
            agent_state_size = self.obs_dim_per_agent  # [x, y, yaw, vx, vy, vyaw] by default
            agent_desired_pos_size = 2  # [desired_x, desired_y]

            # Start and end indices
            agents_end = self.num_agents * agent_state_size
            desired_pos_end = agents_end + (self.num_agents * agent_desired_pos_size)

            # Format agent states
            agents = []
            for i in range(self.num_agents):
                start_idx = i * agent_state_size
                agent_data = {
                    "id": i,
                    "position": {
                        "x": float(state_array[start_idx]),
                        "y": float(state_array[start_idx + 1]),
                        "yaw": float(state_array[start_idx + 2])
                    },
                    "velocity": {
                        "vx": float(state_array[start_idx + 3]),
                        "vy": float(state_array[start_idx + 4]),
                        "vyaw": float(state_array[start_idx + 5])
                    }
                }

                # Add desired position
                desired_start = agents_end + (i * agent_desired_pos_size)
                agent_data["desired_position"] = {
                    "x": float(state_array[desired_start]),
                    "y": float(state_array[desired_start + 1])
                }
                agents.append(agent_data)

            # Format formation data
            formation_data = {
                "centroid": {
                    "x": float(state_array[desired_pos_end]),
                    "y": float(state_array[desired_pos_end + 1])
                },
                "error": float(state_array[desired_pos_end + 2]),
                "cross_track_error": float(state_array[desired_pos_end + 3]),
                "virtual_leader": {
                    "x": float(state_array[desired_pos_end + 4]),
                    "y": float(state_array[desired_pos_end + 5])
                }
            }

            return {
                "agents": agents,
                "formation": formation_data,
                "raw": state_array.tolist()  # Keep raw data for backward compatibility
            }
        except Exception as e:
            self.get_logger().error(f"Error formatting state for logging: {e}")
            # Return raw array if formatting fails
            return state_array.tolist()

    def _format_action_for_logging(self, action_array):
        """Convert flat action array to structured dictionary."""
        try:
            actions = []
            # The issue is that action_array has extra dimensions - flatten it
            action_array = action_array.flatten()

            for i in range(self.num_agents):
                # Use parameterized action dimensions instead of hardcoded 2
                start_idx = i * self.action_dim_per_agent
                actions.append({
                    "agent_id": i,
                    "vyaw_rate": float(action_array[start_idx]),
                    "forward_acceleration": float(action_array[start_idx + 1])
                })
            return {
                "actions": actions,
                "raw": action_array.tolist()  # Keep raw data for backward compatibility
            }
        except Exception as e:
            self.get_logger().error(f"Error formatting action for logging: {e}")
            # Return raw array if formatting fails
            return action_array.tolist()

    def _save_rollout(self, final: bool):
        """Save current episode data to a separate file."""
        if not self.rollout_data:
            return False

        try:
            # Create episode filename
            episode_filename = os.path.join(self.session_dir, f"episode_{self.episode_id}.json")
            
            # Create episode data structure
            episode_data = {
                "episode_id": self.episode_id,
                "start_time": time.time(),
                "start_time_formatted": time.strftime("%H:%M:%S"),
                "is_complete": final,
                "num_transitions": len(self.rollout_data),
                "transitions": self.rollout_data
            }

            # Write to file (single write operation)
            with open(episode_filename, 'w') as f:
                json.dump(episode_data, f, indent=2)

            self.get_logger().info(f"Saved episode {self.episode_id} to {episode_filename} ({len(self.rollout_data)} transitions)")
            return True
        except Exception as e:
            self.get_logger().error(f"Failed to save episode {self.episode_id}: {e}")
            return False

    def _finalize_episode(self, reason: str = 'done'):
        """Save and reset buffers for the next episode."""
        transition_count = len(self.rollout_data)
        valid_episode = transition_count >= self.min_episode_transitions
        if valid_episode:
            self._save_rollout(final=True)
        elif transition_count > 0:
            self.get_logger().warn(
                f"Skipping rollout save for episode {self.episode_id}: "
                f"{transition_count}/{self.min_episode_transitions} transitions "
                f"(reason={reason})"
            )
            if self.training_enabled and hasattr(self, 'training_buffer') and len(self.training_buffer) > 0:
                self.get_logger().warn(
                    "Resetting PPO buffer after short episode to avoid training on reset/out-of-bounds noise"
                )
                self.training_buffer.reset()

        # Reset episodic buffers
        self.rollout_data = []
        self._reset_transition_state()
        self.step_idx = 0
        self.episode_id += 1
        self.get_logger().info(f"Starting episode {self.episode_id}")
        return valid_episode, transition_count

    def collect_rollouts(self, rollout_dir: str | None = None, save_every_steps: int | None = None):
        """Configure where to save rollouts collected from ROS topics.

        Args:
            rollout_dir: Directory to write session folders; defaults to the `rollout_dir` parameter.
            save_every_steps: Deprecated - episodes are now saved individually at completion.
        """
        if rollout_dir is not None:
            self.rollout_dir = os.path.expanduser(rollout_dir)
            os.makedirs(self.rollout_dir, exist_ok=True)
        if save_every_steps is not None:
            self.get_logger().warn("save_every_steps is deprecated - episodes are saved individually")
        self.get_logger().info(
            f"Rollout collection configured: dir={self.rollout_dir}, session={self.session_id}"
        )

    def reward_callback(self, msg):
        self.last_reward = msg.data
        self.current_episode_reward += msg.data

    def _log_debug_stats(self):
        """Periodic lightweight stats to spot collapse/spin issues."""
        try:
            action_mean = None
            action_std = None
            yaw_action_mean = None
            surge_action_mean = None
            if self.last_action is not None:
                action_mean = float(np.mean(self.last_action))
                action_std = float(np.std(self.last_action))
                action_pairs = np.asarray(self.last_action, dtype=np.float32).reshape(-1, self.action_dim_per_agent)
                if action_pairs.shape[1] >= 2:
                    yaw_action_mean = float(np.mean(action_pairs[:, 0]))
                    surge_action_mean = float(np.mean(action_pairs[:, 1]))

            buffer_pos = None
            if self.training_enabled and hasattr(self, 'training_buffer'):
                buffer_pos = int(self.training_buffer.pos)

            msg_parts = [
                f"episode={self.episode_id}",
                f"step={self.step_idx}",
                f"last_reward={self.last_reward if self.last_reward is not None else 'n/a'}",
                f"rollout_len={len(self.rollout_data)}",
                f"buffer_pos={buffer_pos if buffer_pos is not None else 'n/a'}",
                f"action_mean={action_mean if action_mean is not None else 'n/a'}",
                f"action_std={action_std if action_std is not None else 'n/a'}",
                f"yaw_action_mean={yaw_action_mean if yaw_action_mean is not None else 'n/a'}",
                f"surge_action_mean={surge_action_mean if surge_action_mean is not None else 'n/a'}"
            ]
            self.get_logger().info("[ppo_debug] " + " | ".join(msg_parts))
        except Exception as e:
            self.get_logger().warn(f"[ppo_debug] failed to log stats: {e}")

    def done_callback(self, msg):
        """Handle episode completion for training and rollout collection."""
        if not msg.data:
            self.done_latched = False
            self.dones = np.zeros(1, dtype=bool)
            return

        if msg.data:
            if self.done_latched:
                self.get_logger().info('Ignoring repeated latched done.', throttle_duration_sec=2.0)
                return

            if self.reset_in_progress or self.awaiting_reset_state:
                self.get_logger().info('Ignoring repeated done while reset is pending.', throttle_duration_sec=2.0)
                return

            if self.reset_grace_remaining > 0:
                self.get_logger().info('Ignoring stale done during post-reset grace.', throttle_duration_sec=2.0)
                return

            self.done_latched = True
            self.dones = np.ones(1, dtype=bool)

            episode_reward = float(self.current_episode_reward)
            self.episode_rewards.append(episode_reward)
            self.get_logger().info(f"Episode {self.episode_id} finished with total reward: {episode_reward:.4f}")
            self.current_episode_reward = 0.0

            if self.training_enabled:
                # Set episode_start flag for the NEXT step
                self.episode_start = np.ones(1, dtype=bool)

            # Finalize episode for rollout collection
            valid_episode, transition_count = self._finalize_episode(reason='done')
            completed_episode = self.episode_id - 1
            self.get_logger().info(
                f'Episode {completed_episode} finished '
                f'({transition_count} transitions, valid_for_checkpoint={valid_episode})'
            )

            if self.training_enabled:
                if valid_episode and self.save_on_improvement and episode_reward > self.best_episode_reward:
                    self.best_episode_reward = episode_reward
                    self._save_model(f"new best reward {episode_reward:.4f}", best=True)
                elif not valid_episode:
                    self.get_logger().info(
                        f"Episode {completed_episode} too short for best-model comparison "
                        f"({transition_count}/{self.min_episode_transitions})"
                    )

                if (
                    valid_episode
                    and self.auto_save_enabled
                    and int(self.save_frequency) > 0
                    and (self.episode_id - self.last_save_episode) >= int(self.save_frequency)
                ):
                    self._save_model(f"periodic episode {completed_episode}")

            if self.training_enabled and self.episode_id > 0 and self.episode_id % 3 == 0:
                # Try to train every 3 episodes
                self.train_model()

            # Limit number of episodes if needed
            if self.episode_id >= self.max_episodes:
                self.get_logger().info(f"Reached maximum episodes ({self.max_episodes}). Shutting down...")
                self._save_model("max episodes reached")
                rclpy.shutdown()
                return

            self._begin_reset_wait()
            if not self.reset_environment():
                self.reset_in_progress = False
                self.awaiting_reset_state = False

    def _initialize_session_metadata(self):
        """Create metadata file for the training session."""
        metadata_filename = os.path.join(self.session_dir, "metadata.json")
        metadata = {
            "session_id": self.session_id,
            "start_time": self.session_start_time.to_msg().sec,
            "start_time_formatted": time.strftime(
                "%Y-%m-%d %H:%M:%S",
                time.localtime(self.session_start_time.to_msg().sec)
            ),
            "num_agents": self.num_agents,
            "model_path": self.model_path if self.model_path else "new_model",
            "training_enabled": self.training_enabled,
            "max_episodes": self.max_episodes,
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "use_extended_observation": self.use_extended_observation,
            "normalize_observation": self.normalize_observation,
            "reset_grace_steps": self.reset_grace_steps,
            "min_episode_transitions": self.min_episode_transitions,
            "train_frequency": self.train_frequency,
            "min_training_size": self.min_training_size,
            "train_interval_sec": self.train_interval_sec,
            "target_kl": self.target_kl,
            "initial_surge_action_bias": self.initial_surge_action_bias,
            "constant_action_enabled": self.constant_action_enabled,
            "constant_yaw_action": self.constant_yaw_action,
            "constant_surge_action": self.constant_surge_action
        }

        try:
            with open(metadata_filename, 'w') as f:
                json.dump(metadata, f, indent=2)
            self.get_logger().info(f"Initialized session metadata: {metadata_filename}")
        except Exception as e:
            self.get_logger().error(f"Failed to initialize session metadata: {e}")

    def train_model(self):
        """Train the PPO model on collected transitions with progressive approach."""
        if not self.training_enabled:
            return
        if self.training_in_progress:
            self.get_logger().info("Training already in progress; skipping overlapping request")
            return

        # Check how many valid transitions we have
        buffer_size = len(self.training_buffer)

        # Log buffer status with memory information
        memory_usage_kb = (buffer_size * 76) / 1024  # Approximate memory usage
        self.get_logger().info(f"Buffer status: {buffer_size}/{self.buffer_size} transitions ({memory_usage_kb:.1f} KB)")

        if buffer_size < self.min_training_size:
            self.get_logger().info(f"Not enough data for training: {buffer_size}/{self.min_training_size} minimum required")
            return

        self.get_logger().info(f"Training PPO model on {buffer_size} transitions")

        self.training_in_progress = True
        try:
            # Compute returns and advantages
            last_values = th.zeros(1, device=self.model.device)
            self.training_buffer.compute_returns_and_advantage(last_values=last_values, dones=self.dones)

            # Set training mode
            self.model.policy.train()

            # Learning rate schedule
            progress_remaining = max(0.0, 1.0 - (self.episode_id / 1000.0))
            current_lr = self.model.learning_rate

            # Update optimizer learning rate
            for param_group in self.model.optimizer.param_groups:
                param_group["lr"] = current_lr

            # Train for multiple epochs
            clip_range = self.model.clip_range

            for epoch in range(self.n_epochs):
                approx_kl_divs = []

                # Process minibatches
                for rollout_data in self.training_buffer.get(self.batch_size):
                    actions = rollout_data.actions

                    # Evaluate actions
                    values, log_probs, entropy = self.model.policy.evaluate_actions(
                        rollout_data.observations, actions
                    )
                    values = values.flatten()
                    log_probs = log_probs.flatten()
                    old_log_probs = rollout_data.old_log_prob.flatten()

                    # Normalize advantage
                    advantages = rollout_data.advantages.flatten()
                    if len(advantages) > 1:
                        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                    # PPO loss
                    ratio = th.exp(log_probs - old_log_probs)
                    policy_loss_1 = advantages * ratio
                    policy_loss_2 = advantages * th.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
                    policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                    # Value loss (no clipping for simplicity)
                    value_loss = th.nn.functional.mse_loss(rollout_data.returns, values)

                    # Entropy loss
                    if entropy is None:
                        entropy_loss = -th.mean(-log_probs)
                    else:
                        entropy_loss = -th.mean(entropy)

                    # Total loss
                    loss = policy_loss + self.model.ent_coef * entropy_loss + self.model.vf_coef * value_loss

                    # Gradient step
                    self.model.optimizer.zero_grad()
                    loss.backward()
                    # Clip grad norm
                    th.nn.utils.clip_grad_norm_(self.model.policy.parameters(), self.model.max_grad_norm)
                    self.model.optimizer.step()

                    # Log statistics
                    with th.no_grad():
                        log_ratio = log_probs - old_log_probs
                        approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                        approx_kl_divs.append(approx_kl_div)

                mean_kl = np.mean(approx_kl_divs)
                self.get_logger().info(
                    f"Epoch {epoch+1}/{self.n_epochs}, approx_kl={mean_kl:.6f}, lr={current_lr:.6f}"
                )

                # Early stopping
                if self.model.target_kl is not None and mean_kl > 1.5 * self.model.target_kl:
                    self.get_logger().info(f"Early stopping at epoch {epoch+1} due to reaching max KL: {mean_kl:.6f}")
                    break

            # Update training statistics
            self.model._n_updates += 1
            self.training_stats['total_training_sessions'] += 1
            self.training_stats['total_transitions_trained'] += buffer_size
            self.training_stats['average_buffer_size_at_training'] = (
                self.training_stats['total_transitions_trained'] /
                self.training_stats['total_training_sessions']
            )

            # Log training statistics
            self.get_logger().info(
                f"Training complete. Sessions: {self.training_stats['total_training_sessions']}, "
                f"Total transitions: {self.training_stats['total_transitions_trained']}, "
                f"Avg buffer size: {self.training_stats['average_buffer_size_at_training']:.1f}"
            )

            self.training_buffer.reset()
            self.get_logger().info("Reset on-policy rollout buffer after PPO update")
        except Exception as e:
            self.get_logger().error(f"Error during training: {e}")
        finally:
            self.training_in_progress = False

    def _append_rollout_step(self, state: np.ndarray, action: np.ndarray):
        """Add the current state and action to the rollout data.
        
        This method is called from _predict_and_publish to collect state-action 
        pairs for later analysis.
        """
        # Store the current state for later use
        if self.last_state is not None:
            # If we have a previous state, add a full transition
            self._append_transition(state)

        # Update for next time
        self.last_state = state.copy()
        self.last_action = action.copy()

    def reset_environment(self):
        """Request environment reset when agents are out of bounds."""
        if not self.reset_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('Reset service not available, continuing without reset')
            return False

        request = Trigger.Request()
        future = self.reset_client.call_async(request)

        # Setup callback for when reset is complete
        future.add_done_callback(self._reset_done_callback)
        return True

    def _reset_done_callback(self, future):
        """Handle completion of reset service call."""
        try:
            response = future.result()
            if response.success:
                self.get_logger().info('Environment reset successful')
                self._reset_transition_state()
                self.reset_in_progress = False
                self.awaiting_reset_state = True
                self.reset_grace_remaining = self.reset_grace_steps
                self.done_latched = False
                self.dones = np.zeros(1, dtype=bool)
            else:
                self.get_logger().warn(f'Environment reset failed: {response.message}')
                self.reset_in_progress = False
                self.awaiting_reset_state = False
        except Exception as e:
            self.get_logger().error(f'Error in reset callback: {e}')
            self.reset_in_progress = False
            self.awaiting_reset_state = False

    def _agents_out_of_bounds(self, flat_state: np.ndarray) -> bool:
        """Check if any agent is out of bounds."""
        # Define boundaries - updated to match our new coordinate system
        MIN_X, MAX_X = -15.0, 35.0  # Wider than our clipping bounds
        MIN_Y, MAX_Y = -15.0, 35.0  # Wider than our clipping bounds

        for i in range(self.num_agents):
            # Extract agent position (x, y) - uses parameterized dimensions
            agent_x = flat_state[i * self.obs_dim_per_agent]
            agent_y = flat_state[i * self.obs_dim_per_agent + 1]

            # Check if out of bounds
            if (agent_x < MIN_X or agent_x > MAX_X or
                agent_y < MIN_Y or agent_y > MAX_Y):
                return True
        return False

    def check_and_reset(self):
        """Periodically check if agents are stuck and need reset."""
        if not hasattr(self, 'last_position'):
            self.last_position = None
            return

        if self.last_position is not None and self.current_obs is not None:
            # Check if agents haven't moved (using parameterized dimensions)
            # Only check position data for all agents (first obs_dim_per_agent * num_agents elements)
            obs_size = self.obs_dim_per_agent * self.num_agents
            if np.allclose(self.last_position, self.current_obs[:obs_size], atol=0.1):
                self.get_logger().warn("Agents appear stuck. Requesting environment reset")
                self.reset_environment()

        if self.current_obs is not None:
            # Store only the position data for comparison
            obs_size = self.obs_dim_per_agent * self.num_agents
            self.last_position = self.current_obs[:obs_size].copy()

    def _resolve_existing_model_path(self, path):
        if not path:
            return None

        expanded = os.path.expanduser(path)
        if os.path.exists(expanded):
            return expanded

        if expanded.endswith('.zip'):
            legacy_path = expanded[:-4] + '.pth'
            if os.path.exists(legacy_path):
                return legacy_path
        elif expanded.endswith('.pth'):
            zip_path = expanded[:-4] + '.zip'
            if os.path.exists(zip_path):
                return zip_path

        return None

    def _find_and_load_best_model(self):
        """Smart model loading: try multiple sources in order of preference.
        
        Returns the path of the loaded model, or None if no model was loaded.
        """
        model_candidates = []

        seen_paths = set()

        def add_candidate(path, description):
            resolved = self._resolve_existing_model_path(path)
            if resolved and resolved not in seen_paths:
                seen_paths.add(resolved)
                model_candidates.append((resolved, description))

        # 1. First priority: explicit model_path parameter
        if self.model_path:
            add_candidate(self.model_path, "explicit parameter")
            if not model_candidates:
                self.get_logger().warn(f"model_path was provided but does not exist: {self.model_path}")

        if self.autoload_model:
            # 2. Second priority: latest/best models in rollout_dir
            add_candidate(os.path.join(self.rollout_dir, "latest_model.zip"), "latest model")
            add_candidate(os.path.join(self.rollout_dir, "latest_model.pth"), "legacy latest model")
            add_candidate(os.path.join(self.rollout_dir, "best_model.zip"), "best model")
            add_candidate(os.path.join(self.rollout_dir, "best_model.pth"), "legacy best model")

            # 3. Third priority: latest episode model in rollout directory
            if os.path.exists(self.rollout_dir):
                episode_models = []
                for filename in os.listdir(self.rollout_dir):
                    if filename.startswith('ppo_model_ep') and (filename.endswith('.zip') or filename.endswith('.pth')):
                        try:
                            episode_num = int(filename.split('ep')[1].split('.')[0])
                            full_path = os.path.join(self.rollout_dir, filename)
                            episode_models.append((episode_num, full_path))
                        except (ValueError, IndexError):
                            continue

                if episode_models:
                    episode_models.sort(key=lambda x: x[0], reverse=True)
                    latest_episode_model = episode_models[0][1]
                    add_candidate(latest_episode_model, f"latest episode model (ep {episode_models[0][0]})")

                add_candidate(os.path.join(self.rollout_dir, "ppo_model_final.zip"), "final model")
                add_candidate(os.path.join(self.rollout_dir, "ppo_model_final.pth"), "legacy final model")

            # 4. Last priority: common locations
            common_paths = [
                os.path.expanduser("~/Desktop/PPO_Rollouts/latest_model.zip"),
                os.path.expanduser("~/Desktop/PPO_Rollouts/ppo_model_final.zip"),
                "./latest_model.zip",
                "./ppo_model.zip",
                "./models/latest_model.zip",
                "./models/ppo_model.zip"
            ]

            for path in common_paths:
                add_candidate(path, f"common location: {path}")

        # Try to load the first available model
        for model_path, description in model_candidates:
            try:
                self.get_logger().info(f"Attempting to load model from {description}: {model_path}")
                # Load model without environment - we'll validate spaces separately
                obs_dim = self.obs_dim
                action_dim = self.action_dim
                
                try:
                    # Try to load as CustomPPO first
                    self.model = CustomPPO.load(model_path, obs_dim=obs_dim, action_dim=action_dim, device='cpu')
                except Exception as e:
                    self.get_logger().warn(f"Could not load model from {model_path}: {e}")
                    continue  # Try next model candidate

                # Verify the model loaded correctly and spaces match
                if hasattr(self.model, 'policy') and self.model.policy is not None:
                    # STRICT validation: spaces must match EXACTLY for homogeneity
                    model_obs_shape = self.model.observation_space.shape
                    model_act_shape = self.model.action_space.shape
                    expected_obs_shape = self.observation_space.shape
                    expected_act_shape = self.action_space.shape
                    
                    # Check for EXACT match (strict homogeneity)
                    if (model_obs_shape == expected_obs_shape and 
                        model_act_shape == expected_act_shape):
                        self.get_logger().info(f"✓ Model spaces match exactly: obs={model_obs_shape}, action={model_act_shape}")
                        return model_path
                    else:
                        # Reject incompatible models for integrity
                        self.get_logger().error(f"✗ INCOMPATIBLE MODEL - Space shapes don't match:")
                        self.get_logger().error(f"  Expected: obs={expected_obs_shape}, action={expected_act_shape}")
                        self.get_logger().error(f"  Got:      obs={model_obs_shape}, action={model_act_shape}")
                        self.get_logger().error(f"  → Model trained with different space format")
                        self.get_logger().error(f"  → Please retrain model or use compatible checkpoint")
                        # Don't return this model - try next candidate
                        continue
                else:
                    self.get_logger().warn(f"✗ Model loaded but appears invalid from {description}")

            except Exception as e:
                self.get_logger().warn(f"✗ Failed to load model from {description}: {str(e)}")
                continue

        # No model could be loaded
        return None

    def _save_model(self, reason: str = "manual", best: bool = False):
        """Save the current model with proper logging."""
        if not self.training_enabled:
            self.get_logger().info(f"Skipping model save ({reason}) because training is disabled")
            return

        try:
            # Save as latest_model.zip
            latest_model_path = os.path.join(self.rollout_dir, "latest_model.zip")
            self.model.save(latest_model_path)
            
            # Save timestamped version
            timestamp_model_path = os.path.join(self.rollout_dir, f"ppo_model_ep{self.episode_id}.zip")
            self.model.save(timestamp_model_path)

            if best:
                best_model_path = os.path.join(self.rollout_dir, "best_model.zip")
                self.model.save(best_model_path)
                self.get_logger().info(f"Best model saved: {best_model_path}")
            
            self.get_logger().info(f"Model saved ({reason}): {latest_model_path}")
            self.get_logger().info(f"Checkpoint saved: {timestamp_model_path}")
            self.last_save_episode = self.episode_id
            
        except Exception as e:
            self.get_logger().error(f"Failed to save model: {e}")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals by saving model and cleaning up."""
        if self._shutting_down:
            os._exit(0)

        self._shutting_down = True
        self.get_logger().info(f"Received signal {signum}, shutting down PPO node...")
        if self.training_enabled and hasattr(self, 'model') and self.model is not None:
            self._save_model("interrupt/shutdown")
        
        # Graceful shutdown
        if hasattr(self, 'training_timer'):
            self.training_timer.cancel()
        if hasattr(self, 'reset_timer'):
            self.reset_timer.cancel()
            
        self.get_logger().info("Graceful shutdown complete")
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except RuntimeError:
            pass
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)

    def get_training_insights(self):
        """Get current training performance insights."""
        if not self.training_enabled:
            return "Training disabled"

        buffer_size = len(self.training_buffer) if hasattr(self, 'training_buffer') else 0
        memory_usage_kb = (buffer_size * 76) / 1024

        insights = {
            'buffer_status': f"{buffer_size}/{self.buffer_size if hasattr(self, 'buffer_size') else 100}",
            'memory_usage_kb': f"{memory_usage_kb:.1f} KB",
            'training_sessions': self.training_stats['total_training_sessions'],
            'total_transitions': self.training_stats['total_transitions_trained'],
            'memory_resets': self.training_stats['memory_resets'],
            'ready_for_training': buffer_size >= (self.min_training_size if hasattr(self, 'min_training_size') else 50),
            'last_save_episode': self.last_save_episode,
            'episodes_since_save': self.episode_id - self.last_save_episode,
            'best_reward': self.best_episode_reward if self.best_episode_reward != float('-inf') else 'N/A'
        }

        return insights

def main(args=None):
    rclpy.init(args=args)
    node = PPONode()  # Updated to use renamed class (Point 8)
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
