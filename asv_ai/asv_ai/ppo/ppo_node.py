#!/usr/bin/env python3
import json
import os
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
    def __init__(self):
        super().__init__('ppo_node')  # Renamed from asv_ppo_node for generalization

        # Parameters - Environment Configuration
        self.declare_parameter('num_agents', 2)
        self.declare_parameter('model_path', '')
        self.declare_parameter('rollout_dir', os.path.expanduser('~/Desktop/PPO_Rollouts'))
        self.declare_parameter('rollout_save_every', 500)
        self.declare_parameter('rollout_collection_enabled', True)
        
        # Parameters - State/Action Space Dimensions (generalizable for any robot/environment)
        self.declare_parameter('obs_dim_per_agent', 6)  # Default: [x, y, yaw, vx, vy, vyaw]
        self.declare_parameter('action_dim_per_agent', 2)  # Default: [vyaw_rate, acceleration]

        # Resolve parameters
        self.num_agents = self.get_parameter('num_agents').value
        self.model_path = self.get_parameter('model_path').value
        self.rollout_dir = os.path.expanduser(self.get_parameter('rollout_dir').value)
        self.rollout_save_every = int(self.get_parameter('rollout_save_every').value)
        self.rollout_collection_enabled = self.get_parameter('rollout_collection_enabled').value
        
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
        self.declare_parameter('n_epochs', 10)  # PPO training epochs
        self.declare_parameter('batch_size', 64)  # PPO batch size
        self.declare_parameter('gamma', 0.99)  # Discount factor
        self.declare_parameter('gae_lambda', 0.95)  # GAE lambda
        self.declare_parameter('max_episodes', 100000)
        
        # Model saving parameters
        self.declare_parameter('save_frequency', 500)  # Save every N episodes
        self.declare_parameter('auto_save_enabled', True)  # Enable periodic saving
        self.declare_parameter('save_on_improvement', True)  # Save when reward improves

        # Get training parameters
        self.training_enabled = self.get_parameter('training_enabled').value
        self.train_frequency = self.get_parameter('train_frequency').value
        self.n_epochs = self.get_parameter('n_epochs').value
        self.batch_size = self.get_parameter('batch_size').value
        self.gamma = self.get_parameter('gamma').value
        self.gae_lambda = self.get_parameter('gae_lambda').value
        self.max_episodes = self.get_parameter('max_episodes').value
        
        # Get model saving parameters
        self.save_frequency = self.get_parameter('save_frequency').value
        self.auto_save_enabled = self.get_parameter('auto_save_enabled').value
        self.save_on_improvement = self.get_parameter('save_on_improvement').value
        
        # Model saving state
        self.best_episode_reward = float('-inf')
        self.last_save_episode = 0

        # Publishers and Subscribers FIRST to avoid missing early messages
        # Use QoS depth=1 to only process the most recent message and avoid lag
        self.state_sub = self.create_subscription(Float32MultiArray, '/environment/state', self.state_callback, 1)
        self.reward_sub = self.create_subscription(Float32, '/environment/reward', self.reward_callback, 1)
        self.done_sub = self.create_subscription(Bool, '/environment/done', self.done_callback, 1)

        self.action_pub = self.create_publisher(Float32MultiArray, '/ppo/action', 1)

        self.reset_client = self.create_client(Trigger, '/environment/reset')

        # Define observation and action spaces for PPO model (generalizable via parameters)
        # These stay in PPO node as they are model-specific, not environment-specific
        obs_dim = self.num_agents * self.obs_dim_per_agent
        action_dim = self.num_agents * self.action_dim_per_agent

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

        # Smart model loading: try multiple sources in order of preference
        # loaded_model_path = self._find_and_load_best_model()
        # if loaded_model_path:
        #     self.get_logger().info(f'Successfully loaded model from {loaded_model_path}')
        # else:
        # Create new CustomPPO model
        self.model = CustomPPO(
            obs_dim=obs_dim,
            action_dim=action_dim,
            device='cpu',  # Can be changed to 'cuda' if GPU is available
            learning_rate=3e-4,
            clip_range=0.2,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5
        )

        self.model_ready = True
        # If we received a state while loading, process it now
        if self._pending_state is not None:
            self._predict_and_publish(self._pending_state)
            self._pending_state = None

        # Add after model initialization:
        if self.training_enabled:
            # observation_space and action_space already defined above
            
            # Create training buffer with optimized configuration
            self.buffer_size = 200  # 2x larger for better experience diversity
            self.min_training_size = 50  # Train when 50 transitions available
            self.memory_limit = 500  # Reset buffer when it hits memory limit

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
            self.episode_start = np.zeros(1, dtype=bool)  # Episode start flags

            # Create timers (SLOWED DOWN for easier debugging/visualization)
            self.training_timer = self.create_timer(30.0, self.train_model)  # 30 seconds (was 10s)

            self.reset_timer = self.create_timer(30.0, self.check_and_reset)  # 30 seconds (was 20s)

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

    def state_callback(self, msg):
        try:
            # Convert ROS message to NumPy array
            flat_state = DataConverter.ros_to_numpy(msg)

            # If state buffer is full, we need to reset it before adding more
            if hasattr(self, 'training_buffer') and self.training_buffer.full:
                self.training_buffer.reset()

            # Check if agents are out of bounds and log the problematic positions
            if self._agents_out_of_bounds(flat_state):
                problematic_positions = []
                for i in range(self.num_agents):
                    # Use parameterized dimensions instead of hardcoded values
                    agent_x = flat_state[i * self.obs_dim_per_agent]
                    agent_y = flat_state[i * self.obs_dim_per_agent + 1]
                    problematic_positions.append([agent_x, agent_y])

                self.get_logger().warn(f"Agents detected out of bounds at positions: {problematic_positions}")
                self._finalize_episode()  # Save current episode data
                self.reset_environment()  # Request reset
                return

            if not self.model_ready:
                # Buffer the latest state until model is ready
                self._pending_state = flat_state
                return

            self._predict_and_publish(flat_state)
        except Exception as e:
            self.get_logger().error(f'Error in state_callback: {str(e)}')

    def _predict_and_publish(self, flat_state: np.ndarray):
        """Process environment state, predict actions, and publish to ROS topics."""
        try:
            # Format state for the model (returns 1D flat array for homogeneity)
            agent_states = DataConverter.state_to_ppo_input(
                flat_state, 
                self.num_agents, 
                self.obs_dim_per_agent
            )
            self.current_obs = agent_states  # Already flat, no need to reshape

            # Get actions, values, and log_probs from policy network
            with th.no_grad():
                # Reshape to (1, -1) instead of keeping 2D
                obs_tensor = DataConverter.numpy_to_tensor(self.current_obs)
                obs_tensor = obs_tensor.reshape(1, -1)  # Reshape to (1, obs_dim) - batch of 1

                # PPO requires all three: actions for control, values for advantage calculation, log_probs for policy gradients
                actions, values, log_probs = self.model.policy(obs_tensor)

                # Convert to numpy - PPO handles exploration through stochastic policy sampling
                actions_np = actions.cpu().numpy()

            # If training is enabled, add to buffer with smart memory management
            # We create transition (s_t-1, a_t-1, r_t, s_t) using prev_obs from last step
            if self.training_enabled and self.prev_obs is not None and self.last_action is not None:
                try:
                    buffer_size = len(self.training_buffer)  # Use __len__ which returns self.pos

                    # Check if we need to reset buffer due to memory limit
                    if buffer_size >= self.memory_limit:
                        self.training_stats['memory_resets'] += 1
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
                except Exception as e:
                    self.get_logger().error(f"Error adding to training buffer: {e}")

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
            return

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
        except Exception as e:
            self.get_logger().error(f"Failed to save episode {self.episode_id}: {e}")

    def _finalize_episode(self):
        """Save and reset buffers for the next episode."""
        self._save_rollout(final=True)
        # Reset episodic buffers
        self.rollout_data = []
        self.last_state = None
        self.last_action = None
        self.last_reward = None
        self.step_idx = 0
        self.episode_id += 1

    def collect_rollouts(self, rollout_dir: str | None = None, save_every_steps: int | None = None):
        """Configure where to save rollouts collected from ROS topics.

        Args:
            rollout_dir: Directory to write session folders; defaults to the `rollout_dir` parameter.
            save_every_steps: Deprecated - episodes are now saved individually at completion.
        """
        if rollout_dir is not None:
            self.rollout_dir = os.path.expanduser(rollout_dir)
            os.makedirs(self.rollout_dir, exist_ok=True)

    def reward_callback(self, msg):
        self.last_reward = msg.data
        self.current_episode_reward += msg.data

    def done_callback(self, msg):
        """Handle episode completion for training and rollout collection."""
        if msg.data:
            self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0.0

            if self.training_enabled:
                # Set episode_start flag for the NEXT step
                self.episode_start = np.ones(1, dtype=bool)

            # Finalize episode for rollout collection
            self._finalize_episode()

            if self.training_enabled and self.episode_id > 0 and self.episode_id % 3 == 0:
                # Try to train every 3 episodes
                self.train_model()

            # Limit number of episodes if needed
            if self.episode_id >= self.max_episodes:
                self._save_model("max episodes reached")
                rclpy.shutdown()

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
            "max_episodes": self.max_episodes
        }

        try:
            with open(metadata_filename, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            self.get_logger().error(f"Failed to initialize session metadata: {e}")

    def train_model(self):
        """Train the PPO model on collected transitions with progressive approach."""
        if not self.training_enabled:
            return

        # Check how many valid transitions we have
        buffer_size = len(self.training_buffer)  # Use __len__ which returns self.pos

        # Progressive training: train when we have minimum viable batch (not when full)
        if buffer_size < self.min_training_size:
            return

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

                    # Normalize advantage
                    advantages = rollout_data.advantages
                    if len(advantages) > 1:
                        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                    # PPO loss
                    ratio = th.exp(log_probs - rollout_data.old_log_prob)
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
                        log_ratio = log_probs - rollout_data.old_log_prob
                        approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                        approx_kl_divs.append(approx_kl_div)

                mean_kl = np.mean(approx_kl_divs) if approx_kl_divs else 0.0

                # Early stopping
                if self.model.target_kl is not None and mean_kl > 1.5 * self.model.target_kl:
                    break

            # Update training statistics
            self.training_stats['total_training_sessions'] += 1
            self.training_stats['total_transitions_trained'] += buffer_size
            self.training_stats['average_buffer_size_at_training'] = (
                self.training_stats['total_transitions_trained'] /
                self.training_stats['total_training_sessions']
            )
            
            # Log training progress
            avg_reward = np.mean(self.episode_rewards[-10:]) if self.episode_rewards else 0.0
            self.get_logger().info(
                f"Training #{self.training_stats['total_training_sessions']}: "
                f"ep={self.episode_id}, transitions={buffer_size}, "
                f"avg_reward={avg_reward:.2f}, kl={mean_kl:.4f}"
            )

            # Smart buffer management: reset strategically
            current_buffer_size = len(self.training_buffer)
            if current_buffer_size >= self.memory_limit * 0.8:
                self.training_stats['memory_resets'] += 1
                self.training_buffer.reset()
        except Exception as e:
            self.get_logger().error(f"Error during training: {e}")

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
                # Reset internal state variables
                self.last_state = None
                self.last_action = None
                self.last_reward = None
                self.episode_start = np.ones(1, dtype=bool)
            else:
                self.get_logger().warn(f'Environment reset failed: {response.message}')
        except Exception as e:
            self.get_logger().error(f'Error in reset callback: {e}')

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

    def _find_and_load_best_model(self):
        """Smart model loading: try multiple sources in order of preference.
        
        Returns the path of the loaded model, or None if no model was loaded.
        """
        model_candidates = []

        # 1. First priority: explicit model_path parameter
        if self.model_path and os.path.exists(self.model_path):
            model_candidates.append((self.model_path, "explicit parameter"))

        # 2. Second priority: latest_model.zip in rollout directory
        latest_model_path = os.path.join(self.rollout_dir, "latest_model.zip")
        if os.path.exists(latest_model_path):
            model_candidates.append((latest_model_path, "latest model"))

        # 3. Third priority: latest episode model in rollout directory
        if os.path.exists(self.rollout_dir):
            # Look for episode-specific models (ppo_model_ep*.zip)
            episode_models = []
            for filename in os.listdir(self.rollout_dir):
                if filename.startswith('ppo_model_ep') and filename.endswith('.zip'):
                    try:
                        # Extract episode number from filename
                        episode_num = int(filename.split('ep')[1].split('.')[0])
                        full_path = os.path.join(self.rollout_dir, filename)
                        episode_models.append((episode_num, full_path))
                    except (ValueError, IndexError):
                        continue

            # Sort by episode number (latest first)
            if episode_models:
                episode_models.sort(key=lambda x: x[0], reverse=True)
                latest_episode_model = episode_models[0][1]
                model_candidates.append((latest_episode_model, f"latest episode model (ep {episode_models[0][0]})"))

            # 4. Fourth priority: final model
            final_model_path = os.path.join(self.rollout_dir, "ppo_model_final.zip")
            if os.path.exists(final_model_path):
                model_candidates.append((final_model_path, "final model"))

        # 5. Fifth priority: look in common locations
        common_paths = [
            os.path.expanduser("~/Desktop/PPO_Rollouts/latest_model.zip"),
            os.path.expanduser("~/Desktop/PPO_Rollouts/ppo_model_final.zip"),
            "./latest_model.zip",
            "./ppo_model.zip",
            "./models/latest_model.zip",
            "./models/ppo_model.zip"
        ]

        for path in common_paths:
            if os.path.exists(path):
                model_candidates.append((path, f"common location: {path}"))

        # Try to load the first available model
        for model_path, description in model_candidates:
            try:
                # Load model without environment - we'll validate spaces separately
                obs_dim = self.num_agents * self.obs_dim_per_agent
                action_dim = self.num_agents * self.action_dim_per_agent
                
                try:
                    # Try to load as CustomPPO first
                    self.model = CustomPPO.load(model_path, obs_dim=obs_dim, action_dim=action_dim, device='cpu')
                except Exception as e:
                    # For now, create a new CustomPPO model if loading fails
                    self.model = CustomPPO(obs_dim=obs_dim, action_dim=action_dim, device='cpu')
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
                        self.get_logger().info(f"Model loaded: {model_path}")
                        return model_path
                    else:
                        # Reject incompatible models for integrity
                        self.get_logger().error(f"INCOMPATIBLE MODEL: expected obs={expected_obs_shape}, action={expected_act_shape}, got obs={model_obs_shape}, action={model_act_shape}")
                        continue

            except Exception as e:
                continue

        # No model could be loaded
        return None

    def _save_model(self, reason: str = "manual"):
        """Save the current model with proper logging."""
        try:
            # Save as latest_model.zip
            latest_model_path = os.path.join(self.rollout_dir, "latest_model.zip")
            self.model.save(latest_model_path)
            
            # Save timestamped version
            timestamp_model_path = os.path.join(self.rollout_dir, f"ppo_model_ep{self.episode_id}.zip")
            self.model.save(timestamp_model_path)
            
            self.get_logger().info(f"Model saved: {latest_model_path} (ep{self.episode_id}, {reason})")
            
        except Exception as e:
            self.get_logger().error(f"Failed to save model: {e}")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals by saving model and cleaning up."""
        if hasattr(self, 'model') and self.model is not None:
            self._save_model("shutdown")
        
        # Graceful shutdown
        if hasattr(self, 'training_timer'):
            self.training_timer.cancel()
        if hasattr(self, 'reset_timer'):
            self.reset_timer.cancel()
            
        rclpy.shutdown()

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
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
