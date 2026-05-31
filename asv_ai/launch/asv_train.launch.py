from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

from launch import LaunchDescription


def generate_launch_description():
    # Declare all the launch arguments
    num_agents_arg = DeclareLaunchArgument(
        'num_agents', default_value='2',
        description='Number of ASV agents'
    )
    model_path_arg = DeclareLaunchArgument(
        'model_path', default_value='',
        description='Path to pre-trained PPO model'
    )
    train_freq_arg = DeclareLaunchArgument(
        'train_frequency', default_value='1000',
        description='Steps between training updates'
    )
    min_training_size_arg = DeclareLaunchArgument(
        'min_training_size', default_value='0',
        description='Minimum transitions before PPO update; 0 means train_frequency'
    )
    train_interval_sec_arg = DeclareLaunchArgument(
        'train_interval_sec', default_value='0.0',
        description='Optional timer-based PPO update interval; 0 disables partial timer updates'
    )
    learning_rate_arg = DeclareLaunchArgument(
        'learning_rate', default_value='0.0003',
        description='PPO learning rate'
    )
    vf_coef_arg = DeclareLaunchArgument(
        'vf_coef', default_value='0.1',
        description='PPO value loss coefficient'
    )
    initial_log_std_arg = DeclareLaunchArgument(
        'initial_log_std', default_value='-1.5',
        description='Initial PPO actor log standard deviation'
    )
    initial_surge_action_bias_arg = DeclareLaunchArgument(
        'initial_surge_action_bias', default_value='0.0',
        description='Initial deterministic surge action bias for PPO actor mean'
    )
    target_kl_arg = DeclareLaunchArgument(
        'target_kl', default_value='0.05',
        description='PPO KL early-stop threshold'
    )
    ppo_lrn_arg = DeclareLaunchArgument(
        'ppo_node.lrn', default_value='',
        description='Alias for learning_rate, kept for older launch commands'
    )
    batch_size_arg = DeclareLaunchArgument(
        'batch_size', default_value='64',
        description='PPO batch size'
    )
    n_epochs_arg = DeclareLaunchArgument(
        'n_epochs', default_value='10',
        description='PPO training epochs'
    )
    rollout_dir_arg = DeclareLaunchArgument(
        'rollout_dir', default_value='~/Desktop/PPO_Rollouts',
        description='Directory to save rollouts and models'
    )
    enable_viz_arg = DeclareLaunchArgument(
        'enable_visualization', default_value='true',
        description='Enable RViz2 visualization (disable for headless training)'
    )
    training_enabled_arg = DeclareLaunchArgument(
        'training_enabled', default_value='true',
        description='Enable PPO updates; set false to replay/evaluate a loaded model'
    )
    autoload_model_arg = DeclareLaunchArgument(
        'autoload_model', default_value='false',
        description='When model_path is empty, automatically resume the latest checkpoint from rollout_dir/common paths'
    )
    deterministic_inference_arg = DeclareLaunchArgument(
        'deterministic_inference', default_value='true',
        description='Use deterministic policy actions when training is disabled'
    )
    constant_action_enabled_arg = DeclareLaunchArgument(
        'constant_action_enabled', default_value='false',
        description='Publish a fixed action for diagnostics instead of using PPO'
    )
    constant_yaw_action_arg = DeclareLaunchArgument(
        'constant_yaw_action', default_value='0.0',
        description='Fixed normalized yaw action used when constant_action_enabled=true'
    )
    constant_surge_action_arg = DeclareLaunchArgument(
        'constant_surge_action', default_value='0.0',
        description='Fixed normalized surge action used when constant_action_enabled=true; 0 means no thrust'
    )
    use_extended_observation_arg = DeclareLaunchArgument(
        'use_extended_observation', default_value='true',
        description='Feed PPO the full environment state instead of only raw agent states'
    )
    normalize_observation_arg = DeclareLaunchArgument(
        'normalize_observation', default_value='true',
        description='Scale PPO observations to numerically stable ranges'
    )
    reset_grace_steps_arg = DeclareLaunchArgument(
        'reset_grace_steps', default_value='10',
        description='Number of post-reset states that skip out-of-bounds checks'
    )
    min_episode_transitions_arg = DeclareLaunchArgument(
        'min_episode_transitions', default_value='50',
        description='Minimum transitions required before saving/checkpointing an episode'
    )
    training_mode_arg = DeclareLaunchArgument(
        'training_mode', default_value='fast',
        description='Training mode: fast (minimal logging) or debug (full logging)'
    )
    reward_mode_arg = DeclareLaunchArgument(
        'reward_mode', default_value='path_following',
        description='Reward mode: path_following for training or stillness for legacy behavior'
    )
    max_episode_steps_arg = DeclareLaunchArgument(
        'max_episode_steps', default_value='1000',
        description='Maximum environment reward ticks before ending an episode'
    )
    max_episodes_arg = DeclareLaunchArgument(
        'max_episodes', default_value='100000',
        description='Maximum PPO episodes before shutdown'
    )
    save_frequency_arg = DeclareLaunchArgument(
        'save_frequency', default_value='25',
        description='Autosave PPO checkpoint every N completed episodes'
    )
    goal_progress_delta_arg = DeclareLaunchArgument(
        'goal_progress_delta', default_value='12.0',
        description='Path progress required to complete an episode in path_following mode'
    )
    reset_offset_range_arg = DeclareLaunchArgument(
        'reset_offset_range', default_value='2.0',
        description='Random reset offset range around each formation target'
    )
    min_reset_separation_arg = DeclareLaunchArgument(
        'min_reset_separation', default_value='2.0',
        description='Minimum distance between agents after reset'
    )
    reset_yaw_mode_arg = DeclareLaunchArgument(
        'reset_yaw_mode', default_value='random',
        description='Reset yaw sampling mode: random or path_aligned'
    )
    reset_yaw_noise_arg = DeclareLaunchArgument(
        'reset_yaw_noise', default_value='3.14159',
        description='Yaw noise half-range in radians when reset_yaw_mode=path_aligned'
    )
    path_target_speed_arg = DeclareLaunchArgument(
        'path_target_speed', default_value='0.08',
        description='Target forward path speed used to penalize near-stationary path-following'
    )
    path_low_speed_penalty_scale_arg = DeclareLaunchArgument(
        'path_low_speed_penalty_scale', default_value='0.8',
        description='Penalty scale applied when path_speed is below path_target_speed'
    )
    formation_distance_arg = DeclareLaunchArgument(
        'formation_distance', default_value='3.0',
        description='Formation radius around the virtual leader'
    )
    formation_penalty_scale_arg = DeclareLaunchArgument(
        'formation_penalty_scale', default_value='0.35',
        description='Penalty scale for mean formation error'
    )
    formation_bonus_scale_arg = DeclareLaunchArgument(
        'formation_bonus_scale', default_value='0.0',
        description='Reward bonus scale for tight formation'
    )
    formation_bonus_width_arg = DeclareLaunchArgument(
        'formation_bonus_width', default_value='1.0',
        description='Formation error width used by the tight-formation bonus'
    )
    simulation_hz_arg = DeclareLaunchArgument(
        'simulation_hz', default_value='20.0',
        description='Fixed simulation update rate for each ASV agent'
    )
    tf_publish_hz_arg = DeclareLaunchArgument(
        'tf_publish_hz', default_value='20.0',
        description='Maximum TF publish rate for each ASV agent'
    )
    time_scale_arg = DeclareLaunchArgument(
        'time_scale', default_value='1.0',
        description='Simulated seconds advanced per real second at the configured simulation_hz'
    )

    # Get the launch configurations
    num_agents = LaunchConfiguration('num_agents')
    model_path = LaunchConfiguration('model_path')
    train_frequency = LaunchConfiguration('train_frequency')
    min_training_size = LaunchConfiguration('min_training_size')
    train_interval_sec = LaunchConfiguration('train_interval_sec')
    learning_rate = PythonExpression([
        "'", LaunchConfiguration('ppo_node.lrn'), "' if '",
        LaunchConfiguration('ppo_node.lrn'), "' else '",
        LaunchConfiguration('learning_rate'), "'"
    ])
    vf_coef = LaunchConfiguration('vf_coef')
    initial_log_std = LaunchConfiguration('initial_log_std')
    initial_surge_action_bias = LaunchConfiguration('initial_surge_action_bias')
    target_kl = LaunchConfiguration('target_kl')
    batch_size = LaunchConfiguration('batch_size')
    n_epochs = LaunchConfiguration('n_epochs')
    rollout_dir = LaunchConfiguration('rollout_dir')
    enable_viz = LaunchConfiguration('enable_visualization')
    training_enabled = LaunchConfiguration('training_enabled')
    autoload_model = LaunchConfiguration('autoload_model')
    deterministic_inference = LaunchConfiguration('deterministic_inference')
    constant_action_enabled = LaunchConfiguration('constant_action_enabled')
    constant_yaw_action = LaunchConfiguration('constant_yaw_action')
    constant_surge_action = LaunchConfiguration('constant_surge_action')
    use_extended_observation = LaunchConfiguration('use_extended_observation')
    normalize_observation = LaunchConfiguration('normalize_observation')
    reset_grace_steps = LaunchConfiguration('reset_grace_steps')
    min_episode_transitions = LaunchConfiguration('min_episode_transitions')
    training_mode = LaunchConfiguration('training_mode')
    reward_mode = LaunchConfiguration('reward_mode')
    max_episode_steps = LaunchConfiguration('max_episode_steps')
    max_episodes = LaunchConfiguration('max_episodes')
    save_frequency = LaunchConfiguration('save_frequency')
    goal_progress_delta = LaunchConfiguration('goal_progress_delta')
    reset_offset_range = LaunchConfiguration('reset_offset_range')
    min_reset_separation = LaunchConfiguration('min_reset_separation')
    reset_yaw_mode = LaunchConfiguration('reset_yaw_mode')
    reset_yaw_noise = LaunchConfiguration('reset_yaw_noise')
    path_target_speed = LaunchConfiguration('path_target_speed')
    path_low_speed_penalty_scale = LaunchConfiguration('path_low_speed_penalty_scale')
    formation_distance = LaunchConfiguration('formation_distance')
    formation_penalty_scale = LaunchConfiguration('formation_penalty_scale')
    formation_bonus_scale = LaunchConfiguration('formation_bonus_scale')
    formation_bonus_width = LaunchConfiguration('formation_bonus_width')

    # Define nodes that don't depend on agent count
    env_node = Node(
        package='asv_ai',
        executable='asv_env_node',
        name='asv_env_node',
        parameters=[{
            'num_agents': num_agents,
            'training_mode': training_mode,  # Pass training mode for performance optimization
            'reward_mode': reward_mode,
            'max_episode_steps': max_episode_steps,
            'goal_progress_delta': goal_progress_delta,
            'reset_offset_range': reset_offset_range,
            'min_reset_separation': min_reset_separation,
            'reset_yaw_mode': reset_yaw_mode,
            'reset_yaw_noise': reset_yaw_noise,
            'path_target_speed': path_target_speed,
            'path_low_speed_penalty_scale': path_low_speed_penalty_scale,
            'formation_distance': formation_distance,
            'formation_penalty_scale': formation_penalty_scale,
            'formation_bonus_scale': formation_bonus_scale,
            'formation_bonus_width': formation_bonus_width,
            'visualization_enabled': enable_viz
        }],
        output='screen'
    )

    ppo_node = Node(
        package='asv_ai',
        executable='ppo_node',
        name='ppo_node',
        parameters=[{
            'num_agents': num_agents,
            'model_path': model_path,
            'training_enabled': training_enabled,
            'autoload_model': autoload_model,
            'deterministic_inference': deterministic_inference,
            'constant_action_enabled': constant_action_enabled,
            'constant_yaw_action': constant_yaw_action,
            'constant_surge_action': constant_surge_action,
            'use_extended_observation': use_extended_observation,
            'normalize_observation': normalize_observation,
            'reset_grace_steps': reset_grace_steps,
            'min_episode_transitions': min_episode_transitions,
            'train_frequency': train_frequency,
            'min_training_size': min_training_size,
            'train_interval_sec': train_interval_sec,
            'learning_rate': learning_rate,
            'vf_coef': vf_coef,
            'initial_log_std': initial_log_std,
            'initial_surge_action_bias': initial_surge_action_bias,
            'target_kl': target_kl,
            'batch_size': batch_size,
            'n_epochs': n_epochs,
            'max_episodes': max_episodes,
            'save_frequency': save_frequency,
            'rollout_dir': rollout_dir,
            'training_mode': training_mode  # Pass training mode for logging optimization
        }],
        output='screen'
    )

    rviz_config_path = PathJoinSubstitution([
        FindPackageShare('asv_ai'), 'rviz', 'asv.rviz'
    ])

    # Function to create agent nodes - this matches your system launch file
    def launch_setup(context):
        num_agents_str = LaunchConfiguration('num_agents').perform(context)
        num_agents_value = int(num_agents_str)
        enable_viz_str = LaunchConfiguration('enable_visualization').perform(context)
        enable_viz_value = enable_viz_str.lower() == 'true'
        training_mode_str = LaunchConfiguration('training_mode').perform(context)
        simulation_hz_value = float(LaunchConfiguration('simulation_hz').perform(context))
        tf_publish_hz_value = float(LaunchConfiguration('tf_publish_hz').perform(context))
        time_scale_value = float(LaunchConfiguration('time_scale').perform(context))

        urdf_path = PathJoinSubstitution([
            FindPackageShare('yf_description'),  # Changed from asv_description to yf_description
            'urdf',
            'asv_loyola.urdf.xacro'  # Using the same URDF as in system launch
        ])

        nodes_to_launch = []

        # Conditional RViz2 launch for performance (only if visualization enabled)
        if enable_viz_value:
            rviz_config_path = PathJoinSubstitution([
                FindPackageShare('asv_ai'), 'rviz', 'asv.rviz'
            ])

            rviz_node = Node(
                package='rviz2',
                executable='rviz2',
                name='rviz2',
                arguments=['-d', rviz_config_path],
                output='screen'
            )
            nodes_to_launch.append(rviz_node)

        for i in range(num_agents_value):
            robot_description = ParameterValue(
                Command(['xacro "', urdf_path, '"', f' id:={i}', ' own:=true']),
                value_type=str
            )

            agent_node = Node(
                package='asv_ai',
                executable='asv_agent_node',
                name=f'asv_agent_node_{i}',
                namespace=f'agent_{i}',
                parameters=[{
                    'agent_id': i,
                    'training_mode': training_mode_str,
                    'simulation_hz': simulation_hz_value,
                    'tf_publish_hz': tf_publish_hz_value,
                    'time_scale': time_scale_value
                }],
                output='screen' if training_mode_str == 'debug' else 'log'  # Reduce terminal output in fast mode
            )
            nodes_to_launch.append(agent_node)

            if enable_viz_value:
                # Robot state publisher is only needed for RViz/RobotModel.
                robot_state_publisher_node = Node(
                    package='robot_state_publisher',
                    executable='robot_state_publisher',
                    name=f'robot_state_publisher_{i}',
                    namespace=f'agent_{i}',
                    parameters=[{
                        'robot_description': robot_description,
                        'publish_frequency': 30.0
                    }],
                    remappings=[
                        ('/robot_description', '/robot_description')
                    ],
                    output='log'
                )
                nodes_to_launch.append(robot_state_publisher_node)

        return nodes_to_launch

    # Add all to launch description
    ld = LaunchDescription()

    # Add all arguments
    ld.add_action(num_agents_arg)
    ld.add_action(model_path_arg)
    ld.add_action(train_freq_arg)
    ld.add_action(min_training_size_arg)
    ld.add_action(train_interval_sec_arg)
    ld.add_action(learning_rate_arg)
    ld.add_action(vf_coef_arg)
    ld.add_action(initial_log_std_arg)
    ld.add_action(initial_surge_action_bias_arg)
    ld.add_action(target_kl_arg)
    ld.add_action(ppo_lrn_arg)
    ld.add_action(batch_size_arg)
    ld.add_action(n_epochs_arg)
    ld.add_action(rollout_dir_arg)
    ld.add_action(enable_viz_arg)
    ld.add_action(training_enabled_arg)
    ld.add_action(autoload_model_arg)
    ld.add_action(deterministic_inference_arg)
    ld.add_action(constant_action_enabled_arg)
    ld.add_action(constant_yaw_action_arg)
    ld.add_action(constant_surge_action_arg)
    ld.add_action(use_extended_observation_arg)
    ld.add_action(normalize_observation_arg)
    ld.add_action(reset_grace_steps_arg)
    ld.add_action(min_episode_transitions_arg)
    ld.add_action(training_mode_arg)
    ld.add_action(reward_mode_arg)
    ld.add_action(max_episode_steps_arg)
    ld.add_action(max_episodes_arg)
    ld.add_action(save_frequency_arg)
    ld.add_action(goal_progress_delta_arg)
    ld.add_action(reset_offset_range_arg)
    ld.add_action(min_reset_separation_arg)
    ld.add_action(reset_yaw_mode_arg)
    ld.add_action(reset_yaw_noise_arg)
    ld.add_action(path_target_speed_arg)
    ld.add_action(path_low_speed_penalty_scale_arg)
    ld.add_action(formation_distance_arg)
    ld.add_action(formation_penalty_scale_arg)
    ld.add_action(formation_bonus_scale_arg)
    ld.add_action(formation_bonus_width_arg)
    ld.add_action(simulation_hz_arg)
    ld.add_action(tf_publish_hz_arg)
    ld.add_action(time_scale_arg)

    # Add core nodes (env and ppo)
    ld.add_action(env_node)
    ld.add_action(ppo_node)
    # Note: RViz2 is now conditionally launched within launch_setup
    ld.add_action(OpaqueFunction(function=launch_setup))

    return ld
