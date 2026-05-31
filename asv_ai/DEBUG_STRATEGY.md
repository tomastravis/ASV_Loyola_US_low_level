# Debug Strategy: PPO Circling Behavior in `asv_ai`

Goal: debug the system in layers: first verify that the boats can stay still with stable ROS2 publishers/TF, then move back to forward path-following training.

## Run a Short Debug Session

Build and launch from the ROS2 workspace inside the container:

```bash
colcon build --symlink-install --executor sequential --packages-select yf_description asv_ai
source install/setup.bash
ros2 launch asv_ai asv_train.launch.py num_agents:=2 enable_visualization:=true training_mode:=debug reward_mode:=stillness simulation_hz:=20.0 tf_publish_hz:=20.0 reset_grace_steps:=10 min_episode_transitions:=50
```

Use `reward_mode:=path_following` after the stillness case is clean: one publisher per topic, no TF/model jumps, and `spin_cnt=0`.

## First Checks

Open separate terminals after sourcing the workspace:

```bash
ros2 topic hz /environment/state
ros2 topic hz /ppo/action
ros2 topic echo /environment/reward
ros2 topic echo /environment/done
```

Expected signs:

- `/environment/state` and `/ppo/action` should both keep publishing.
- `[ppo_debug] buffer_pos` should climb from 0 toward `train_frequency`.
- In `stillness`, speed and yaw rate should trend near zero.
- In `path_following`, `[env_debug] progress ... d=...` should often be positive once the boats move forward.
- In `path_following`, `path_speed` should become positive. If `path_speed` is negative while speed is non-zero, the boats are moving against the path direction.
- `spin_cnt` should not steadily climb to the spin window.
- `done=running` should remain true until `goal_progress` or `max_episode_steps`, not immediately after reset.
- After a reset, PPO may log `Ignoring stale done during post-reset grace`; that is good. It means a stale latched `done=True` was filtered instead of closing a 0-step episode.

## Isolate the Simulator Before PPO

Use the PPO node's constant-action diagnostic mode so there is only one `/ppo/action` publisher. The action order is `[yaw, surge]` for each agent. Surge is centered: `0.0` means no thrust, positive values push forward, and negative values push backward.

No thrust:

```bash
ros2 launch asv_ai asv_train.launch.py num_agents:=2 enable_visualization:=true training_mode:=debug training_enabled:=false reward_mode:=stillness constant_action_enabled:=true constant_yaw_action:=0.0 constant_surge_action:=0.0 simulation_hz:=30.0 time_scale:=1.0 tf_publish_hz:=30.0
```

Gentle straight thrust:

```bash
ros2 launch asv_ai asv_train.launch.py num_agents:=2 enable_visualization:=true training_mode:=debug training_enabled:=false reward_mode:=path_following constant_action_enabled:=true constant_yaw_action:=0.0 constant_surge_action:=0.25 simulation_hz:=30.0 time_scale:=1.0 tf_publish_hz:=30.0
```

If gentle straight thrust still creates strong yaw, the problem is below PPO: action mapping, dynamics, frame convention, or reset orientation.
For no thrust, `[env_debug]` should show `speed` and `yaw_rate` near zero, `still` near `5`, and `drift` close to zero after reset.

## Path-Following Run

Once the stillness run is stable:

```bash
ros2 launch asv_ai asv_train.launch.py num_agents:=2 enable_visualization:=true training_mode:=debug reward_mode:=path_following max_episode_steps:=1000 goal_progress_delta:=12.0 reset_offset_range:=2.0 min_reset_separation:=2.0 learning_rate:=0.0001 use_extended_observation:=true normalize_observation:=true reset_grace_steps:=10 min_episode_transitions:=50
```

## Path-Following Curriculum

If the model learns safe stillness but does not learn forward motion, start with resets that face the path. This keeps the first path-following task close to "move gently forward", then later you can widen the yaw noise or go back to random yaw.

```bash
ros2 launch asv_ai asv_train.launch.py num_agents:=2 enable_visualization:=false training_mode:=fast training_enabled:=true autoload_model:=false reward_mode:=path_following reset_yaw_mode:=path_aligned reset_yaw_noise:=0.2 reset_offset_range:=0.5 goal_progress_delta:=4.0 path_target_speed:=0.08 path_low_speed_penalty_scale:=0.8 simulation_hz:=100.0 time_scale:=3.0 tf_publish_hz:=0.0 rollout_dir:=~/Desktop/PPO_Rollouts_path_aligned_v4 learning_rate:=0.000005 vf_coef:=0.01 initial_log_std:=-1.2 initial_surge_action_bias:=0.2 n_epochs:=1 train_frequency:=1000 min_training_size:=0 train_interval_sec:=0.0 target_kl:=0.02 save_frequency:=5 use_extended_observation:=true normalize_observation:=true reset_grace_steps:=10 min_episode_transitions:=50
```

Validation targets for this curriculum:

- `path_speed` should be positive more often than negative.
- `low_speed_pen` should be close to `0` when the boats move forward and negative when they drift/stall.
- `initial_surge_action_bias:=0.2` starts the actor close to the constant-action diagnostic that moved straight, instead of asking PPO to discover forward thrust from symmetric noise.
- `progress ... d=...` should be positive often enough to finish some episodes by `goal_progress`, not only by `max_episode_steps`.
- `action_mean` should not collapse to exactly zero immediately; small positive surge with low yaw is a good early sign.
- `approx_kl` should stay near or below `target_kl`. If it stays much higher, reduce `learning_rate` again before increasing exploration.

## Formation Curriculum

Once the boats consistently move forward on the line, continue from the best path-aligned model and increase the formation terms without changing the yaw-reset difficulty yet:

```bash
ros2 launch asv_ai asv_train.launch.py num_agents:=2 enable_visualization:=false training_mode:=fast training_enabled:=true autoload_model:=false model_path:=~/Desktop/PPO_Rollouts_path_aligned_v4/best_model.zip reward_mode:=path_following reset_yaw_mode:=path_aligned reset_yaw_noise:=0.2 reset_offset_range:=0.5 goal_progress_delta:=4.0 path_target_speed:=0.08 path_low_speed_penalty_scale:=0.8 formation_distance:=3.0 formation_penalty_scale:=0.8 formation_bonus_scale:=0.6 formation_bonus_width:=1.0 simulation_hz:=100.0 time_scale:=3.0 tf_publish_hz:=0.0 rollout_dir:=~/Desktop/PPO_Rollouts_path_formation_v1 learning_rate:=0.000003 vf_coef:=0.01 initial_log_std:=-1.2 n_epochs:=1 train_frequency:=1000 min_training_size:=0 train_interval_sec:=0.0 target_kl:=0.02 save_frequency:=5 use_extended_observation:=true normalize_observation:=true reset_grace_steps:=10 min_episode_transitions:=50
```

Validation targets for this curriculum:

- `path_speed` should stay mostly positive. If it collapses, reduce `formation_penalty_scale` before changing exploration.
- `form_err` should trend down compared with the path-only run.
- `form_pen` shows the active penalty from formation error; `form_bonus` grows only when formation is tight.
- Episodes should still finish by `goal_progress` sometimes. If they only finish by `max_episode_steps`, the formation objective is too heavy or forward speed has collapsed.
- For a visual check, load `~/Desktop/PPO_Rollouts_path_formation_v1/best_model.zip` with `training_enabled:=false` and `deterministic_inference:=true`.

## Fast Headless Training

Use this when the ROS graph is clean and you want samples fast, not pretty rendering:

```bash
ros2 launch asv_ai asv_train.launch.py num_agents:=2 enable_visualization:=false training_mode:=fast training_enabled:=true autoload_model:=false reward_mode:=path_following simulation_hz:=200.0 time_scale:=5.0 tf_publish_hz:=0.0 rollout_dir:=~/Desktop/PPO_Rollouts_clean learning_rate:=0.0001 vf_coef:=0.1 initial_log_std:=-1.5 n_epochs:=2 train_frequency:=1000 min_training_size:=0 train_interval_sec:=0.0 target_kl:=0.05 save_frequency:=10 use_extended_observation:=true normalize_observation:=true reset_grace_steps:=10 min_episode_transitions:=50
```

Notes:

- `simulation_hz` controls how often the agents integrate/publish state.
- `time_scale` controls how much simulated time advances per real second.
- `tf_publish_hz:=0.0` disables agent TF publishing for headless runs.
- `min_training_size:=0` means "wait for a full `train_frequency` buffer".
- `train_interval_sec:=0.0` disables timer-based partial PPO updates.
- `target_kl:=0.05` stops an update early if the policy jumps too far.
- `autoload_model:=false` starts fresh. Set `autoload_model:=true` or pass `model_path:=...` to resume.
- Use a fresh `rollout_dir` after reward/observation changes. Old 12-dimensional checkpoints are intentionally incompatible with the new 22-dimensional extended observation.
- If `approx_kl` stays very high, reduce `n_epochs` first, then reduce `learning_rate`.

## Visual Evaluation of a Trained Model

After headless training has written `latest_model.zip`:

```bash
ros2 launch asv_ai asv_train.launch.py num_agents:=2 enable_visualization:=true training_mode:=fast training_enabled:=false deterministic_inference:=true reward_mode:=path_following model_path:=~/Desktop/PPO_Rollouts_clean/latest_model.zip simulation_hz:=30.0 time_scale:=1.0 tf_publish_hz:=30.0 use_extended_observation:=true normalize_observation:=true
```

## Record Evidence

Capture short bags around the failure mode:

```bash
ros2 bag record /environment/state /environment/reward /environment/done /ppo/action /agent_0/state_update /agent_1/state_update
```

Useful questions for each bag:

- Do actions saturate to a constant yaw command?
- Does reward increase when progress increases?
- Does the episode reset after `/environment/done`?
- Does `buffer_pos` climb, or does PPO repeatedly train on an empty/fixed-size buffer?

## Breakpoint Alternative

ROS2 nodes are event-driven, so logs and bags are usually faster than interactive breakpoints. If a real breakpoint is needed, add `debugpy.listen(...)` inside one node and attach from VS Code, but keep this for narrow cases after the topic/bag checks above.
