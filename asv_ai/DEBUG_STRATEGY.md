# Debug Strategy: PPO Circling Behavior in `asv_ai`

Goal: stop the failure mode where both boats fall into synchronized circular motion and abandon the path, and understand precisely when/why it happens. This is ROS2-friendly and log/console driven (no interactive breakpoints needed, though debugpy is optional).

## What to log/observe
- Topics to record: `/environment/state`, `/ppo/action`, `/environment/reward`, `/environment/done`.
- PPO internals (throttled): entropy, policy loss, value loss, clip fraction, learning rate, mean/std of actions.
- Env metrics: cross-track error, heading error, along-track progress, yaw rate, reward breakdown, termination reason.
- Episode summaries: return, length, spin detector triggers, save/load events.

## Instrumentation (minimal code changes)
1) `ppo_node.py`
- Add throttled logs every N steps (e.g., 200) with: entropy, policy_loss, value_loss, clip_fraction, lr, mean_action, std_action.
- On episode end: log episode_return, episode_len, avg|max yaw_rate_cmd (if available), avg cross_track_error (passed from env), spin_detector flag.
- Optional: expose a param to set log throttle interval.

2) `asv_env_node.py`
- Log reward breakdown per step (throttled) and termination reason.
- Publish or log a compact debug vector: `[cross_track_error, heading_error, along_progress, yaw_rate, collision_flag]`.
- Add a simple spin detector: consecutive steps with high yaw_rate and low along-track progress → warn and mark in episode summary.

3) Recording
- Standard rosbag capture during suspect windows: record the four key topics above for post-hoc analysis.

## Experiments (run order)
1) Eval-only check: Run a saved model with training disabled to see if circling is baked into the policy vs. emerging during training.
2) Short, high-logging sessions: Run with dense logging for a few episodes to capture the transition into circling (look at entropy collapse, action saturation, reward plateau).
3) Reset diversity: Increase randomness in initial pose/heading to break symmetry; observe if circling frequency drops.
4) Entropy/LR/clip sweep: Slightly increase entropy bonus, lower learning rate, and/or tighten clip range; rerun short sessions to see if circling disappears.
5) Reward shaping probe: Add penalties for |yaw_rate| and sustained heading error; reward along-track progress; early-terminate when cross-track or heading error remains high for T seconds.
6) Action scaling audit: Verify env action mapping matches PPO’s `[-1,1]`; check for actuator clipping or bias that might favor spins.

## Detection & success criteria
- Spin detector triggers should go to zero (or near-zero) over evaluation rollouts.
- Episode return should improve or remain stable after mitigation.
- Cross-track error and along-track progress should trend better in logs.
- Entropy should not crash to near-zero too early; action std should not collapse to a constant spin pattern.

## Optional: interactive debugging
- For deeper dives, run `ppo_node.py` with `debugpy` and attach from VS Code; otherwise rely on logs and bags to stay lightweight.
