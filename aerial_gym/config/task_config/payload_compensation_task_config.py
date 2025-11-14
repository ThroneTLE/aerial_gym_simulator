class task_config:
    seed = 1
    sim_name = "base_sim"
    env_name = "empty_env"
    robot_name = "base_quadrotor"
    controller_name = "lee_position_control_with_compensation"
    args = {}
    num_envs = 1024
    use_warp = False
    headless = False
    device = "cuda:0"

    observation_space_dim = 24  # 13 base + 11 payload features (for 4 payloads)
    privileged_observation_space_dim = 0
    action_space_dim = 3  # compensation torques only
    controller_action_dim = 7  # 4 Lee inputs + 3 compensation commands

    episode_len_steps = 1800
    return_state_before_reset = False

    reward_parameters = {
        "position_weight": 0.5,
        "crash_penalty": -20.0,
        "attitude_penalty_coef": 1.2,
        "release_attitude_boost": 2.0,
        "comp_torque_penalty_coef": 0.05,
    }

    payload_parameters = {
        "payload_mass": 0.025,
        "offsets": [
            [0.4, 0.4, -0.4],
            [0.4, -0.4, -0.4],
            [-0.4, 0.4, -0.4],
            [-0.4, -0.4, -0.4],
        ],
        "release_start": 400,
        "release_interval": 200,
        "release_start_range": [300, 500],
        "release_interval_range": [150, 300],
        "warning_steps": 20,
    }
