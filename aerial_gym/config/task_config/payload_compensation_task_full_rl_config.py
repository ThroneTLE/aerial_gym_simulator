class task_config:
    seed = 1
    sim_name = "base_sim"
    env_name = "empty_env"
    robot_name = "base_quadrotor"
    controller_name = "lee_position_control"
    args = {}
    num_envs = 1024
    use_warp = False
    headless = False
    device = "cuda:0"

    observation_space_dim = 24  # 13 base + 11 payload features
    privileged_observation_space_dim = 0
    action_space_dim = 4  # RL 输出 [x, y, z, yaw]
    episode_len_steps = 1800
    return_state_before_reset = False

    reward_parameters = {
        "pos_error_gain1": [2.0, 2.0, 2.0],
        "pos_error_exp1": [1 / 3.5, 1 / 3.5, 1 / 3.5],
        "pos_error_gain2": [2.0, 2.0, 2.0],
        "pos_error_exp2": [2.0, 2.0, 2.0],
        "dist_reward_coefficient": 7.5,
        "max_dist": 15.0,
        "action_diff_penalty_gain": [1.0, 1.0, 1.0],
        "absolute_action_reward_gain": [2.0, 2.0, 2.0],
        "crash_penalty": -100,
    }

    crash_distance_threshold = 3.0
    crash_tilt_threshold_deg = 55.0

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
        "randomize_release": False,
    }
