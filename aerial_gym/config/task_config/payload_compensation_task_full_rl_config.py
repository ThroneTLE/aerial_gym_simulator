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
    episode_len_steps = 400
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
        "crash_penalty": -200,
        "tilt_warning_deg": 35.0,
        "tilt_warning_penalty": -20.0,
        "height_warning": 0.6,
        "height_warning_penalty": -20.0,
        "velocity_penalty_coef": 0.05,
        "angvel_penalty_coef": 0.05,
        "release_tilt_limit_deg": 10.0,
        "release_tilt_penalty": -50.0,
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
        "release_start_range": None,
        "release_interval_range": None,
        "warning_steps": 20,
        "randomize_release": False,
    }

    randomization_parameters = {
        # 初始状态扰动
        "initial_position_noise": [0.0, 0.0, 0.0],
        "initial_orientation_noise_deg": [0.0, 0.0, 0.0],
        # 目标点随机范围（XYZ min/max）
        "target_position_range": [
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 1.0],
        ],
        # 观测噪声（高斯标准差）
        "obs_noise_std": {
            "position_error": 0.0,
            "linear_velocity": 0.0,
            "angular_velocity": 0.0,
        },
        # 质量/惯量/推力随机化配置（默认关闭，后续训练再开启）
        "mass_jitter": {
            "enabled": False,
            "relative_range": 0.05,
        },
        "inertia_jitter": {
            "enabled": False,
            "relative_range": 0.05,
        },
        "thrust_scale_jitter": {
            "enabled": False,
            "relative_range": 0.05,
        },
    }
