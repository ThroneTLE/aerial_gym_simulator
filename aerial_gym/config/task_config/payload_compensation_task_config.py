class task_config:
    seed = 1
    sim_name = "base_sim"
    env_name = "empty_env"
    robot_name = "base_quadrotor"
    controller_name = "lee_position_control_with_compensation"
    args = {}
    num_envs = 1024
    use_warp = False
    headless = True
    device = "cuda:0"

    observation_space_dim = 29  # 3 pos err + 9 rot mat + lin/ang vel + payload features
    privileged_observation_space_dim = 0
    action_space_dim = 4  # thrust + 3 compensation torques
    controller_action_dim = 8  # 4 Lee inputs + thrust + 3 torque compensation commands

    episode_len_steps = 2000
    return_state_before_reset = False

    reward_parameters = {
        # 仅关注释放期的姿态/角速度/速度抑制
        "position_weight": 0.0,
        # 在警告期/释放扰动下可轻微减弱
        "crash_penalty": -120.0,
        "attitude_penalty_coef": 1.0,
        "release_attitude_boost": 2.0,
        # 先减轻惩罚以鼓励探索补偿动作
        "comp_torque_penalty_coef": 0.02,
        "comp_thrust_penalty_coef": 0.02,
        # 放松速度/角速度惩罚，专注抑制释放瞬间动量
        "velocity_penalty_coef": 0.4,
        "angvel_penalty_coef": 0.15,
        "action_smoothness_coef": 0.0,
        "tilt_warning_deg": 0.0,
        "tilt_warning_penalty": 0.0,
        # 高度相关奖励/惩罚关闭
        "height_warning": 0.0,
        "height_warning_penalty": 0.0,
        "height_safe_bonus": 0.0,
        "release_tilt_limit_deg": 0.0,
        "release_tilt_penalty": 0.0,
        "stability_radius": 0.0,
        "stability_tilt_deg": 0.0,
        "stability_penalty": 0.0,
        "stability_velocity_penalty": 0.0,
        "position_error_penalty_coef": 0.0,
        "z_error_penalty_coef": 1.0,
        "yaw_penalty_coef": 0.0,
        # Hover 奖励关闭
        "hover_bonus_radius": 0.0,
        "hover_bonus_tilt_deg": 0.0,
        "hover_bonus_velocity": 0.0,
        "hover_bonus": 0.0,
        # When a payload was just released, temporarily boost stability incentives.
        "release_stability_steps": 30,
        "release_hover_boost": 2.0,
        "release_angvel_boost": 1.5,
        # 奖励距离误差缩小量，仅在预警/释放窗口内生效
        "delta_error_bonus_coef": 0.0,
        "delta_error_window_steps": 0,
        "delta_error_bonus_clip": 0.0,
        # 奖励/惩罚仅在预警与释放后窗口内生效的步数
        "release_reward_window_steps": 300,
        # 加速度惩罚：远离目标的加速度重罚，朝向目标的加速度随距离变近惩罚加重
        "accel_penalty_away_coef": 2.5,
        "accel_penalty_toward_coef": 0.02,
        # 补偿分段加重惩罚
        "comp_penalty_high_threshold": 1.1,  # 提高阈值，允许更大补偿
        "comp_torque_penalty_high_coef": 0.04,  # 降低高段惩罚
        "comp_thrust_penalty_high_coef": 0.04,
        # 预警/释放窗口内额外放松惩罚、鼓励动作
        "comp_window_penalty_scale": 0.35,
        "vel_window_penalty_scale": 0.35,
        "smooth_window_penalty_scale": 0.6,
        "comp_activation_bonus_coef": 0.0,
    }

    crash_distance_threshold = 5.0  # meters
    crash_tilt_threshold_deg = 8.0

    payload_parameters = {
        "payload_mass": 0.025,
        "offsets": [
            [0.4, 0.4, -0.4],
            [0.4, -0.4, -0.4],
            [-0.4, 0.4, -0.4],
            [-0.4, -0.4, -0.4],
        ],
        "release_start": 300,
        "release_interval": 300,
        "release_start_range": [200, 400],
        "release_interval_range": [200, 400],
        "warning_steps": 120,
        "randomize_release": False,
        "log_release_events": False,
    }

    randomization_parameters = None

    curriculum_parameters = None
