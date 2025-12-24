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

    # 特权 raw 仅放入 priviliged_obs，由策略侧 encoder 压到 8 维；obs 只保留基础 29 维
    observation_space_dim = 29
    privileged_observation_space_dim = 41
    action_space_dim = 4  # thrust + 3 compensation torques
    controller_action_dim = 8  # 4 Lee inputs + thrust + 3 torque compensation commands
    dagger_frac = 0.0          # 初始教师动作占比
    dagger_decay_reward = 15000.0  # 均值奖励达到此值后开始衰减
    dagger_decay_rate = 0.99      # 每步衰减系数
    dagger_min_frac = 0.0          # 衰减下限（保留少量教师）
    imitation_err_threshold = 0.5# 仅当误差低于该阈值才衰减
    dagger_use_postmix_err = True  # 使用混合后的 imitation err 作为衰减判定，和 TB 曲线一致
    fix_yaw_residual_zero = True   # 教师残差的 yaw 力矩固定为 0

    episode_len_steps = 3000
    return_state_before_reset = False
    teacher_mode = True  # 启用特权/模仿
    release_feedforward_parameters = {
        "enable": True,
        "steps": 25,
        "decay": 0.85,
        "torque_scale": 1.0,
        "thrust_scale": 0.0,
        "log": False,
        "log_path": "logs/release_ff.log",
    }

    # 奖励仿照 xadapt：存活奖励为主，角速度/线加速度/动作振荡惩罚，移除位置/补偿项，模仿不计入 reward
    reward_parameters = {
        "position_weight": 0.0,
        "crash_penalty": -10.0,
        "attitude_penalty_coef": 0.0,
        "release_attitude_boost": 1.0,
        "comp_torque_penalty_coef": 0.0,
        "comp_thrust_penalty_coef": 0.0,
        # 线速度惩罚关闭，改用线加速度惩罚
        "velocity_penalty_coef": 0.0,
        "angvel_penalty_coef": 0.600, #0.2
        "action_smoothness_coef": 1.8000, #0.06
        "tilt_warning_deg": 0.0,
        "tilt_warning_penalty": 0.0,
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
        "z_error_penalty_coef": 0.0,
        "yaw_penalty_coef": 0.0,
        "hover_bonus_radius": 0.0,
        "hover_bonus_tilt_deg": 0.0,
        "hover_bonus_velocity": 0.0,
        "hover_bonus": 0.0,
        "release_stability_steps": 50,
        "release_hover_boost": 0.10,
        "release_angvel_boost": 0.5,
        "delta_error_bonus_coef": 0.0,
        "delta_error_window_steps": 0,
        "delta_error_bonus_clip": 0.0,
        "release_reward_window_steps": 200,
        "accel_penalty_away_coef": 0.01,
        "accel_penalty_toward_coef": 0.0,
        "comp_penalty_high_threshold": 1.0,
        "comp_torque_penalty_high_coef": 0.0,
        "comp_thrust_penalty_high_coef": 0.0,
        "comp_window_penalty_scale": 1,
        "vel_window_penalty_scale": 1,
        "smooth_window_penalty_scale": 1,
        "comp_activation_bonus_coef": 0.0,
        "vel_away_penalty_coef": 0.0,
        "tilt_excess_threshold_deg": 0.0,
        "tilt_excess_coef": 0.0,
        "tilt_excess_exp": 0.0,
        # 模仿不计入 reward，如需监督请在损失里加
        "imitation_weight": 16.0,  # legacy fallback
        "imitation_thrust_weight": 0.0,
        "imitation_torque_weight": 0.0,
        # 存活奖励
        "survive_bonus": 10.0,
    }

    crash_distance_threshold = 30.0
    crash_tilt_threshold_deg = 200.0

    payload_parameters = {
        "enable_payload": True,
        "payload_mass": 0.02,
        "log_payload_torque": True,
        "payload_torque_log_path": "logs/payload_torque.log",
        "payload_torque_log_interval": 1,
        "offsets": [
            [0.4, 0.4, -0.4],
            [0.4, -0.4, -0.4],
            [-0.4, 0.4, -0.4],
            [-0.4, -0.4, -0.4],
        ],
        "release_start": 50,
        "release_interval": 300,
        "release_start_range": [50, 100],
        "release_interval_range": [300, 350],
        "warning_steps": 50,
        "randomize_release":   True,  # 初始验证先固定
        "log_release_events": False,
    }

    trajectory_parameters = {
        "enable": False,
        "type": "random_mix",
        "space_min": [-3.0, -3.0, -1.0],
        "space_max": [3.0, 3.0, 4.0],
        "max_speed": 0.6,
        "max_accel": 0.6,
        "ramp_steps": 400,
        "num_harmonics": 1,
        "freq_range_hz": [0.02, 0.05],
        "amp_range": [0.5, 1.2],
        "spiral": False,
        "spiral_radius_range": [0.5, 1.5],
        "spiral_radius_mod_range": [0.2, 0.6],
        "spiral_radius_freq_range": [0.05, 0.15],
        "spiral_theta_freq_range": [0.05, 0.15],
        "z_drift_range": [-0.1, 0.1],
        "loop": False,
        "randomize_each_reset": True,
    }

    randomization_parameters = False

    curriculum_parameters = None
