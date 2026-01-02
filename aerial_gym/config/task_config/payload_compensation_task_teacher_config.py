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

    # 新观测结构: 旋转矩阵(9) + 角速度(3) + 4位附着掩码(4) + 预警(1) + 上一时刻动作(3) = 20
    observation_space_dim = 20
    privileged_observation_space_dim = 7
    action_space_dim = 3  # thrust + roll torque + pitch torque (无 yaw)
    controller_action_dim = 8  # 保持与 Lee 控制器兼容，yaw 补偿位置设为 0
    dagger_frac = 0.0          # 初始教师动作占比
    dagger_decay_reward = 15000.0  # 均值奖励达到此值后开始衰减
    dagger_decay_rate = 0.99      # 每步衰减系数
    dagger_min_frac = 0.0          # 衰减下限（保留少量教师）
    imitation_err_threshold = 0.5# 仅当误差低于该阈值才衰减
    dagger_use_postmix_err = True  # 使用混合后的 imitation err 作为衰减判定，和 TB 曲线一致
    fix_yaw_residual_zero = True   # 教师残差的 yaw 力矩固定为 0

    episode_len_steps = 1500
    return_state_before_reset = False
    teacher_mode = True  # 启用特权/模仿

    # 奖励仿照 xadapt：存活奖励为主，角速度/线加速度/动作振荡惩罚，移除位置/补偿项，模仿不计入 reward
    reward_parameters = {
        "position_weight": 0.0,
        "crash_penalty": -10.0,
        "attitude_penalty_coef": 0.00,
        "release_attitude_boost": 1.0,
        "comp_torque_penalty_coef": 0.0,
        "comp_thrust_penalty_coef": 0.0,
        # 线速度惩罚关闭，改用线加速度惩罚
        "velocity_penalty_coef": 0.0,
        "angvel_penalty_coef": 0.000, #0.2
        "action_smoothness_coef": 0.00, #0.06
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
        "release_hover_boost": 0.000000,
        "release_angvel_boost": 0.00000,
        "delta_error_bonus_coef": 0.0,
        "delta_error_window_steps": 0,
        "delta_error_bonus_clip": 0.0,
        "release_reward_window_steps": 200,
        "accel_penalty_away_coef": 0.00000,
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
        # 模仿权重 - 分离推力和力矩
        "imitation_weight": 0.0,  # 统一权重（当 thrust/torque 未指定时使用）
        "imitation_weight_thrust": 0.0,  # 推力模仿权重（action[0]）
        "imitation_weight_torque": 0.0,  # 力矩模仿权重（action[1:3]）
        # 动作幅度惩罚 - 防止不必要的残差输出和抖动
        "action_magnitude_penalty_coef": 0.0,  # 惩罚系数，越大越抑制输出 惩罚 = thrust² × thrust_coef + mean(torque²) × torque_coef
        "action_magnitude_penalty_thrust": 0.000,  # thrust 惩罚（可选单独设置）
        "action_magnitude_penalty_torque": 0.000,  # torque 惩罚（可选单独设置）
        # 存活奖励
        "survive_bonus": 10.0,
    }

    crash_distance_threshold = 5.0
    crash_tilt_threshold_deg = 20.0

    payload_parameters = {
        "payload_mass": 0.02,
        "payload_mass_range": [0.00, 0.04],
        "randomize_payload_mass": True,
        "randomize_offsets_on_plane": True,
        "offset_plane_radial_jitter": 0.4,  #沿机臂方向的“半径扰动”，均匀分布 [-jitter, +jitter]
        "offset_plane_z_jitter": 0.8,  #垂直方向的“高度扰动”，均匀分布 [-jitter, +jitter]
        "offset_plane_r_max": 0.4,   #机臂方向最大偏移距离
        "offset_plane_z_max": 0.4,      #垂直方向最大偏移距离
        "force_offset_torque_scale": 0.00,  # 等效力矩系数：tau_eq = - r_com x F_total，1.0=全量补偿，0=关闭
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
        "warning_steps": 100,
        "randomize_release":   True,  # 初始验证先固定
        "log_release_events": False,
    }

    randomization_parameters = False
    observation_parameters = {
        "include_payload_mass": False,  # base obs 是否包含载荷质量
        "include_payload_com": False,  # base obs 是否包含载荷质心偏移
        "include_last_release_mass": False,  # base obs 是否包含上次释放质量
    }

    curriculum_parameters = None
