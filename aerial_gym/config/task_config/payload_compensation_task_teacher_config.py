class task_config:
    seed = 1
    sim_name = "base_sim_2ms"
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
    dagger_frac = 0.7          # 初始教师动作占比
    dagger_decay_reward = 15000.0  # 均值奖励达到此值后开始衰减
    dagger_decay_rate = 0.99      # 每步衰减系数
    dagger_min_frac = 0.1          # 衰减下限（保留少量教师）
    imitation_err_threshold = 0.3  # 仅当误差低于该阈值才衰减
    dagger_use_postmix_err = True  # 使用混合后的 imitation err 作为衰减判定，和 TB 曲线一致
    fix_yaw_residual_zero = True   # 教师残差的 yaw 力矩固定为 0

    # dt 从 0.01 -> 0.002，步数相关参数按 5x 放大以保持物理时长一致
    episode_len_steps = 7500
    return_state_before_reset = False
    teacher_mode = True  # 启用特权/模仿

    reward_parameters = {
        "position_weight": 4.0,
        "crash_penalty": -120.0,
        "attitude_penalty_coef": 0.010,
        "release_attitude_boost": 1.0,
        "comp_torque_penalty_coef": 0.01,
        "comp_thrust_penalty_coef": 0.02,
        "velocity_penalty_coef": 0.1,
        "angvel_penalty_coef": 0.1,
        "action_smoothness_coef": 0.1,
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
        "release_stability_steps": 250,
        "release_hover_boost": 0.10,
        "release_angvel_boost": 0.5,
        "delta_error_bonus_coef": 0.0,
        "delta_error_window_steps": 0,
        "delta_error_bonus_clip": 0.0,
        "release_reward_window_steps": 1000,
        "accel_penalty_away_coef": 0.0,
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
        # 模仿专家残差的权重
        "imitation_weight": 32.0,
    }

    crash_distance_threshold = 5.0
    crash_tilt_threshold_deg = 20.0

    payload_parameters = {
        "payload_mass": 0.02,
        "offsets": [
            [0.4, 0.4, -0.4],
            [0.4, -0.4, -0.4],
            [-0.4, 0.4, -0.4],
            [-0.4, -0.4, -0.4],
        ],
        "release_start": 250,
        "release_interval": 1500,
        "release_start_range": [250, 500],
        "release_interval_range": [1500, 1750],
        "warning_steps": 250,
        "randomize_release":   True,  # 初始验证先固定
        "log_release_events": False,
    }

    randomization_parameters = False

    curriculum_parameters = None
