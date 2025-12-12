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

    # 奖励分组：优先调“跟踪主信号”，其余保持 0 可视作关闭
    reward_parameters = {
        # 跟踪主信号：距离惩罚 + 连续跟踪奖励
        "tracking_tolerance": 0.1,          # 米，误差低于该值计入跟踪时长
        "tracking_reward_per_step": 5.0,   # 连续跟踪每步奖励，时长越久越大
        "tracking_penalty_coef": 1.0,       # 距离惩罚系数，-coef * dist

        # 动作/速度正则与抖动抑制
        "velocity_penalty_coef": 0.0,       # 线速度范数惩罚
        "angvel_penalty_coef": 0.000,       # 角速度惩罚
        "action_smoothness_coef": 0.0000,   # 动作变化惩罚
        "accel_penalty_away_coef": 0.00,    # 远离目标方向的加速度惩罚
        "accel_penalty_toward_coef": 0.0,   # 朝向目标的加速度惩罚
        "vel_away_penalty_coef": 0.0,       # 远离目标方向的速度惩罚

        # 补偿/幅值惩罚（目前关闭）
        "comp_torque_penalty_coef": 0.0,
        "comp_thrust_penalty_coef": 0.0,
        "comp_penalty_high_threshold": 1.0,
        "comp_torque_penalty_high_coef": 0.0,
        "comp_thrust_penalty_high_coef": 0.0,
        "comp_window_penalty_scale": 1,
        "comp_activation_bonus_coef": 0.0,

        # 安全/约束（当前为 0 即关闭）
        "crash_penalty": -10.0,
        "tilt_warning_deg": 0.0,
        "tilt_warning_penalty": 0.0,
        "height_warning": 0.0,
        "height_warning_penalty": 0.0,
        "height_safe_bonus": 0.0,
        "release_tilt_limit_deg": 0.0,
        "release_tilt_penalty": 0.0,
        "tilt_excess_threshold_deg": 0.0,
        "tilt_excess_coef": 0.0,
        "tilt_excess_exp": 0.0,

        # 兼容保留项（轨迹场景下通常为 0）
        "position_weight": 0.0,
        "attitude_penalty_coef": 0.0,
        "release_attitude_boost": 0.0,
        "stability_radius": 0.0,
        "stability_tilt_deg": 0.0,
        "stability_penalty": 0.0,
        "stability_velocity_penalty": 0.0,
        "position_error_penalty_coef": 0.0,
        "z_error_penalty_coef": 0.0,
        "vel_window_penalty_scale": 1,
        "smooth_window_penalty_scale": 1,
        "yaw_penalty_coef": 0.0,
        "hover_bonus_radius": 0.0,
        "hover_bonus_tilt_deg": 0.0,
        "hover_bonus_velocity": 0.0,
        "hover_bonus": 0.0,
        "release_stability_steps": 50,
        "release_hover_boost": 1.0,
        "release_angvel_boost": 1.0,
        "delta_error_bonus_coef": 0.0,
        "delta_error_window_steps": 0,
        "delta_error_bonus_clip": 0.0,
        "release_reward_window_steps": 200,

        # 模仿/生存
        "imitation_weight": 10.0,
        "survive_bonus": 10.0,
    }

    crash_distance_threshold = 3.0
    crash_tilt_threshold_deg = 15.0

    payload_parameters = {
        "payload_mass": 0.02,
        "offsets": [
            [0.4, 0.4, -0.4],
            [0.4, -0.4, -0.4],
            [-0.4, 0.4, -0.4],
            [-0.4, -0.4, -0.4],
        ],
        "release_start": 50, #以1500为基准
        "release_interval": 600,
        "release_start_range": [50, 100],
        "release_interval_range": [300, 350],
        "warning_steps": 100,
        "randomize_release":   False,  # 初始验证先固定
        "log_release_events": False,
    }

    # 轨迹配置：默认改为 XY 圆轨迹追踪
    trajectory_parameters = {
        "type": "circle",
        "radius": 1.0,
        "angular_speed_rad_per_step": 0.01,
        "center": [0.0, 0.0, 0.0],
        "z_height": 0.0,
        "phase_random": True,
    }

    randomization_parameters = False

    curriculum_parameters = None
'''
在你不改这些基准数值、只把 episode_len_steps 乘 5 的情况下，当前代码会自动按比例放大释放节奏（默认基准 1500）：

缩放系数 scale = 7500/1500 = 5。
release_start 从 50 → 250；release_interval 从 300 → 1500；range 也同步乘 5。
randomize_release=False 时就是固定时刻：约 250、1750、3250、4750（4 个挂点）。
warning_steps 不缩放，还是 100，相对整个回合的比例变小。
所以不用改数值，释放时刻会被自动推迟/拉开，覆盖更长的 episode；如果希望预警窗口也按比例放大，可再把 warning_steps 乘同样的系数。'''
