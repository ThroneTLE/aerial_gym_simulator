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
        # 降低主奖励权重，让补偿相关信号不被淹没
        "position_weight": 1.2,
        # 在警告期/释放扰动下可轻微减弱
        "crash_penalty": -120.0,
        "attitude_penalty_coef": 1.0,
        "release_attitude_boost": 2.0,
        # 先减轻惩罚以鼓励探索补偿动作
        "comp_torque_penalty_coef": 0.02,
        "comp_thrust_penalty_coef": 0.02,
        # 放松速度/角速度/平滑惩罚，避免策略因惩罚不敢动作
        "velocity_penalty_coef": 0.25,
        "angvel_penalty_coef": 0.12,
        "action_smoothness_coef": 0.05,
        "tilt_warning_deg": 20.0,
        "tilt_warning_penalty": -2.0,
        # 平滑高度约束 + 安全高度小奖励
        "height_warning": 0.2,
        "height_warning_penalty": 1.0,  # 线性系数（正数），内部会取负号
        "height_safe_bonus": 0.05,
        "release_tilt_limit_deg": 25.0,
        "release_tilt_penalty": -5.0,
        "stability_radius": 0.5,
        "stability_tilt_deg": 8.0,
        "stability_penalty": 0.35,  # 在小半径内给予小奖励
        "stability_velocity_penalty": 0.08,
        "position_error_penalty_coef": 1.5,
        "yaw_penalty_coef": 0.0,
        # Hover bonus: give a small reward when staying close to target, upright and slow.
        "hover_bonus_radius": 0.3,  # 收紧半径
        "hover_bonus_tilt_deg": 6.0,  # 收紧倾角
        "hover_bonus_velocity": 0.5,
        "hover_bonus": 0.17,  # 略升奖励
        # When a payload was just released, temporarily boost stability incentives.
        "release_stability_steps": 30,
        "release_hover_boost": 2.0,
        "release_angvel_boost": 1.5,
        # 奖励距离误差缩小量，仅在预警/释放窗口内生效
        "delta_error_bonus_coef": 150.0,
        "delta_error_window_steps": 300,
        "delta_error_bonus_clip": 0.2,  # 防止偶发大步进
        # 补偿分段加重惩罚
        "comp_penalty_high_threshold": 0.9,
        "comp_torque_penalty_high_coef": 0.08,
        "comp_thrust_penalty_high_coef": 0.08,
        # 预警/释放窗口内额外放松惩罚、鼓励动作
        "comp_window_penalty_scale": 0.35,
        "vel_window_penalty_scale": 0.35,
        "smooth_window_penalty_scale": 0.6,
        "comp_activation_bonus_coef": 0.04,
    }

    crash_distance_threshold = 9.0  # meters
    crash_tilt_threshold_deg = 25.0

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
