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



    episode_len_steps = 1500
    return_state_before_reset = False
    teacher_mode = True  # 启用特权/模仿

    # 奖励仿照 xadapt：存活奖励为主，角速度/线加速度/动作振荡惩罚，移除位置/补偿项，模仿不计入 reward
    reward_parameters = {
        "position_weight": 0.0,  # xadapt 不用位置奖励
        "crash_penalty": -10.0,  # xadapt: -10
        "attitude_penalty_coef": 0.00,
        "release_attitude_boost": 1.0,

        # 线速度惩罚关闭，改用线加速度惩罚
        "velocity_penalty_coef": 0.0,
        "angvel_penalty_coef": 10.00,  # xadapt: -0.2 (roll/pitch/yaw各0.2)
        "action_smoothness_coef": 0.0,  # xadapt: -0.06 (oscillate_coeff)

        "position_error_penalty_coef": 0.0,
        "z_error_penalty_coef": 0.0,

        "accel_penalty_away_coef": 0.0,  # xadapt: -0.01 (lin_accel_coeff)
        "accel_penalty_toward_coef": 0.0,

        # 模仿权重 - 分离推力和力矩
        "imitation_weight": 8.0,  # 模仿通过 BC loss 实现，不在 reward 中
        "imitation_weight_thrust": 8.0,
        "imitation_weight_torque": 8.0,

        # 物理感知模仿奖励 - 考虑质量和惯量的误差惩罚
        # 将动作误差转换为实际物理效应（加速度/角加速度）
        # 物理自然决定重要性：力矩误差 / 小惯量 >> 推力误差 / 大质量
        "physics_imitation_weight": 0.0,  # 总权重系数 (建议 0.01~0.1，因为角加速度量级大)
        # 存活奖励 - xadapt: 10
        "survive_bonus": 10.0,
    }

    crash_distance_threshold = 1.0
    crash_tilt_threshold_deg = 20.0

    # 补偿限制（物理量级）
    # 最大总载荷 = 4 × max_mass = 4 × 0.04 = 0.16 kg
    # thrust = 0.16 × 9.81 = 1.57 N → 留余量设为 2.0
    # torque = 0.16 × 9.81 × 0.4 = 0.628 N·m → 设为 1.0
    compensation_thrust_limit = 1.0  # N，匹配 controller config (覆盖4个载荷总重)
    compensation_torque_limits = [1.0, 1.0, 0.2]  # [roll, pitch, yaw] N·m，匹配 controller config

    payload_parameters = {
        "payload_mass": 0.02,
        "payload_mass_range": [0.00, 0.03],
        "randomize_payload_mass": True,
        "randomize_offsets_on_plane": False,
        "offset_plane_radial_jitter": 0.4,  #沿机臂方向的“半径扰动”，均匀分布 [-jitter, +jitter]
        "offset_plane_z_jitter": 0.8,  #垂直方向的“高度扰动”，均匀分布 [-jitter, +jitter]
        "offset_plane_r_max": 0.4,   #机臂方向最大偏移距离
        "offset_plane_z_max": 0.4,      #垂直方向最大偏移距离
        "force_offset_torque_scale": 0.00,  # 等效力矩系数：tau_eq = - r_com x F_total，1.0=全量补偿，0=关闭
        "offsets": [
            [0.4, 0.0, -0.4],
            [0.0, -0.4, -0.4],
            [-0.4, 0.0, -0.4],
            [0.0, 0.4, -0.4],
        ],
        "release_start": 50,
        "release_interval": 300,
        "release_start_range": [50, 100],
        "release_interval_range": [300, 350],
        "warning_steps": 0,
        "randomize_release":   True,  # 初始验证先固定
        "log_release_events": False,
    }

    randomization_parameters = False
    observation_parameters = {      
        # 特权观测控制 (Privileged Obs Control)
        "include_priv_mass": False,    # 索引 0: payload mass
        "include_priv_com": False,     # 索引 1-3: COM offset
        "include_priv_inertia": False, # 索引 4-6: true inertia
        
        # 基础观测控制 (Basic Obs Control) - 对 [observations] 向量进行屏蔽
        "include_base_rot": True,         # 索引 0-8: 旋转矩阵
        "include_base_angvel": True,      # 索引 9-11: 机体角速度
        "include_base_attached": True,    # 索引 12-15: 附着掩码
        "include_base_warning": True,     # 索引 16: 预警标志
        "include_base_prev_action": True, # 索引 17-19: 上一动作
    }

    curriculum_parameters = None
