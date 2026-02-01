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
    privileged_observation_space_dim = 18
    action_space_dim = 3  # thrust + roll torque + pitch torque (无 yaw)
    controller_action_dim = 8  # 保持与 Lee 控制器兼容，yaw 补偿位置设为 0


    episode_len_steps = 1500 # 主要用于释放任务训练
    return_state_before_reset = False
    teacher_mode = True  # 启用特权/模仿

    # 奖励仿照 xadapt：存活奖励为主，角速度/线加速度/动作振荡惩罚，移除位置/补偿项，模仿不计入 reward
    # 奖励参数配置 (Reward Parameters)
    # ⚠️ 重要提示：请仔细阅读符号说明！
    # 1. 奖励项 (Bonus): 使得总分增加，系数应为【正数 (+)】。
    # 2. 惩罚项 (Penalty): 使得总分减少，代码中通常处理为 `-coef * error`。
    #    所以只需提供【正数 (+)】系数即可代表惩罚力度。
    #    若在此处填负数，会导致 `- (-coef) = +coef`，反而变成奖励误差！
    
    reward_parameters = {
        # === 核心奖励 (Positives) ===
        # 位置权重: 乘在 pos_reward(指数函数,距离越近值越大) 上。
        # 【正数 (+)】: 鼓励靠近目标 (3.0 推荐)
        # 【0.0】: 关闭显式引导，靠存活奖励隐式引导 (xadapt模式)
        # 【禁止负数】: 负数会导致无人机主动飞离目标！
        "position_weight": 0.0, 
        
        # 存活奖励: 每一步存活给予的固定分值。
        # 【正数 (+)】: 鼓励活得更久 (10.0 推荐)
        "survive_bonus": 10.0,
        
        # === 惩罚项 (Penalties) - 请填【正数 (+)】代表惩罚力度 ===
        
        # 坠毁惩罚: 触发坠毁时的额外扣分。
        # 【负数 (-)】: 这里是个特例，代码直接累加 `rewards += crash_penalty`
        # 所以必须填负数！(如 -10.0)
        "crash_penalty": -10.0,

        # 姿态惩罚系数: 代码逻辑 `reward -= coef * roll_pitch_error`
        # 【正数 (+)】: 惩罚倾斜 (推荐 0.0 或较小值，避免限制机动性)
        "attitude_penalty_coef": 0.900,
        "release_attitude_boost": 1.0, # 释放阶段的姿态惩罚倍率

        # --- 动作与稳定性惩罚 ---
        # 速度/角速度惩罚: 代码逻辑 `reward -= coef * norm(vel)`
        # 【正数 (+)】: 限制速度/角速度，减少过冲和震荡
        "velocity_penalty_coef": 0.0,   # 线速度惩罚 (通常关)
        "angvel_penalty_coef": 0.200,    # 角速度惩罚 (抑制高频震荡, 推荐 0.1~1.0)
        "action_smoothness_coef": 2.0,  # 动作平滑惩罚 (抑制动作突变, 推荐 0.05~2.0)

        # 显式误差惩罚 (可选)
        "position_error_penalty_coef": 0.0, # 距离误差的线性惩罚 (通常用 position_weight 的指数奖励代替)
        "z_error_penalty_coef": 0.0,        # Z轴高度误差惩罚

        # 加速度惩罚 (xadapt):
        # 【正数 (+)】: 惩罚背离目标的加速度 (Away) 或朝向目标的加速度 (Toward)
        "accel_penalty_away_coef": 0.0,   
        "accel_penalty_toward_coef": 0.0,

        # === 模仿学习奖励 (Imitation) ===
        # 模仿惩罚: 代码逻辑 `reward -= weight * (policy - expert)^2`
        # 【正数 (+)】: 强迫策略动作接近专家动作
        # 注意: 只有在 teacher_mode=True 且 imitation_w > 0 时生效
        "imitation_weight": 8.0,          # 通用权重
        "imitation_weight_thrust": 8.0,   # 推力通道独立权重
        "imitation_weight_torque": 8.0,   # 力矩通道独立权重

        # 物理感知模仿 (Experimental)
        "physics_imitation_weight": 0.0, 
    }

    crash_distance_threshold = 2.0
    crash_tilt_threshold_deg = 25.0

    # 补偿限制（物理量级）
    # 最大总载荷 = 4 × max_mass = 4 × 0.4 = 1.6 kg
    # thrust = 1.6 × 9.81 × 1.25 = 19.6 N → 设为 20.0
    # torque = 1.6 × 9.81 × 0.4 = 6.3 N·m → 设为 12.0 (Supported by 40N motors -> 12.7Nm max)
    compensation_thrust_limit = 20.0  # N，覆盖4个载荷总重
    compensation_torque_limits = [12.0, 12.0, 1.0]  # [roll, pitch, yaw] N·m (Upgraded)

    payload_parameters = {
        "payload_mass": 0.2,  # 默认单载荷质量 (kg)
        "payload_mass_range": [0.0, 0.35],  # 随机化范围 0-0.4kg
        "randomize_payload_mass": False,
        "randomize_offsets_on_plane": False,
        "offset_plane_radial_jitter": 0.2,  # 沿机臂方向的半径扰动 (450mm轴距)
        "offset_plane_z_jitter": 0.4,  #垂直方向的“高度扰动”，均匀分布 [-jitter, +jitter]
        "offset_plane_r_max": 0.2,   # 机臂方向最大偏移距离 (450mm轴距)
        "offset_plane_z_max": 0.4,      #垂直方向最大偏移距离
        "force_offset_torque_scale": 0.00,  # 等效力矩系数：tau_eq = - r_com x F_total，1.0=全量补偿，0=关闭
        "offsets": [
            [0.2, 0.0, -0.1],
            [0.0, -0.2, -0.1],
            [-0.2, 0.0, -0.1],
            [0.0, 0.2, -0.1],
        ],
        "release_start": 200,  # 第一阶段：尽早开始释放任务 
        "release_interval": 300,
        "release_start_range": [250, 350],  # 随机化释放开始时间
        "release_interval_range": [300, 350],
        "warning_steps": 0,
        "randomize_release":   True,  # 初始验证先固定
        "log_release_events": False,
    }

    randomization_parameters = {
        # 初始状态噪声 (Initial State Noise)
        "initial_position_noise": [0.0, 0.0, 0.0],  # 初始位置偏移
        "initial_orientation_noise_deg": [0.0, 0.0, 0.0],  # 初始姿态角偏移（度）
        
        # 扩展物理参数随机化 (Extended Physical Params Randomization)
        # 电机参数 (Motor Params)
        "randomize_motor_thrust_constant": False,  # 随机化电机推力系数
        "motor_thrust_constant_range_scale": [0.7, 1.0], # 推力系数缩放比例 (相对于标称值)
        
        "randomize_motor_time_constant": False,  # 随机化电机时间常数
        "motor_time_constant_range": [0.02, 0.08], # 时间常数范围 (秒)
        
        # 阻力系数 (Drag Coeffs)
        "randomize_drag_coefficients": False,  # 随机化阻力系数
        "lin_drag_coeff_range": [0.0, 0.2],  # 线性阻力系数范围
        "ang_drag_coeff_range": [0.0, 0.05], # 角阻力系数范围
        
        # 恒定外力干扰 (Wind/Constant External Disturbance)
        "randomize_external_disturbance": False,  # 启用恒定外力
        "external_force_range": [0.0, 0.20], # 外力大小范围 (牛顿)
        "external_torque_range": [0.0, 0.0], # 外力矩范围 (牛顿·米)
    }

    observation_parameters = {      
        # 特权观测控制 (Privileged Obs Control)
        # 0: mass, 1-3: com, 4-6: inertia
        "include_priv_mass": True,
        "include_priv_com": True,
        "include_priv_inertia": True,
        
        # New Extended Privileged Info
        "include_priv_motor_thrust": True, # 7: motor thrust scale
        "include_priv_motor_tc": True,    # 8: motor time constant
        "include_priv_drag_lin": True,    # 9-11: linear drag coeffs
        "include_priv_drag_ang": True,    # 12-14: angular drag coeffs
        "include_priv_disturbance": True, # 15-17: external force (wind)
        
        # 基础观测控制 (Basic Obs Control) - 对 [observations] 向量进行屏蔽
        "include_base_rot": True,         # 索引 0-8: 旋转矩阵
        "include_base_linvel": True,     # 索引 9-11: 线速度 (REVERTED)
        "include_base_angvel": True,      # 索引 12-14: 机体角速度
        "include_base_attached": True,    # 索引 15-18: 附着掩码 (REVERTED)
        "include_base_warning": True,     # 索引 19: 预警标志 (REVERTED)
        "include_base_prev_action": True, # 索引 20-22: 上一动作
    }

    curriculum_parameters = None
