import torch # 导入 PyTorch

from aerial_gym.utils.math import * # 导入数学工具函数（如四元数、旋转矩阵操作）

from aerial_gym.utils.logging import CustomLogger # 导入自定义日志工具

logger = CustomLogger("base_lee_controller") # 初始化日志记录器

logger.setLevel("DEBUG") # 设置日志级别为 DEBUG


import pytorch3d.transforms as p3d_transforms # 导入 PyTorch3D 的变换库（用于旋转矩阵转四元数）

from aerial_gym.control.controllers.base_controller import * # 导入控制器基类


class BaseLeeController(BaseController):
    """
    此类将作为所有（Lee）控制器的基类。
    它将被特定的控制器子类继承。
    """

    def __init__(self, control_config, num_envs, device, mode="robot"):
        # 调用父类构造函数
        super().__init__(control_config, num_envs, device, mode)
        self.cfg = control_config # 控制器配置
        self.num_envs = num_envs
        self.device = device
        # 初始化滤波后的惯量矩阵（与当前 robot_inertia 同形）
        self.I_filt = None
        self.inertia_alpha = 0.2  # 0.0~1.0, 数值越小过渡越慢，按需要调
        self.J_sim_const = None  # 首次捕获的物理惯量(常值)
        self.use_inertia_virtualization = True  # 开关，必要时可设为 False


    def init_tensors(self, global_tensor_dict):
        """初始化控制器的所有张量，特别是控制增益 (Gains)。"""
        super().init_tensors(global_tensor_dict)

        # 从配置中读取增益（K）的最大值和最小值，并扩展到所有环境
        # K_pos_tensor: 位置增益 (P 项)
        self.K_pos_tensor_max = torch.tensor(
            self.cfg.K_pos_tensor_max, device=self.device, requires_grad=False
        ).expand(self.num_envs, -1)
        self.K_pos_tensor_min = torch.tensor(
            self.cfg.K_pos_tensor_min, device=self.device, requires_grad=False
        ).expand(self.num_envs, -1)
        # K_linvel_tensor: 线性速度增益 (D 项)
        self.K_linvel_tensor_max = torch.tensor(
            self.cfg.K_vel_tensor_max, device=self.device, requires_grad=False
        ).expand(self.num_envs, -1)
        self.K_linvel_tensor_min = torch.tensor(
            self.cfg.K_vel_tensor_min, device=self.device, requires_grad=False
        ).expand(self.num_envs, -1)
        # K_rot_tensor: 旋转/姿态误差增益 (P 项)
        self.K_rot_tensor_max = torch.tensor(
            self.cfg.K_rot_tensor_max, device=self.device, requires_grad=False
        ).expand(self.num_envs, -1)
        self.K_rot_tensor_min = torch.tensor(
            self.cfg.K_rot_tensor_min, device=self.device, requires_grad=False
        ).expand(self.num_envs, -1)
        # K_angvel_tensor: 角速度增益 (D 项)
        self.K_angvel_tensor_max = torch.tensor(
            self.cfg.K_angvel_tensor_max, device=self.device, requires_grad=False
        ).expand(self.num_envs, -1)
        self.K_angvel_tensor_min = torch.tensor(
            self.cfg.K_angvel_tensor_min, device=self.device, requires_grad=False
        ).expand(self.num_envs, -1)

        # 将当前增益设置为最大值和最小值的平均值（初始值）
        self.K_pos_tensor_current = (self.K_pos_tensor_max + self.K_pos_tensor_min) / 2.0
        
        # --- [ I 控制代码已移除 ] ---

        self.K_linvel_tensor_current = (self.K_linvel_tensor_max + self.K_linvel_tensor_min) / 2.0
        self.K_rot_tensor_current = (self.K_rot_tensor_max + self.K_rot_tensor_min) / 2.0
        self.K_angvel_tensor_current = (self.K_angvel_tensor_max + self.K_angvel_tensor_min) / 2.0

        # --- 控制器内部张量 (Internal Tensors) ---
        self.accel = torch.zeros((self.num_envs, 3), device=self.device) # 计算出的加速度指令
        # 最终的力和力矩指令 [fx, fy, fz, tx, ty, tz]
        self.wrench_command = torch.zeros(
            (self.num_envs, 6), device=self.device
        )  # [fx, fy, fz, tx, ty, tz]

        # 期望的姿态和角速度
        self.desired_quat = torch.zeros_like(self.robot_orientation)
        self.desired_body_angvel = torch.zeros_like(self.robot_body_angvel)
        self.euler_angle_rates = torch.zeros_like(self.robot_body_angvel)

        # 缓冲区张量，供 torch.jit 函数用于各种目的
        self.buffer_tensor = torch.zeros((self.num_envs, 3, 3), device=self.device)

    def __call__(self, *args, **kwargs):
        """使控制器对象可调用，等同于调用 update 方法。"""
        return self.update(*args, **kwargs)

    def reset_commands(self):
        """重置力和力矩指令。"""
        self.wrench_command[:] = 0.0

    def reset(self):
        """重置所有环境的控制器状态（参数随机化）。"""
        self.reset_idx(env_ids=None)

    def reset_idx(self, env_ids):
        """重置指定索引环境的控制器状态。"""
        if env_ids is None:
            # 如果 env_ids 为 None，则重置所有环境
            env_ids = torch.arange(self.K_rot_tensor_max.shape[0]) # 匹配 .expand() 后的 shape
        self.randomize_params(env_ids) # 随机化控制参数
        # 注意: 积分状态 self.pos_integral 的重置通常在 EnvManager/Task 的 reset 中处理

    def randomize_params(self, env_ids):
        """
        随机化指定环境索引的控制器增益参数（领域随机化）。
        """
        if self.cfg.randomize_params == False:
            # 如果配置禁用参数随机化，则直接返回
            return
            
        # 在 min 和 max 范围内随机生成当前增益
        self.K_pos_tensor_current[env_ids] = torch_rand_float_tensor(
            self.K_pos_tensor_min[env_ids], self.K_pos_tensor_max[env_ids]
        )
        self.K_linvel_tensor_current[env_ids] = torch_rand_float_tensor(
            self.K_linvel_tensor_min[env_ids], self.K_linvel_tensor_max[env_ids]
        )
        self.K_rot_tensor_current[env_ids] = torch_rand_float_tensor(
            self.K_rot_tensor_min[env_ids], self.K_rot_tensor_max[env_ids]
        )
        self.K_angvel_tensor_current[env_ids] = torch_rand_float_tensor(
            self.K_angvel_tensor_min[env_ids], self.K_angvel_tensor_max[env_ids]
        )

    def compute_acceleration(self, setpoint_position, setpoint_velocity):
        """
        计算实现位置和速度跟踪所需的加速度指令 (PD 控制)。
        """
        # 计算世界坐标系下的位置误差
        position_error_world_frame = setpoint_position - self.robot_position
        # 将期望速度从相对坐标系（假设是载具坐标系）旋转到世界坐标系
        setpoint_velocity_world_frame = quat_rotate(self.robot_vehicle_orientation, setpoint_velocity)
        # 计算速度误差
        velocity_error = setpoint_velocity_world_frame - self.robot_linvel

        # --- [ I 控制代码已移除 ] ---

        # 计算加速度指令 (PD 控制律)
        accel_command = (
            self.K_pos_tensor_current * position_error_world_frame # P 项
            + self.K_linvel_tensor_current * velocity_error         # D 项
        )
        return accel_command

    def compute_body_torque(self, setpoint_orientation, setpoint_angvel):
        """
        根据 Lee 姿态控制律计算所需的本体力矩。
        """
        # 低通惯量：I_filt = alpha * I_new + (1-alpha) * I_filt_prev
        I_cur = self.robot_inertia  # 来自外部同步的新惯量 3x3 或 [B,3,3]
        if self.I_filt is None:
            self.I_filt = I_cur.clone()
        else:
            self.I_filt = self.inertia_alpha * I_cur + (1.0 - self.inertia_alpha) * self.I_filt

        # 对期望的偏航角速度进行限幅
        setpoint_angvel[:, 2] = torch.clamp(
            setpoint_angvel[:, 2], -self.cfg.max_yaw_rate, self.cfg.max_yaw_rate
        )
        
        # 计算旋转误差四元数: R^T * R_d (从实际姿态到期望姿态的旋转)
        RT_Rd_quat = quat_mul(quat_inverse(self.robot_orientation), setpoint_orientation)
        # 转换为旋转矩阵
        RT_Rd = quat_to_rotation_matrix(RT_Rd_quat)
        
        # 计算旋转误差向量 (Lee 姿态控制的核心项)
        # rotation_error = 1/2 * vee_map(R_d^T * R - R^T * R_d)
        rotation_error = 0.5 * compute_vee_map(torch.transpose(RT_Rd, -2, -1) - RT_Rd)
        
        # 计算角速度误差
        # setpoint_angvel 需要被旋转到实际体坐标系：R^T * \omega_d
        angvel_error = self.robot_body_angvel - quat_rotate(RT_Rd_quat, setpoint_angvel)
        if self.I_filt is None:
            self.I_filt = self.robot_inertia.clone()
        else:
            # 简单 EMA；可把 alpha 绑定到 cfg，例如 alpha = self.cfg.inertia_filter_alpha
            alpha = 0.2
            self.I_filt = alpha * self.robot_inertia + (1 - alpha) * self.I_filt

        I_use = self.I_filt  # 用滤后的惯量做前馈
        # 原来的前馈行：
        # feed_forward_body_rates = torch.cross(
        #     self.robot_body_angvel,
        #     torch.bmm(self.robot_inertia, self.robot_body_angvel.unsqueeze(2)).squeeze(2),
        #     dim=1,
        # )
        # 改成：
        feed_forward_body_rates = torch.cross(
            self.robot_body_angvel,
            torch.bmm(I_use, self.robot_body_angvel.unsqueeze(2)).squeeze(2),
            dim=1,
        )
        
        # 计算最终力矩
        torque = (
            -self.K_rot_tensor_current * rotation_error # P 项 (姿态误差)
            - self.K_angvel_tensor_current * angvel_error # D 项 (角速度误差)
            + feed_forward_body_rates # 前馈项
        )
        return torque


# --- JIT 脚本函数 (Torch JIT Scripted Functions) ---

@torch.jit.script
def calculate_desired_orientation_from_forces_and_yaw(forces_command, yaw_setpoint):
    """
    根据力和期望偏航角计算所需的期望姿态（四元数）。
    这是 Lee 控制器中将推力向量投影到姿态的方法（较简化的版本）。
    """
    # 假设 forces_command 是在世界坐标系下，且 \vec{f} = m * (\ddot{x}_d + g)
    # 且 \vec{f} = R \vec{e}_3 T_c = R b_3 T_c
    # R b_3 = [c\phi s\theta c\psi + s\phi s\psi, c\phi s\theta s\psi - s\phi c\psi, c\phi c\theta]^T
    # 实际使用的是推力向量 \vec{f}_c = -m (\ddot{x}_d - g)
    c_phi_s_theta = forces_command[:, 0]
    s_phi = -forces_command[:, 1]
    c_phi_c_theta = forces_command[:, 2]

    # 计算期望的俯仰角和滚转角
    pitch_setpoint = torch.atan2(c_phi_s_theta, c_phi_c_theta)
    roll_setpoint = torch.atan2(s_phi, torch.sqrt(c_phi_c_theta**2 + c_phi_s_theta**2))
    
    # 组合滚转、俯仰、期望偏航角为四元数
    quat_desired = quat_from_euler_xyz_tensor(
        torch.stack((roll_setpoint, pitch_setpoint, yaw_setpoint), dim=1)
    )
    return quat_desired


# @torch.jit.script # 注意：此函数被注释掉 JIT 编译
def calculate_desired_orientation_for_position_velocity_control(
    forces_command, yaw_setpoint, rotation_matrix_desired
):
    """
    根据力和期望偏航角计算所需的期望姿态（旋转矩阵 -> 四元数）。
    更严格的 Lee 控制器方法：构建完整的期望旋转矩阵 R_d。
    """
    # 期望的 Z 轴方向 (b3_c): 沿推力方向，单位化
    b3_c = torch.div(forces_command, torch.norm(forces_command, dim=1).unsqueeze(1))
    
    # 期望的 X 轴方向 (temp_dir): 由期望偏航角确定
    temp_dir = torch.zeros_like(forces_command)
    temp_dir[:, 0] = torch.cos(yaw_setpoint)
    temp_dir[:, 1] = torch.sin(yaw_setpoint)

    # 期望的 Y 轴方向 (b2_c): 正交于 b3_c 和 temp_dir，然后单位化
    b2_c = torch.cross(b3_c, temp_dir, dim=1)
    b2_c = torch.div(b2_c, torch.norm(b2_c, dim=1).unsqueeze(1))
    
    # 期望的 X 轴方向 (b1_c): 正交于 b2_c 和 b3_c
    b1_c = torch.cross(b2_c, b3_c, dim=1)

    # 构造期望旋转矩阵 R_d
    rotation_matrix_desired[:, :, 0] = b1_c
    rotation_matrix_desired[:, :, 1] = b2_c
    rotation_matrix_desired[:, :, 2] = b3_c
    
    # 旋转矩阵转换为四元数 (使用 pytorch3d 库)
    q = p3d_transforms.matrix_to_quaternion(rotation_matrix_desired)
    # 注意: pytorch3d 返回 [w, x, y, z]，此处转换为 [x, y, z, w] 格式
    quat_desired = torch.stack((q[:, 1], q[:, 2], q[:, 3], q[:, 0]), dim=1)

    # 处理四元数符号（可选，确保 w 分量为正）
    sign_qw = torch.sign(quat_desired[:, 3])
    # quat_desired = quat_desired * sign_qw.unsqueeze(1) # 此行被注释

    return quat_desired


# quat_from_rotation_matrix(rotation_matrix_desired) # 此行为注释

@torch.jit.script
def euler_rates_to_body_rates(robot_euler_angles, desired_euler_rates, rotmat_euler_to_body_rates):
    """
    将欧拉角变化率（Roll, Pitch, Yaw Rate）转换为本体角速度（p, q, r）。
    """
    s_pitch = torch.sin(robot_euler_angles[:, 1]) # sin(俯仰角)
    c_pitch = torch.cos(robot_euler_angles[:, 1]) # cos(俯仰角)

    s_roll = torch.sin(robot_euler_angles[:, 0]) # sin(滚转角)
    c_roll = torch.cos(robot_euler_angles[:, 0]) # cos(滚转角)

    # 构造变换矩阵 (M)
    rotmat_euler_to_body_rates[:, 0, 0] = 1.0
    rotmat_euler_to_body_rates[:, 1, 1] = c_roll
    rotmat_euler_to_body_rates[:, 0, 2] = -s_pitch
    rotmat_euler_to_body_rates[:, 2, 1] = -s_roll
    rotmat_euler_to_body_rates[:, 1, 2] = s_roll * c_pitch
    rotmat_euler_to_body_rates[:, 2, 2] = c_roll * c_pitch

    # 本体角速度 = 变换矩阵 * 欧拉角变化率
    return torch.bmm(rotmat_euler_to_body_rates, desired_euler_rates.unsqueeze(2)).squeeze(2)