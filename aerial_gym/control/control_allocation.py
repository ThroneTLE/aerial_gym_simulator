import torch # 导入 PyTorch
from aerial_gym.control.motor_model import MotorModel # 导入电机模型
from aerial_gym.utils.logging import CustomLogger # 导入自定义日志工具

logger = CustomLogger("control_allocation") # 初始化日志记录器


class ControlAllocator:
    def __init__(self, num_envs, dt, config, device):
        self.num_envs = num_envs # 环境数量
        self.dt = dt # 模拟步长
        self.cfg = config # 控制分配器配置（通常来自 BaseQuadCfg）
        self.device = device # 设备
        
        # 力施加级别：决定是将总力矩施加到根链接 ('root_link') 还是将推力施加到电机链接 ('motor_link')
        self.force_application_level = self.cfg.force_application_level
        
        # 电机旋转方向张量（用于计算偏航扭矩）
        self.motor_directions = torch.tensor(self.cfg.motor_directions, device=self.device)
        
        # 力矩分配矩阵 (Allocation Matrix): 6xN 矩阵，将 N 个电机推力映射到 6 维力和力矩
        self.force_torque_allocation_matrix = torch.tensor(
            self.cfg.allocation_matrix, device=self.device, dtype=torch.float32
        )
        # 伪逆分配矩阵 (Pseudo-inverse): 用于将 6 维指令反向映射到期望的电机推力
        self.inv_force_torque_allocation_matrix = torch.linalg.pinv(
            self.force_torque_allocation_matrix
        )
        
        # 最终输出的 6 维合力/力矩 (Wrench) 张量
        self.output_wrench = torch.zeros((self.num_envs, 6), device=self.device)

        # 检查分配矩阵的尺寸
        assert (
            len(self.cfg.allocation_matrix[0]) == self.cfg.num_motors
        ), "Allocation matrix must have 6 rows and num_motors columns."

        # 再次初始化分配矩阵（确保类型正确）
        self.force_torque_allocation_matrix = torch.tensor(
            self.cfg.allocation_matrix, device=self.device, dtype=torch.float32
        )
        
        # 检查分配矩阵的秩 (Rank)
        alloc_matrix_rank = torch.linalg.matrix_rank(self.force_torque_allocation_matrix)
        if alloc_matrix_rank < 6:
            # 对于四旋翼（4 个电机），矩阵秩通常小于 6，这是正常的，但会发出警告
            print("WARNING: allocation matrix is not full rank. Rank: {}".format(alloc_matrix_rank))
            
        # 将矩阵扩展到环境数量 (num_envs, 6, num_motors)
        self.force_torque_allocation_matrix = self.force_torque_allocation_matrix.expand(
            self.num_envs, -1, -1
        )
        # 将伪逆矩阵扩展到环境数量 (num_envs, num_motors, 6)
        self.inv_force_torque_allocation_matrix = torch.linalg.pinv(
            torch.tensor(self.cfg.allocation_matrix, device=self.device, dtype=torch.float32)
        ).expand(self.num_envs, -1, -1)
        
        # 实例化电机动力学模型
        self.motor_model = MotorModel(
            num_envs=self.num_envs,
            dt=self.dt,
            motors_per_robot=self.cfg.num_motors,
            config=self.cfg.motor_model_config,
            device=self.device,
        )
        # 警告：此分配器未考虑执行器饱和（推力限幅）问题，可能导致分配不理想
        logger.warning(
            f"Control allocation does not account for actuator limits. This leads to suboptimal allocation"
        )

    def allocate_output(self, command, output_mode):
        """
        根据控制指令和施加级别，计算最终施加的力和力矩。
        command: 控制器输出的指令（可以是力和力矩，也可以是纯推力）。
        output_mode: 'forces'（纯力指令）或 'wrench'（力和力矩指令）。
        """
        if self.force_application_level == "motor_link":
            # --- 模式 A: 将推力施加到单个电机链接 ---
            if output_mode == "forces":
                # 简单推力指令（通常不用于 Lee 控制器）
                motor_thrusts = self.update_motor_thrusts_with_forces(command)
            else:
                # Wrench 指令（用于 Lee 控制器）
                motor_thrusts = self.update_motor_thrusts_with_wrench(command)
                
            # 将电机推力转换为施加到电机链接上的 3D 力和力矩
            forces, torques = self.calc_motor_forces_torques_from_thrusts(motor_thrusts)

        else:
            # --- 模式 B: 将总合力/力矩施加到根链接 ---
            output_wrench = self.update_wrench(command)
            # 提取合力（unsqueeze(1) 确保形状正确 (N, 1, 3)）
            forces = output_wrench[:, 0:3].unsqueeze(1)
            # 提取合力矩
            torques = output_wrench[:, 3:6].unsqueeze(1)

        return forces, torques

    def update_wrench(self, ref_wrench):
        """
        [模式 B] 更新输出合力/力矩（用于施加到根链接）。
        """
        # 1. 分配：使用伪逆矩阵将期望的 6D Wrench 转换为期望的 N 个电机推力
        ref_motor_thrusts = torch.bmm(
            self.inv_force_torque_allocation_matrix, ref_wrench.unsqueeze(-1)
        ).squeeze(-1)

        # 2. 动力学：应用电机模型，计算实际的电机推力
        current_motor_thrust = self.motor_model.update_motor_thrusts(ref_motor_thrusts)

        # 3. 反分配：将实际推力映射回实际输出的 6D Wrench
        self.output_wrench[:] = torch.bmm(
            self.force_torque_allocation_matrix, current_motor_thrust.unsqueeze(-1)
        ).squeeze(-1)

        return self.output_wrench

    def update_motor_thrusts_with_forces(self, ref_forces):
        """
        [模式 A, forces] 仅更新电机推力（如果命令已经是期望推力）。
        """
        current_motor_thrust = self.motor_model.update_motor_thrusts(ref_forces)
        return current_motor_thrust

    def update_motor_thrusts_with_wrench(self, ref_wrench):
        """
        [模式 A, wrench] 将 6D Wrench 转换为电机推力，并应用电机模型。
        """
        # 1. 分配：将 6D Wrench 转换为期望推力
        ref_motor_thrusts = torch.bmm(
            self.inv_force_torque_allocation_matrix, ref_wrench.unsqueeze(-1)
        ).squeeze(-1)

        # 2. 动力学：应用电机模型，计算实际电机推力
        current_motor_thrust = self.motor_model.update_motor_thrusts(ref_motor_thrusts)

        return current_motor_thrust

    def reset_idx(self, env_ids):
        """重置指定环境索引的电机模型参数。"""
        self.motor_model.reset_idx(env_ids)
        # 这里可以根据需要随机化分配矩阵

    def reset(self):
        """重置所有环境的电机模型参数。"""
        self.motor_model.reset()
        # 这里可以根据需要随机化分配矩阵

    def calc_motor_forces_torques_from_thrusts(self, motor_thrusts):
        """
        [模式 A] 从单个电机的推力（标量）计算出施加到电机链接上的 3D 力和力矩。
        """
        # 电机力 (Forces): 推力始终沿 Z 轴（向下或向上，取决于惯例）
        motor_forces = torch.stack(
            [
                torch.zeros_like(motor_thrusts), # Fx = 0
                torch.zeros_like(motor_thrusts), # Fy = 0
                motor_thrusts,                   # Fz = 推力
            ],
            dim=2, # 结果形状 (N, num_motors, 3)
        )
        
        # 电机扭矩 (Torques): 由推力引起的力矩（反作用力矩）
        # T = Cq * F * direction
        cq = self.cfg.motor_model_config.thrust_to_torque_ratio # 推力到扭矩的比例 (Cq)
        # 力矩 = Cq * 推力 * 电机旋转方向
        # motor_directions[None, :, None] 用于在环境维度和 3D 维度上扩展
        motor_torques = cq * motor_forces * (-self.motor_directions[None, :, None])
        
        return motor_forces, motor_torques

    def set_single_allocation_matrix(self, alloc_matrix):
        """
        设置或更新分配矩阵。
        """
        if alloc_matrix.shape != (6, self.cfg.num_motors):
            raise ValueError("Allocation matrix must have shape (6, num_motors)")
            
        # 更新并扩展分配矩阵
        self.force_torque_allocation_matrix[:] = torch.tensor(
            alloc_matrix, device=self.device, dtype=torch.float32
        ).expand(self.num_envs, -1, -1)
        
        # 重新计算并更新伪逆矩阵
        self.inv_force_torque_allocation_matrix[:] = torch.linalg.pinv(
            self.force_torque_allocation_matrix
        ).expand(self.num_envs, -1, -1)