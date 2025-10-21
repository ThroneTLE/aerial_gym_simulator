# 导入机器人基类
from aerial_gym.robots.base_robot import BaseRobot

# 导入控制分配器
from aerial_gym.control.control_allocation import ControlAllocator
# 导入控制器注册表
from aerial_gym.registry.controller_registry import controller_registry

import torch # 导入 PyTorch
import numpy as np # 导入 NumPy

# 导入数学工具函数（如四元数、欧拉角转换）
from aerial_gym.utils.math import *

from aerial_gym.utils.logging import CustomLogger # 导入自定义日志工具

logger = CustomLogger("base_multirotor") # 初始化日志记录器


class BaseMultirotor(BaseRobot):
    """
    多旋翼机器人（如四旋翼）的基类。
    它继承自 BaseRobot，并负责处理控制分配、空气动力学和扰动。
    
    此基类应被具有特定传感器或执行器的子类继承。
    机器人的控制器配置用于初始化其控制器。
    """

    def __init__(self, robot_config, controller_name, env_config, device):
        logger.debug("Initializing BaseQuadrotor")
        super().__init__(
            robot_config=robot_config,
            controller_name=controller_name,
            env_config=env_config,
            device=device,
        )
        logger.warning(f"Creating {self.num_envs} multirotors.")
        
        # 力施加级别：'root_link' 或 'motor_link'
        self.force_application_level = self.cfg.control_allocator_config.force_application_level
        
        # 根据控制器名称设置输出模式
        if controller_name == "no_control":
            self.output_mode = "forces" # 如果没有控制器，直接处理力指令
        else:
            self.output_mode = "wrench" # 否则处理力和力矩指令 (Wrench)

        # 逻辑检查：如果力施加到 'root_link'，则不能使用 'no_control'
        if self.force_application_level == "root_link" and controller_name == "no_control":
            raise ValueError(
                "Force application level 'root_link' cannot be used with 'no_control'."
            )

        # 初始化为 None 的内部张量和对象
        self.robot_state = None
        self.robot_force_tensors = None
        self.robot_torque_tensors = None
        self.action_tensor = None
        self.max_init_state = None
        self.min_init_state = None
        self.max_force_and_torque_disturbance = None
        self.max_torque_disturbance = None # 注意：这个变量似乎未被使用
        self.controller_input = None
        self.control_allocator = None
        self.output_forces = None
        self.output_torques = None

        logger.debug("[DONE] Initializing BaseQuadrotor")

    def init_tensors(self, global_tensor_dict):
        """
        初始化机器人的状态、力、扭矩和动作张量。
        这些张量作为环境主张量的切片被传入。
        """
        super().init_tensors(global_tensor_dict)
        
        # --- 初始化并添加到全局字典的派生状态张量 ---
        # 载具坐标系方向（通常是偏航角被移除后的方向）
        self.robot_vehicle_orientation = torch.zeros_like(
            self.robot_orientation, requires_grad=False, device=self.device
        )
        # 载具坐标系线速度
        self.robot_vehicle_linvel = torch.zeros_like(
            self.robot_linvel, requires_grad=False, device=self.device
        )
        # 本体坐标系角速度
        self.robot_body_angvel = torch.zeros_like(
            self.robot_linvel, requires_grad=False, device=self.device
        )
        # 本体坐标系线速度
        self.robot_body_linvel = torch.zeros_like(
            self.robot_linvel, requires_grad=False, device=self.device
        )
        # 欧拉角 (Roll, Pitch, Yaw)
        self.robot_euler_angles = torch.zeros_like(
            self.robot_linvel, requires_grad=False, device=self.device
        )
        
        # 添加到全局张量字典
        global_tensor_dict["robot_vehicle_orientation"] = self.robot_vehicle_orientation
        global_tensor_dict["robot_vehicle_linvel"] = self.robot_vehicle_linvel
        global_tensor_dict["robot_body_angvel"] = self.robot_body_angvel
        global_tensor_dict["robot_body_linvel"] = self.robot_body_linvel
        global_tensor_dict["robot_euler_angles"] = self.robot_euler_angles

        # 机器人动作空间维度
        global_tensor_dict["num_robot_actions"] = self.controller_config.num_actions

        # 初始化控制器内部张量
        self.controller.init_tensors(global_tensor_dict)
        
        # 初始化动作输入张量
        self.action_tensor = torch.zeros(
            (self.num_envs, self.controller_config.num_actions),
            device=self.device,
            requires_grad=False,
        )

        # --- 初始化机器人初始状态参数 (Init State Params) ---
        # [x, y, z, roll, pitch, yaw, 1.0, vx, vy, vz, wx, wy, wz] 的最小/最大张量
        self.min_init_state = torch.tensor(
            self.cfg.init_config.min_init_state, device=self.device, requires_grad=False
        ).expand(self.num_envs, -1)
        self.max_init_state = torch.tensor(
            self.cfg.init_config.max_init_state, device=self.device, requires_grad=False
        ).expand(self.num_envs, -1)

        # --- 扰动参数 (Disturbance Params) ---
        # [fx, fy, fz, tx, ty, tz] 的最大值张量
        self.max_force_and_torque_disturbance = torch.tensor(
            self.cfg.disturbance.max_force_and_torque_disturbance,
            device=self.device,
            requires_grad=False,
        ).expand(self.num_envs, -1)

        # --- 控制器和分配器初始化 ---
        # 控制器输入（占位符，如果不需要）
        self.controller_input = torch.zeros(
            (self.num_envs, self.num_actions), device=self.device, requires_grad=False
        )
        # 实例化控制分配器
        self.control_allocator = ControlAllocator(
            num_envs=self.num_envs,
            dt=self.dt,
            config=self.cfg.control_allocator_config,
            device=self.device,
        )

        # --- 阻尼系数 (Damping Coefficients) ---
        self.body_vel_linear_damping_coefficient = torch.tensor(
            self.cfg.damping.linvel_linear_damping_coefficient,
            device=self.device,
            requires_grad=False,
        )
        self.body_vel_quadratic_damping_coefficient = torch.tensor(
            self.cfg.damping.linvel_quadratic_damping_coefficient,
            device=self.device,
            requires_grad=False,
        )
        self.angvel_linear_damping_coefficient = torch.tensor(
            self.cfg.damping.angular_linear_damping_coefficient,
            device=self.device,
            requires_grad=False,
        )
        self.angvel_quadratic_damping_coefficient = torch.tensor(
            self.cfg.damping.angular_quadratic_damping_coefficient,
            device=self.device,
            requires_grad=False,
        )

        # --- 施力掩码 (Application Mask) ---
        if self.force_application_level == "motor_link":
            # 如果施力到电机链接，使用分配器配置中的掩码（电机链接索引）
            self.application_mask = torch.tensor(
                self.cfg.control_allocator_config.application_mask,
                device=self.device,
                requires_grad=False,
            )
        else:
            # 如果施力到根链接，掩码只包含根链接（索引 0）
            self.application_mask = torch.tensor([0], device=self.device, requires_grad=False)

        # 电机旋转方向
        self.motor_directions = torch.tensor(
            self.cfg.control_allocator_config.motor_directions,
            device=self.device,
            requires_grad=False,
        )

        # 初始化输出力和扭矩张量（形状与全局力/扭矩张量相同）
        self.output_forces = torch.zeros_like(
            global_tensor_dict["robot_force_tensor"], device=self.device
        )
        self.output_torques = torch.zeros_like(
            global_tensor_dict["robot_torque_tensor"], device=self.device
        )

    def reset(self):
        """重置所有环境的机器人状态和控制器。"""
        self.reset_idx(torch.arange(self.num_envs))

    def reset_idx(self, env_ids):
        """
        重置指定环境 ID 的机器人状态。
        """
        if len(env_ids) == 0:
            return
            
        # 1. 随机采样初始状态（比例值和欧拉角）
        # [ratio_x, ratio_y, ratio_z, roll_rad, pitch_rad, yaw_rad, 1.0, vx, vy, vz, wx, wy, wz]
        random_state = torch_rand_float_tensor(self.min_init_state, self.max_init_state)

        # 2. 计算实际位置（通过在环境边界内插值比例值）
        self.robot_state[env_ids, 0:3] = torch_interpolate_ratio(
            self.env_bounds_min, self.env_bounds_max, random_state[:, 0:3]
        )[env_ids]

        # 3. 欧拉角转换为四元数
        self.robot_state[env_ids, 3:7] = quat_from_euler_xyz_tensor(random_state[env_ids, 3:6])

        # 4. 设置初始速度和角速度
        self.robot_state[env_ids, 7:10] = random_state[env_ids, 7:10]
        self.robot_state[env_ids, 10:13] = random_state[env_ids, 10:13]

        # 5. 重置控制器参数和分配器
        self.controller.randomize_params(env_ids=env_ids)
        self.control_allocator.reset_idx(env_ids)

        # 在重置后更新状态（计算本体速度、欧拉角等派生状态）
        self.update_states()

    def clip_actions(self):
        """
        将动作张量限幅到控制器输入范围。
        """
        # 简单地将动作限幅到 [-10.0, 10.0]
        self.action_tensor[:] = torch.clamp(self.action_tensor, -10.0, 10.0)

    def apply_disturbance(self):
        """
        以一定概率向机器人施加随机的力和扭矩扰动。
        """
        if not self.cfg.disturbance.enable_disturbance:
            return
            
        # 根据概率 (prob_apply_disturbance) 生成一个伯努利分布的张量，决定哪些环境施加扰动
        disturbance_occurence = torch.bernoulli(
            self.cfg.disturbance.prob_apply_disturbance
            * torch.ones((self.num_envs), device=self.device)
        )
        
        # 对根链接（索引 0）施加随机力扰动 [fx, fy, fz]
        self.robot_force_tensors[:, 0, 0:3] += torch_rand_float_tensor(
            -self.max_force_and_torque_disturbance[:, 0:3],
            self.max_force_and_torque_disturbance[:, 0:3],
        ) * disturbance_occurence.unsqueeze(1)
        
        # 对根链接（索引 0）施加随机扭矩扰动 [tx, ty, tz]
        self.robot_torque_tensors[:, 0, 0:3] += torch_rand_float_tensor(
            -self.max_force_and_torque_disturbance[:, 3:6],
            self.max_force_and_torque_disturbance[:, 3:6],
        ) * disturbance_occurence.unsqueeze(1)

    def control_allocation(self, command_wrench, output_mode):
        """
        将力和扭矩指令分配给电机，并应用电机模型。
        """
        # 调用控制分配器计算最终施加的力和扭矩
        forces, torques = self.control_allocator.allocate_output(command_wrench, output_mode)

        # 将分配器计算的力和扭矩写入到机器人的输出张量中
        self.output_forces[:, self.application_mask, :] = forces
        self.output_torques[:, self.application_mask, :] = torques

    def call_controller(self):
        """
        调用控制器，进行动作限幅，执行控制分配，并将结果写入机器人的力/扭矩张量。
        """
        self.clip_actions() # 限幅动作输入
        
        # 调用控制器，控制器将动作张量转换为 wrench 命令
        controller_output = self.controller(self.action_tensor)
        # 执行控制分配
        self.control_allocation(controller_output, self.output_mode)

        # 将计算出的力和扭矩写入到 Isaac Gym 将使用的张量中
        self.robot_force_tensors[:] = self.output_forces
        self.robot_torque_tensors[:] = self.output_torques

    def simulate_drag(self):
        """
        模拟机器人的空气阻力（线速度和角速度的阻尼）。
        """
        # --- 线速度阻尼 (Linear Velocity Drag) ---
        # 线性阻尼: -C_L * v
        self.robot_body_vel_drag_linear = (
            -self.body_vel_linear_damping_coefficient * self.robot_body_linvel
        )
        # 二次阻尼: -C_Q * |v| * v
        self.robot_body_vel_drag_quadratic = (
            -self.body_vel_quadratic_damping_coefficient
            * torch.norm(self.robot_body_linvel, dim=-1).unsqueeze(-1)
            * self.robot_body_linvel
        )
        self.robot_body_vel_drag = (
            self.robot_body_vel_drag_linear + self.robot_body_vel_drag_quadratic
        )
        # 将阻力添加到根链接的力张量上
        self.robot_force_tensors[:, 0, 0:3] += self.robot_body_vel_drag

        # --- 角速度阻尼 (Angular Velocity Drag) ---
        # 线性阻尼: -C_L * \omega
        self.robot_body_angvel_drag_linear = (
            -self.angvel_linear_damping_coefficient * self.robot_body_angvel
        )
        # 二次阻尼: -C_Q * |\omega| * \omega
        self.robot_body_angvel_drag_quadratic = (
            -self.angvel_quadratic_damping_coefficient
            * self.robot_body_angvel.abs()
            * self.robot_body_angvel
        )
        self.robot_body_angvel_drag = (
            self.robot_body_angvel_drag_linear + self.robot_body_angvel_drag_quadratic
        )
        # 将阻力添加到根链接的扭矩张量上
        self.robot_torque_tensors[:, 0, 0:3] += self.robot_body_angvel_drag

    def update_states(self):
        """
        计算并更新机器人的派生状态（如欧拉角、本体速度等）。
        """
        # 欧拉角（确保在 [-pi, pi] 范围内）
        self.robot_euler_angles[:] = ssa(get_euler_xyz_tensor(self.robot_orientation))
        
        # 载具坐标系方向（四元数）
        self.robot_vehicle_orientation[:] = vehicle_frame_quat_from_quat(self.robot_orientation)
        
        # 载具坐标系线速度（世界速度旋转到载具坐标系）
        self.robot_vehicle_linvel[:] = quat_rotate_inverse(
            self.robot_vehicle_orientation, self.robot_linvel
        )
        # 本体坐标系线速度（世界速度旋转到本体坐标系）
        self.robot_body_linvel[:] = quat_rotate_inverse(self.robot_orientation, self.robot_linvel)
        # 本体坐标系角速度（世界角速度旋转到本体坐标系）
        self.robot_body_angvel[:] = quat_rotate_inverse(self.robot_orientation, self.robot_angvel)

    def step(self, action_tensor):
        """
        更新四旋翼的状态，该函数在每个模拟步被调用。
        """
        # 1. 更新派生状态
        self.update_states()
        
        # 2. 检查动作张量维度
        if action_tensor.shape[0] != self.num_envs:
            raise ValueError("Action tensor does not have the correct number of environments")
            
        self.action_tensor[:] = action_tensor
        
        # 3. 调用控制器，执行控制分配（Call Controller -> Control Allocation -> Motor Model）
        self.call_controller()
        
        # 4. 模拟空气阻力
        self.simulate_drag()
        
        # 5. 施加外部扰动
        self.apply_disturbance()