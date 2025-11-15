import torch
import pytorch3d.transforms as p3d_transforms
from aerial_gym.utils.math import *

from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger("lee_position_controller")


from aerial_gym.control.controllers.base_lee_controller import *


class LeePositionController(BaseLeeController):
    def __init__(self, config, num_envs, device):
        super().__init__(config, num_envs, device)

    def init_tensors(self, global_tensor_dict=None):
        super().init_tensors(global_tensor_dict)

    def update(self, command_actions):
        """
        改进版 Lee 位置控制器（修正推力缩放错误）
        """
        self.reset_commands()
        
        # 期望加速度（PD 控制器）
        self.accel[:] = self.compute_acceleration(
            setpoint_position=command_actions[:, 0:3],
            setpoint_velocity=torch.zeros_like(self.robot_vehicle_linvel),
        )

        # 当前姿态旋转矩阵
        R = quat_to_rotation_matrix(self.robot_orientation)
        
        # 当前重力向量（通常 [0, 0, -9.81]）
        g_vec = self.gravity

        # 计算世界坐标下的总力
        forces = self.mass * (self.accel - g_vec)

        # 计算推力（Lee 论文公式，沿机体 z 轴）
        thrust_world_z = torch.sum(forces * R[:, :, 2], dim=1)

        # ✅ 不再归一化，直接输出物理推力
        self.wrench_command[:, 2] = thrust_world_z

        # 姿态控制部分（保持不变）
        self.desired_quat[:] = calculate_desired_orientation_for_position_velocity_control(
            forces, command_actions[:, 3], self.buffer_tensor
        )

        self.euler_angle_rates[:] = 0.0
        self.desired_body_angvel[:] = 0.0
        self.wrench_command[:, 3:6] = self.compute_body_torque(
            self.desired_quat, self.desired_body_angvel
        )
        #print("控制器实时质量:", self.mass)
        return self.wrench_command


class LeePositionControllerWithCompensation(LeePositionController):
    """
    Lee position controller that accepts additional torque-compensation inputs.

    Expected command layout per environment:
        [x, y, z, yaw, τx_cmd, τy_cmd, τz_cmd]
    The first four entries are identical to the classic Lee controller. The
    remaining entries are scaled (see config) and added directly to the body-frame
    torque output to counteract external disturbances such as suspended payloads.
    """

    _BASE_ACTION_DIM = 4

    def __init__(self, config, num_envs, device):
        super().__init__(config, num_envs, device)
        self.compensation_dims = getattr(self.cfg, "compensation_dims", 3)
        self.comp_torque_limits = None
        self.comp_thrust_limit = None

    def init_tensors(self, global_tensor_dict=None):
        super().init_tensors(global_tensor_dict)
        torque_limits = getattr(self.cfg, "compensation_torque_limits", None)
        thrust_limit = getattr(self.cfg, "compensation_thrust_limit", None)
        if thrust_limit is None:
            raise AttributeError(
                "compensation_thrust_limit must be provided in the controller config."
            )
        if torque_limits is None:
            raise AttributeError(
                "compensation_torque_limits must be provided in the controller config."
            )

        if len(torque_limits) != self.compensation_dims - 1:
            raise ValueError(
                "compensation_torque_limits length must be compensation_dims - 1 "
                f"({self.compensation_dims - 1})."
            )
        self.comp_thrust_limit = torch.tensor(
            [thrust_limit], dtype=torch.float32, device=self.device
        ).view(1, 1)
        self.comp_torque_limits = torch.tensor(
            torque_limits, dtype=torch.float32, device=self.device
        ).view(1, -1)

    def update(self, command_actions):
        if command_actions.shape[1] != self._BASE_ACTION_DIM + self.compensation_dims:
            raise ValueError(
                "Expected command_actions with "
                f"{self._BASE_ACTION_DIM + self.compensation_dims} columns, got "
                f"{command_actions.shape[1]}."
            )

        base_actions = command_actions[:, : self._BASE_ACTION_DIM]
        compensation_actions = command_actions[:, self._BASE_ACTION_DIM :]

        # Run the standard Lee position control on the first four commands.
        wrench = super().update(base_actions)

        # Scale and add compensation torques (Clamp to [-1, 1] to match usual action range).
        compensation_actions = torch.clamp(compensation_actions, -1.0, 1.0)
        thrust_action = compensation_actions[:, :1] * self.comp_thrust_limit
        torque_actions = compensation_actions[:, 1:] * self.comp_torque_limits

        wrench[:, 2:3] += thrust_action
        wrench[:, 3:6] += torque_actions
        return wrench
