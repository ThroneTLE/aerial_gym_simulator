# 导入任务基类
from aerial_gym.task.base_task import BaseTask
# 导入模拟环境构建器
from aerial_gym.sim.sim_builder import SimBuilder
import torch # 导入 PyTorch
import numpy as np # 导入 NumPy

from aerial_gym.utils.math import * # 导入数学工具函数（如四元数操作）

from aerial_gym.utils.logging import CustomLogger # 导入自定义日志工具

import gymnasium as gym # 导入 Gymnasium (Gym) 库
from gym.spaces import Dict, Box # 导入 Gym 的空间定义（Dict, Box）

logger = CustomLogger("position_setpoint_task") # 初始化日志记录器


def dict_to_class(dict):
    """
    辅助函数：将字典转换为一个包含字典键值作为属性的简单类。
    """
    return type("ClassFromDict", (object,), dict)


class PositionSetpointTask(BaseTask):
    def __init__(
        self, 
        task_config, 
        seed=None, 
        num_envs=None, 
        headless=None, 
        device=None, 
        use_warp=None
    ):
        # 如果用户提供了参数，则覆盖任务配置中的参数
        if seed is not None:
            task_config.seed = seed
        if num_envs is not None:
            task_config.num_envs = num_envs
        if headless is not None:
            task_config.headless = headless
        if device is not None:
            task_config.device = device
        if use_warp is not None:
            task_config.use_warp = use_warp

        # 调用 BaseTask 的构造函数
        super().__init__(task_config)
        self.device = self.task_config.device
        
        # 将奖励参数字典中的每个元素转换为 PyTorch 张量
        for key in self.task_config.reward_parameters.keys():
            self.task_config.reward_parameters[key] = torch.tensor(
                self.task_config.reward_parameters[key], device=self.device
            )
            
        logger.info("Building environment for position setpoint task.")
        logger.info(
            "\nSim Name: {},\nEnv Name: {},\nRobot Name: {}, \nController Name: {}".format(
                self.task_config.sim_name,
                self.task_config.env_name,
                self.task_config.robot_name,
                self.task_config.controller_name,
            )
        )
        logger.info(
            "\nNum Envs: {},\nUse Warp: {},\nHeadless: {}".format(
                self.task_config.num_envs,
                self.task_config.use_warp,
                self.task_config.headless,
            )
        )

        # 使用 SimBuilder 构建模拟环境
        self.sim_env = SimBuilder().build_env(
            sim_name=self.task_config.sim_name,
            env_name=self.task_config.env_name,
            robot_name=self.task_config.robot_name,
            controller_name=self.task_config.controller_name,
            args=self.task_config.args,
            device=self.device,
            num_envs=self.task_config.num_envs,
            use_warp=self.task_config.use_warp,
            headless=self.task_config.headless,
        )

        # 初始化动作张量
        self.actions = torch.zeros(
            (self.sim_env.num_envs, self.task_config.action_space_dim),
            device=self.device,
            requires_grad=False,
        )
        self.prev_actions = torch.zeros_like(self.actions) # 上一步的动作
        self.counter = 0

        # 初始化目标位置张量（所有环境的目标位置）
        self.target_position = torch.zeros(
            (self.sim_env.num_envs, 3), device=self.device, requires_grad=False
        )

        # 从环境获取观测字典，张量会原地更新，避免数据在函数间不断复制
        self.obs_dict = self.sim_env.get_obs()
        self.obs_dict["num_obstacles_in_env"] = 1 # 假设障碍物数量为 1 (可能被后续代码覆盖)
        self.terminations = self.obs_dict["crashes"] # 终止 (崩溃) 状态
        self.truncations = self.obs_dict["truncations"] # 截断状态 (通常指达到最大步长)
        self.rewards = torch.zeros(self.truncations.shape[0], device=self.device) # 奖励张量

        # 定义 Gymnasium 的观测空间
        self.observation_space = Dict(
            {"observations": Box(low=-1.0, high=1.0, shape=(13,), dtype=np.float32)}
        )
        # 定义 Gymnasium 的动作空间
        self.action_space = Box(
            low=-1.0,
            high=1.0,
            shape=(self.task_config.action_space_dim,),
            dtype=np.float32,
        )
        # self.action_transformation_function = self.sim_env.robot_manager.robot.action_transformation_function

        self.num_envs = self.sim_env.num_envs

        self.counter = 0

        # 任务观测字典，用于返回给 RL 智能体
        # 目前只将 "observations" 发送给 Actor 和 Critic。
        # "priviliged_obs" 尚未在 sample-factory 中处理
        self.task_obs = {
            "observations": torch.zeros(
                (self.sim_env.num_envs, self.task_config.observation_space_dim),
                device=self.device,
                requires_grad=False,
            ),
            "priviliged_obs": torch.zeros(
                (
                    self.sim_env.num_envs,
                    self.task_config.privileged_observation_space_dim,
                ),
                device=self.device,
                requires_grad=False,
            ),
            "collisions": torch.zeros(
                (self.sim_env.num_envs, 1), device=self.device, requires_grad=False
            ),
            "rewards": torch.zeros(
                (self.sim_env.num_envs, 1), device=self.device, requires_grad=False
            ),
        }

    def close(self):
        """关闭环境并清理资源。"""
        self.sim_env.delete_env()

    def reset(self):
        """重置所有环境。"""
        # 将目标位置设为 (0, 0, 0)
        self.target_position[:, 0:3] = 0.0  # torch.rand_like(self.target_position) * 10.0
        self.infos = {}
        self.sim_env.reset() # 调用模拟环境的重置
        return self.get_return_tuple() # 返回 Gym 风格的元组

    def reset_idx(self, env_ids):
        """重置指定索引的环境。"""
        # 将目标位置设为 (0, 0, 0)
        self.target_position[:, 0:3] = (
            0.0  # (torch.rand_like(self.target_position[env_ids]) * 10.0)
        )
        self.infos = {}
        self.sim_env.reset_idx(env_ids)
        return

    def render(self):
        """渲染环境（任务级别）。"""
        return None

    def step(self, actions):
        """
        执行模拟的一个时间步。
        actions: RL 智能体计算的动作。
        """
        self.counter += 1
        self.prev_actions[:] = self.actions
        self.actions = actions

        # 执行模拟步（包括应用动作、物理更新、后处理）
        self.sim_env.step(actions=self.actions)

        # 计算奖励和崩溃状态
        # 这一步必须在 sim_env.post_reward_calculation_step() 之前完成。
        self.rewards[:], self.terminations[:] = self.compute_rewards_and_crashes(self.obs_dict)

        # 如果配置要求在重置前返回状态，则先获取返回元组
        if self.task_config.return_state_before_reset == True:
            return_tuple = self.get_return_tuple()

        # 计算截断状态：如果模拟步数超过最大步长，则截断
        self.truncations[:] = torch.where(
            self.sim_env.sim_steps > self.task_config.episode_len_steps, 1, 0
        )
        # 执行奖励计算后的步骤（包括重置已终止/截断的环境）
        self.sim_env.post_reward_calculation_step()

        self.infos = {}  # self.obs_dict["infos"]

        # 如果配置要求在重置后返回状态，则在此获取返回元组
        if self.task_config.return_state_before_reset == False:
            return_tuple = self.get_return_tuple()

        return return_tuple

    def get_return_tuple(self):
        """准备并返回 Gym 风格的 (obs, reward, terminated, truncated, info) 元组。"""
        self.process_obs_for_task()
        return (
            self.task_obs,
            self.rewards,
            self.terminations,
            self.truncations,
            self.infos,
        )

    def process_obs_for_task(self):
        """将原始环境观测处理为任务所需的观测格式。"""
        # 观测 0: 目标位置 - 机器人当前位置（位置误差）
        self.task_obs["observations"][:, 0:3] = (
            self.target_position - self.obs_dict["robot_position"]
        )
        # 观测 1: 机器人方向（四元数）
        self.task_obs["observations"][:, 3:7] = self.obs_dict["robot_orientation"]
        # 观测 2: 机器人本体线速度
        self.task_obs["observations"][:, 7:10] = self.obs_dict["robot_body_linvel"]
        # 观测 3: 机器人本体角速度
        self.task_obs["observations"][:, 10:13] = self.obs_dict["robot_body_angvel"]
        # 将奖励和终止/截断状态添加到 task_obs 字典（供 critic 使用）
        self.task_obs["rewards"] = self.rewards
        self.task_obs["terminations"] = self.terminations
        self.task_obs["truncations"] = self.truncations

    def compute_rewards_and_crashes(self, obs_dict):
        """计算当前步的奖励和崩溃状态。"""
        robot_position = obs_dict["robot_position"]
        target_position = self.target_position
        robot_linvel = obs_dict["robot_linvel"]
        robot_vehicle_orientation = obs_dict["robot_vehicle_orientation"]
        robot_orientation = obs_dict["robot_orientation"]
        
        # 目标方向（保持默认四元数 [0, 0, 0, 1]）
        target_orientation = torch.zeros_like(robot_orientation, device=self.device)
        target_orientation[:, 3] = 1.0 
        
        angular_velocity = obs_dict["robot_body_angvel"]
        root_quats = obs_dict["robot_orientation"]

        # 计算在机器人载具坐标系下的位置误差（将世界坐标系误差转换到载具坐标系）
        pos_error_vehicle_frame = quat_apply_inverse(
            robot_vehicle_orientation, (target_position - robot_position)
        )
        
        # 调用 JIT 脚本函数计算最终奖励
        return compute_reward(
            pos_error_vehicle_frame, # 位置误差（载具坐标系）
            robot_linvel,
            root_quats,
            angular_velocity,
            obs_dict["crashes"], # 初始崩溃状态
            1.0,  # obs_dict["curriculum_level_multiplier"], # 课程学习乘数（此处为 1.0）
            self.actions,
            self.prev_actions,
            self.task_config.reward_parameters, # 奖励参数
        )


# --- 奖励辅助函数 (Reward Helper Functions) ---

@torch.jit.script
def exp_func(x, gain, exp):
    # type: (Tensor, float, float) -> Tensor
    """高斯（指数）函数： gain * exp(-exp * x^2)"""
    return gain * torch.exp(-exp * x * x)


@torch.jit.script
def exp_penalty_func(x, gain, exp):
    # type: (Tensor, float, float) -> Tensor
    """指数惩罚函数： gain * (exp(-exp * x^2) - 1)"""
    return gain * (torch.exp(-exp * x * x) - 1)


@torch.jit.script
def compute_reward(
    pos_error,
    lin_vels,
    robot_quats,
    robot_angvels,
    crashes,
    curriculum_level_multiplier,
    current_action,
    prev_actions,
    parameter_dict,
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor, Tensor, Dict[str, Tensor]) -> Tuple[Tensor, Tensor]
    """
    计算位置设定点任务的奖励和更新崩溃状态。
    """
    
    # 计算到目标的距离（欧几里得范数）
    dist = torch.norm(pos_error, dim=1)

    # 1. 位置奖励 (Pos Reward): 距离越近奖励越高，使用高斯函数
    pos_reward = exp_func(dist, 3.0, 8.0) + exp_func(dist, 2.0, 4.0)

    # 2. 距离奖励 (Dist Reward): 简单的线性奖励项
    dist_reward = (20 - dist) / 40.0  

    # 3. 姿态/向上奖励 (Up Reward): 惩罚倾斜
    ups = quat_axis(robot_quats, 2) # 获取四元数表示的 Z 轴方向
    tiltage = torch.abs(1 - ups[..., 2]) # 机器人 Z 轴与世界 Z 轴的点积（期望为 1）的误差
    up_reward = 0.2 / (0.1 + tiltage * tiltage) # 误差小时奖励高

    # 4. 角速度奖励 (Ang Vel Reward): 惩罚高速旋转
    spinnage = torch.norm(robot_angvels, dim=1) # 角速度范数
    ang_vel_reward = (1.0 / (1.0 + spinnage * spinnage)) * 3 # 角速度小时奖励高

    # 5. 总奖励
    total_reward = (
        pos_reward + dist_reward + pos_reward * (up_reward + ang_vel_reward)
    )
    # 应用课程学习乘数（此处为 1.0）
    total_reward[:] = curriculum_level_multiplier * total_reward

    # 6. 更新崩溃状态 (Crashes)
    # 如果距离目标超过 8.0 米，则视为崩溃
    crashes[:] = torch.where(dist > 8.0, torch.ones_like(crashes), crashes)

    # 7. 施加惩罚
    # 如果发生崩溃 (crashes > 0.0)，则施加巨大的负奖励 -20
    total_reward[:] = torch.where(crashes > 0.0, -20 * torch.ones_like(total_reward), total_reward)
    
    

    return total_reward, crashes