# 导入 Isaac Gym 的 API 和 Tensor 相关的库
from isaacgym import gymapi # 导入 Isaac Gym 核心 API
from isaacgym import gymtorch # 导入 Isaac Gym PyTorch 张量工具

from isaacgym import gymutil # 导入 Isaac Gym 实用工具

# 导入 aerial_gym 内部的管理器和工具
from aerial_gym.env_manager.base_env_manager import BaseManager # 导入环境管理器基类
from aerial_gym.env_manager.asset_manager import AssetManager # 导入资产管理器（未直接使用，但可能相关）
from aerial_gym.env_manager.IGE_viewer_control import IGEViewerControl # 导入 Isaac Gym 查看器控制类
import torch # 导入 PyTorch

import os # 导入操作系统接口

from aerial_gym.utils.math import torch_rand_float_tensor # 导入张量随机数生成工具

from aerial_gym.utils.helpers import (
    get_args, # 获取命令行参数
    update_cfg_from_args, # 使用命令行参数更新配置
    class_to_dict, # 将类转换为字典
    parse_sim_params, # 解析模拟参数
)
import numpy as np # 导入 NumPy
from aerial_gym.utils.logging import CustomLogger # 导入自定义日志工具

logger = CustomLogger("IsaacGymEnvManager") # 初始化日志记录器


class IsaacGymEnv(BaseManager):
    """
    负责与 Isaac Gym 模拟器交互的类。它创建和管理 Isaac Gym 模拟、环境、资产和张量。
    """
    def __init__(self, config, sim_config, has_IGE_cameras, device):
        # 调用 BaseManager 构造函数，设置配置和设备
        super().__init__(config, device)
        self.sim_config = sim_config # 模拟器配置
        self.env_tensor_bounds_min = None
        self.env_tensor_bounds_max = None
        self.asset_handles = [] # 存储每个环境中的资产句柄列表
        self.env_handles = [] # 存储环境句柄列表
        self.num_rigid_bodies_robot = None # 机器人的刚体数量
        self.has_IGE_cameras = has_IGE_cameras # 是否使用 Isaac Gym 内部的相机
        self.sim_has_dof = False # 模拟中是否包含自由度 (DoF)
        self.dof_control_mode = "none" # 自由度控制模式

        logger.info("Creating Isaac Gym Environment")
        # 创建 Isaac Gym 核心对象
        self.gym, self.sim = self.create_sim()
        logger.info("Created Isaac Gym Environment")

        # --- 环境边界张量初始化 (Environment Bounds Tensors Initialization) ---
        # 最小/最大边界的最小/最大值（用于随机化环境大小）
        self.env_lower_bound_min = torch.tensor(
            self.cfg.env.lower_bound_min, device=self.device, requires_grad=False
        ).expand(self.cfg.env.num_envs, -1)
        self.env_lower_bound_max = torch.tensor(
            self.cfg.env.lower_bound_max, device=self.device, requires_grad=False
        ).expand(self.cfg.env.num_envs, -1)
        self.env_upper_bound_min = torch.tensor(
            self.cfg.env.upper_bound_min, device=self.device, requires_grad=False
        ).expand(self.cfg.env.num_envs, -1)
        self.env_upper_bound_max = torch.tensor(
            self.cfg.env.upper_bound_max, device=self.device, requires_grad=False
        ).expand(self.cfg.env.num_envs, -1)

        # 随机化生成每个环境的实际边界
        self.env_lower_bound = torch_rand_float_tensor(
            self.env_lower_bound_min, self.env_lower_bound_max
        )
        self.env_upper_bound = torch_rand_float_tensor(
            self.env_upper_bound_min, self.env_upper_bound_max
        )

        self.viewer = None
        self.graphics_are_stepped = True # 图形是否已更新

    def create_sim(self):
        """
        创建一个 gym 对象，并用适当的模拟参数进行初始化。
        """
        logger.info("Acquiring gym object")
        self.gym = gymapi.acquire_gym() # 获取 Isaac Gym 实例
        logger.info("Acquired gym object")
        
        # 参数解析：命令行参数 > 模拟配置
        args = get_args()
        
        # 结合模拟和环境配置，并用命令行参数覆盖
        sim_config_dict = dict(class_to_dict(self.sim_config))
        env_config_dict = dict(class_to_dict(self.cfg))
        combined_dict = {**sim_config_dict, **env_config_dict}
        sim_cfg = update_cfg_from_args(combined_dict, args)
        self.simulator_params = parse_sim_params(args, sim_cfg) # 解析模拟器参数

        logger.info("Fixing devices")
        args.sim_device = self.device
        self.sim_device_type, self.sim_device_id = gymutil.parse_device_str(args.sim_device)
        
        # 检查 GPU 管线和 CPU 设备的兼容性
        if self.sim_device_type == "cpu" and self.simulator_params.use_gpu_pipeline == True:
            logger.warning(
                "The use_gpu_pipeline is set to True in the sim_config, but the device is set to CPU. Running the simulation on the CPU."
            )
            self.simulator_params.use_gpu_pipeline = False
            
        # 警告：未启用 GPU 管线可能导致模拟变慢
        if self.simulator_params.use_gpu_pipeline == False:
            logger.critical(
                "The use_gpu_pipeline is set to False, this will result in slower simulation times"
            )
        else:
            logger.info(
                "Using GPU pipeline for simulation."
            )
        logger.info(
            "Sim Device type: {}, Sim Device ID: {}".format(
                self.sim_device_type, self.sim_device_id
            )
        )
        
        # 设置图形设备 ID
        if self.sim_config.viewer.headless and not self.has_IGE_cameras:
            # 无头模式且不使用 IGE 相机时，无需图形设备
            self.graphics_device_id = -1
            logger.critical(
                "\n Setting graphics device to -1."
                + "\n This is done because the simulation is run in headless mode and no Isaac Gym cameras are used."
                + "\n No need to worry. The simulation and warp rendering will work as expected."
            )
        else:
            # 否则使用模拟设备作为图形设备
            self.graphics_device_id = self.sim_device_id
        logger.info("Graphics Device ID: {}".format(self.graphics_device_id))
        
        logger.info("Creating Isaac Gym Simulation Object")
        # 警告：关于 CUDA 设备和图形设备不匹配的问题
        warn_msg1 = (
            "If you have set the CUDA_VISIBLE_DEVICES environment variable, please ensure that you set it\n"
            + "to a particular one that works for your system to use the viewer or Isaac Gym cameras.\n"
            + "If you want to run parallel simulations on multiple GPUs with camera sensors,\n"
            + "please disable Isaac Gym and use warp (by setting use_warp=True), set the viewer to headless."
        )
        logger.warning(warn_msg1)
        warn_msg2 = (
            "If you see a segfault in the next lines, it is because of the discrepancy between the CUDA device and the graphics device.\n"
            + "Please ensure that the CUDA device and the graphics device are the same."
        )
        logger.warning(warn_msg2)
        
        # 创建 Isaac Gym 模拟对象
        self.sim = self.gym.create_sim(
            self.sim_device_id, self.graphics_device_id, args.physics_engine, self.simulator_params
        )
        logger.info("Created Isaac Gym Simulation Object")
        return self.gym, self.sim

    def create_ground_plane(self):
        """
        创建地面平面。
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0) # 法线朝向 Z 轴正向
        self.gym.add_ground(self.sim, plane_params)
        return

    def create_env(self, env_id):
        """
        创建给定 ID 的环境。
        """
        # 从配置中获取环境边界
        min_bound_vec3 = gymapi.Vec3(
            self.cfg.env.lower_bound_min[0],
            self.cfg.env.lower_bound_min[1],
            self.cfg.env.lower_bound_min[2],
        )
        max_bound_vec3 = gymapi.Vec3(
            self.cfg.env.upper_bound_max[0],
            self.cfg.env.upper_bound_max[1],
            self.cfg.env.upper_bound_max[2],
        )
        
        # 创建环境。int(np.sqrt(self.cfg.env.num_envs)) 是每行的环境数量。
        env_handle = self.gym.create_env(
            self.sim,
            min_bound_vec3,
            max_bound_vec3,
            int(np.sqrt(self.cfg.env.num_envs)),
        )
        
        # 存储环境句柄和初始化资产句柄列表
        if len(self.env_handles) <= env_id:
            self.env_handles.append(env_handle)
            self.asset_handles.append([])
        else:
            raise ValueError("Environment already exists")
            
        return env_handle

    def reset(self):
        """
        重置所有环境（通过调用 reset_idx）。
        """
        self.reset_idx(torch.arange(self.num_envs, device=self.device))

    def add_asset_to_env(
        self,
        asset_info_dict,
        env_handle,
        env_id,
        global_asset_counter,
        segmentation_counter,
    ):
        """
        向环境添加一个资产（Actor）。
        """

        local_segmentation_ctr_for_isaacgym_asset = segmentation_counter
        # 处理语义 ID
        if asset_info_dict["semantic_id"] < 0:
            asset_segmentation_id = local_segmentation_ctr_for_isaacgym_asset
            local_segmentation_ctr_for_isaacgym_asset += 1
        else:
            asset_segmentation_id = asset_info_dict["semantic_id"]
            local_segmentation_ctr_for_isaacgym_asset += 1

        # 创建 Actor (资产)
        asset_handle = self.gym.create_actor(
            env_handle,
            asset_info_dict["isaacgym_asset"].asset, # Isaac Gym 资产句柄
            gymapi.Transform(), # 初始变换（通常为原点）
            "env_asset_" + str(global_asset_counter), # Actor 名称
            env_id,
            asset_info_dict["collision_mask"], # 碰撞掩码
            asset_segmentation_id, # 语义 ID
        )

        # 如果是机器人资产，记录刚体数量
        if asset_info_dict["asset_type"] == "robot":
            self.num_rigid_bodies_robot = self.gym.get_actor_rigid_body_count(
                env_handle, asset_handle
            )

        # 处理按链接的语义分割
        if asset_info_dict["per_link_semantic"]:
            rigid_body_names_all = self.gym.get_actor_rigid_body_names(env_handle, asset_handle)

            if not type(asset_info_dict["semantic_masked_links"]) == dict:
                raise ValueError("semantic_masked_links should be a dictionary")

            links_to_label = asset_info_dict["semantic_masked_links"].keys()
            if len(links_to_label) == 0:
                links_to_label = rigid_body_names_all

            for name in rigid_body_names_all:

                # 跳过字典中已预定义的值，确保语义 ID 不重复
                while (
                    local_segmentation_ctr_for_isaacgym_asset
                    in asset_info_dict["semantic_masked_links"].values()
                ):
                    local_segmentation_ctr_for_isaacgym_asset += 1

                if name in links_to_label:
                    if name in asset_info_dict["semantic_masked_links"]:
                        # 使用字典中预定义的值
                        segmentation_value = asset_info_dict["semantic_masked_links"][name]
                        logger.debug(f"Setting segmentation id for {name} to {segmentation_value}")
                    else:
                        # 使用递增的值
                        segmentation_value = local_segmentation_ctr_for_isaacgym_asset
                        local_segmentation_ctr_for_isaacgym_asset += 1
                        logger.debug(f"Setting segmentation id for {name} to {segmentation_value}")
                else:
                    # 对于未标记的链接，使用递增的值
                    segmentation_value = local_segmentation_ctr_for_isaacgym_asset
                    logger.debug(f"Setting segmentation id for {name} to {segmentation_value}")
                
                # 设置刚体的语义 ID
                index = rigid_body_names_all.index(name)
                self.gym.set_rigid_body_segmentation_id(
                    env_handle, asset_handle, index, segmentation_value
                )

        # 设置资产的颜色
        color = asset_info_dict["color"]
        if asset_info_dict["color"] is None:
            # 如果未指定颜色，则随机生成颜色
            color = np.random.randint(low=50, high=200, size=3)

        self.gym.set_rigid_body_color(
            env_handle,
            asset_handle,
            0, # 刚体索引
            gymapi.MESH_VISUAL, # 可视化网格
            gymapi.Vec3(color[0] / 255, color[1] / 255, color[2] / 255),
        )

        self.asset_handles[env_id].append(asset_handle)
        
        # 返回资产句柄和所使用的语义 ID 数量
        return (
            asset_handle,
            local_segmentation_ctr_for_isaacgym_asset - segmentation_counter,
        )

    def prepare_for_simulation(self, env_manager, global_tensor_dict):
        """
        在开始模拟循环之前，获取 Isaac Gym 张量并进行准备工作。
        """
        # 准备模拟：这是在创建完所有 Actor 后，开始运行物理模拟前的最后一步
        if not self.gym.prepare_sim(self.sim):
            raise RuntimeError("Failed to prepare Isaac Gym Environment")

        self.create_viewer(env_manager)
        self.has_viewer = self.viewer is not None

        # 检查所有环境是否具有相同数量的资产
        self.num_envs = len(self.env_handles)
        self.num_assets_per_env = [len(assets) for assets in self.asset_handles]
        if not all(
            [num_assets == self.num_assets_per_env[0] for num_assets in self.num_assets_per_env]
        ):
            raise ValueError("All environments should have the same number of assets")
        self.num_assets_per_env = self.num_assets_per_env[0]

        # 检查所有环境是否具有相同数量的刚体
        self.num_rigid_bodies_per_env = [
            self.gym.get_env_rigid_body_count(self.env_handles[i]) for i in range(self.num_envs)
        ]
        if not all(
            [
                num_rigid_bodies == self.num_rigid_bodies_per_env[0]
                for num_rigid_bodies in self.num_rigid_bodies_per_env
            ]
        ):
            raise ValueError("All environments should have the same number of rigid bodies.")
        self.num_rigid_bodies_per_env = self.num_rigid_bodies_per_env[0]

        # --- 获取和包装 Isaac Gym 张量 (Acquire and Wrap Isaac Gym Tensors) ---
        
        # 获取 Actor 根状态张量（包含所有资产）
        self.unfolded_vec_root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.unfolded_vec_root_tensor = gymtorch.wrap_tensor(self.unfolded_vec_root_tensor)

        # 获取接触力张量
        self.global_contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.global_contact_force_tensor = gymtorch.wrap_tensor(
            self.global_contact_force_tensor
        ).view(self.num_envs, self.num_rigid_bodies_per_env, -1)

        # 重新塑形根状态张量为 (num_envs, num_assets_per_env, state_dim)
        self.vec_root_tensor = self.unfolded_vec_root_tensor.view(
            self.num_envs, self.num_assets_per_env, -1
        )

        self.global_tensor_dict = global_tensor_dict

        # --- 填充通用环境张量 (Populate Common Environment Tensors) ---
        self.global_tensor_dict["vec_root_tensor"] = self.vec_root_tensor
        # 假设机器人是每个环境中的第一个资产 (索引 0)
        self.global_tensor_dict["robot_state_tensor"] = self.vec_root_tensor[:, 0, :]
        # 环境资产状态张量 (索引 1 及以后)
        self.global_tensor_dict["env_asset_state_tensor"] = self.vec_root_tensor[:, 1:, :]
        self.global_tensor_dict["unfolded_env_asset_state_tensor"] = self.unfolded_vec_root_tensor
        # 备份一份根状态张量，用于重置时作为常量 (克隆是为了安全)
        self.global_tensor_dict["unfolded_env_asset_state_tensor_const"] = self.global_tensor_dict[
            "unfolded_env_asset_state_tensor"
        ].clone()

        # 刚体状态张量（包含位置、旋转、线速度、角速度 for *所有*刚体）
        self.global_tensor_dict["rigid_body_state_tensor"] = gymtorch.wrap_tensor(
            self.gym.acquire_rigid_body_state_tensor(self.sim)
        )
        # 初始化全局力和扭矩张量（用于施加力）
        self.global_tensor_dict["global_force_tensor"] = torch.zeros(
            (self.global_tensor_dict["rigid_body_state_tensor"].shape[0], 3),
            device=self.device,
            requires_grad=False,
        )
        self.global_tensor_dict["global_torque_tensor"] = torch.zeros(
            (self.global_tensor_dict["rigid_body_state_tensor"].shape[0], 3),
            device=self.device,
            requires_grad=False,
        )

        # 自由度 (DoF) 状态张量（包含关节位置和速度）
        self.global_tensor_dict["unfolded_dof_state_tensor"] = gymtorch.wrap_tensor(
            self.gym.acquire_dof_state_tensor(self.sim)
        )
        # 如果 DoF 张量不为空，则设置 sim_has_dof 标志
        if not self.global_tensor_dict["unfolded_dof_state_tensor"] is None:
            self.sim_has_dof = True
            self.global_tensor_dict["dof_state_tensor"] = self.global_tensor_dict[
                "unfolded_dof_state_tensor"
            ].view(self.num_envs, -1, 2) # (num_envs, num_dofs, [pos, vel])

        self.global_tensor_dict["global_contact_force_tensor"] = self.global_contact_force_tensor
        # 机器人接触力张量（假设机器人是第一个刚体，索引 0）
        self.global_tensor_dict["robot_contact_force_tensor"] = self.global_contact_force_tensor[
            :, 0, :
        ]

        # --- 填充机器人状态子张量 (Populate Robot State Sub-Tensors) ---
        self.global_tensor_dict["robot_position"] = self.global_tensor_dict["robot_state_tensor"][
            :, :3
        ]
        self.global_tensor_dict["robot_orientation"] = self.global_tensor_dict[
            "robot_state_tensor"
        ][:, 3:7] # 四元数
        self.global_tensor_dict["robot_linvel"] = self.global_tensor_dict["robot_state_tensor"][
            :, 7:10
        ] # 世界坐标系线速度
        self.global_tensor_dict["robot_angvel"] = self.global_tensor_dict["robot_state_tensor"][
            :, 10:
        ] # 世界坐标系角速度
        # 以下是占位符或待计算的张量
        self.global_tensor_dict["robot_body_angvel"] = torch.zeros_like(
            self.global_tensor_dict["robot_state_tensor"][:, 10:13]
        )
        self.global_tensor_dict["robot_body_linvel"] = torch.zeros_like(
            self.global_tensor_dict["robot_state_tensor"][:, 7:10]
        )
        self.global_tensor_dict["robot_euler_angles"] = torch.zeros_like(
            self.global_tensor_dict["robot_state_tensor"][:, 7:10]
        )

        # 提取机器人相关的力和扭矩张量
        idx = self.num_rigid_bodies_robot
        self.global_tensor_dict["robot_force_tensor"] = self.global_tensor_dict[
            "global_force_tensor"
        ].view(self.num_envs, self.num_rigid_bodies_per_env, 3)[:, :idx, :]

        self.global_tensor_dict["robot_torque_tensor"] = self.global_tensor_dict[
            "global_torque_tensor"
        ].view(self.num_envs, self.num_rigid_bodies_per_env, 3)[:, :idx, :]

        # --- 填充障碍物状态子张量 (Populate Obstacle State Sub-Tensors) ---
        if self.num_assets_per_env > 0:
            # 障碍物状态张量 (索引 1 及以后)
            self.global_tensor_dict["obstacle_position"] = self.global_tensor_dict[
                "env_asset_state_tensor"
            ][:, :, 0:3]
            self.global_tensor_dict["obstacle_orientation"] = self.global_tensor_dict[
                "env_asset_state_tensor"
            ][:, :, 3:7]
            self.global_tensor_dict["obstacle_linvel"] = self.global_tensor_dict[
                "env_asset_state_tensor"
            ][:, :, 7:10]
            self.global_tensor_dict["obstacle_angvel"] = self.global_tensor_dict[
                "env_asset_state_tensor"
            ][:, :, 10:]
            # 占位符或待计算的张量
            self.global_tensor_dict["obstacle_body_angvel"] = torch.zeros_like(
                self.global_tensor_dict["env_asset_state_tensor"][:, :, 10:13]
            )
            self.global_tensor_dict["obstacle_body_linvel"] = torch.zeros_like(
                self.global_tensor_dict["env_asset_state_tensor"][:, :, 7:10]
            )
            self.global_tensor_dict["obstacle_euler_angles"] = torch.zeros_like(
                self.global_tensor_dict["env_asset_state_tensor"][:, :, 7:10]
            )

            # 提取障碍物相关的力和扭矩张量
            # 假设每个障碍物都折叠为一个基链接
            self.global_tensor_dict["obstacle_force_tensor"] = self.global_tensor_dict[
                "global_force_tensor"
            ].view(self.num_envs, self.num_rigid_bodies_per_env, 3)[:, idx:, :]

            self.global_tensor_dict["obstacle_torque_tensor"] = self.global_tensor_dict[
                "global_torque_tensor"
            ].view(self.num_envs, self.num_rigid_bodies_per_env, 3)[:, idx:, :]

        # --- 填充其他环境信息 (Populate Other Environment Info) ---
        self.global_tensor_dict["env_bounds_max"] = self.env_upper_bound
        self.global_tensor_dict["env_bounds_min"] = self.env_lower_bound
        # 重力向量
        self.global_tensor_dict["gravity"] = torch.tensor(
            self.sim_config.sim.gravity, device=self.device, requires_grad=False
        ).expand(self.num_envs, -1)
        self.global_tensor_dict["dt"] = self.sim_config.sim.dt
        
        # 初始化查看器张量
        if self.viewer is not None:
            self.viewer.init_tensors(global_tensor_dict)
            
        return True

    def create_viewer(self, env_manager):
        """
        创建并配置 Isaac Gym 查看器 (Viewer)。
        """
        # 机器人句柄是每个环境中的第一个资产
        self.robot_handles = [ah[0] for ah in self.asset_handles]
        logger.warning(f"Headless: {self.sim_config.viewer.headless}")
        if not self.sim_config.viewer.headless:
            logger.info("Creating viewer")
            # 实例化查看器控制类
            self.viewer = IGEViewerControl(
                self.gym, self.sim, env_manager, self.sim_config.viewer, self.device
            )
            self.viewer.set_actor_and_env_handles(self.robot_handles, self.env_handles)
            self.viewer.set_camera_lookat() # 设置摄像机焦点
            logger.info("Created viewer")
        else:
            logger.info("Headless mode. Viewer not created.")
        return

    def pre_physics_step(self, actions):
        """
        在物理步之前执行任何必要的操作（应用力和设置 DoF 目标）。
        """
        # 如果配置要求，将张量写回模拟器
        if self.cfg.env.write_to_sim_at_every_timestep:
            self.write_to_sim()
            
        # 应用全局力和扭矩到刚体上
        self.gym.apply_rigid_body_force_tensors(
            self.sim,
            gymtorch.unwrap_tensor(self.global_tensor_dict["global_force_tensor"]),
            gymtorch.unwrap_tensor(self.global_tensor_dict["global_torque_tensor"]),
            gymapi.LOCAL_SPACE, # 在局部坐标系 (Actor/刚体坐标系) 中施加力
        )
        
        # 处理自由度 (DoF) 控制
        if self.sim_has_dof:
            self.dof_control_mode = self.global_tensor_dict["dof_control_mode"]

            if self.dof_control_mode == "position":
                self.dof_application_function = self.gym.set_dof_position_target_tensor
                self.dof_application_tensor = gymtorch.unwrap_tensor(
                    self.global_tensor_dict["dof_position_setpoint_tensor"]
                )
            elif self.dof_control_mode == "velocity":
                self.dof_application_function = self.gym.set_dof_velocity_target_tensor
                self.dof_application_tensor = gymtorch.unwrap_tensor(
                    self.global_tensor_dict["dof_velocity_setpoint_tensor"]
                )
            elif self.dof_control_mode == "effort":
                self.dof_application_function = self.gym.set_dof_actuation_force_tensor
                self.dof_application_tensor = gymtorch.unwrap_tensor(
                    self.global_tensor_dict["dof_effort_tensor"]
                )
            else:
                raise ValueError("Invalid dof control mode")
            
            # 施加 DoF 目标或力矩
            self.dof_application_function(self.sim, self.dof_application_tensor)
        return

    def physics_step(self):
        """
        执行物理模拟步。
        """
        self.gym.simulate(self.sim)
        self.graphics_are_stepped = False
        return

    def post_physics_step(self):
        """
        在物理步之后执行任何必要的操作（获取结果和刷新张量）。
        """
        # 获取模拟结果
        self.gym.fetch_results(self.sim, True)
        self.refresh_tensors() # 刷新张量
        return

    def refresh_tensors(self):
        """
        从 Isaac Gym 刷新所有重要的状态张量。
        """
        self.gym.refresh_rigid_body_state_tensor(self.sim) # 刚体状态
        self.gym.refresh_force_sensor_tensor(self.sim) # 力传感器（IMU）
        self.gym.refresh_actor_root_state_tensor(self.sim) # 根状态
        self.gym.refresh_net_contact_force_tensor(self.sim) # 接触力
        self.gym.refresh_dof_state_tensor(self.sim) # DoF 状态

    def step_graphics(self):
        """
        执行图形更新。
        """
        if not self.graphics_are_stepped:
            self.gym.step_graphics(self.sim)
            self.graphics_are_stepped = True

    def render_viewer(self):
        """
        渲染查看器。
        """
        if self.viewer is not None:
            # 如果图形未更新且查看器同步启用，则先更新图形
            if not self.graphics_are_stepped and self.viewer.enable_viewer_sync:
                self.step_graphics()
            self.viewer.render() # 渲染查看器
        return

    def reset_idx(self, env_ids):
        """
        为给定的环境 ID 随机化环境边界。
        注意：此处只随机化了边界张量，实际的 Actor 状态重置在 EnvManager 中处理。
        """
        # 随机化环境的下边界
        self.env_lower_bound[env_ids, :] = torch_rand_float_tensor(
            self.env_lower_bound_min, self.env_lower_bound_max
        )[env_ids]
        # 随机化环境的上边界
        self.env_upper_bound[env_ids, :] = torch_rand_float_tensor(
            self.env_upper_bound_min, self.env_upper_bound_max
        )[env_ids]

    def write_to_sim(self):
        """
        将张量中的状态数据写回模拟器（用于强制设置状态）。
        """
        # 设置 Actor 根状态（包括机器人和资产）
        self.gym.set_actor_root_state_tensor(
            self.sim,
            gymtorch.unwrap_tensor(self.global_tensor_dict["unfolded_env_asset_state_tensor"]),
        )
        # 如果存在 DoF，则设置 DoF 状态
        if self.sim_has_dof:
            self.gym.set_dof_state_tensor(
                self.sim,
                gymtorch.unwrap_tensor(self.global_tensor_dict["unfolded_dof_state_tensor"]),
            )
        return