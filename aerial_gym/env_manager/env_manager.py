# 从 aerial_gym.env_manager 导入 IsaacGymEnv，这是 Isaac Gym 环境接口的封装
from aerial_gym.env_manager.IGE_env_manager import IsaacGymEnv

# 导入各种管理器和加载器
from aerial_gym.env_manager.base_env_manager import BaseManager # 基础管理器
from aerial_gym.env_manager.asset_manager import AssetManager # 资产管理器
from aerial_gym.env_manager.warp_env_manager import WarpEnv # Warp 物理引擎环境（可选）
from aerial_gym.env_manager.asset_loader import AssetLoader # 资产加载器
from aerial_gym.robots.robot_manager import RobotManagerIGE # Isaac Gym 机器人管理器
from aerial_gym.env_manager.obstacle_manager import ObstacleManager # 障碍物管理器


# 导入配置注册表
from aerial_gym.registry.env_registry import env_config_registry # 环境配置注册表
from aerial_gym.registry.sim_registry import sim_config_registry # 模拟配置注册表
from aerial_gym.registry.robot_registry import robot_registry # 机器人配置注册表

import torch # 导入 PyTorch
from aerial_gym.utils.logging import CustomLogger # 导入自定义日志工具

import math, random # 导入数学和随机库

logger = CustomLogger("env_manager") # 初始化日志记录器


class EnvManager(BaseManager):
    """
    该类用于管理整个模拟环境。它可以处理机器人、环境和资产管理器的创建。
    该类负责张量（tensor）的创建和销毁。

    此外，环境管理器可以在主环境类中被调用，通过抽象化接口来操控环境。

    此脚本应尽可能保持通用性，以处理不同类型的环境，同时可以在各个机器人或环境管理器中
    进行更改以处理特定情况。
    """

    def __init__(
        self,
        sim_name, # 模拟器名称
        env_name, # 环境名称
        robot_name, # 机器人名称
        controller_name, # 控制器名称
        device, # 设备（如 'cuda:0' 或 'cpu'）
        args=None, # 额外的参数
        num_envs=None, # 并行环境数量（可选覆盖配置）
        use_warp=None, # 是否使用 Warp（可选覆盖配置）
        headless=None, # 是否无头模式（可选覆盖配置）
    ):
        self.robot_name = robot_name
        self.controller_name = controller_name
        # 从注册表创建模拟配置
        self.sim_config = sim_config_registry.make_sim(sim_name)

        # 调用 BaseManager 的构造函数，创建环境配置
        super().__init__(env_config_registry.make_env(env_name), device)

        # 处理可选的参数覆盖
        if num_envs is not None:
            self.cfg.env.num_envs = num_envs
        if use_warp is not None:
            self.cfg.env.use_warp = use_warp
        if headless is not None:
            self.sim_config.viewer.headless = headless

        self.num_envs = self.cfg.env.num_envs
        self.use_warp = self.cfg.env.use_warp

        self.asset_manager = None
        self.tensor_manager = None # 注意：这个变量似乎未被使用
        self.env_args = args

        self.keep_in_env = None

        self.global_tensor_dict = {} # 全局张量字典，用于存储所有共享数据

        logger.info("Populating environments.")
        # 填充环境，创建模拟器实例、机器人和资产
        self.populate_env(env_cfg=self.cfg, sim_cfg=self.sim_config)
        logger.info("[DONE] Populating environments.")
        # 准备模拟，初始化管理器和张量
        self.prepare_sim()

        # 初始化模拟步数张量
        self.sim_steps = torch.zeros(
            self.num_envs, dtype=torch.int32, requires_grad=False, device=self.device
        )

    def create_sim(self, env_cfg, sim_cfg):
        """
        该函数创建环境和机器人管理器。它为 IsaacGym 环境实例创建了必要的设置。
        """
        logger.info("Creating simulation instance.")
        logger.info("Instantiating IGE object.")

        # === 需要在此处检查，否则 IGE 在使用不同 CUDA GPU 时会崩溃 (segfault) ====
        has_IGE_cameras = False
        robot_config = robot_registry.get_robot_config(self.robot_name)
        # 如果启用了相机且未使用 Warp，则需要 IGE 相机
        if robot_config.sensor_config.enable_camera == True and self.use_warp == False:
            has_IGE_cameras = True
        # ===============================================================================================

        # 创建 Isaac Gym 环境实例
        self.IGE_env = IsaacGymEnv(env_cfg, sim_cfg, has_IGE_cameras, self.device)

        # 定义一个全局字典，存储模拟对象和在环境、资产和机器人管理器之间共享的重要参数
        self.global_sim_dict = {}
        self.global_sim_dict["gym"] = self.IGE_env.gym # Isaac Gym API
        self.global_sim_dict["sim"] = self.IGE_env.sim # Isaac Gym 模拟句柄
        self.global_sim_dict["env_cfg"] = self.cfg
        self.global_sim_dict["use_warp"] = self.IGE_env.cfg.env.use_warp
        self.global_sim_dict["num_envs"] = self.IGE_env.cfg.env.num_envs
        self.global_sim_dict["sim_cfg"] = sim_cfg

        logger.info("IGE object instantiated.")

        # 如果配置使用 Warp，则创建 Warp 环境
        if self.cfg.env.use_warp:
            logger.info("Creating warp environment.")
            self.warp_env = WarpEnv(self.global_sim_dict, self.device)
            logger.info("Warp environment created.")

        # 创建资产加载器
        self.asset_loader = AssetLoader(self.global_sim_dict, self.device)

        logger.info("Creating robot manager.")
        # 创建 Isaac Gym 机器人管理器
        self.robot_manager = RobotManagerIGE(
            self.global_sim_dict, self.robot_name, self.controller_name, self.device
        )
        self.global_sim_dict["robot_config"] = self.robot_manager.cfg
        logger.info("[DONE] Creating robot manager.")

        logger.info("[DONE] Creating simulation instance.")

    def populate_env(self, env_cfg, sim_cfg):
        """
        该函数使用必要的资产和机器人填充环境。
        """
        # 创建包含环境和机器人管理器的模拟实例
        self.create_sim(env_cfg, sim_cfg)

        # 在模拟中创建机器人资产（例如加载 URDF）
        self.robot_manager.create_robot(self.asset_loader)

        # 首先为环境选择资产：
        # global_asset_dicts: 每个环境的资产信息
        # keep_in_env_num: 永久保留在环境中的资产数量
        self.global_asset_dicts, keep_in_env_num = self.asset_loader.select_assets_for_sim()

        # 检查并设置 keep_in_env 数量的一致性
        if self.keep_in_env == None:
            self.keep_in_env = keep_in_env_num
        elif self.keep_in_env != keep_in_env_num:
            raise Exception(
                "Inconsistent number of assets kept in the environment. The number of keep_in_env assets must be equal for all environments. Check."
            )

        # 将资产添加到环境
        segmentation_ctr = 100 # 语义分割计数器起始值

        self.global_asset_counter = 0
        self.step_counter = 0

        self.asset_min_state_ratio = None
        self.asset_max_state_ratio = None

        # 初始化崩溃和截断张量 (Boolean 类型)
        self.global_tensor_dict["crashes"] = torch.zeros(
            (self.num_envs), device=self.device, requires_grad=False, dtype=torch.bool
        )
        self.global_tensor_dict["truncations"] = torch.zeros(
            (self.num_envs), device=self.device, requires_grad=False, dtype=torch.bool
        )

        self.num_env_actions = self.cfg.env.num_env_actions
        self.global_tensor_dict["num_env_actions"] = self.num_env_actions
        self.global_tensor_dict["env_actions"] = None # 当前环境动作
        self.global_tensor_dict["prev_env_actions"] = None # 上一个环境动作

        self.collision_tensor = self.global_tensor_dict["crashes"]
        self.truncation_tensor = self.global_tensor_dict["truncations"]

        # 在填充环境之前，创建地面平面
        if self.cfg.env.create_ground_plane:
            logger.info("Creating ground plane in Isaac Gym Simulation.")
            self.IGE_env.create_ground_plane()
            logger.info("[DONE] Creating ground plane in Isaac Gym Simulation")

        # 循环创建并填充每个环境实例
        for i in range(self.cfg.env.num_envs):
            logger.debug(f"Populating environment {i}")
            if i % 1000 == 0:
                logger.info(f"Populating environment {i}")

            # 在 Isaac Gym 中创建环境句柄
            env_handle = self.IGE_env.create_env(i)
            # 如果使用 Warp，则在 Warp 中创建环境
            if self.cfg.env.use_warp:
                self.warp_env.create_env(i)

            # 在环境中添加机器人资产
            self.robot_manager.add_robot_to_env(
                self.IGE_env, env_handle, self.global_asset_counter, i, segmentation_ctr
            )
            self.global_asset_counter += 1

            self.num_obs_in_env = 0
            # 在环境中添加常规资产/障碍物
            for asset_info_dict in self.global_asset_dicts[i]:
                # 在 IGE 中添加资产
                asset_handle, ige_seg_ctr = self.IGE_env.add_asset_to_env(
                    asset_info_dict,
                    env_handle,
                    i,
                    self.global_asset_counter,
                    segmentation_ctr,
                )
                self.num_obs_in_env += 1
                warp_segmentation_ctr = 0
                # 如果使用 Warp，则在 Warp 中添加资产
                if self.cfg.env.use_warp:
                    empty_handle, warp_segmentation_ctr = self.warp_env.add_asset_to_env(
                        asset_info_dict,
                        i,
                        self.global_asset_counter,
                        segmentation_ctr,
                    )
                # 更新全局资产计数器和分割计数器
                self.global_asset_counter += 1
                segmentation_ctr += max(ige_seg_ctr, warp_segmentation_ctr)
                
                # 收集资产状态比例（用于重置/随机化）
                if self.asset_min_state_ratio is None or self.asset_max_state_ratio is None:
                    self.asset_min_state_ratio = torch.tensor(
                        asset_info_dict["min_state_ratio"], requires_grad=False
                    ).unsqueeze(0)
                    self.asset_max_state_ratio = torch.tensor(
                        asset_info_dict["max_state_ratio"], requires_grad=False
                    ).unsqueeze(0)
                else:
                    self.asset_min_state_ratio = torch.vstack(
                        (
                            self.asset_min_state_ratio,
                            torch.tensor(asset_info_dict["min_state_ratio"], requires_grad=False),
                        )
                    )
                    self.asset_max_state_ratio = torch.vstack(
                        (
                            self.asset_max_state_ratio,
                            torch.tensor(asset_info_dict["max_state_ratio"], requires_grad=False),
                        )
                    )

        # 检查环境中是否有资产，然后处理状态比例张量
        if self.asset_min_state_ratio is not None:
            self.asset_min_state_ratio = self.asset_min_state_ratio.to(self.device)
            self.asset_max_state_ratio = self.asset_max_state_ratio.to(self.device)
            # 重新塑形为 (num_envs, num_assets_per_env, 13)
            self.global_tensor_dict["asset_min_state_ratio"] = self.asset_min_state_ratio.view(
                self.cfg.env.num_envs, -1, 13
            )
            self.global_tensor_dict["asset_max_state_ratio"] = self.asset_max_state_ratio.view(
                self.cfg.env.num_envs, -1, 13
            )
        else:
            # 如果没有资产，则创建空的张量
            self.global_tensor_dict["asset_min_state_ratio"] = torch.zeros(
                (self.cfg.env.num_envs, 0, 13), device=self.device
            )
            self.global_tensor_dict["asset_max_state_ratio"] = torch.zeros(
                (self.cfg.env.num_envs, 0, 13), device=self.device
            )

        self.global_tensor_dict["num_obstacles_in_env"] = self.num_obs_in_env # 每个环境中的障碍物数量

    def prepare_sim(self):
        """
        该函数为环境准备模拟。
        """
        # 告知 Isaac Gym 环境准备进行模拟，并传入全局张量字典
        if not self.IGE_env.prepare_for_simulation(self, self.global_tensor_dict):
            raise Exception("Failed to prepare the simulation")
        # 告知 Warp 环境准备进行模拟
        if self.cfg.env.use_warp:
            if not self.warp_env.prepare_for_simulation(self.global_tensor_dict):
                raise Exception("Failed to prepare the simulation")

        # 创建资产管理器并准备
        self.asset_manager = AssetManager(self.global_tensor_dict, self.keep_in_env)
        self.asset_manager.prepare_for_sim()
        # 准备机器人管理器
        self.robot_manager.prepare_for_sim(self.global_tensor_dict)
        # 创建并准备障碍物管理器
        self.obstacle_manager = ObstacleManager(
            self.IGE_env.num_assets_per_env, self.cfg, self.device
        )
        self.obstacle_manager.prepare_for_sim(self.global_tensor_dict)
        # 从机器人管理器获取机器人动作数量
        self.num_robot_actions = self.global_tensor_dict["num_robot_actions"]

    def reset_idx(self, env_ids=None):
        """
        该函数为给定的环境索引重置环境。
        """
        # 1. 首先重置 Isaac Gym 环境 (确定环境边界)
        # 2. 然后重置资产管理器 (重新定位环境中的资产)
        # 3. 然后重置 Warp 环境 (如果使用，它会读取资产的状态张量并转换网格)
        # 4. 最后重置机器人管理器 (重置机器人状态张量和传感器)
        # logger.debug(f"Resetting environments {env_ids}.")
        self.IGE_env.reset_idx(env_ids)
        self.asset_manager.reset_idx(env_ids, self.global_tensor_dict["num_obstacles_in_env"])
        if self.cfg.env.use_warp:
            self.warp_env.reset_idx(env_ids)
        self.robot_manager.reset_idx(env_ids)
        # 将重置后的状态写入模拟器
        self.IGE_env.write_to_sim()
        self.sim_steps[env_ids] = 0 # 重置模拟步数计数器

    def log_memory_use(self):
        """
        该函数记录 GPU 的内存使用情况。
        """
        # 打印当前分配的 GPU 内存
        logger.warning(
            f"torch.cuda.memory_allocated: {torch.cuda.memory_allocated(0)/1024/1024/1024}GB"
        )
        # 打印当前预留的 GPU 内存
        logger.warning(
            f"torch.cuda.memory_reserved: {torch.cuda.memory_reserved(0)/1024/1024/1024}GB"
        )
        # 打印历史最大预留 GPU 内存
        logger.warning(
            f"torch.cuda.max_memory_reserved: {torch.cuda.max_memory_reserved(0)/1024/1024/1024}GB"
        )

        # 计算该类的对象所使用的系统 RAM 内存用量
        total_memory = 0
        for key, value in self.__dict__.items():
            total_memory += value.__sizeof__()
        logger.warning(
            f"Total memory used by the objects of this class: {total_memory/1024/1024}MB"
        )

    def reset(self):
        # 重置所有环境
        self.reset_idx(env_ids=torch.arange(self.cfg.env.num_envs))

    def pre_physics_step(self, actions, env_actions):
        # 1. 首先让机器人计算动作
        self.robot_manager.pre_physics_step(actions)
        # 2. 资产管理器在此处应用环境动作
        self.asset_manager.pre_physics_step(env_actions)
        # 3. 对障碍物管理器应用环境动作
        self.obstacle_manager.pre_physics_step(env_actions)
        # 4. 模拟器在此处应用它们（如机器人关节力/扭矩）
        self.IGE_env.pre_physics_step(actions)
        # 5. 如果使用 warp，warp 环境在此处应用动作
        # 注意：如果网格发生变化，需要调用 refit()（开销很大）。
        if self.use_warp:
            self.warp_env.pre_physics_step(actions)

    def reset_tensors(self):
        # 重置碰撞和截断张量为 0/False
        self.collision_tensor[:] = 0
        self.truncation_tensor[:] = 0

    def simulate(self, actions, env_actions):
        # 物理步前的准备工作
        self.pre_physics_step(actions, env_actions)
        # 执行物理模拟步
        self.IGE_env.physics_step()
        # 物理步后的处理
        self.post_physics_step(actions, env_actions)

    def post_physics_step(self, actions, env_actions):
        # IGE 环境的后处理
        self.IGE_env.post_physics_step()
        # 机器人管理器的后处理
        self.robot_manager.post_physics_step()
        # Warp 环境的后处理
        if self.use_warp:
            self.warp_env.post_physics_step()
        # 资产管理器的后处理
        self.asset_manager.post_physics_step()

    def compute_observations(self):
        # 计算碰撞状态：如果机器人接触力（第二维）的范数超过阈值，则将碰撞张量置为 True
        self.collision_tensor[:] += (
            torch.norm(self.global_tensor_dict["robot_contact_force_tensor"], dim=1)
            > self.cfg.env.collision_force_threshold
        )

    def reset_terminated_and_truncated_envs(self):
        # 获取发生碰撞的环境索引
        collision_envs = self.collision_tensor.nonzero(as_tuple=False).squeeze(-1)
        # 获取发生截断的环境索引
        truncation_envs = self.truncation_tensor.nonzero(as_tuple=False).squeeze(-1)
        
        # 计算需要重置的环境索引：(如果配置允许碰撞重置 AND 发生碰撞) OR 发生截断
        envs_to_reset = (
            (self.collision_tensor * int(self.cfg.env.reset_on_collision) + self.truncation_tensor)
            .nonzero(as_tuple=False)
            .squeeze(-1)
        )
        # 重置需要重置的环境
        if len(envs_to_reset) > 0:
            self.reset_idx(envs_to_reset)
        return envs_to_reset

    def render(self, render_components="sensors"):
        # 根据参数选择渲染查看器还是传感器
        if render_components == "viewer":
            self.render_viewer()
        elif render_components == "sensors":
            self.render_sensors()

    def render_sensors(self):
        # 在物理步之后渲染传感器
        if self.robot_manager.has_IGE_sensors:
            self.IGE_env.step_graphics() # IGE 传感器需要更新图形
        self.robot_manager.capture_sensors() # 捕获机器人传感器数据

    def render_viewer(self):
        # 渲染查看器 GUI
        self.IGE_env.render_viewer()

    def post_reward_calculation_step(self):
        # 重置已终止和已截断的环境
        envs_to_reset = self.reset_terminated_and_truncated_envs()
        # 在重置后执行渲染，以确保传感器数据已从新的机器人状态中更新。
        self.render(render_components="sensors")
        return envs_to_reset

    def step(self, actions, env_actions=None):
        """
        该函数执行环境的模拟步。
        actions: 发送给机器人的动作。
        env_actions: 发送给环境实体的动作。
        """
        self.reset_tensors() # 重置碰撞和截断张量
        
        # 处理环境动作
        if env_actions is not None:
            if self.global_tensor_dict["env_actions"] is None:
                # 首次设置环境动作张量
                self.global_tensor_dict["env_actions"] = env_actions
                self.global_tensor_dict["prev_env_actions"] = env_actions
                self.prev_env_actions = self.global_tensor_dict["prev_env_actions"]
                self.env_actions = self.global_tensor_dict["env_actions"]
            logger.warning(
                f"Env actions shape: {env_actions.shape}, Previous env actions shape: {self.env_actions.shape}"
            )
            # 更新环境动作张量
            self.prev_env_actions[:] = self.env_actions
            self.env_actions[:] = env_actions
        
        # 根据配置（平均值和标准差）采样每次环境步要执行的物理步数
        num_physics_step_per_env_step = max(
            math.floor(
                random.gauss(
                    self.cfg.env.num_physics_steps_per_env_step_mean,
                    self.cfg.env.num_physics_steps_per_env_step_std,
                )
            ),
            0,
        )
        
        # 执行多次物理模拟步
        for timestep in range(num_physics_step_per_env_step):
            self.simulate(actions, env_actions)
            self.compute_observations() # 在每一步后计算观测

        self.sim_steps[:] = self.sim_steps[:] + 1 # 更新模拟步数
        self.step_counter += 1
        
        # 定期渲染查看器
        if self.step_counter % self.cfg.env.render_viewer_every_n_steps == 0:
            self.render(render_components="viewer")

    def get_obs(self):
        # 返回包含所有张量的字典。任务需要计算奖励的部分可以使用这些张量。
        return self.global_tensor_dict