# 从环境对象配置中导入各种资产参数类
from aerial_gym.config.asset_config.env_object_config import (
    panel_asset_params,  # 板状资产参数
    thin_asset_params,   # 薄形资产参数
    tree_asset_params,   # 树木资产参数
    object_asset_params, # 通用对象资产参数
    tile_asset_params,   # 地砖/瓦片资产参数
)
# 导入构成环境边界的墙体资产参数
from aerial_gym.config.asset_config.env_object_config import (
    left_wall,    # 左墙
    right_wall,   # 右墙
    back_wall,    # 后墙
    front_wall,   # 前墙
    bottom_wall,  # 底墙（地板）
    top_wall,     # 顶墙（天花板）
)

import numpy as np # 导入 NumPy 库（尽管在类定义中未直接使用，但通常用于配置相关操作）


# 解释: 这个类定义了一个包含障碍物的环境配置。
class EnvWithObstaclesCfg:
    class env:
        # 并行环境的数量。如果任务配置中使用了 num_envs 参数，则会被覆盖。
        num_envs = 64  # 如果任务配置中使用，则会被 num_envs 参数覆盖
        # 由环境自身处理的动作数量。
        num_env_actions = 4  # 由环境处理的动作数量
        # 这些动作可能部分来自于 RL 智能体用于控制机器人，
        # 部分可用于控制环境中的各种实体，例如障碍物的运动等。
        # potentially some of these can be input from the RL agent for the robot and
        # some of them can be used to control various entities in the environment
        # e.g. motion of obstacles, etc.
        env_spacing = 5.0  # 环境之间的间距。[对于高度场(heightfields)/三角网格(trimeshes)不使用]。

        # 两次摄像机渲染之间的物理步长数的平均值。
        num_physics_steps_per_env_step_mean = 10  # 两次摄像机渲染之间的步数平均值
        # 两次摄像机渲染之间的物理步长数的标准差。
        num_physics_steps_per_env_step_std = 0  # 两次摄像机渲染之间的步数标准差

        render_viewer_every_n_steps = 1  # 每 n 个环境步渲染一次查看器。
        # 当四旋翼飞行器上的接触力超过阈值时，是否重置环境。
        reset_on_collision = (
            True  # 当四旋翼飞行器上的接触力超过阈值时重置环境
        )
        collision_force_threshold = 0.05  # [N] 碰撞力阈值（牛顿）。
        create_ground_plane = False  # 是否创建地面平面。
        # 是否对时间步长进行采样以增加延迟噪声。
        sample_timestep_for_latency = True  # 对时间步长进行采样以计算延迟噪声
        perturb_observations = True  # 是否对观测值添加扰动/噪声。
        keep_same_env_for_num_episodes = 1  # 保持同一环境配置的集数。
        # 是否在每个时间步都将数据写入模拟器。
        write_to_sim_at_every_timestep = False  # 在每个时间步写入模拟

        use_warp = True  # 是否使用 Warp 物理引擎/框架。
        # 环境空间的最小边界的下限 (x, y, z)。
        lower_bound_min = [-2.0, -4.0, -3.0]  # 环境空间的最小边界下限
        # 环境空间的最小边界的上限 (x, y, z)。
        lower_bound_max = [-1.0, -2.5, -2.0]  # 环境空间的最小边界上限
        # 环境空间的最大边界的下限 (x, y, z)。
        upper_bound_min = [9.0, 2.5, 2.0]  # 环境空间的最大边界下限
        # 环境空间的最大边界的上限 (x, y, z)。
        upper_bound_max = [10.0, 4.0, 3.0]  # 环境空间的最大边界上限

    class env_config:
        # 要包含在环境中的资产类型（及其启用/禁用状态）。
        include_asset_type = {
            "panels": True,    # 包含板状物
            "tiles": False,    # 不包含地砖/瓦片
            "thin": False,     # 不包含薄形物
            "trees": False,    # 不包含树木
            "objects": True,   # 包含通用对象
            "left_wall": True, # 包含左墙
            "right_wall": True, # 包含右墙
            "back_wall": True,  # 包含后墙
            "front_wall": True, # 包含前墙
            "top_wall": True,   # 包含顶墙
            "bottom_wall": True, # 包含底墙
        }

        # 将上述名称映射到定义资产的类。它们可以通过上面的 include_asset_type 字典启用或禁用。
        asset_type_to_dict_map = {
            "panels": panel_asset_params,
            "thin": thin_asset_params,
            "trees": tree_asset_params,
            "objects": object_asset_params,
            "left_wall": left_wall,
            "right_wall": right_wall,
            "back_wall": back_wall,
            "front_wall": front_wall,
            "bottom_wall": bottom_wall,
            "top_wall": top_wall,
            "tiles": tile_asset_params,
        }