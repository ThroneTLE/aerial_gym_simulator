# nothing to import here as the no other modules are needed to define base class
# 此处无需导入，因为定义基类不需要其他模块

# 解释: 这个类定义了一个空环境的基础配置参数 (Empty Environment Configuration)。
class EmptyEnvCfg:
    class env:
        num_envs = 3  # 环境的数量，即并行模拟的实例数。
        # 这是一个由环境自身处理的动作数量。
        num_env_actions = 0  # 由环境处理的动作数量
        # 这些动作被发送给环境中的实体，并可能用于控制环境中的各种元素，
        # 例如障碍物的运动等。
        # these are the actions that are sent to environment entities
        # and some of them may be used to control various entities in the environment
        # e.g. motion of obstacles, etc.
        env_spacing = 1.0  # 环境之间的间距。[对于高度场(heightfields)/三角网格(trimeshes)不使用]。
        # 两次摄像机渲染之间的物理步长数的平均值。
        num_physics_steps_per_env_step_mean = 1  # 两次摄像机渲染之间的步数平均值
        # 两次摄像机渲染之间的物理步长数的标准差（用于增加随机性）。
        num_physics_steps_per_env_step_std = 0  # 两次摄像机渲染之间的步数标准差
        render_viewer_every_n_steps = 10  # 每 n 个环境步渲染一次查看器。
        collision_force_threshold = 0.010  # 碰撞力阈值。
        manual_camera_trigger = False  # 是否手动触发摄像机捕获图像。
        # 当四旋翼飞行器上的接触力超过阈值时，是否重置环境。
        reset_on_collision = (
            True  # 当四旋翼飞行器上的接触力超过阈值时重置环境
        )
        create_ground_plane = False  # 是否创建地面平面。
        # 是否对时间步长进行采样以增加延迟噪声。
        sample_timestep_for_latency = True  # 对时间步长进行采样以计算延迟噪声
        perturb_observations = True  # 是否对观测值添加扰动/噪声。
        keep_same_env_for_num_episodes = 1  # 保持同一环境配置的集数。
        # 是否在每个时间步都将数据写入模拟器。
        write_to_sim_at_every_timestep = False  # 在每个时间步写入模拟

        use_warp = False  # 是否使用 Warp 物理引擎/框架。
        e_s = env_spacing  # 将 env_spacing 赋值给 e_s 方便使用。
        # 环境空间的最小边界的下限 (x, y, z)。
        lower_bound_min = [-e_s, -e_s, -e_s]  # 环境空间的最小边界下限
        # 环境空间的最小边界的上限 (x, y, z)。
        lower_bound_max = [-e_s, -e_s, -e_s]  # 环境空间的最小边界上限
        # 环境空间的最大边界的下限 (x, y, z)。
        upper_bound_min = [e_s, e_s, e_s]  # 环境空间的最大边界下限
        # 环境空间的最大边界的上限 (x, y, z)。
        upper_bound_max = [e_s, e_s, e_s]  # 环境空间的最大边界上限

    class env_config:
        # 包含在环境中的资产类型（及其配置）。
        include_asset_type = {}

        # 资产类型到配置字典的映射。
        asset_type_to_dict_map = {}