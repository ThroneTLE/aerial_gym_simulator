# 解释: 这个类定义了基础的模拟配置参数，主要包括可视化查看器 (viewer) 和物理模拟器 (sim) 的设置。
class BaseSimConfig:
    # -----------------------------------------------------------------------
    # 模拟环境可视化查看器配置 (Viewer Camera Configuration)
    class viewer:
        headless = False  # 是否以无头模式运行（不显示图形界面）。False 表示显示。
        ref_env = 0  # 摄像机跟随或观察的参考环境索引。
        camera_position = [-5, -5, 4]  # [m] 摄像机的初始位置 (x, y, z)。
        lookat = [0, 0, 0]  # [m] 摄像机注视的点 (x, y, z)。
        camera_orientation_euler_deg = [0, 0, 0]  # [deg] 摄像机的欧拉角朝向（度）。
        # 摄像机跟随类型。FOLLOW_TRANSFORM 表示跟随某个变换矩阵。
        camera_follow_type = "FOLLOW_TRANSFORM"
        width = 1280  # 窗口宽度（像素）。
        height = 720  # 窗口高度（像素）。
        max_range = 100.0  # [m] 摄像机远裁剪面距离。
        min_range = 0.1  # 摄像机近裁剪面距离。
        horizontal_fov_deg = 90  # 摄像机的水平视野角度（度）。
        use_collision_geometry = False  # 是否使用碰撞几何体进行渲染（通常用于调试）。
        # 摄像机跟随目标变换的局部坐标系偏移量 (x, y, z)。
        camera_follow_transform_local_offset = [-1.0, 0.0, 0.3]  # m
        # 摄像机跟随目标位置的全局坐标系偏移量 (x, y, z)。
        camera_follow_position_global_offset = [-1.0, 0.0, 0.3]  # m

    # -----------------------------------------------------------------------
    # 基础模拟配置 (Core Simulation Configuration)
    class sim:
        dt = 0.01  # 仿真步长/时间间隔（秒）。
        substeps = 1  # 每次仿真步长内的子步数。
        gravity = [0.0, 0.0, -9.81]  # [m/s^2] 重力向量 (x, y, z)。
        up_axis = 1  # 向上轴的定义：0 是 Y 轴，1 是 Z 轴。
        use_gpu_pipeline = True  # 是否使用 GPU 加速的仿真管线。

        # PhysX 物理引擎配置 (NVIDIA PhysX Configuration)
        class physx:
            num_threads = 10  # PhysX 求解器使用的 CPU 线程数。
            # 求解器类型：0: PGS (Sequential impulse), 1: TGS (Temporal Gauss-Seidel)。
            solver_type = 1  # 0: pgs (位置/速度分离), 1: tgs (时间高斯-赛德尔)
            num_position_iterations = 4  # 求解器位置迭代次数（影响穿透校正的精度）。
            num_velocity_iterations = 1  # 求解器速度迭代次数（影响速度和摩擦的精度）。
            contact_offset = 0.002  # [m] 接触偏移量，用于定义接触生成的距离。
            rest_offset = 0.001  # [m] 静止偏移量，用于稳定堆叠。
            bounce_threshold_velocity = 0.1  # [m/s] 发生弹跳所需的最小相对速度。
            max_depenetration_velocity = 1.0  # 最大反穿透速度，限制物体分离的速度。
            # 最大 GPU 接触对数。对于大规模环境（如 8000 个环境）需要设置较大值。
            max_gpu_contact_pairs = 2**24  # 2**24 -> 适用于 8000 个以上环境
            default_buffer_size_multiplier = 10  # 默认缓冲区大小的乘数。
            # 接触收集模式：0: 从不, 1: 最后一个子步, 2: 所有子步 (默认=2)。
            contact_collection = 1  # 0: never, 1: last sub-step, 2: all sub-steps (默认=2)