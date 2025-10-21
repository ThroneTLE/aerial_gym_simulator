import numpy as np # 导入 NumPy

from aerial_gym import AERIAL_GYM_DIRECTORY # 导入 AERIAL_GYM 的根目录

# 导入各种传感器配置基类
from aerial_gym.config.sensor_config.camera_config.base_depth_camera_config import (
    BaseDepthCameraConfig, # 基础深度相机配置
)
from aerial_gym.config.sensor_config.lidar_config.base_lidar_config import (
    BaseLidarConfig, # 基础激光雷达配置
)
from aerial_gym.config.sensor_config.camera_config.base_normal_faceID_camera_config import (
    BaseNormalFaceIDCameraConfig, # 基础法线/FaceID 相机配置
)
from aerial_gym.config.sensor_config.camera_config.stereo_camera_config import (
    StereoCameraConfig, # 立体相机配置
)
from aerial_gym.config.sensor_config.lidar_config.osdome_64_config import OSDome_64_Config # OSDome 64 线雷达配置
from aerial_gym.config.sensor_config.imu_config.base_imu_config import BaseImuConfig # 基础 IMU 配置


class BaseQuadCfg:
    """基础四旋翼飞行器配置类。"""

    class init_config:
        """机器人初始状态配置。"""
        # 初始状态张量格式: [ratio_x, ratio_y, ratio_z, roll_rad, pitch_rad, yaw_rad, 1.0, vx, vy, vz, wx, wy, wz]
        # 这里的比例 (ratio) 是相对于环境边界的。
        min_init_state = [
            0.1, 0.15, 0.15, # 最小位置比例
            0, 0, -np.pi / 6, # 最小旋转角度（Roll, Pitch, Yaw）
            1.0, # 保持形状的占位符
            -0.2, -0.2, -0.2, # 最小初始线速度 (vx, vy, vz)
            -0.2, -0.2, -0.2, # 最小初始角速度 (wx, wy, wz)
        ]
        max_init_state = [
            0.2, 0.85, 0.85, # 最大位置比例
            0, 0, np.pi / 6, # 最大旋转角度（Roll, Pitch, Yaw）
            1.0,
            0.2, 0.2, 0.2, # 最大初始线速度
            0.2, 0.2, 0.2, # 最大初始角速度
        ]

    class sensor_config:
        """机器人传感器配置。"""
        enable_camera = False # 是否启用相机
        camera_config = BaseDepthCameraConfig  # 默认相机配置（深度相机）

        enable_lidar = False # 是否启用激光雷达
        lidar_config = BaseLidarConfig  # 默认激光雷达配置

        enable_imu = False # 是否启用 IMU（惯性测量单元）
        imu_config = BaseImuConfig

    class disturbance:
        """外部扰动配置。"""
        enable_disturbance = False # 是否启用外部扰动
        prob_apply_disturbance = 0.00 # 每一步施加扰动的概率
        # 最大力和力矩扰动 [fx, fy, fz, tx, ty, tz]
        max_force_and_torque_disturbance = [0.75, 0.75, 0.75, 0.004, 0.004, 0.004]

    class damping:
        """空气阻尼配置。"""
        # 沿机体轴 [x, y, z] 的线速度线性阻尼系数
        linvel_linear_damping_coefficient = [0.0, 0.0, 0.0]
        # 沿机体轴 [x, y, z] 的线速度二次阻尼系数
        linvel_quadratic_damping_coefficient = [0.0, 0.0, 0.0]
        # 沿机体轴 [x, y, z] 的角速度线性阻尼系数
        angular_linear_damping_coefficient = [0.0, 0.0, 0.0]
        # 沿机体轴 [x, y, z] 的角速度二次阻尼系数
        angular_quadratic_damping_coefficient = [0.0, 0.0, 0.0]

    class robot_asset:
        """机器人资产配置（模型和物理属性）。"""
        asset_folder = f"{AERIAL_GYM_DIRECTORY}/resources/robots/quad" # 资产文件夹
        file = "quad.urdf" # 资产文件
        name = "base_quadrotor"  # 演员/机器人名称
        base_link_name = "base_link" # 基础链接名称
        disable_gravity = False # 是否禁用重力
        collapse_fixed_joints = False  # 是否合并由固定关节连接的物体
        fix_base_link = False  # 是否固定机器人基础链接
        collision_mask = 0  # 碰撞掩码：1 为禁用，0 为启用（位过滤器）
        # 是否用胶囊体替换碰撞圆柱体（有助于模拟稳定）
        replace_cylinder_with_capsule = False
        # 某些 .obj 网格是否需要从 Y 轴向上翻转到 Z 轴向上
        flip_visual_attachments = True
        density = 0.000001 # 密度
        angular_damping = 0.01 # 角阻尼
        linear_damping = 0.01 # 线性阻尼
        max_angular_velocity = 100.0 # 最大角速度
        max_linear_velocity = 100.0 # 最大线速度
        armature = 0.001 # 骨架/电枢

        semantic_id = 0 # 语义 ID
        per_link_semantic = False # 是否为每个链接使用独立语义 ID

        # 重置时的最小状态比例
        min_state_ratio = [
            0.1, 0.1, 0.1, 0, 0, -np.pi, 1.0, 0, 0, 0, 0, 0, 0,
        ]  # [ratio_x, ratio_y, ratio_z, roll_rad, pitch_rad, yaw_rad, 1.0, vx, vy, vz, wx, wy, wz]
        # 重置时的最大状态比例
        max_state_ratio = [
            0.3, 0.9, 0.9, 0, 0, np.pi, 1.0, 0, 0, 0, 0, 0, 0,
        ]  # [ratio_x, ratio_y, ratio_z, roll_rad, pitch_rad, yaw_rad, 1.0, vx, vy, vz, wx, wy, wz]

        # 最大的外力/力矩扰动 [fx, fy, fz, tx, ty, tz]
        max_force_and_torque_disturbance = [
            0.1, 0.1, 1000, 0.05, 0.05, 0.05,
        ]

        color = None # 颜色
        semantic_masked_links = {} # 语义掩码链接
        keep_in_env = True  # 对机器人无效
        
        # 最小/最大位置比例（如果使用，会覆盖 state_ratio 中的位置部分）
        min_position_ratio = None
        max_position_ratio = None

        min_euler_angles = [-np.pi, -np.pi, -np.pi] # 最小欧拉角（用于随机化）
        max_euler_angles = [np.pi, np.pi, np.pi] # 最大欧拉角

        # 是否放置力传感器（如果需要 IMU，通常设置为 True）
        place_force_sensor = True  # 如果需要 IMU，设置为 True
        force_sensor_parent_link = "base_link" # 力传感器附着的链接
        # 力传感器变换：[x, y, z, qx, qy, qz, qw]
        force_sensor_transform = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

        use_collision_mesh_instead_of_visual = False  # 对机器人无效

    class control_allocator_config:
        """控制分配器配置（将抽象控制输入映射到电机推力/扭矩）。"""
        num_motors = 4 # 电机数量
        # 力施加级别: "motor_link" 或 "root_link"
        force_application_level = "motor_link"  # 决定将合力施加到根链接还是单个电机链接

        # 施力掩码：指定接收力/力矩输入的链接索引
        application_mask = [1 + 4 + i for i in range(0, 4)]
        motor_directions = [1, -1, 1, -1] # 电机旋转方向（影响偏航扭矩）

        # 分配矩阵（将期望的力和力矩 [fz, tx, ty, tz] 映射到电机指令）
        # 格式: [fx, fy, fz, tx, ty, tz] = Allocation_Matrix * [m1, m2, m3, m4]
        allocation_matrix = [
            [0.0, 0.0, 0.0, 0.0], # fx
            [0.0, 0.0, 0.0, 0.0], # fy
            [1.0, 1.0, 1.0, 1.0], # fz (总推力)
            [-0.13, -0.13, 0.13, 0.13], # tx (俯仰力矩)
            [-0.13, 0.13, 0.13, -0.13], # ty (滚转力矩)
            [-0.01, 0.01, -0.01, 0.01], # tz (偏航力矩)
        ]

        class motor_model_config:
            """电机模型配置。"""
            use_rps = True # 是否使用 RPS（每秒转数）作为电机输出单位

            motor_thrust_constant_min = 0.00000926312 # 最小推力常数
            motor_thrust_constant_max = 0.00001826312 # 最大推力常数

            motor_time_constant_increasing_min = 0.04 # 推力增加时的时间常数（最小）
            motor_time_constant_increasing_max = 0.04 # 推力增加时的时间常数（最大）

            motor_time_constant_decreasing_min = 0.04 # 推力减小时的时间常数（最小）
            motor_time_constant_decreasing_max = 0.04 # 推力减小时的时间常数（最大）

            max_thrust = 2 # 最大推力
            min_thrust = 0 # 最小推力

            max_thrust_rate = 100000.0 # 最大推力变化率
            thrust_to_torque_ratio = 0.01 # 推力与扭矩之比
            # 是否使用离散近似计算推力变化率
            use_discrete_approximation = (
                True  # 设置为 False 将基于差值和时间常数计算 f'
            )

# --- 派生配置类 (Derived Configuration Classes) ---

class BaseQuadWithImuCfg(BaseQuadCfg):
    """带 IMU 的基础四旋翼配置。"""
    class sensor_config(BaseQuadCfg.sensor_config):
        enable_imu = True
        imu_config = BaseImuConfig


class BaseQuadWithCameraCfg(BaseQuadCfg):
    """带深度相机的基础四旋翼配置。"""
    class sensor_config(BaseQuadCfg.sensor_config):
        enable_camera = True
        camera_config = BaseDepthCameraConfig

class BaseQuadWithCameraImuCfg(BaseQuadCfg):
    """带深度相机和 IMU 的基础四旋翼配置。"""
    class sensor_config(BaseQuadCfg.sensor_config):
        enable_camera = True
        camera_config = BaseDepthCameraConfig

        enable_imu = True
        imu_config = BaseImuConfig

class BaseQuadWithLidarCfg(BaseQuadCfg):
    """带激光雷达的基础四旋翼配置。"""
    class sensor_config(BaseQuadCfg.sensor_config):
        enable_lidar = True
        lidar_config = BaseLidarConfig

class BaseQuadWithFaceIDNormalCameraCfg(BaseQuadCfg):
    """带法线/FaceID 相机的基础四旋翼配置。"""
    class sensor_config(BaseQuadCfg.sensor_config):
        enable_camera = True
        camera_config = BaseNormalFaceIDCameraConfig

class BaseQuadWithStereoCameraCfg(BaseQuadCfg):
    """带立体相机的基础四旋翼配置。"""
    class sensor_config(BaseQuadCfg.sensor_config):
        enable_camera = True
        camera_config = StereoCameraConfig