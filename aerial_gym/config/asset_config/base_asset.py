from aerial_gym import AERIAL_GYM_DIRECTORY  # 从 aerial_gym 导入 AERIAL_GYM_DIRECTORY

# 解释: 这个类定义了用于创建模拟环境中基础资产的参数。
class BaseAssetParams:
    num_assets = 1  # 资产的数量，默认为 1。

    # 资产模型文件的根文件夹路径。
    asset_folder = f"{AERIAL_GYM_DIRECTORY}/resources/models/environment_assets"
    # 指定要使用的资产文件。如果为 None，将随机选择资产。
    file = None  # 如果 file=None，将选择随机资产。如果不为 None，将使用此文件。

    # 资产最小位置的比例，相对于环境边界。
    min_position_ratio = [0.5, 0.5, 0.5]  # 相对于边界的最小位置比例
    # 资产最大位置的比例，相对于环境边界。
    max_position_ratio = [0.5, 0.5, 0.5]  # 相对于边界的最大位置比例

    collision_mask = 1  # 碰撞掩码，用于定义哪些对象之间会发生碰撞。

    disable_gravity = False  # 是否禁用资产的重力。
    # 是否用胶囊体替换碰撞圆柱体。这通常会带来更快/更稳定的仿真。
    replace_cylinder_with_capsule = (
        True  # 用胶囊体替换碰撞圆柱体，带来更快/更稳定的仿真
    )
    # 某些 .obj 网格可能需要从 y 轴向上翻转到 z 轴向上。
    flip_visual_attachments = True  # 某些 .obj 网格必须从 y 轴向上翻转到 z 轴向上
    density = 0.000001  # 资产的密度。
    angular_damping = 0.0001  # 角阻尼。
    linear_damping = 0.0001  # 线性阻尼。
    max_angular_velocity = 100.0  # 最大角速度。
    max_linear_velocity = 100.0  # 最大线速度。
    armature = 0.001  # 骨架（Armature）参数，通常用于物理模拟中的稳定性。

    collapse_fixed_joints = True  # 是否合并固定关节，有助于简化物理模型。
    fix_base_link = True  # 是否固定基础链接（base link），使其不可移动。
    color = None  # 资产的颜色。如果为 None，则使用模型自带颜色或默认颜色。
    keep_in_env = False  # 是否保持资产在环境中（防止其被移除或超出边界）。

    body_semantic_label = 0  # 整体的语义标签。
    link_semantic_label = 0  # 链接的语义标签。
    per_link_semantic = False  # 是否为每个链接设置独立的语义标签。
    semantic_masked_links = {}  # 带有语义标签掩码的链接字典。
    place_force_sensor = False  # 是否放置力传感器。
    force_sensor_parent_link = "base_link"  # 力传感器连接的父链接名称。
    # 力传感器的变换矩阵：[x, y, z, quat_x, quat_y, quat_z, quat_w]
    force_sensor_transform = [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ]  # 位置 (x, y, z)，四元数 (x, y, z, w)
    # 是否使用碰撞网格代替可视化网格进行渲染（通常用于调试或简化显示）。
    use_collision_mesh_instead_of_visual = False