from aerial_gym import AERIAL_GYM_DIRECTORY # 导入 aerial_gym 的根目录路径

import numpy as np # 导入 NumPy 库，用于数学操作（如 np.pi）

# --- 语义标签常量定义 (Semantic Label Constants) ---
THIN_SEMANTIC_ID = 1 # 薄形资产的语义 ID
TREE_SEMANTIC_ID = 2 # 树木资产的语义 ID
OBJECT_SEMANTIC_ID = 3 # 通用对象的语义 ID
PANEL_SEMANTIC_ID = 20 # 面板资产的语义 ID
FRONT_WALL_SEMANTIC_ID = 9 # 前墙的语义 ID
BACK_WALL_SEMANTIC_ID = 10 # 后墙的语义 ID
LEFT_WALL_SEMANTIC_ID = 11 # 左墙的语义 ID
RIGHT_WALL_SEMANTIC_ID = 12 # 右墙的语义 ID
BOTTOM_WALL_SEMANTIC_ID = 13 # 底墙（地板）的语义 ID
TOP_WALL_SEMANTIC_ID = 14 # 顶墙（天花板）的语义 ID


# --- 基础资产状态参数类 (Base Asset State Parameters) ---
class asset_state_params:
    num_assets = 1  # 默认包含的资产数量

    # 资产模型文件的根文件夹路径。
    asset_folder = f"{AERIAL_GYM_DIRECTORY}/resources/models/environment_assets"
    # 指定要使用的资产文件。如果为 None，将随机选择资产。
    file = None  # 如果 file=None，随机选择资产。如果不为 None，使用此文件。

    # 最小/最大位置比例，相对于环境边界（用于随机化位置）。
    min_position_ratio = [0.5, 0.5, 0.5]  # 相对于边界的最小位置比例
    max_position_ratio = [0.5, 0.5, 0.5]  # 相对于边界的最大位置比例

    collision_mask = 1 # 碰撞掩码。

    disable_gravity = False # 是否禁用重力。
    # 用胶囊体替换碰撞圆柱体，通常导致更快/更稳定的模拟。
    replace_cylinder_with_capsule = (
        True  # 用胶囊体替换碰撞圆柱体，带来更快/更稳定的模拟
    )
    # 某些 .obj 网格可能需要从 Y 轴向上翻转到 Z 轴向上。
    flip_visual_attachments = True  # 某些 .obj 网格必须从 Y 轴向上翻转到 Z 轴向上
    density = 0.001 # 密度。
    angular_damping = 0.1 # 角阻尼。
    linear_damping = 0.1 # 线性阻尼。
    max_angular_velocity = 100.0 # 最大角速度。
    max_linear_velocity = 100.0 # 最大线速度。
    armature = 0.001 # 骨架/电枢参数。

    collapse_fixed_joints = True # 是否合并固定关节。
    fix_base_link = True # 是否固定基础链接。
    # 如果不为 None，则使用此特定文件路径而不是随机选择文件夹。
    specific_filepath = None  # 如果不为 None，使用此文件夹而不是随机化
    color = None # 资产颜色。
    keep_in_env = False # 是否永久保留在环境中（不随重置而消失）。

    body_semantic_label = 0 # 整体语义标签。
    link_semantic_label = 0 # 链接语义标签。
    per_link_semantic = False # 是否为每个链接设置独立的语义标签。
    semantic_masked_links = {} # 带有语义掩码的链接字典。
    place_force_sensor = False # 是否放置力传感器。
    force_sensor_parent_link = "base_link" # 力传感器附着的父链接。
    # 力传感器变换：[位置 x, y, z, 四元数 x, y, z, w]
    force_sensor_transform = [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ]  # 位置, 四元数 x, y, z, w

    # 是否使用碰撞网格代替可视化网格进行渲染。
    use_collision_mesh_instead_of_visual = False


# --- 面板资产参数 (Panel Asset Parameters) ---
# 用于创建可用于穿越或作为障碍物的板状结构。
class panel_asset_params(asset_state_params):
    num_assets = 3 # 包含 3 个面板资产。

    # 指定面板资产的文件夹。
    asset_folder = f"{AERIAL_GYM_DIRECTORY}/resources/models/environment_assets/panels"

    collision_mask = 1  # 碰撞掩码（相同掩码的对象不会碰撞）。

    # 定义面板的随机位置范围。
    min_position_ratio = [0.3, 0.05, 0.05]
    max_position_ratio = [0.85, 0.95, 0.95]

    # 指定固定位置，如果值大于 -900，则使用此值而非随机化比例。
    specified_position = [
        -1000.0,
        -1000.0,
        -1000.0,
    ]  # 如果 > -900, 使用此值而非随机化比例

    # 最小/最大欧拉角（滚动、俯仰、偏航）。
    min_euler_angles = [0.0, 0.0, -np.pi / 3.0]  # 最小欧拉角 (弧度)
    max_euler_angles = [0.0, 0.0, np.pi / 3.0]  # 最大欧拉角 (弧度)

    # 最小状态比例 [pos x, y, z, rot x, y, z, w, vel x..z, ang_vel x..z]
    min_state_ratio = [
        0.3, 0.05, 0.05,  # 位置比例
        0.0, 0.0, -np.pi / 3.0, 1.0,  # 旋转（欧拉角，w 分量为 1.0）
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 速度/角速度
    ]
    max_state_ratio = [
        0.85, 0.95, 0.95,
        0.0, 0.0, np.pi / 3.0, 1.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ]

    keep_in_env = True # 永久保留在环境中。

    collapse_fixed_joints = True
    per_link_semantic = False
    # 语义 ID 为 -1，表示将按实例递增分配。
    semantic_id = -1  # 将按实例递增分配
    color = [170, 66, 66] # 资产颜色 (RGB)


# --- 地砖/瓦片资产参数 (Tile Asset Parameters) ---
class tile_asset_params(asset_state_params):
    num_assets = 1

    # 指定地砖资产的文件夹。
    asset_folder = f"{AERIAL_GYM_DIRECTORY}/resources/models/environment_assets/tile_meshes"

    collision_mask = 1

    # 位置比例范围。
    min_position_ratio = [0.3, 0.05, 0.05]
    max_position_ratio = [0.85, 0.95, 0.95]

    specified_position = [
        -1000.0,
        -1000.0,
        -1000.0,
    ]  # 如果 > -900, 使用此值

    # 欧拉角范围（此处为固定）。
    min_euler_angles = [0.0, 0.0, 0.0]
    max_euler_angles = [0.0, 0.0, 0.0]

    # 最小/最大状态比例（固定在中心位置，无旋转）。
    min_state_ratio = [
        0.5, 0.5, 0.5,
        0.0, 0.0, 0.0, 1.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ]
    max_state_ratio = [
        0.5, 0.5, 0.5,
        0.0, 0.0, 0.0, 1.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ]

    keep_in_env = True

    collapse_fixed_joints = True
    per_link_semantic = False
    semantic_id = -1 # 将按实例递增分配
    # color = [170, 66, 66] # 颜色被注释掉，使用模型自带或默认颜色


# --- 薄形资产参数 (Thin Asset Parameters) ---
# 用于创建细长或薄的障碍物。
class thin_asset_params(asset_state_params):
    num_assets = 0 # 默认不包含此类资产。

    asset_folder = f"{AERIAL_GYM_DIRECTORY}/resources/models/environment_assets/thin"

    collision_mask = 1

    # 最小/最大状态比例（允许在所有轴上进行大范围的旋转）。
    min_state_ratio = [
        0.3, 0.05, 0.05,
        -np.pi, -np.pi, -np.pi, 1.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ]
    max_state_ratio = [
        0.85, 0.95, 0.95,
        np.pi, np.pi, np.pi, 1.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ]

    collapse_fixed_joints = True
    per_link_semantic = False
    semantic_id = -1 # 将按实例递增分配
    color = [170, 66, 66]


# --- 树木资产参数 (Tree Asset Parameters) ---
class tree_asset_params(asset_state_params):
    num_assets = 1

    asset_folder = f"{AERIAL_GYM_DIRECTORY}/resources/models/environment_assets/trees"

    collision_mask = 1

    # 最小/最大状态比例。
    min_state_ratio = [
        0.1, 0.1, 0.0, # 位置比例：Z 轴从底部 (0.0) 开始
        0, -np.pi / 6.0, -np.pi, 1.0, # 旋转：只允许俯仰和偏航范围内的随机化
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ]
    max_state_ratio = [
        0.9, 0.9, 0.0,
        0, np.pi / 6.0, np.pi, 1.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ]

    collapse_fixed_joints = True
    per_link_semantic = True # 为每个链接设置独立的语义标签。
    keep_in_env = True

    # semantic_id = -1 # 原始值被注释。此处未明确使用 TREE_SEMANTIC_ID。
    semantic_id = -1  # 将按实例递增分配 (注意: 如果 per_link_semantic 为 True，此值可能被忽略或用于初始/默认值)
    color = [70, 200, 100]

    semantic_masked_links = {}


# --- 通用对象资产参数 (Object Asset Parameters) ---
# 用于创建各种随机形状的障碍物。
class object_asset_params(asset_state_params):
    num_assets = 35 # 包含 35 个通用对象资产。

    asset_folder = f"{AERIAL_GYM_DIRECTORY}/resources/models/environment_assets/objects"

    # 最小/最大状态比例（允许大范围的位置和旋转随机化）。
    min_state_ratio = [
        0.30, 0.05, 0.05,
        -np.pi, -np.pi, -np.pi, 1.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ]
    max_state_ratio = [
        0.85, 0.9, 0.9,
        np.pi, np.pi, np.pi, 1.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ]

    keep_in_env = False # 不永久保留在环境中。
    per_link_semantic = False
    semantic_id = -1 # 将按实例递增分配

    # color = [80,255,100] # 颜色被注释掉


# --- 边界墙体资产参数 (Boundary Wall Asset Parameters) ---

# 左墙 (Left Wall)
class left_wall(asset_state_params):
    num_assets = 1
    asset_folder = f"{AERIAL_GYM_DIRECTORY}/resources/models/environment_assets/walls"
    file = "left_wall.urdf" # 指定 URDF 文件

    collision_mask = 1

    # 最小/最大状态比例（固定位置：X 轴中心 0.5，Y 轴边界 1.0，Z 轴中心 0.5）。
    min_state_ratio = [0.5, 1.0, 0.5, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,]
    max_state_ratio = [0.5, 1.0, 0.5, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,]

    keep_in_env = True
    collapse_fixed_joints = True
    specific_filepath = "cube.urdf" # 可能用于指定内部使用的简单碰撞体
    per_link_semantic = False
    semantic_id = LEFT_WALL_SEMANTIC_ID # 使用左墙的语义 ID
    color = [100, 200, 210]


# 右墙 (Right Wall)
class right_wall(asset_state_params):
    num_assets = 1
    asset_folder = f"{AERIAL_GYM_DIRECTORY}/resources/models/environment_assets/walls"
    file = "right_wall.urdf"

    # 最小/最大状态比例（固定位置：X 轴中心 0.5，Y 轴边界 0.0，Z 轴中心 0.5）。
    min_state_ratio = [0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,]
    max_state_ratio = [0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,]

    keep_in_env = True
    collapse_fixed_joints = True
    per_link_semantic = False
    specific_filepath = "cube.urdf"
    semantic_id = RIGHT_WALL_SEMANTIC_ID
    color = [100, 200, 210]


# 顶墙/天花板 (Top Wall)
class top_wall(asset_state_params):
    num_assets = 1
    asset_folder = f"{AERIAL_GYM_DIRECTORY}/resources/models/environment_assets/walls"
    file = "top_wall.urdf"

    collision_mask = 1

    # 最小/最大状态比例（固定位置：X, Y 轴中心 0.5，Z 轴边界 1.0）。
    min_state_ratio = [0.5, 0.5, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,]
    max_state_ratio = [0.5, 0.5, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,]

    keep_in_env = True
    collapse_fixed_joints = True
    specific_filepath = "cube.urdf"
    per_link_semantic = False
    semantic_id = TOP_WALL_SEMANTIC_ID
    color = [100, 200, 210]


# 底墙/地板 (Bottom Wall)
class bottom_wall(asset_state_params):
    num_assets = 1
    asset_folder = f"{AERIAL_GYM_DIRECTORY}/resources/models/environment_assets/walls"
    file = "bottom_wall.urdf"

    collision_mask = 1

    # 最小/最大状态比例（固定位置：X, Y 轴中心 0.5，Z 轴边界 0.0）。
    min_state_ratio = [0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,]
    max_state_ratio = [0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,]

    keep_in_env = True
    collapse_fixed_joints = True
    specific_filepath = "cube.urdf"
    per_link_semantic = False
    semantic_id = BOTTOM_WALL_SEMANTIC_ID
    color = [100, 150, 150]


# 前墙 (Front Wall)
class front_wall(asset_state_params):
    num_assets = 1
    asset_folder = f"{AERIAL_GYM_DIRECTORY}/resources/models/environment_assets/walls"
    file = "front_wall.urdf"

    collision_mask = 1

    # 最小/最大状态比例（固定位置：X 轴边界 1.0，Y, Z 轴中心 0.5）。
    min_state_ratio = [1.0, 0.5, 0.5, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,]
    max_state_ratio = [1.0, 0.5, 0.5, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,]

    keep_in_env = True
    collapse_fixed_joints = True
    specific_filepath = "cube.urdf"
    per_link_semantic = False
    semantic_id = FRONT_WALL_SEMANTIC_ID
    color = [100, 200, 210]


# 后墙 (Back Wall)
class back_wall(asset_state_params):
    num_assets = 1
    asset_folder = f"{AERIAL_GYM_DIRECTORY}/resources/models/environment_assets/walls"
    file = "back_wall.urdf"

    collision_mask = 1

    # 最小/最大状态比例（固定位置：X 轴边界 0.0，Y, Z 轴中心 0.5）。
    min_state_ratio = [0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,]
    max_state_ratio = [0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,]

    keep_in_env = True
    collapse_fixed_joints = True
    specific_filepath = "cube.urdf"
    per_link_semantic = False
    semantic_id = BACK_WALL_SEMANTIC_ID
    color = [100, 200, 210]