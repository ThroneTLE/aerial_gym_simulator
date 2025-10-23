import numpy as np

class control:
    """
    ... (注释不变) ...
    """

    num_actions = 4
    max_inclination_angle_rad = np.pi / 3.0
    max_yaw_rate = np.pi / 3.0

    # --- 恢复为通用的、有范围的默认值 ---
    
    K_pos_tensor_max = [3.0, 3.0, 2.0]  # (原始默认值)
    K_pos_tensor_min = [2.0, 2.0, 1.0]  # (原始默认值)

    # (你的 PPO 结果 - 注释掉)
    # K_pos_tensor_max = [3.8515, 3.3746, 2.9375]
    # K_pos_tensor_min = [0.8515, 1.3746, 2.9375]

    K_vel_tensor_max = [3.0, 3.0, 3.0]  # (原始默认值)
    K_vel_tensor_min = [2.0, 2.0, 2.0]  # (原始默认值)

    # (你的 PPO 结果 - 注释掉)
    # K_vel_tensor_max = [2.7403, 3.1098, 2.7561]
    # K_vel_tensor_min = [0.7403, 0.1098, 2.7561]

    # 位置积分增益（这个可以保留，PPO 脚本并未使用它）
    K_pos_i_tensor_max = [0.0, 0.0, 5.6]
    K_pos_i_tensor_min = [0.0, 0.0, 0.01]

    K_rot_tensor_max = [1.2, 1.2, 0.6]  # (原始默认值)
    K_rot_tensor_min = [0.6, 0.6, 0.3]  # (原始默认值)

    # (你的 PPO 结果 - 注释掉)
    # K_rot_tensor_max = [0.9800, 0.8548, 0.9800]
    # K_rot_tensor_min = [0.9800, 0.8548, 0.9800]

    K_angvel_tensor_max = [0.2, 0.2, 0.2]  # (原始默认值)
    K_angvel_tensor_min = [0.1, 0.1, 0.1]  # (原始默认值)

    # (你的 PPO 结果 - 注释掉)
    # K_angvel_tensor_max = [1.3782, 1.4058, 0.4015]
    # K_angvel_tensor_min = [0.3782, 0.4058, 0.4015]

    randomize_params = False