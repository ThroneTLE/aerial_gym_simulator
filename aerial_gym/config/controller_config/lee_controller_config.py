'''
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

    
    
import numpy as np

class control:
    """
    Configuración del controlador Lee Position Controller con ganancias
    optimizadas por rl-games (basado en los resultados proporcionados).

    Resultados de la ejecución:
    {
      "reward": -2.637246239045635,
      "gains": [
        2.2221944332122803, 2.81489634513855, 4.887500286102295,    // K_pos
        2.2952847480773926, 2.751523017883301, 2.641634464263916,    // K_vel
        4.887500286102295, 4.887500286102295, 2.187039852142334,    // K_rot
        0.17375002801418304, 0.17375002801418304, 0.33314138650894165 // K_angvel
      ],
      "loss": 2.637246239045635
    }
"""

    num_actions = 4
    max_inclination_angle_rad = np.pi / 3.0
    max_yaw_rate = np.pi / 3.0

    # --- Ganancias PPO (K_pos) ---
    # Valores: [2.2221, 2.8148, 4.8875]
    K_pos_tensor_max = [2.2221944332122803, 2.81489634513855, 4.887500286102295]
    K_pos_tensor_min = [2.2221944332122803, 2.81489634513855, 4.887500286102295]

    # --- Ganancias PPO (K_vel) ---
    # Valores: [2.2952, 2.7515, 2.6416]
    K_vel_tensor_max = [2.2952847480773926, 2.751523017883301, 2.641634464263916]
    K_vel_tensor_min = [2.2952847480773926, 2.751523017883301, 2.641634464263916]

    # --- Ganancias Integrales (K_pos_i) - No optimizadas por RL, valores por defecto ---
    # 保持原始文件的默认值
    K_pos_i_tensor_max = [0.0, 0.0, 0.0]
    K_pos_i_tensor_min = [0.0, 0.0, 0.0]

    # --- Ganancias PPO (K_rot) ---
    # Valores: [4.8875, 4.8875, 2.1870]
    K_rot_tensor_max = [4.887500286102295, 4.887500286102295, 2.187039852142334]
    K_rot_tensor_min = [4.887500286102295, 4.887500286102295, 2.187039852142334]

    # --- Ganancias PPO (K_angvel) ---
    # Valores: [0.1737, 0.1737, 0.3331]
    K_angvel_tensor_max = [0.17375002801418304, 0.17375002801418304, 0.33314138650894165]
    K_angvel_tensor_min = [0.17375002801418304, 0.17375002801418304, 0.33314138650894165]

    # Deshabilitar aleatorización ya que min y max son iguales
    randomize_params = False
    
'''

import numpy as np

class control:
    """
    Configuración del controlador Lee Position Controller con ganancias
    optimizadas manualmente (basado en los valores proporcionados).

    Valores:
    K_pos: [6.762563705444336, 7.672436237335205, 13.746875762939453]
    K_vel: [7.846602439880371, 7.9705023765563965, 11.299003601074219]
    K_rot: [13.746875762939453, 13.746875762939453, 8.456195831298828]
    K_angvel: [2.955031156539917, 2.955031156539917, 4.277930736541748]
    """

    num_actions = 4
    max_inclination_angle_rad = np.pi / 3.0
    max_yaw_rate = np.pi / 3.0

    # --- Ganancias PPO (K_pos) ---
    K_pos_tensor_max = [9.762563705444336, 9.672436237335205, 13.746875762939453]
    K_pos_tensor_min = [3.762563705444336, 3.672436237335205, 13.746875762939453]

    # --- Ganancias PPO (K_vel) ---
    K_vel_tensor_max = [8.846602439880371, 8.9705023765563965, 11.299003601074219]
    K_vel_tensor_min = [6.846602439880371, 6.9705023765563965, 11.299003601074219]

    # --- Ganancias Integrales (K_pos_i) - No optimizadas, valores por defecto ---
    # Se mantienen los valores del archivo original
    K_pos_i_tensor_max = [0.0, 0.0, 0]
    K_pos_i_tensor_min = [0.0, 0.0, 0]

    # --- Ganancias PPO (K_rot) ---
    K_rot_tensor_max = [13.746875762939453, 13.746875762939453, 8.456195831298828]
    K_rot_tensor_min = [13.746875762939453, 13.746875762939453, 8.456195831298828]

    # --- Ganancias PPO (K_angvel) ---
    K_angvel_tensor_max = [2.955031156539917, 2.955031156539917, 4.277930736541748]
    K_angvel_tensor_min = [2.955031156539917, 2.955031156539917, 4.277930736541748]

    # Deshabilitar aleatorización ya que min y max son iguales
    randomize_params = False