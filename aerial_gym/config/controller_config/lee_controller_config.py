import numpy as np


class control:
    """
    Control parameters
    controller:
        lee_position_control: command_actions = [x, y, z, yaw] in environment frame scaled between -1 and 1
        lee_velocity_control: command_actions = [vx, vy, vz, yaw_rate] in vehicle frame scaled between -1 and 1
        lee_attitude_control: command_actions = [thrust, roll, pitch, yaw_rate] in vehicle frame scaled between -1 and 1
    kP: gains for position
    kV: gains for velocity
    kR: gains for attitude
    kOmega: gains for angular velocity
    """

    num_actions = 4
    max_inclination_angle_rad = np.pi / 3.0
    max_yaw_rate = np.pi / 3.0

    #K_pos_tensor_max = [3.0, 3.0, 2.0]  # used for lee_position_control only
    #K_pos_tensor_min = [2.0, 2.0, 1.0]  # used for lee_position_control only
    # Updated for 1.5kg mass (original 0.25kg, ~6x increase, ×2 for conservative scaling)
    # Optimized PID Gains for 1.5kg Quadrotor (Tuned via Parallel CMA-ES)
    # Cost: 0.7084 | K_pos ~ 3.3, K_vel ~ 2.8, K_rot ~ 1.5, K_angvel ~ 0.3
    K_pos_tensor_min = [3.3, 3.3, 6.6] # Z axis usually higher
    K_pos_tensor_max = [3.3, 3.3, 6.6]
    
    K_vel_tensor_min = [2.8, 2.8, 2.8]
    K_vel_tensor_max = [2.8, 2.8, 2.8]
    
    K_rot_tensor_min = [1.5, 1.5, 1.5]
    K_rot_tensor_max = [1.5, 1.5, 1.5]
    
    K_angvel_tensor_min = [0.3, 0.3, 0.3]
    K_angvel_tensor_max = [0.3, 0.3, 0.3]

    # Optimized PID Gains for 1.6kg Offset Payload (Tuned via Parallel CMA-ES)
    # Cost: 0.4064 | K_pos ~ 22.7, K_vel ~ 1.9, K_rot ~ 12.5, K_angvel ~ 9.8
    # K_pos_tensor_min = [22.7, 22.7, 45.4] 

    # --- Gain Schedule for Omniscient Teacher (0 to 4 Payloads) ---
    # tuned via auto_tune_pid.py (Isotropic Gains)
    # Format: Index = Num Payloads. Value = {K_pos, K_vel, K_rot, K_angvel}
    K_gain_schedule = [
        # 0 Payloads: Kp=24.19, Kv=3.15, Kr=6.83, Kw=8.57
        {
            "K_pos": [24.19, 24.19, 24.19], 
            "K_vel": [3.15, 3.15, 3.15],
            "K_rot": [6.83, 6.83, 6.83],
            "K_angvel": [8.57, 8.57, 8.57]
        },
        # 1 Payload: Kp=26.33, Kv=13.23, Kr=1.30, Kw=2.91
        {
            "K_pos": [26.33, 26.33, 26.33], 
            "K_vel": [13.23, 13.23, 13.23],
            "K_rot": [1.30, 1.30, 1.30],
            "K_angvel": [2.91, 2.91, 2.91]
        },
        # 2 Payloads: Kp=24.19, Kv=3.15, Kr=6.83, Kw=8.57
        {
            "K_pos": [24.19, 24.19, 24.19], 
            "K_vel": [3.15, 3.15, 3.15],
            "K_rot": [6.83, 6.83, 6.83],
            "K_angvel": [8.57, 8.57, 8.57]
        },
        # 3 Payloads: Assumed identical to 0/2 due to perfect Phys Comp
        {
            "K_pos": [24.19, 24.19, 24.19], 
            "K_vel": [3.15, 3.15, 3.15],
            "K_rot": [6.83, 6.83, 6.83],
            "K_angvel": [8.57, 8.57, 8.57]
        },
        # 4 Payloads: Kp=32.30, Kv=7.10, Kr=31.10, Kw=48.58
        {
            "K_pos": [32.30, 32.30, 32.30], 
            "K_vel": [7.10, 7.10, 7.10],
            "K_rot": [31.10, 31.10, 31.10],
            "K_angvel": [48.58, 48.58, 48.58]
        },
    ]
    # K_pos_tensor_max = [22.7, 22.7, 45.4]
    # K_vel_tensor_min = [1.9, 1.9, 1.9]
    # K_vel_tensor_max = [1.9, 1.9, 1.9]
    # K_rot_tensor_min = [12.5, 12.5, 12.5]
    # K_rot_tensor_max = [12.5, 12.5, 12.5]
    # K_angvel_tensor_min = [9.8, 9.8, 9.8]
    # K_angvel_tensor_max = [9.8, 9.8, 9.8]

    randomize_params = False
