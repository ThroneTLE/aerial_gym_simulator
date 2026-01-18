"""
Ablation: w/o Physics Randomization

This config disables all physics parameter randomization to demonstrate
the importance of physics-consistent randomization in generalization.

Only payload mass/position randomization remains enabled (core task requirement).
"""


class task_config:
    seed = 1
    sim_name = "base_sim"
    env_name = "empty_env"
    robot_name = "base_quadrotor"
    controller_name = "lee_position_control_with_compensation"
    args = {}
    num_envs = 1024
    use_warp = False
    headless = True
    device = "cuda:0"

    # Same as teacher config
    observation_space_dim = 20
    privileged_observation_space_dim = 18
    action_space_dim = 3
    controller_action_dim = 8

    episode_len_steps = 1500
    return_state_before_reset = False
    teacher_mode = True

    # Same reward parameters as teacher
    reward_parameters = {
        "position_weight": 0.0,
        "survive_bonus": 10.0,
        "crash_penalty": -10.0,
        "attitude_penalty_coef": 1.00,
        "release_attitude_boost": 1.0,
        "velocity_penalty_coef": 0.0,
        "angvel_penalty_coef": 1.00,
        "action_smoothness_coef": 2.0,
        "position_error_penalty_coef": 0.0,
        "z_error_penalty_coef": 0.0,
        "accel_penalty_away_coef": 0.0,
        "accel_penalty_toward_coef": 0.0,
        "imitation_weight": 8.0,
        "imitation_weight_thrust": 8.0,
        "imitation_weight_torque": 8.0,
        "physics_imitation_weight": 0.0,
    }

    crash_distance_threshold = 1.0
    crash_tilt_threshold_deg = 20.0

    compensation_thrust_limit = 1.0
    compensation_torque_limits = [1.0, 1.0, 0.2]

    # Payload randomization (kept for task requirement)
    payload_parameters = {
        "payload_mass": 0.02,
        "payload_mass_range": [0.00, 0.03],
        "randomize_payload_mass": True,
        "randomize_offsets_on_plane": False,
        "offset_plane_radial_jitter": 0.4,
        "offset_plane_z_jitter": 0.8,
        "offset_plane_r_max": 0.4,
        "offset_plane_z_max": 0.4,
        "force_offset_torque_scale": 0.00,
        "offsets": [
            [0.4, 0.0, -0.4],
            [0.0, -0.4, -0.4],
            [-0.4, 0.0, -0.4],
            [0.0, 0.4, -0.4],
        ],
        "release_start": 400,
        "release_interval": 300,
        "release_start_range": [80, 120],
        "release_interval_range": [300, 350],
        "warning_steps": 0,
        "randomize_release": True,
        "log_release_events": False,
    }

    # ============================================================
    # ABLATION: ALL PHYSICS RANDOMIZATION DISABLED
    # ============================================================
    randomization_parameters = {
        "initial_position_noise": [0.0, 0.0, 0.0],
        "initial_orientation_noise_deg": [0.0, 0.0, 0.0],
        
        # DISABLED: Motor params randomization
        "randomize_motor_thrust_constant": False,  # <-- ABLATION
        "motor_thrust_constant_range_scale": [1.0, 1.0],  # Fixed to nominal
        
        "randomize_motor_time_constant": False,  # <-- ABLATION
        "motor_time_constant_range": [0.05, 0.05],  # Fixed to nominal
        
        # DISABLED: Drag coefficients randomization
        "randomize_drag_coefficients": False,  # <-- ABLATION
        "lin_drag_coeff_range": [0.0, 0.0],  # No drag
        "ang_drag_coeff_range": [0.0, 0.0],
        
        # DISABLED: External disturbance
        "randomize_external_disturbance": False,
        "external_force_range": [0.0, 0.0],
        "external_torque_range": [0.0, 0.0],
    }

    observation_parameters = {
        "include_priv_mass": True,
        "include_priv_com": True,
        "include_priv_inertia": True,
        "include_priv_motor_thrust": True,
        "include_priv_motor_tc": True,
        "include_priv_drag_lin": True,
        "include_priv_drag_ang": True,
        "include_priv_disturbance": True,
        "include_base_rot": True,
        "include_base_angvel": True,
        "include_base_attached": True,
        "include_base_warning": True,
        "include_base_prev_action": True,
    }

    curriculum_parameters = None
