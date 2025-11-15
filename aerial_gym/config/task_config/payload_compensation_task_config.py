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

    observation_space_dim = 26  # 13 base + roll/pitch error + 11 payload features
    privileged_observation_space_dim = 0
    action_space_dim = 4  # thrust + 3 compensation torques
    controller_action_dim = 8  # 4 Lee inputs + thrust + 3 torque compensation commands

    episode_len_steps = 800
    return_state_before_reset = False

    reward_parameters = {
        "position_weight": 2.5,
        "crash_penalty": -60.0,
        "attitude_penalty_coef": 1.2,
        "release_attitude_boost": 2.0,
        "comp_torque_penalty_coef": 0.08,
        "comp_thrust_penalty_coef": 0.08,
        "velocity_penalty_coef": 0.3,
        "angvel_penalty_coef": 0.15,
        "action_smoothness_coef": 0.08,
        "tilt_warning_deg": 0.0,
        "tilt_warning_penalty": 0.0,
        "height_warning": 0.0,
        "height_warning_penalty": 0.0,
        "release_tilt_limit_deg": 0.0,
        "release_tilt_penalty": 0.0,
        "stability_radius": 0.0,
        "stability_tilt_deg": 0.0,
        "stability_penalty": 0.0,
        "stability_velocity_penalty": 0.0,
    }

    crash_distance_threshold = 3.0  # meters
    crash_tilt_threshold_deg = 55.0

    payload_parameters = {
        "payload_mass": 0.025,
        "offsets": [
            [0.4, 0.4, -0.4],
            [0.4, -0.4, -0.4],
            [-0.4, 0.4, -0.4],
            [-0.4, -0.4, -0.4],
        ],
        "release_start": 300,
        "release_interval": 150,
        "release_start_range": None,
        "release_interval_range": None,
        "warning_steps": 20,
        "randomize_release": False,
        "log_release_events": False,
    }

    randomization_parameters = None

    curriculum_parameters = None
