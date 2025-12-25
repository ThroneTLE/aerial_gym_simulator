class task_config:
    seed = 1
    sim_name = "base_sim"
    env_name = "empty_env"
    robot_name = "base_quadrotor_with_imu"
    controller_name = "lee_position_control"
    args = {}
    num_envs = 1024
    use_warp = False
    headless = True
    device = "cuda:0"

    observation_space_dim = 15  # pos err (3) + imu (6) + body lin/ang vel (6)
    privileged_observation_space_dim = 0
    action_space_dim = 4  # [x, y, z, yaw]
    episode_len_steps = 500
    return_state_before_reset = False

    reward_parameters = {
        "pos_error_gain1": [2.0, 2.0, 2.0],
        "pos_error_exp1": [1 / 3.5, 1 / 3.5, 1 / 3.5],
        "pos_error_gain2": [2.0, 2.0, 2.0],
        "pos_error_exp2": [2.0, 2.0, 2.0],
        "dist_reward_coefficient": 7.5,
        "max_dist": 15.0,
        "action_diff_penalty_gain": [1.0, 1.0, 1.0],
        "absolute_action_reward_gain": [2.0, 2.0, 2.0],
        "crash_penalty": -100,
    }

    payload_parameters = {
        "payload_mass": 0.04,
        "payload_mass_range": [0.05, 0.001],
        "randomize_payload_mass": True,
        "randomize_payload_count": False,
        "randomize_offsets_on_plane": True,
        "offset_plane_radial_jitter": 0.05,
        "offset_plane_z_jitter": 0.0,
        "offsets": [
            [0.4, 0.4, -0.4],
            [0.4, -0.4, -0.4],
            [-0.4, 0.4, -0.4],
            [-0.4, -0.4, -0.4],
        ],
    }

    randomization_parameters = {
        "initial_position_noise": [0.0, 0.0, 0.0],
        "initial_orientation_noise_deg": [0.0, 0.0, 0.0],
        "target_position_range": [
            [-0.5, 0.5],
            [-0.5, 0.5],
            [0.5, 1.0],
        ],
    }

    trajectory_parameters = {
        "enable": True,
        "mode": "sine",
        "position_range": None,  # None -> use target_position_range
        "amplitude_scale": [0.6, 0.6, 0.4],
        "cycles_range": [1.0, 2.5],
        "yaw_center_deg_range": [0.0, 0.0],
        "yaw_amplitude_deg": 0.0,
        "yaw_cycles_range": [0.5, 1.5],
    }

    data_collection_parameters = {
        "enable": True,
        "log_path": "logs/payload_estimation_data.csv",
        "log_env_id": 0,
        "log_all_envs": True,
        "log_interval": 1,
        "max_steps": 0,
        "flush_every": 200,
    }
