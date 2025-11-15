import numpy as np
import torch
from isaacgym import gymtorch

from aerial_gym.task.position_setpoint_task.position_setpoint_task import PositionSetpointTask
from aerial_gym.task.payload_compensation_task.payload_compensation_task import (
    PayloadManager,
    PayloadConfig,
)
from aerial_gym.utils.math import (
    quat_axis,
    quat_from_euler_xyz_tensor,
    quat_mul,
    get_euler_xyz_tensor,
)


class PayloadCompensationTaskFullRL(PositionSetpointTask):
    """
    纯 RL 控制的 payload 释放任务：基于 PositionSetpointTask，
    但额外叠加 payload 质量/质心变化与释放逻辑。
    """

    def __init__(self, task_config, seed=None, num_envs=None, headless=None, device=None, use_warp=None):
        super().__init__(
            task_config=task_config,
            seed=seed,
            num_envs=num_envs,
            headless=headless,
            device=device,
            use_warp=use_warp,
        )

        payload_cfg = PayloadConfig(
            payload_mass=self.task_config.payload_parameters["payload_mass"],
            offsets=self.task_config.payload_parameters["offsets"],
            release_start=self.task_config.payload_parameters["release_start"],
            release_interval=self.task_config.payload_parameters["release_interval"],
            warning_steps=self.task_config.payload_parameters["warning_steps"],
            release_start_range=self.task_config.payload_parameters.get("release_start_range"),
            release_interval_range=self.task_config.payload_parameters.get("release_interval_range"),
            randomize_release=self.task_config.payload_parameters.get("randomize_release", True),
        )
        self.payload_manager = PayloadManager(self.sim_env, payload_cfg)
        self.payload_manager.reset()
        self._patch_pre_physics_step()

        self.crash_distance_threshold = getattr(self.task_config, "crash_distance_threshold", 8.0)
        tilt_deg = getattr(self.task_config, "crash_tilt_threshold_deg", 90.0)
        self.crash_tilt_threshold_rad = np.deg2rad(tilt_deg)

        rand_cfg = getattr(self.task_config, "randomization_parameters", {})
        self.initial_position_noise = torch.tensor(
            rand_cfg.get("initial_position_noise", [0.0, 0.0, 0.0]), device=self.device
        )
        self.initial_orientation_noise = torch.tensor(
            rand_cfg.get("initial_orientation_noise_deg", [0.0, 0.0, 0.0]), device=self.device
        )
        self.initial_orientation_noise = self.initial_orientation_noise * np.pi / 180.0
        self.target_position_range = rand_cfg.get("target_position_range")
        self.obs_noise_std = rand_cfg.get("obs_noise_std", {})
        # 预留的质量/惯量/推力随机化配置（默认关闭，未来可扩展）
        self.mass_jitter_cfg = rand_cfg.get("mass_jitter", {"enabled": False})
        self.inertia_jitter_cfg = rand_cfg.get("inertia_jitter", {"enabled": False})
        self.thrust_scale_jitter_cfg = rand_cfg.get("thrust_scale_jitter", {"enabled": False})

    def _patch_pre_physics_step(self):
        robot_manager = self.sim_env.robot_manager
        orig_pre_physics_step = robot_manager.pre_physics_step
        payload_manager = self.payload_manager
        sim_env = self.sim_env

        def patched_pre_physics_step(actions, _orig=orig_pre_physics_step):
            _orig(actions)
            orientations = sim_env.IGE_env.global_tensor_dict["robot_orientation"]
            body_torque = payload_manager.compute_body_torque(orientations)
            sim_env.robot_manager.robot.robot_torque_tensors[:, 0, :] += body_torque

        robot_manager.pre_physics_step = patched_pre_physics_step

    def reset(self):
        self.payload_manager.reset()
        super().reset()
        self._randomize_target_positions()
        self._apply_initial_state_noise()
        return self.get_return_tuple()

    def reset_idx(self, env_ids):
        env_tensor = torch.as_tensor(env_ids, device=self.device)
        self.payload_manager.reset(env_tensor)
        super().reset_idx(env_ids)
        self._randomize_target_positions(env_tensor)
        self._apply_initial_state_noise(env_tensor)

    def step(self, actions):
        self.payload_manager.step()
        task_obs, rewards, terminations, truncations, infos = super().step(actions)
        self._apply_safety_penalties(rewards)

        env_id = 0
        distance = torch.norm(self.obs_dict["robot_position"][env_id] - self.target_position[env_id])
        quat = self.obs_dict["robot_orientation"][env_id : env_id + 1]
        body_z = quat_axis(quat, 2)
        dot = torch.clamp(body_z[:, 2], -1.0, 1.0)
        tilt = torch.acos(dot)
        if distance.item() > self.crash_distance_threshold or tilt.item() > self.crash_tilt_threshold_rad:
            terminations[env_id] = 1

        return task_obs, rewards, terminations, truncations, infos

    def process_obs_for_task(self):
        super().process_obs_for_task()
        self._apply_observation_noise()
        payload_obs = self.payload_manager.get_observation_features()
        observations = self.task_obs["observations"]
        idx = 13
        observations[:, idx] = payload_obs["payload_mass"]
        idx += 1
        observations[:, idx : idx + 3] = payload_obs["com_offset"]
        idx += 3
        num_payloads = payload_obs["attached_mask"].shape[1]
        observations[:, idx : idx + num_payloads] = payload_obs["attached_mask"]
        idx += num_payloads
        observations[:, idx] = payload_obs["last_release_norm"]
        idx += 1
        observations[:, idx] = payload_obs["last_release_mass"]
        idx += 1
        observations[:, idx] = payload_obs["warning_flag"]

    def _apply_safety_penalties(self, rewards: torch.Tensor) -> None:
        cfg = self.task_config.reward_parameters
        orientations = self.obs_dict["robot_orientation"]
        body_z = quat_axis(orientations, 2)

        tilt_warn_deg = float(cfg.get("tilt_warning_deg", 0.0))
        if tilt_warn_deg > 0.0:
            tilt_warn_rad = np.deg2rad(tilt_warn_deg)
            tilt_penalty = float(cfg.get("tilt_warning_penalty", 0.0))
            if tilt_penalty != 0.0:
                tilt = torch.acos(torch.clamp(body_z[:, 2], -1.0, 1.0))
                rewards[tilt > tilt_warn_rad] += tilt_penalty

        height_warn = float(cfg.get("height_warning", 0.0))
        if height_warn > 0.0:
            height_penalty = float(cfg.get("height_warning_penalty", 0.0))
            if height_penalty != 0.0:
                rewards[self.obs_dict["robot_position"][:, 2] < height_warn] += height_penalty

        vel_coef = float(cfg.get("velocity_penalty_coef", 0.0))
        if vel_coef > 0.0:
            lin_vel = torch.norm(self.obs_dict["robot_body_linvel"], dim=1)
            rewards -= vel_coef * lin_vel

        ang_coef = float(cfg.get("angvel_penalty_coef", 0.0))
        if ang_coef > 0.0:
            ang_vel = torch.norm(self.obs_dict["robot_body_angvel"], dim=1)
            rewards -= ang_coef * ang_vel

        release_limit_deg = float(cfg.get("release_tilt_limit_deg", 0.0))
        release_penalty = float(cfg.get("release_tilt_penalty", 0.0))
        if release_limit_deg > 0.0 and release_penalty != 0.0:
            mask = self.payload_manager.just_released_flag
            if mask.any():
                eulers = get_euler_xyz_tensor(orientations)
                roll = torch.abs(eulers[:, 0])
                pitch = torch.abs(eulers[:, 1])
                limit_rad = np.deg2rad(release_limit_deg)
                violation = (roll > limit_rad) | (pitch > limit_rad)
                rewards[mask & violation] += release_penalty

    def _get_env_tensor(self, env_ids=None):
        if env_ids is None:
            return torch.arange(self.sim_env.num_envs, device=self.device, dtype=torch.long)
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(self.device).long()
        return torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

    def _randomize_target_positions(self, env_ids=None):
        if self.target_position_range is None:
            return
        env_tensor = self._get_env_tensor(env_ids)
        if env_tensor.numel() == 0:
            return
        ranges = torch.tensor(self.target_position_range, device=self.device, dtype=torch.float32)
        lows = ranges[:, 0]
        highs = ranges[:, 1]
        span = highs - lows
        samples = torch.rand((env_tensor.shape[0], 3), device=self.device) * span + lows
        self.target_position[env_tensor] = samples

    def _apply_initial_state_noise(self, env_ids=None):
        has_pos_noise = torch.any(self.initial_position_noise > 0)
        has_rot_noise = torch.any(self.initial_orientation_noise > 0)
        if (not has_pos_noise and not has_rot_noise) or not hasattr(self.sim_env, "IGE_env"):
            return
        vec_root = getattr(self.sim_env.IGE_env, "vec_root_tensor", None)
        if vec_root is None:
            return
        env_tensor = self._get_env_tensor(env_ids)
        if env_tensor.numel() == 0:
            return

        if has_pos_noise:
            noise = (torch.rand((env_tensor.shape[0], 3), device=self.device) * 2 - 1.0)
            noise = noise * self.initial_position_noise
            vec_root[env_tensor, 0, 0:3] += noise

        if has_rot_noise:
            angles = (torch.rand((env_tensor.shape[0], 3), device=self.device) * 2 - 1.0)
            angles = angles * self.initial_orientation_noise
            delta_quat = quat_from_euler_xyz_tensor(angles)
            root_quat = vec_root[env_tensor, 0, 3:7]
            vec_root[env_tensor, 0, 3:7] = quat_mul(delta_quat, root_quat)

        gym = self.sim_env.IGE_env.gym
        sim = self.sim_env.IGE_env.sim
        gym.set_actor_root_state_tensor(
            sim, gymtorch.unwrap_tensor(self.sim_env.IGE_env.unfolded_vec_root_tensor)
        )

    def _apply_observation_noise(self):
        if not self.obs_noise_std:
            return
        observations = self.task_obs["observations"]
        pos_std = float(self.obs_noise_std.get("position_error", 0.0))
        if pos_std > 0:
            observations[:, 0:3] += torch.randn_like(observations[:, 0:3]) * pos_std
        lin_std = float(self.obs_noise_std.get("linear_velocity", 0.0))
        if lin_std > 0:
            observations[:, 7:10] += torch.randn_like(observations[:, 7:10]) * lin_std
        ang_std = float(self.obs_noise_std.get("angular_velocity", 0.0))
        if ang_std > 0:
            observations[:, 10:13] += torch.randn_like(observations[:, 10:13]) * ang_std
