import numpy as np
import torch

from aerial_gym.task.position_setpoint_task.position_setpoint_task import PositionSetpointTask
from aerial_gym.task.payload_compensation_task.payload_compensation_task import (
    PayloadManager,
    PayloadConfig,
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
        return super().reset()

    def reset_idx(self, env_ids):
        env_tensor = torch.as_tensor(env_ids, device=self.device)
        self.payload_manager.reset(env_tensor)
        return super().reset_idx(env_ids)

    def step(self, actions):
        self.payload_manager.step()
        task_obs, rewards, terminations, truncations, infos = super().step(actions)

        env_id = 0
        distance = torch.norm(self.obs_dict["robot_position"][env_id] - self.target_position[env_id])
        tilt = torch.acos(
            torch.clamp(self.obs_dict["robot_orientation"][env_id, 2], -1.0, 1.0)
        )
        if distance.item() > self.crash_distance_threshold or tilt.item() > self.crash_tilt_threshold_rad:
            terminations[env_id] = 1

        return task_obs, rewards, terminations, truncations, infos

    def process_obs_for_task(self):
        super().process_obs_for_task()
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
