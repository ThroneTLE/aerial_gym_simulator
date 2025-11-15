from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import torch
from isaacgym import gymapi, gymtorch

from aerial_gym.task.base_task import BaseTask
from aerial_gym.sim.sim_builder import SimBuilder
from aerial_gym.utils.logging import CustomLogger
from aerial_gym.utils.math import (
    quat_apply_inverse,
    quat_axis,
    quat_rotate_inverse,
    quat_from_euler_xyz_tensor,
    quat_mul,
)

logger = CustomLogger("payload_compensation_task")


def _mat33_to_np(mat: gymapi.Mat33) -> np.ndarray:
    return np.array(
        [
            [mat.x.x, mat.y.x, mat.z.x],
            [mat.x.y, mat.y.y, mat.z.y],
            [mat.x.z, mat.y.z, mat.z.z],
        ],
        dtype=np.float32,
    )


def _np_to_mat33(arr: np.ndarray) -> gymapi.Mat33:
    mat = gymapi.Mat33()
    mat.x = gymapi.Vec3(arr[0, 0], arr[1, 0], arr[2, 0])
    mat.y = gymapi.Vec3(arr[0, 1], arr[1, 1], arr[2, 1])
    mat.z = gymapi.Vec3(arr[0, 2], arr[1, 2], arr[2, 2])
    return mat


def point_mass_inertia(mass: float, offset: np.ndarray) -> np.ndarray:
    r_sq = float(np.dot(offset, offset))
    return mass * (r_sq * np.eye(3) - np.outer(offset, offset))


@dataclass
class PayloadConfig:
    payload_mass: float
    offsets: Sequence[Sequence[float]]
    release_start: int
    release_interval: int
    warning_steps: int
    release_start_range: Optional[Sequence[int]] = None
    release_interval_range: Optional[Sequence[int]] = None
    randomize_release: bool = False
    log_release_events: bool = False


class PayloadManager:
    """Handles payload release schedule, mass/inertia updates, and equivalent torque injection."""

    def __init__(self, env_manager, payload_cfg: PayloadConfig):
        self.env_manager = env_manager
        self.device = env_manager.device
        self.num_envs = env_manager.num_envs
        self.gym = env_manager.IGE_env.gym
        self.sim = env_manager.IGE_env.sim
        self.env_handles = env_manager.IGE_env.env_handles
        self.robot_handles = env_manager.robot_manager.robot_handles
        self.robot = env_manager.robot_manager.robot
        self.controller = env_manager.robot_manager.robot.controller

        self.cfg = payload_cfg

        self.payload_mass = payload_cfg.payload_mass
        self.offsets = torch.as_tensor(
            np.array(payload_cfg.offsets, dtype=np.float32), device=self.device
        )
        self.num_payloads = self.offsets.shape[0]

        self.attached_mask = torch.ones(
            (self.num_envs, self.num_payloads), dtype=torch.bool, device=self.device
        )
        self.step_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.next_release_step = torch.full(
            (self.num_envs,), payload_cfg.release_start, dtype=torch.long, device=self.device
        )
        self.release_start = payload_cfg.release_start
        self.release_interval = payload_cfg.release_interval
        self.warning_steps = payload_cfg.warning_steps
        self.release_start_range = payload_cfg.release_start_range
        self.release_interval_range = payload_cfg.release_interval_range
        self.randomize_release = payload_cfg.randomize_release
        self.log_release_events = payload_cfg.log_release_events

        self.last_release_index = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self.last_release_mass = torch.zeros(self.num_envs, device=self.device)
        self.just_released_flag = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

        self.current_payload_mass = torch.full(
            (self.num_envs,), self.num_payloads * self.payload_mass, device=self.device
        )
        self.com_offset_body = torch.zeros((self.num_envs, 3), device=self.device)
        self.release_warning_flag = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

        self.gravity = env_manager.IGE_env.global_tensor_dict["gravity"][0].to(self.device)

        self.actor_props: List[List[gymapi.RigidBodyProperties]] = []
        self.base_mass = torch.zeros(self.num_envs, device=self.device)
        self.base_inertia = torch.zeros((self.num_envs, 3, 3), device=self.device)
        self._cache_rigid_body_props()

        self.controller_mass_tensor = self.controller.mass
        self.robot_masses = env_manager.robot_manager.robot_masses
        self.release_orders = torch.zeros(
            (self.num_envs, self.num_payloads), dtype=torch.long, device=self.device
        )
        self.release_cursor = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def _cache_rigid_body_props(self):
        for env_id in range(self.num_envs):
            env_handle = self.env_handles[env_id]
            robot_handle = self.robot_handles[env_id]
            props = self.gym.get_actor_rigid_body_properties(env_handle, robot_handle)
            self.actor_props.append(props)
            base_prop = props[0]
            self.base_mass[env_id] = base_prop.mass
            self.base_inertia[env_id] = torch.from_numpy(_mat33_to_np(base_prop.inertia))

    def reset(self, env_ids: Optional[torch.Tensor] = None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self.attached_mask[env_ids] = True
        self.step_counter[env_ids] = 0
        self._assign_random_release_start(env_ids)
        self._assign_random_release_order(env_ids)
        self.release_cursor[env_ids] = 0
        self.last_release_index[env_ids] = -1
        self.last_release_mass[env_ids] = 0.0
        self.just_released_flag[env_ids] = False
        self.current_payload_mass[env_ids] = self.num_payloads * self.payload_mass
        self.com_offset_body[env_ids] = 0.0
        self.release_warning_flag[env_ids] = False
        self._update_mass_properties(env_ids)

    def step(self):
        self.just_released_flag[:] = False
        self.last_release_mass[:] = 0.0
        self.step_counter += 1

        candidates = torch.nonzero(
            (self.step_counter >= self.next_release_step) & self.attached_mask.any(dim=1)
        ).squeeze(-1)

        for env_id in candidates.tolist():
            self._release_payload(env_id)

        remaining_steps = self.next_release_step - self.step_counter
        self.release_warning_flag = torch.where(
            (remaining_steps <= self.warning_steps) & self.attached_mask.any(dim=1),
            torch.ones_like(self.release_warning_flag),
            torch.zeros_like(self.release_warning_flag),
        )

    def _release_payload(self, env_id: int):
        attach_row = self.attached_mask[env_id]
        if not attach_row.any():
            return

        cursor = int(self.release_cursor[env_id].item())
        payload_id = None
        while cursor < self.num_payloads:
            candidate = int(self.release_orders[env_id, cursor].item())
            if attach_row[candidate]:
                payload_id = candidate
                break
            cursor += 1

        if payload_id is None:
            return

        attach_row[payload_id] = False
        self.release_cursor[env_id] = cursor + 1
        self.last_release_index[env_id] = payload_id
        self.last_release_mass[env_id] = self.payload_mass
        self.just_released_flag[env_id] = True
        if self.log_release_events:
            logger.info(
                "[ReleaseEvent] env=%d payload=%d offset=%s step=%d",
                env_id,
                payload_id,
                self.offsets[payload_id].tolist(),
                int(self.step_counter[env_id].item()),
            )
        self.step_counter[env_id] = 0
        if attach_row.any():
            self.next_release_step[env_id] = self._sample_interval()
        else:
            self.next_release_step[env_id] = torch.iinfo(torch.int64).max

        self._update_mass_properties(torch.tensor([env_id], device=self.device, dtype=torch.long))

    def _assign_random_release_order(self, env_ids: torch.Tensor):
        for env_id in env_ids.long().tolist():
            if self.randomize_release:
                self.release_orders[env_id] = torch.randperm(
                    self.num_payloads, device=self.device, dtype=torch.long
                )
            else:
                self.release_orders[env_id] = torch.arange(
                    self.num_payloads, device=self.device, dtype=torch.long
                )

    def _assign_random_release_start(self, env_ids: torch.Tensor):
        start_values = []
        for _ in env_ids.long().tolist():
            start_values.append(self._sample_start())
        self.next_release_step[env_ids] = torch.tensor(
            start_values, device=self.device, dtype=torch.long
        )

    def _sample_between(self, bounds: Optional[Sequence[int]], default: int) -> int:
        if bounds is None or not self.randomize_release:
            return int(default)
        low, high = bounds
        if low > high:
            low, high = high, low
        if low == high:
            return int(low)
        return int(torch.randint(low, high + 1, (1,), device=self.device).item())

    def _sample_start(self) -> int:
        return self._sample_between(self.release_start_range, self.release_start)

    def _sample_interval(self) -> int:
        return self._sample_between(self.release_interval_range, self.release_interval)

    def _update_mass_properties(self, env_ids: torch.Tensor):
        env_id_list = env_ids.long().tolist()
        for env_id in env_id_list:
            attached = self.attached_mask[env_id]
            payload_count = int(attached.sum().item())
            payload_mass_sum = payload_count * self.payload_mass
            self.current_payload_mass[env_id] = payload_mass_sum

            base_mass = float(self.base_mass[env_id].item())
            total_mass = base_mass + payload_mass_sum
            self._sync_controller_mass(env_id, total_mass)

            inertia_np = self.base_inertia[env_id].cpu().numpy().copy()
            weighted_offset = np.zeros(3, dtype=np.float32)
            for idx, attached_flag in enumerate(attached.tolist()):
                if not attached_flag:
                    continue
                offset = self.offsets[idx].cpu().numpy()
                inertia_np += point_mass_inertia(self.payload_mass, offset)
                weighted_offset += self.payload_mass * offset

            if total_mass > 0.0:
                self.com_offset_body[env_id] = torch.as_tensor(
                    weighted_offset / total_mass, device=self.device, dtype=torch.float32
                )
            else:
                self.com_offset_body[env_id] = 0.0

            props = self.actor_props[env_id]
            props[0].mass = total_mass
            props[0].inertia = _np_to_mat33(inertia_np)
            self.gym.set_actor_rigid_body_properties(
                self.env_handles[env_id], self.robot_handles[env_id], props, recomputeInertia=False
            )

    def _sync_controller_mass(self, env_id: int, new_mass: float):
        if isinstance(self.controller_mass_tensor, torch.Tensor):
            self.controller_mass_tensor[env_id, 0] = new_mass
        self.robot_masses[env_id] = new_mass

    def compute_world_torque(self) -> torch.Tensor:
        payload_mass = self.attached_mask.sum(dim=1) * self.payload_mass
        total_mass = self.base_mass + payload_mass

        weighted_offsets = (
            self.attached_mask.float().unsqueeze(-1) * self.offsets.unsqueeze(0)
        ).sum(dim=1)
        weighted_offsets *= self.payload_mass

        torque = torch.zeros((self.num_envs, 3), device=self.device)
        valid = total_mass > 0
        if valid.any():
            com_offset = torch.zeros_like(weighted_offsets)
            com_offset[valid] = weighted_offsets[valid] / total_mass[valid].unsqueeze(-1)
            gravity_vec = self.gravity.unsqueeze(0).expand(self.num_envs, -1)
            torque[valid] = torch.cross(
                com_offset[valid], gravity_vec[valid] * total_mass[valid].unsqueeze(-1)
            )
        return torque

    def compute_body_torque(self, orientations: torch.Tensor) -> torch.Tensor:
        world_torque = self.compute_world_torque()
        if world_torque.shape[0] != orientations.shape[0]:
            world_torque = world_torque[: orientations.shape[0]]
        return quat_rotate_inverse(orientations, world_torque)

    def get_observation_features(self):
        attached_mask = self.attached_mask.float()
        denom = max(self.num_payloads - 1, 1)
        last_release_norm = torch.where(
            self.last_release_index >= 0,
            (self.last_release_index.float() / denom) * 2.0 - 1.0,
            torch.full(
                (self.num_envs,), -1.0, device=self.device, dtype=torch.float32
            ),
        )
        return {
            "payload_mass": self.current_payload_mass,
            "com_offset": self.com_offset_body,
            "attached_mask": attached_mask,
            "last_release_norm": last_release_norm,
            "last_release_mass": self.last_release_mass,
            "warning_flag": self.release_warning_flag.float(),
        }


class PayloadCompensationTask(BaseTask):
    def __init__(
        self,
        task_config,
        seed=None,
        num_envs=None,
        headless=None,
        device=None,
        use_warp=None,
    ):
        if seed is not None:
            task_config.seed = seed
        if num_envs is not None:
            task_config.num_envs = num_envs
        if headless is not None:
            task_config.headless = headless
        if device is not None:
            task_config.device = device
        if use_warp is not None:
            task_config.use_warp = use_warp

        super().__init__(task_config)
        self.device = self.task_config.device

        for key in self.task_config.reward_parameters.keys():
            self.task_config.reward_parameters[key] = torch.tensor(
                self.task_config.reward_parameters[key], device=self.device
            )

        logger.info("Building environment for payload compensation task.")
        self.sim_builder = SimBuilder()
        self.sim_env = self.sim_builder.build_env(
            sim_name=self.task_config.sim_name,
            env_name=self.task_config.env_name,
            robot_name=self.task_config.robot_name,
            controller_name=self.task_config.controller_name,
            args=self.task_config.args,
            device=self.device,
            num_envs=self.task_config.num_envs,
            use_warp=self.task_config.use_warp,
            headless=self.task_config.headless,
        )

        self.actions = torch.zeros(
            (self.sim_env.num_envs, self.task_config.action_space_dim),
            device=self.device,
            requires_grad=False,
        )
        self.prev_actions = torch.zeros_like(self.actions)
        self.controller_actions = torch.zeros(
            (self.sim_env.num_envs, self.task_config.controller_action_dim),
            device=self.device,
            requires_grad=False,
        )

        self.target_position = torch.zeros(
            (self.sim_env.num_envs, 3), device=self.device, requires_grad=False
        )

        self.obs_dict = self.sim_env.get_obs()
        self.terminations = self.obs_dict["crashes"]
        self.truncations = self.obs_dict["truncations"]
        self.rewards = torch.zeros(self.truncations.shape[0], device=self.device)
        self._debug_reset_count = 0

        self.observation_space_dim = self.task_config.observation_space_dim
        self.action_space_dim = self.task_config.action_space_dim

        self.observation_space = None
        self.action_space = None

        self.task_obs = {
            "observations": torch.zeros(
                (self.sim_env.num_envs, self.observation_space_dim),
                device=self.device,
                requires_grad=False,
            ),
            "priviliged_obs": torch.zeros(
                (self.sim_env.num_envs, self.task_config.privileged_observation_space_dim),
                device=self.device,
                requires_grad=False,
            ),
            "collisions": torch.zeros(
                (self.sim_env.num_envs, 1), device=self.device, requires_grad=False
            ),
            "rewards": torch.zeros(
                (self.sim_env.num_envs, 1), device=self.device, requires_grad=False
            ),
        }

        payload_cfg = PayloadConfig(
            payload_mass=self.task_config.payload_parameters["payload_mass"],
            offsets=self.task_config.payload_parameters["offsets"],
            release_start=self.task_config.payload_parameters["release_start"],
            release_interval=self.task_config.payload_parameters["release_interval"],
            warning_steps=self.task_config.payload_parameters["warning_steps"],
            release_start_range=self.task_config.payload_parameters.get("release_start_range"),
            release_interval_range=self.task_config.payload_parameters.get("release_interval_range"),
            randomize_release=self.task_config.payload_parameters.get("randomize_release", True),
            log_release_events=self.task_config.payload_parameters.get("log_release_events", False),
        )
        self.payload_manager = PayloadManager(self.sim_env, payload_cfg)
        self.payload_manager.reset()
        self._patch_pre_physics_step()
        self._initialize_vehicle_state()

        rand_cfg = getattr(self.task_config, "randomization_parameters", None) or {}
        self.initial_position_noise = torch.tensor(
            rand_cfg.get("initial_position_noise", [0.0, 0.0, 0.0]), device=self.device
        )
        self.initial_orientation_noise = torch.tensor(
            rand_cfg.get("initial_orientation_noise_deg", [0.0, 0.0, 0.0]), device=self.device
        )
        self.initial_orientation_noise = self.initial_orientation_noise * np.pi / 180.0
        self.target_position_range = rand_cfg.get("target_position_range")
        self.current_target_range_tensor = (
            torch.tensor(self.target_position_range, device=self.device, dtype=torch.float32)
            if self.target_position_range is not None
            else None
        )
        self.obs_noise_std = rand_cfg.get("obs_noise_std", {})

        curriculum_cfg = getattr(self.task_config, "curriculum_parameters", None)
        self.curriculum_target_ranges = None
        self.curriculum_stage_steps = None
        self.curriculum_stage = 0
        self.curriculum_stage_progress = 0
        if curriculum_cfg:
            ranges = curriculum_cfg.get("target_ranges")
            if ranges:
                self.curriculum_target_ranges = torch.tensor(
                    ranges, device=self.device, dtype=torch.float32
                )
                self.current_target_range_tensor = self.curriculum_target_ranges[0].clone()
            self.curriculum_stage_steps = int(curriculum_cfg.get("steps_per_stage", 0))

        self.counter = 0
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

    def close(self):
        if hasattr(self, "sim_builder") and self.sim_builder is not None:
            self.sim_builder.delete_env()

    def reset(self):
        self.target_position[:, 0:3] = 0.0
        self.infos = {}
        self.payload_manager.reset()
        self.sim_env.reset()
        self._initialize_vehicle_state()
        self._randomize_target_positions()
        self._apply_initial_state_noise()
        self._log_debug_reset(env_tensor=None)
        return self.get_return_tuple()

    def reset_idx(self, env_ids):
        self.target_position[:, 0:3] = 0.0
        env_tensor = torch.as_tensor(env_ids, device=self.device)
        self.payload_manager.reset(env_ids=env_tensor)
        self.sim_env.reset_idx(env_ids)
        self._initialize_vehicle_state(env_ids=env_tensor)
        self._randomize_target_positions(env_tensor)
        self._apply_initial_state_noise(env_tensor)
        self._log_debug_reset(env_tensor)

    def render(self):
        return None

    def step(self, actions):
        self.counter += 1
        self.prev_actions[:] = self.actions
        self.actions = actions

        self.payload_manager.step()
        self._advance_curriculum_if_needed()

        self.controller_actions[:, 0:3] = self.target_position
        self.controller_actions[:, 3] = 0.0
        clamped_actions = torch.clamp(self.actions, -1.0, 1.0)
        self.controller_actions[:, 4] = clamped_actions[:, 0]
        self.controller_actions[:, 5:] = clamped_actions[:, 1:]

        self.sim_env.step(actions=self.controller_actions)

        base_rewards, self.terminations[:] = compute_reward(
            quat_apply_inverse(
                self.obs_dict["robot_vehicle_orientation"],
                (self.target_position - self.obs_dict["robot_position"]),
            ),
            self.obs_dict["robot_linvel"],
            self.obs_dict["robot_orientation"],
            self.obs_dict["robot_body_angvel"],
            self.obs_dict["crashes"],
            1.0,
            self.controller_actions,
            self.controller_actions,
            self.task_config.reward_parameters,
            self.crash_distance_threshold,
            self.crash_tilt_threshold_rad,
        )
        self.rewards[:] = base_rewards
        self.rewards += self._compute_payload_penalties(clamped_actions)

        self.truncations[:] = torch.where(
            self.sim_env.sim_steps > self.task_config.episode_len_steps, 1, 0
        )
        self.sim_env.post_reward_calculation_step()

        self.infos = {
            "last_release_index": self.payload_manager.last_release_index.clone().detach().cpu(),
            "just_released": self.payload_manager.just_released_flag.clone().detach().cpu(),
        }
        return self.get_return_tuple()

    def _compute_payload_penalties(self, clamped_actions):
        reward_cfg = self.task_config.reward_parameters
        euler = self.obs_dict["robot_euler_angles"]
        pos_error = self.target_position - self.obs_dict["robot_position"]
        roll_pitch_error = torch.norm(euler[:, 0:2], dim=1)
        base_penalty = -reward_cfg["attitude_penalty_coef"] * roll_pitch_error

        release_multiplier = torch.ones_like(base_penalty)
        release_multiplier = torch.where(
            self.payload_manager.just_released_flag,
            release_multiplier * reward_cfg["release_attitude_boost"],
            release_multiplier,
        )
        attitude_term = base_penalty * release_multiplier

        torque_actions = clamped_actions[:, 1:]
        comp_penalty = -reward_cfg["comp_torque_penalty_coef"] * torch.sum(
            torque_actions**2, dim=1
        )

        thrust_penalty = -reward_cfg.get("comp_thrust_penalty_coef", 0.0) * torch.abs(
            clamped_actions[:, 0]
        )

        vel_coef = reward_cfg.get("velocity_penalty_coef", 0.0)
        smooth_coef = reward_cfg.get("action_smoothness_coef", 0.0)
        velocity_penalty = -vel_coef * torch.norm(
            self.obs_dict["robot_body_linvel"], dim=1
        )
        action_delta = self.actions - self.prev_actions
        smooth_penalty = -smooth_coef * torch.norm(action_delta, dim=1)

        ang_coef = reward_cfg.get("angvel_penalty_coef", 0.0)
        ang_penalty = -ang_coef * torch.norm(self.obs_dict["robot_body_angvel"], dim=1)

        tilt_warn = reward_cfg.get("tilt_warning_deg", 0.0)
        tilt_penalty = 0.0
        if tilt_warn > 0.0:
            tilt_warn_rad = np.deg2rad(float(tilt_warn))
            tilt_penalty = reward_cfg.get("tilt_warning_penalty", 0.0) * (
                (torch.abs(euler[:, 0]) > tilt_warn_rad)
                | (torch.abs(euler[:, 1]) > tilt_warn_rad)
            ).float()

        height_warn = reward_cfg.get("height_warning", 0.0)
        height_penalty = 0.0
        if height_warn > 0.0:
            height_penalty = reward_cfg.get("height_warning_penalty", 0.0) * (
                self.obs_dict["robot_position"][:, 2] < height_warn
            ).float()

        release_limit = reward_cfg.get("release_tilt_limit_deg", 0.0)
        release_penalty = 0.0
        if release_limit > 0.0:
            limit_rad = np.deg2rad(float(release_limit))
            violation = (torch.abs(euler[:, 0]) > limit_rad) | (
                torch.abs(euler[:, 1]) > limit_rad
            )
            release_penalty = reward_cfg.get("release_tilt_penalty", 0.0) * (
                violation & self.payload_manager.just_released_flag
            ).float()

        stability_radius = float(reward_cfg.get("stability_radius", 0.0))
        stability_penalty_coef = reward_cfg.get("stability_penalty", 0.0)
        stability_vel_coef = reward_cfg.get("stability_velocity_penalty", 0.0)
        stability_tilt_deg = float(reward_cfg.get("stability_tilt_deg", 0.0))
        stability_term = torch.zeros_like(roll_pitch_error)
        if stability_radius > 0.0:
            pos_norm = torch.norm(pos_error, dim=1)
            near_mask = (pos_norm < stability_radius).float()
            stability_term = stability_penalty_coef * near_mask

            if stability_tilt_deg > 0.0:
                limit_rad = np.deg2rad(stability_tilt_deg)
                tilt_violation = (
                    (torch.abs(euler[:, 0]) > limit_rad)
                    | (torch.abs(euler[:, 1]) > limit_rad)
                ).float()
                stability_term = stability_term * tilt_violation

            if stability_vel_coef != 0.0:
                vel_norm = torch.norm(self.obs_dict["robot_body_linvel"], dim=1)
                stability_term += -stability_vel_coef * vel_norm * near_mask

        return (
            attitude_term
            + comp_penalty
            + thrust_penalty
            + velocity_penalty
            + ang_penalty
            + smooth_penalty
            + tilt_penalty
            + height_penalty
            + release_penalty
            + stability_term
        )

    def get_return_tuple(self):
        self.process_obs_for_task()
        return (
            self.task_obs,
            self.rewards,
            self.terminations,
            self.truncations,
            self.infos,
        )

    def process_obs_for_task(self):
        pos_error = self.target_position - self.obs_dict["robot_position"]
        self.task_obs["observations"][:, 0:3] = pos_error
        self.task_obs["observations"][:, 3:7] = self.obs_dict["robot_orientation"]
        self.task_obs["observations"][:, 7:10] = self.obs_dict["robot_body_linvel"]
        self.task_obs["observations"][:, 10:13] = self.obs_dict["robot_body_angvel"]

        euler = self.obs_dict["robot_euler_angles"]
        self.task_obs["observations"][:, 13] = euler[:, 0]
        self.task_obs["observations"][:, 14] = euler[:, 1]

        payload_obs = self.payload_manager.get_observation_features()
        idx = 15
        self.task_obs["observations"][:, idx] = payload_obs["payload_mass"]
        idx += 1
        self.task_obs["observations"][:, idx : idx + 3] = payload_obs["com_offset"]
        idx += 3
        num_payloads = payload_obs["attached_mask"].shape[1]
        self.task_obs["observations"][:, idx : idx + num_payloads] = payload_obs[
            "attached_mask"
        ]
        idx += num_payloads
        self.task_obs["observations"][:, idx] = payload_obs["last_release_norm"]
        idx += 1
        self.task_obs["observations"][:, idx] = payload_obs["last_release_mass"]
        idx += 1
        self.task_obs["observations"][:, idx] = payload_obs["warning_flag"]

        self._apply_observation_noise()

        self.task_obs["rewards"] = self.rewards
        self.task_obs["terminations"] = self.terminations
        self.task_obs["truncations"] = self.truncations

    def _get_env_tensor(self, env_ids=None):
        if env_ids is None:
            return torch.arange(self.sim_env.num_envs, device=self.device, dtype=torch.long)
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(self.device).long()
        return torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

    def _randomize_target_positions(self, env_ids=None):
        range_tensor = self.current_target_range_tensor
        if range_tensor is None:
            return
        env_tensor = self._get_env_tensor(env_ids)
        if env_tensor.numel() == 0:
            return
        ranges = range_tensor.to(self.device)
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
        obs = self.task_obs["observations"]
        pos_std = float(self.obs_noise_std.get("position_error", 0.0))
        if pos_std > 0:
            obs[:, 0:3] += torch.randn_like(obs[:, 0:3]) * pos_std
        lin_std = float(self.obs_noise_std.get("linear_velocity", 0.0))
        if lin_std > 0:
            obs[:, 7:10] += torch.randn_like(obs[:, 7:10]) * lin_std
        ang_std = float(self.obs_noise_std.get("angular_velocity", 0.0))
        if ang_std > 0:
            obs[:, 10:13] += torch.randn_like(obs[:, 10:13]) * ang_std

    def _advance_curriculum_if_needed(self):
        if (
            self.curriculum_target_ranges is None
            or self.curriculum_stage_steps is None
            or self.curriculum_stage_steps <= 0
        ):
            return
        self.curriculum_stage_progress += self.sim_env.num_envs
        if (
            self.curriculum_stage < self.curriculum_target_ranges.shape[0] - 1
            and self.curriculum_stage_progress >= self.curriculum_stage_steps
        ):
            self.curriculum_stage_progress = 0
            self.curriculum_stage += 1
            self.current_target_range_tensor = self.curriculum_target_ranges[
                self.curriculum_stage
            ].clone()
            self._randomize_target_positions()

    def _log_debug_reset(self, env_tensor):
        # 仅记录前几次 reset，避免刷屏
        if self._debug_reset_count >= 5:
            return
        env_ids = (
            env_tensor.tolist()
            if env_tensor is not None
            else list(range(min(3, self.sim_env.num_envs)))
        )
        release_info = {
            int(env_id): {
                "order": self.payload_manager.release_orders[int(env_id)].tolist(),
                "next_release": int(self.payload_manager.next_release_step[int(env_id)].item()),
            }
            for env_id in env_ids
        }
        target_samples = self.target_position[env_ids].detach().cpu().numpy().tolist()
        logger.info(
            "[DebugReset %d] env_ids=%s targets=%s release_info=%s",
            self._debug_reset_count,
            env_ids,
            target_samples,
            release_info,
        )
        self._debug_reset_count += 1

    def _initialize_vehicle_state(self, env_ids=None):
        if not hasattr(self.sim_env, "IGE_env"):
            return
        gym = self.sim_env.IGE_env.gym
        sim = self.sim_env.IGE_env.sim
        vec_root = self.sim_env.IGE_env.vec_root_tensor
        if vec_root is None:
            return
        single_state = torch.zeros(13, device=self.device)
        single_state[6] = 1.0
        if env_ids is None:
            vec_root[:, 0, :] = single_state
        else:
            vec_root[env_ids.long(), 0, :] = single_state
        gym.set_actor_root_state_tensor(
            sim, gymtorch.unwrap_tensor(self.sim_env.IGE_env.unfolded_vec_root_tensor)
        )


def exp_func(x: torch.Tensor, gain: float, exp_coeff: float) -> torch.Tensor:
    return gain * torch.exp(-exp_coeff * x * x)


def compute_reward(
    pos_error,
    lin_vels,
    robot_quats,
    robot_angvels,
    crashes,
    curriculum_level_multiplier,
    current_action,
    prev_actions,
    parameter_dict,
    crash_distance_threshold,
    crash_tilt_threshold_rad,
):
    dist = torch.norm(pos_error, dim=1)
    pos_reward = exp_func(dist, 3.0, 8.0) + exp_func(dist, 2.0, 4.0)
    dist_reward = (20 - dist) / 40.0

    ups = quat_axis(robot_quats, 2)
    tiltage = torch.abs(1 - ups[..., 2])
    up_reward = 0.2 / (0.1 + tiltage * tiltage)

    spinnage = torch.norm(robot_angvels, dim=1)
    ang_vel_reward = (1.0 / (1.0 + spinnage * spinnage)) * 3

    total_reward = (
        parameter_dict["position_weight"] * (pos_reward + dist_reward)
        + pos_reward * (up_reward + ang_vel_reward)
    )
    total_reward[:] = curriculum_level_multiplier * total_reward

    crashes[:] = torch.where(dist > crash_distance_threshold, torch.ones_like(crashes), crashes)
    tilt_angle = torch.acos(torch.clamp(ups[..., 2], -1.0, 1.0))
    crashes[:] = torch.where(tilt_angle > crash_tilt_threshold_rad, torch.ones_like(crashes), crashes)
    total_reward[:] = torch.where(
        crashes > 0.0, parameter_dict["crash_penalty"] * torch.ones_like(total_reward), total_reward
    )
    return total_reward, crashes
