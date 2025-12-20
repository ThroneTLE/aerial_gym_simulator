from dataclasses import dataclass
from typing import List, Optional, Sequence

import atexit
import numpy as np
import torch
import os
from datetime import datetime
from isaacgym import gymapi, gymtorch
from torch.utils.tensorboard import SummaryWriter

from aerial_gym.task.base_task import BaseTask
from aerial_gym.sim.sim_builder import SimBuilder
from aerial_gym.config.controller_config import lee_controller_with_comp_config
from aerial_gym.config.robot_config.base_quad_config import BaseQuadCfg
from aerial_gym.utils.logging import CustomLogger
from aerial_gym.utils.math import (
    quat_apply_inverse,
    quat_axis,
    quat_rotate_inverse,
    quat_from_euler_xyz_tensor,
    quat_mul,
    quat_to_rotation_matrix,
)

logger = CustomLogger("payload_compensation_task")


def _mean_detached(t, device=None):
    if not torch.is_tensor(t):
        t = torch.as_tensor(t, device=device)
    if t.numel() == 0:
        return 0.0
    return float(t.mean().item())


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
        # 保留名义惯量副本（用于 tau_inertia 计算）
        self.base_inertia_nominal = self.base_inertia.clone()

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
        self.teacher_mode = getattr(self.task_config, "teacher_mode", False)
        self.dagger_frac = float(getattr(self.task_config, "dagger_frac", 0.0))
        self._dagger_init_frac = self.dagger_frac
        self._dagger_updates = 0
        self.imitation_err_threshold = float(getattr(self.task_config, "imitation_err_threshold", 1e9))
        self.fix_yaw_residual_zero = bool(getattr(self.task_config, "fix_yaw_residual_zero", False))
        self.dagger_use_postmix_err = bool(getattr(self.task_config, "dagger_use_postmix_err", False))
        self._last_policy_imitation_err = 0.0
        self._last_decay_imitation_err = 0.0
        self.dagger_decay_reward = float(getattr(self.task_config, "dagger_decay_reward", 0.0))
        self.dagger_decay_rate = float(getattr(self.task_config, "dagger_decay_rate", 1.0))
        self.dagger_min_frac = float(getattr(self.task_config, "dagger_min_frac", 0.0))
        log_dir = os.environ.get("AERIAL_TB_LOGDIR")
        if log_dir:
            log_dir = self._make_timestamped_log_dir(log_dir)
        self.tb_writer = SummaryWriter(log_dir=log_dir) if log_dir else None
        self.tb_log_interval = int(os.environ.get("AERIAL_TB_INTERVAL", "200"))

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
        # Teacher 模式：残差目标与补偿限幅
        self.teacher_residual = torch.zeros(
            (self.sim_env.num_envs, self.task_config.action_space_dim), device=self.device
        )
        self.comp_thrust_limit = getattr(
            lee_controller_with_comp_config.control, "compensation_thrust_limit", 0.3
        )
        torque_limits = getattr(
            lee_controller_with_comp_config.control, "compensation_torque_limits", [0.5, 0.5, 0.1]
        )
        self.comp_torque_limits = torch.as_tensor(torque_limits, device=self.device)
        # 教师模式：特权向量（载荷/惯量/扰动/分配矩阵等），raw 形式输出，由策略侧可训练编码器处理
        self.priv_vec_dim = 41
        self.priv_embed_dim = self.priv_vec_dim

        self.target_position = torch.zeros(
            (self.sim_env.num_envs, 3), device=self.device, requires_grad=False
        )
        # 记录上一帧的位置误差范数，用于计算误差缩小奖励
        self.prev_pos_dist = torch.zeros(self.sim_env.num_envs, device=self.device)
        # 记录上一帧线速度，用于近似计算加速度惩罚
        self.prev_linvel = torch.zeros_like(self.target_position)
        # Teacher 模式下缓存最后一次 payload torque
        self._last_tau_payload = torch.zeros((self.sim_env.num_envs, 3), device=self.device)

        self.obs_dict = self.sim_env.get_obs()
        self.terminations = self.obs_dict["crashes"]
        self.truncations = self.obs_dict["truncations"]
        self.rewards = torch.zeros(self.truncations.shape[0], device=self.device)
        self._debug_reset_count = 0
        self._last_reward_components = {}

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
        if not isinstance(rand_cfg, dict):
            rand_cfg = {}
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
        self._init_rollout_logger()

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
        self._flush_rollout_logger()
        if hasattr(self, "sim_builder") and self.sim_builder is not None:
            self.sim_builder.delete_env()

    def reset(self):
        self.infos = {}
        self.payload_manager.reset()
        self.sim_env.reset()
        self.prev_pos_dist[:] = 0.0
        self._refresh_env_state(env_ids=None, reset_payload_manager=False)
        return self.get_return_tuple()

    def reset_idx(self, env_ids):
        env_tensor = self._get_env_tensor(env_ids)
        self.payload_manager.reset(env_ids=env_tensor)
        self.sim_env.reset_idx(env_ids)
        self._refresh_env_state(env_ids=env_tensor, reset_payload_manager=False)

    def _refresh_env_state(self, env_ids=None, reset_payload_manager=False):
        env_tensor = None if env_ids is None else self._get_env_tensor(env_ids)
        if env_tensor is not None and env_tensor.numel() == 0:
            return

        if env_tensor is None:
            self.target_position[:, 0:3] = 0.0
            self.prev_pos_dist[:] = 0.0
        else:
            self.target_position[env_tensor.long(), 0:3] = 0.0
            self.prev_pos_dist[env_tensor.long()] = 0.0

        if reset_payload_manager:
            self.payload_manager.reset(env_ids=env_tensor)

        self._initialize_vehicle_state(env_ids=env_tensor)
        self._randomize_target_positions(env_tensor)
        self._apply_initial_state_noise(env_tensor)
        self._log_debug_reset(env_tensor if env_tensor is not None else None)

    def render(self):
        return None

    def step(self, actions):
        # 保证外部传入的动作在正确设备/类型上（避免 CPU→GPU 混合引发 device mismatch）
        if not torch.is_tensor(actions):
            actions = torch.as_tensor(actions, device=self.device, dtype=torch.float32)
        else:
            if actions.device != self.device:
                actions = actions.to(self.device)
            if actions.dtype != torch.float32:
                actions = actions.float()
        self.counter += 1
        self.prev_actions[:] = self.actions
        self.actions = actions

        self.payload_manager.step()
        self._advance_curriculum_if_needed()
        if self.teacher_mode:
            self._update_teacher_residual()
            clamped_policy_actions = torch.clamp(self.actions, -1.0, 1.0)
            policy_imitation_err = torch.norm(
                clamped_policy_actions - self.teacher_residual, dim=1
            ).mean().item()
            self._last_policy_imitation_err = policy_imitation_err
            decay_err = policy_imitation_err
            if self.dagger_use_postmix_err:
                mixed_preview = self.dagger_frac * self.teacher_residual + (1.0 - self.dagger_frac) * clamped_policy_actions
                decay_err = torch.norm(mixed_preview - self.teacher_residual, dim=1).mean().item()
            self._last_decay_imitation_err = decay_err
            # 按回合衰减 DAgger 比例（类似 SB3 BC alpha^updates）
            if self.counter > 0 and self.counter % self.task_config.episode_len_steps == 0:
                # 仅在模仿误差足够低时衰减（可选使用混合后的误差）
                if decay_err <= self.imitation_err_threshold:
                    prev_frac = self.dagger_frac
                    self._dagger_updates += 1
                    target_frac = self._dagger_init_frac * (self.dagger_decay_rate ** self._dagger_updates)
                    self.dagger_frac = max(self.dagger_min_frac, target_frac)
                    print(
                        f"[DaggerDecay] step={self.counter} err={decay_err:.4f} "
                        f"raw={policy_imitation_err:.4f} frac={prev_frac:.3f}->{self.dagger_frac:.3f}"
                    )
            # DAgger 风格：用教师动作与策略残差混合，早期偏向教师
            dagger_frac = max(0.0, min(1.0, self.dagger_frac))
            if dagger_frac > 0.0:
                actions = dagger_frac * self.teacher_residual + (1.0 - dagger_frac) * clamped_policy_actions
                self.actions = actions

        self.controller_actions[:, 0:3] = self.target_position
        self.controller_actions[:, 3] = 0.0
        clamped_actions = torch.clamp(self.actions, -1.0, 1.0)
        # 基础奖励/补偿常开，不再使用窗口掩码
        reward_params = self.task_config.reward_parameters
        reward_window_steps = int(reward_params.get("release_reward_window_steps", 0))
        reward_window_mask = torch.ones_like(
            self.payload_manager.release_warning_flag, dtype=torch.bool, device=self.device
        )
        reward_window_mask_f = reward_window_mask.float()

        self.controller_actions[:, 4] = clamped_actions[:, 0]
        self.controller_actions[:, 5:] = clamped_actions[:, 1:]

        self.sim_env.step(actions=self.controller_actions)

        self._log_rollout(clamped_actions)
        pos_error_body = quat_apply_inverse(
            self.obs_dict["robot_vehicle_orientation"],
            (self.target_position - self.obs_dict["robot_position"]),
        )

        base_rewards, self.terminations[:] = compute_reward(
            pos_error_body,
            self.obs_dict["robot_linvel"],
            self.obs_dict["robot_orientation"],
            self.obs_dict["robot_body_angvel"],
            self.obs_dict["crashes"],
            1.0,
            self.controller_actions,
            self.controller_actions,
            reward_params,
            self.crash_distance_threshold,
            self.crash_tilt_threshold_rad,
        )
        self.rewards[:] = base_rewards
        survive_bonus = float(reward_params.get("survive_bonus", 0.0))
        if survive_bonus != 0.0:
            self.rewards += survive_bonus

        # 额外奖励：在预警或释放后窗口内，鼓励距离误差减小
        dist_norm = torch.norm(self.target_position - self.obs_dict["robot_position"], dim=1)
        delta = self.prev_pos_dist - dist_norm
        self.prev_pos_dist = dist_norm
        bonus_coef = reward_params.get("delta_error_bonus_coef", 0.0)
        bonus_clip = reward_params.get("delta_error_bonus_clip", None)
        window_steps = int(reward_params.get("delta_error_window_steps", 0))
        window_mask = torch.zeros_like(self.payload_manager.step_counter, dtype=torch.bool)
        if window_steps > 0:
            window_mask = self.payload_manager.step_counter < window_steps
        delta_for_log = delta
        if bonus_coef != 0.0:
            delta_clamped = torch.clamp_min(delta, 0.0)
            if bonus_clip is not None and bonus_clip > 0.0:
                delta_clamped = torch.clamp(delta_clamped, max=bonus_clip)
            bonus = delta_clamped * bonus_coef * window_mask.float()
            self.rewards += bonus
            delta_for_log = delta_clamped

        # 记录位置相关奖励分量，便于 TB 观察
        dist = torch.norm(pos_error_body, dim=1)
        pos_reward = exp_func(dist, 3.0, 8.0) + exp_func(dist, 2.0, 4.0)
        dist_reward = (20 - dist) / 40.0
        ups = quat_axis(self.obs_dict["robot_orientation"], 2)
        tiltage = torch.abs(1 - ups[..., 2])
        up_reward = 0.2 / (0.1 + tiltage * tiltage)
        spinnage = torch.norm(self.obs_dict["robot_body_angvel"], dim=1)
        ang_vel_reward = (1.0 / (1.0 + spinnage * spinnage)) * 3

        self.rewards += self._compute_payload_penalties(
            clamped_actions,
            delta=delta_for_log,
            bonus_coef=bonus_coef,
            window_mask=window_mask,
            reward_window_mask=reward_window_mask,
        )

        # 模仿专家残差（仅 Teacher 模式生效）
        imitation_w = float(reward_params.get("imitation_weight", 0.0))
        if self.teacher_mode and imitation_w > 0.0:
            imit_penalty = torch.mean((clamped_actions - self.teacher_residual) ** 2, dim=1)
            self.rewards -= imitation_w * imit_penalty

        # 补充 TB 记录：位置/姿态基础项（均为 batch 均值）
        if hasattr(self, "_last_reward_components") and isinstance(self._last_reward_components, dict):
            if survive_bonus != 0.0:
                self._last_reward_components["survive_bonus"] = survive_bonus
            self._last_reward_components.update(
                {
                    "pos_reward": _mean_detached(pos_reward),
                    "dist_reward": _mean_detached(dist_reward),
                    "pos_weighted": _mean_detached(
                        reward_params["position_weight"] * (pos_reward + dist_reward)
                    ),
                    "up_reward": _mean_detached(pos_reward * up_reward),
                    "ang_vel_reward": _mean_detached(pos_reward * ang_vel_reward),
                }
            )

        # 暴露扰动/教师信息供 BC/优势加权使用，独立于 TB 开关
        disturb_flag = self.payload_manager.release_warning_flag | self.payload_manager.just_released_flag
        self.infos = {
            "last_release_index": self.payload_manager.last_release_index.clone(),
            "just_released": self.payload_manager.just_released_flag.clone(),
            "is_disturbance": disturb_flag.clone(),
        }
        if self.teacher_mode:
            self.infos["teacher_actions"] = self.teacher_residual.clone()

        # 更新上一帧速度，用于下一步的加速度估计
        self.prev_linvel = self.obs_dict["robot_body_linvel"].detach()

        self.truncations[:] = torch.where(
            self.sim_env.sim_steps > self.task_config.episode_len_steps, 1, 0
        )
        reset_envs = self.sim_env.post_reward_calculation_step()
        if reset_envs is not None and reset_envs.numel() > 0:
            self._refresh_env_state(env_ids=reset_envs, reset_payload_manager=True)

        if (
            self.tb_writer is not None
            and self._last_reward_components
            and self.counter % self.tb_log_interval == 0
        ):
            # 额外诊断：观测/动作范数，便于定位模仿误差飙升原因
            obs_base_norm = torch.norm(self.task_obs["observations"], dim=1).mean().item()
            self.tb_writer.add_scalar("debug/obs_base_norm", obs_base_norm, self.counter)
            if self.teacher_mode and "priviliged_obs" in self.task_obs:
                priv_norm = torch.norm(self.task_obs["priviliged_obs"], dim=1).mean().item()
                self.tb_writer.add_scalar("debug/priv_norm", priv_norm, self.counter)
            # 策略/教师动作范数
            policy_norm = torch.norm(clamped_actions, dim=1).mean().item()
            self.tb_writer.add_scalar("debug/policy_action_norm", policy_norm, self.counter)
            if self.teacher_mode and hasattr(self, "teacher_residual"):
                teacher_norm = torch.norm(self.teacher_residual, dim=1).mean().item()
                self.tb_writer.add_scalar("debug/teacher_action_norm", teacher_norm, self.counter)
                # 范围检查，确认教师/策略动作尺度一致
                self.tb_writer.add_scalar("debug/policy_action_min", clamped_actions.min().item(), self.counter)
                self.tb_writer.add_scalar("debug/policy_action_max", clamped_actions.max().item(), self.counter)
                self.tb_writer.add_scalar("debug/teacher_action_min", self.teacher_residual.min().item(), self.counter)
                self.tb_writer.add_scalar("debug/teacher_action_max", self.teacher_residual.max().item(), self.counter)

            total_mean = float(self.rewards.mean().item()) if torch.is_tensor(self.rewards) else 0.0
            denom = total_mean if abs(total_mean) > 1e-6 else 1e-6
            for name, mean_val in self._last_reward_components.items():
                self.tb_writer.add_scalar(f"reward_components/{name}", mean_val, self.counter)
                self.tb_writer.add_scalar(f"reward_components/{name}_ratio", mean_val / denom, self.counter)
            self.tb_writer.add_scalar("reward_components/total_reward", total_mean, self.counter)
            self.tb_writer.flush()

            # 动作饱和率监控（补偿通道接近 -1/1 的占比）
            sat_mask = torch.abs(clamped_actions) >= 0.99
            sat_rate = sat_mask.float().mean().item()
            self.tb_writer.add_scalar("actions/saturation_rate", sat_rate, self.counter)
            comp_sat = sat_mask[:, 1:].float().mean().item()
            self.tb_writer.add_scalar("actions/comp_saturation_rate", comp_sat, self.counter)
            # 模仿误差监控（仅教师模式）
            if self.teacher_mode:
                imit_err = torch.norm(clamped_actions - self.teacher_residual, dim=1).mean().item()
                self.tb_writer.add_scalar("imitation/err", imit_err, self.counter)
                self.tb_writer.add_scalar("imitation/policy_err_raw", self._last_policy_imitation_err, self.counter)
                self.tb_writer.add_scalar("imitation/err_for_decay", self._last_decay_imitation_err, self.counter)
            self.tb_writer.flush()
        # 记录/打印当前 dagger_frac，便于逐“epoch”（tb_log_interval）观察
        if self.teacher_mode:
            if self.tb_writer is not None:
                self.tb_writer.add_scalar("imitation/dagger_frac", self.dagger_frac, self.counter)
            if self.counter % self.tb_log_interval == 0:
                print(f"[DaggerFrac] step={self.counter} frac={self.dagger_frac:.8f}")
            if self.tb_writer is not None:
                self.tb_writer.flush()
        else:
            if self.tb_writer is not None:
                self.tb_writer.flush()
        return self.get_return_tuple()

    def _compute_payload_penalties(
        self, clamped_actions, delta=None, bonus_coef=0.0, window_mask=None, reward_window_mask=None
    ):
        reward_cfg = self.task_config.reward_parameters
        euler = self.obs_dict["robot_euler_angles"]
        pos_error = self.target_position - self.obs_dict["robot_position"]
        roll_pitch_error = torch.norm(euler[:, 0:2], dim=1)
        yaw_error = torch.abs(euler[:, 2])
        base_penalty = -reward_cfg["attitude_penalty_coef"] * roll_pitch_error

        release_multiplier = torch.ones_like(base_penalty)
        release_multiplier = torch.where(
            self.payload_manager.just_released_flag,
            release_multiplier * reward_cfg["release_attitude_boost"],
            release_multiplier,
        )
        attitude_term = base_penalty * release_multiplier

        # 分段补偿惩罚：小幅动作惩罚轻，大幅补偿额外加重
        torque_actions = clamped_actions[:, 1:]
        torque_coef = reward_cfg["comp_torque_penalty_coef"]
        torque_high_coef = reward_cfg.get("comp_torque_penalty_high_coef", torque_coef)
        comp_high_thresh = reward_cfg.get("comp_penalty_high_threshold", 1.0)
        torque_nominal = torch.sum(torque_actions**2, dim=1)
        torque_excess = torch.clamp(torch.abs(torque_actions) - comp_high_thresh, min=0.0)
        torque_excess_penalty = torch.sum(torque_excess**2, dim=1)
        comp_penalty = -(torque_coef * torque_nominal + torque_high_coef * torque_excess_penalty)

        thrust_abs = torch.abs(clamped_actions[:, 0])
        thrust_coef = reward_cfg.get("comp_thrust_penalty_coef", 0.0)
        thrust_high_coef = reward_cfg.get("comp_thrust_penalty_high_coef", thrust_coef)
        thrust_excess = torch.clamp(thrust_abs - comp_high_thresh, min=0.0)
        thrust_penalty = -(thrust_coef * thrust_abs + thrust_high_coef * thrust_excess**2)

        # 加速度惩罚：远离目标的加速度始终重罚，朝向目标的加速度随距离变近惩罚加重
        accel_away_coef = reward_cfg.get("accel_penalty_away_coef", 0.0)
        accel_toward_coef = reward_cfg.get("accel_penalty_toward_coef", 0.0)
        accel_penalty = torch.zeros_like(thrust_penalty)
        if accel_away_coef > 0.0 or accel_toward_coef > 0.0:
            accel_vec = self.obs_dict["robot_body_linvel"] - self.prev_linvel
            dir_to_target = pos_error / (torch.norm(pos_error, dim=1, keepdim=True) + 1e-6)
            accel_world = self.obs_dict.get("robot_linvel", self.obs_dict["robot_body_linvel"]) - self.prev_linvel
            proj = torch.sum(accel_world * dir_to_target, dim=1)
            dist = torch.norm(pos_error, dim=1) + 1e-6
            toward_acc = torch.clamp(proj, min=0.0)
            away_acc = torch.clamp(-proj, min=0.0)
            accel_penalty = -(accel_away_coef * away_acc + accel_toward_coef * toward_acc / dist)

        vel_coef = reward_cfg.get("velocity_penalty_coef", 0.0)
        smooth_coef = reward_cfg.get("action_smoothness_coef", 0.0)
        velocity_penalty = -vel_coef * torch.norm(self.obs_dict["robot_body_linvel"], dim=1)
        action_delta = self.actions - self.prev_actions
        smooth_penalty = -smooth_coef * torch.norm(action_delta, dim=1)

        ang_coef = reward_cfg.get("angvel_penalty_coef", 0.0)
        ang_penalty = -ang_coef * torch.norm(self.obs_dict["robot_body_angvel"], dim=1)

        yaw_penalty_coef = reward_cfg.get("yaw_penalty_coef", 0.0)
        yaw_penalty = -yaw_penalty_coef * yaw_error

        pos_penalty_coef = reward_cfg.get("position_error_penalty_coef", 0.0)
        pos_penalty = -pos_penalty_coef * torch.norm(pos_error, dim=1)
        z_penalty_coef = reward_cfg.get("z_error_penalty_coef", 0.0)
        z_penalty = -z_penalty_coef * torch.abs(pos_error[:, 2])

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
        height_safe_bonus = reward_cfg.get("height_safe_bonus", 0.0)
        if height_warn > 0.0:
            height_below = torch.clamp(height_warn - self.obs_dict["robot_position"][:, 2], min=0.0)
            height_warn_scaled = height_below / max(height_warn, 1e-3)
            height_penalty_coef = reward_cfg.get("height_warning_penalty", 0.0)
            height_penalty = -height_penalty_coef * height_warn_scaled
            if height_safe_bonus != 0.0:
                safe_mask = (self.obs_dict["robot_position"][:, 2] >= height_warn).float()
                height_penalty += height_safe_bonus * safe_mask

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

        # 在释放预警期间，减弱姿态/位置惩罚，让策略有余地预备动作
        warn_mask = self.payload_manager.release_warning_flag.float()
        warn_scale_att = 0.7
        warn_scale_pos = 0.85
        if warn_mask.any():
            base_penalty = base_penalty * (1.0 - warn_mask + warn_scale_att * warn_mask)
            pos_penalty = pos_penalty * (1.0 - warn_mask + warn_scale_pos * warn_mask)
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

        # Hover bonus: reward staying close to target with low tilt and low velocity
        hover_bonus_radius = float(reward_cfg.get("hover_bonus_radius", 0.0))
        hover_bonus_tilt_deg = float(reward_cfg.get("hover_bonus_tilt_deg", 0.0))
        hover_bonus_vel = float(reward_cfg.get("hover_bonus_velocity", 0.0))
        hover_bonus = reward_cfg.get("hover_bonus", 0.0)
        hover_term = torch.zeros_like(roll_pitch_error)
        release_stability_steps = int(reward_cfg.get("release_stability_steps", 0))
        release_hover_boost = float(reward_cfg.get("release_hover_boost", 1.0))
        release_angvel_boost = float(reward_cfg.get("release_angvel_boost", 1.0))
        if hover_bonus_radius > 0.0 and hover_bonus != 0.0:
            pos_norm = torch.norm(pos_error, dim=1)
            near_mask = (pos_norm < hover_bonus_radius).float()
            tilt_mask = torch.ones_like(near_mask)
            if hover_bonus_tilt_deg > 0.0:
                limit_rad = np.deg2rad(hover_bonus_tilt_deg)
                tilt_mask = (
                    (torch.abs(euler[:, 0]) < limit_rad) & (torch.abs(euler[:, 1]) < limit_rad)
                ).float()
            vel_mask = torch.ones_like(near_mask)
            if hover_bonus_vel > 0.0:
                vel_norm = torch.norm(self.obs_dict["robot_body_linvel"], dim=1)
                vel_mask = (vel_norm < hover_bonus_vel).float()
            hover_term = hover_bonus * near_mask * tilt_mask * vel_mask

        # Boost stability signals shortly after each release.
        post_release_mask = torch.zeros_like(hover_term)
        if release_stability_steps > 0:
            post_release_mask = (
                (self.payload_manager.step_counter < release_stability_steps)
                & self.payload_manager.attached_mask.any(dim=1)
            ).float()
            decay = 1.0 - torch.clamp(
                self.payload_manager.step_counter.float() / release_stability_steps, min=0.0, max=1.0
            )
            if release_hover_boost != 1.0:
                hover_term = hover_term * (1.0 + (release_hover_boost - 1.0) * post_release_mask * decay)
            if release_angvel_boost != 1.0:
                ang_penalty = ang_penalty * (1.0 + (release_angvel_boost - 1.0) * post_release_mask * decay)

        # Default delta/window tracking for logging if caller passes nothing.
        if delta is None:
            delta = torch.zeros_like(roll_pitch_error)
        if window_mask is None:
            window_mask = torch.zeros_like(roll_pitch_error, dtype=torch.bool)
        if not torch.is_tensor(window_mask):
            window_mask = torch.as_tensor(window_mask, device=self.device)
        window_mask = window_mask.to(self.device)
        window_mask_float = window_mask.float()

        if reward_window_mask is None:
            reward_window_mask = window_mask
        if not torch.is_tensor(reward_window_mask):
            reward_window_mask = torch.as_tensor(reward_window_mask, device=self.device)
        reward_window_mask = reward_window_mask.to(self.device)
        reward_window_mask_float = reward_window_mask.float()

        comp_activation_coef = reward_cfg.get("comp_activation_bonus_coef", 0.0)
        comp_activation_term = 0.0
        if comp_activation_coef != 0.0:
            comp_action_mag = torch.norm(torque_actions, dim=1)
            comp_activation_term = comp_activation_coef * comp_action_mag * window_mask_float

        if window_mask.any():
            comp_scale = reward_cfg.get("comp_window_penalty_scale", 0.5)
            vel_scale = reward_cfg.get("vel_window_penalty_scale", 0.5)
            smooth_scale = reward_cfg.get("smooth_window_penalty_scale", 0.7)
            comp_penalty = comp_penalty * (1.0 - window_mask_float + comp_scale * window_mask_float)
            thrust_penalty = thrust_penalty * (1.0 - window_mask_float + comp_scale * window_mask_float)
            velocity_penalty = velocity_penalty * (1.0 - window_mask_float + vel_scale * window_mask_float)
            ang_penalty = ang_penalty * (1.0 - window_mask_float + vel_scale * window_mask_float)
            smooth_penalty = smooth_penalty * (1.0 - window_mask_float + smooth_scale * window_mask_float)

        raw_components = {
            "attitude": _mean_detached(attitude_term),
            "position": _mean_detached(pos_penalty),
            "z_position": _mean_detached(z_penalty),
            "yaw": _mean_detached(yaw_penalty),
            "velocity": _mean_detached(velocity_penalty),
            "angular_velocity": _mean_detached(ang_penalty),
            "smooth": _mean_detached(smooth_penalty),
            "comp_torque": _mean_detached(comp_penalty),
            "comp_thrust": _mean_detached(thrust_penalty),
            "comp_activation": _mean_detached(comp_activation_term),
            "stability": _mean_detached(stability_term),
            "tilt_warn": _mean_detached(tilt_penalty),
            "height_warn": _mean_detached(height_penalty),
            "release_tilt": _mean_detached(release_penalty),
            "hover_bonus": _mean_detached(hover_term),
            "delta_error_bonus": _mean_detached(
                torch.clamp_min(delta, 0.0) * bonus_coef * window_mask.float()
            ),
            "acceleration": _mean_detached(accel_penalty),
        }
        # 过滤掉恒为零的条目，避免空白 TB 图
        self._last_reward_components = {
            k: v for k, v in raw_components.items() if abs(v) > 1e-9
        }

        return reward_window_mask_float * (
            attitude_term
            + comp_penalty
            + thrust_penalty
            + accel_penalty
            + velocity_penalty
            + ang_penalty
            + smooth_penalty
            + tilt_penalty
            + height_penalty
            + release_penalty
            + stability_term
            + hover_term
            + pos_penalty
            + z_penalty
            + yaw_penalty
            + comp_activation_term
        )

    def _get_tb_log_dir(self) -> str:
        # 1) 优先用环境变量显式指定
        env_dir = os.environ.get("AERIAL_TB_LOGDIR")
        if env_dir:
            return self._make_timestamped_log_dir(env_dir)
        # 2) 尝试找到当前 runs/ 下最新的 summaries 目录，与 rl-games 默认输出靠近
        runs_root = os.path.join(os.getcwd(), "runs")
        latest = None
        latest_mtime = -1
        if os.path.isdir(runs_root):
            for entry in os.listdir(runs_root):
                summary_dir = os.path.join(runs_root, entry, "summaries")
                if os.path.isdir(summary_dir):
                    mtime = os.path.getmtime(summary_dir)
                    if mtime > latest_mtime:
                        latest_mtime = mtime
                        latest = summary_dir
        if latest:
            return latest
        # 3) 回退：默认写入 runs/reward_components
        return os.path.join(runs_root, "reward_components")

    def _make_timestamped_log_dir(self, base_dir: str) -> str:
        """Append timestamp to avoid overwriting existing diagnostics payload logs."""
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"{base_dir}_{ts}"

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
        rot_mat = quat_to_rotation_matrix(self.obs_dict["robot_orientation"]).reshape(
            self.sim_env.num_envs, 9
        )
        self.task_obs["observations"][:, 3:12] = rot_mat
        self.task_obs["observations"][:, 12:15] = self.obs_dict["robot_body_linvel"]
        self.task_obs["observations"][:, 15:18] = self.obs_dict["robot_body_angvel"]

        payload_obs = self.payload_manager.get_observation_features()
        idx = 18
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

        if self.teacher_mode:
            # 特权向量：当前载荷 + 惯量/扰动/分配矩阵（去掉电机模型超大值，保持慢变量）
            priv_vec = torch.zeros((self.sim_env.num_envs, self.priv_vec_dim), device=self.device)
            idx = 0
            # 0: payload mass
            priv_vec[:, idx] = payload_obs["payload_mass"]
            idx += 1
            # 1-3: COM offset
            priv_vec[:, idx : idx + 3] = payload_obs["com_offset"]
            idx += 3
            # 4-6: base inertia diag
            base_inertia_diag = torch.diagonal(self.payload_manager.base_inertia, dim1=1, dim2=2)
            if base_inertia_diag.shape[0] >= self.sim_env.num_envs:
                priv_vec[:, idx : idx + 3] = base_inertia_diag[: self.sim_env.num_envs]
            idx += 3
            # 7-9: last payload torque
            priv_vec[:, idx : idx + 3] = self._last_tau_payload
            idx += 3
            # 10-33: allocation matrix (flatten 24)
            alloc = np.array(BaseQuadCfg.control_allocator_config.allocation_matrix, dtype=np.float32).flatten()
            alloc_t = torch.as_tensor(alloc, device=self.device)
            end_alloc = idx + alloc_t.numel()
            if end_alloc <= self.priv_vec_dim:
                priv_vec[:, idx:end_alloc] = alloc_t
            idx = end_alloc
            # 34-39: disturbance max force/torque (6)
            disturb = BaseQuadCfg.disturbance.max_force_and_torque_disturbance
            disturb_t = torch.as_tensor(disturb, device=self.device, dtype=torch.float32)
            end_disturb = idx + disturb_t.numel()
            if end_disturb <= self.priv_vec_dim:
                priv_vec[:, idx:end_disturb] = disturb_t
            idx = end_disturb
            # 40: prob_apply_disturbance
            if idx < self.priv_vec_dim:
                priv_vec[:, idx] = getattr(BaseQuadCfg.disturbance, "prob_apply_disturbance", 0.0)
            # 其余预留字段保持 0
            # raw 特权直接输出，由策略侧编码
            self.task_obs["priviliged_obs"] = priv_vec

        self.task_obs["rewards"] = self.rewards
        self.task_obs["terminations"] = self.terminations
        self.task_obs["truncations"] = self.truncations

    def _get_env_tensor(self, env_ids=None):
        if env_ids is None:
            return torch.arange(self.sim_env.num_envs, device=self.device, dtype=torch.long)
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(self.device).long()
        return torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

    def _update_teacher_residual(self):
        """Compute teacher residual (normalized) using privileged mass/COM."""
        orientations = self.obs_dict["robot_orientation"]
        tau_payload = self.payload_manager.compute_body_torque(orientations)
        self._last_tau_payload = tau_payload.detach()
        residual = torch.zeros_like(self.teacher_residual)
        # 惯量差补偿：使用 gyroscopic 项近似 tau_true - tau_base
        angvel = self.obs_dict["robot_body_angvel"]
        I_true = self._compute_true_inertia()
        I_nom = self.payload_manager.base_inertia_nominal
        Iw_true = torch.bmm(I_true, angvel.unsqueeze(-1)).squeeze(-1)
        Iw_nom = torch.bmm(I_nom, angvel.unsqueeze(-1)).squeeze(-1)
        gyro_true = torch.cross(angvel, Iw_true, dim=1)
        gyro_nom = torch.cross(angvel, Iw_nom, dim=1)
        tau_inertia = gyro_true - gyro_nom

        torque_limits = self.comp_torque_limits.view(1, 3).clamp(min=1e-6)
        total_tau = -tau_payload + tau_inertia
        residual[:, 1:] = torch.clamp(total_tau / torque_limits, -1.0, 1.0)
        if self.fix_yaw_residual_zero:
            residual[:, 3] = 0.0
        # thrust 补偿：名义控制未包含载荷质量，补齐 payload 重力
        mass_delta = self.payload_manager.current_payload_mass  # 真实-名义
        if self.comp_thrust_limit > 1e-6:
            thrust_extra = torch.abs(self.payload_manager.gravity[2]) * mass_delta
            residual[:, 0] = torch.clamp(thrust_extra / self.comp_thrust_limit, -1.0, 1.0)
        self.teacher_residual = residual

    def _compute_true_inertia(self):
        """Recompute true inertia tensor from base + attached payloads."""
        inertia = self.payload_manager.base_inertia_nominal.clone()
        attached = self.payload_manager.attached_mask
        for idx in range(self.payload_manager.num_payloads):
            mask = attached[:, idx].float().unsqueeze(-1)
            offset = self.payload_manager.offsets[idx].unsqueeze(0)  # (1,3)
            point_I = point_mass_inertia(self.payload_manager.payload_mass, offset.squeeze(0).cpu().numpy())
            point_I = torch.as_tensor(point_I, device=self.device, dtype=torch.float32).unsqueeze(0)
            inertia += point_I * mask.view(-1, 1, 1)
        return inertia

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

    def _init_rollout_logger(self):
        path = os.environ.get("AERIAL_ROLLOUT_LOG")
        if not path:
            self._rollout_logger = None
            return
        max_steps = int(os.environ.get("AERIAL_ROLLOUT_STEPS", "4000"))
        self._rollout_logger = {
            "path": path,
            "max_steps": max_steps,
            "steps": [],
            "z": [],
            "euler": [],
            "policy": [],
            "teacher": [],
            "just_released": [],
            "last_release_index": [],
        }
        logger.info("[RolloutLog] enabled path=%s max_steps=%d (env0 only)", path, max_steps)
        # 进程退出时也尝试刷盘，防止未显式 close 时丢数据
        atexit.register(self._flush_rollout_logger)

    def _log_rollout(self, clamped_actions):
        log = getattr(self, "_rollout_logger", None)
        if not log or len(log["steps"]) >= log["max_steps"]:
            return
        env_id = 0
        log["steps"].append(int(self.counter))
        pos = self.obs_dict["robot_position"][env_id]
        log["z"].append(float(pos[2].item()))
        euler = self.obs_dict["robot_euler_angles"][env_id].detach().cpu().numpy().astype(np.float32)
        log["euler"].append(euler)
        log["policy"].append(clamped_actions[env_id].detach().cpu().numpy().astype(np.float32))
        if self.teacher_mode and hasattr(self, "teacher_residual"):
            log["teacher"].append(self.teacher_residual[env_id].detach().cpu().numpy().astype(np.float32))
        else:
            log["teacher"].append(None)
        log["just_released"].append(bool(self.payload_manager.just_released_flag[env_id].item()))
        log["last_release_index"].append(int(self.payload_manager.last_release_index[env_id].item()))
        if len(log["steps"]) >= log["max_steps"]:
            self._flush_rollout_logger()

    def _flush_rollout_logger(self):
        log = getattr(self, "_rollout_logger", None)
        if not log or len(log["steps"]) == 0:
            return
        path = log["path"]
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        data = {
            "step": np.array(log["steps"], dtype=np.int64),
            "z": np.array(log["z"], dtype=np.float32),
            "euler": np.vstack(log["euler"]) if log["euler"] else np.empty((0, 3), dtype=np.float32),
            "policy": np.vstack(log["policy"]) if log["policy"] else np.empty(
                (0, self.task_config.action_space_dim), dtype=np.float32
            ),
            "just_released": np.array(log["just_released"], dtype=bool),
            "last_release_index": np.array(log["last_release_index"], dtype=np.int64),
        }
        if log["teacher"] and any(t is not None for t in log["teacher"]):
            filled = [
                (t if t is not None else np.zeros(self.task_config.action_space_dim, dtype=np.float32))
                for t in log["teacher"]
            ]
            data["teacher"] = np.vstack(filled)
        np.savez(path, **data)
        logger.info("[RolloutLog] saved %d steps to %s", len(log["steps"]), path)
        # 防止重复写入
        self._rollout_logger = None

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

    # 远离目标方向的速度惩罚：速度在 pos_error 方向上的正投影
    vel_away_coef = float(parameter_dict.get("vel_away_penalty_coef", 0.0))
    if vel_away_coef != 0.0:
        dir_vec = torch.zeros_like(pos_error)
        nonzero = dist > 1e-6
        dir_vec[nonzero] = pos_error[nonzero] / dist[nonzero].unsqueeze(1)
        v_away = torch.sum(lin_vels * dir_vec, dim=1)
        v_away = torch.clamp(v_away, min=0.0)
        total_reward -= vel_away_coef * v_away

    tilt_excess_coef = float(parameter_dict.get("tilt_excess_coef", 0.0))
    tilt_excess_exp = float(parameter_dict.get("tilt_excess_exp", 0.0))
    tilt_excess_threshold = float(parameter_dict.get("tilt_excess_threshold_deg", 0.0))
    if tilt_excess_coef != 0.0 and tilt_excess_exp > 0.0 and tilt_excess_threshold > 0.0:
        threshold_rad = np.deg2rad(tilt_excess_threshold)
        excess = torch.clamp(tilt_angle - threshold_rad, min=0.0)
        if torch.any(excess > 0):
            total_reward -= tilt_excess_coef * (torch.exp(tilt_excess_exp * excess) - 1.0)
    total_reward[:] = torch.where(
        crashes > 0.0, parameter_dict["crash_penalty"] * torch.ones_like(total_reward), total_reward
    )
    return total_reward, crashes
