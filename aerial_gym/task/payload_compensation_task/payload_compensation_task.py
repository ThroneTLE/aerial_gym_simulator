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
from aerial_gym.config.controller_config import lee_controller_config
from aerial_gym.config.robot_config.base_quad_config import BaseQuadCfg
from aerial_gym.control.controllers.position_control import LeePositionController
from aerial_gym.registry.controller_registry import controller_registry
from aerial_gym.utils.logging import CustomLogger
from aerial_gym.utils.math import (
    quat_apply_inverse,
    quat_axis,
    quat_rotate,
    quat_rotate_inverse,
    quat_from_euler_xyz_tensor,
    quat_mul,
    quat_to_rotation_matrix,
)
from aerial_gym.utils.nan_prevention_utils import (
    clip_rewards,
    clip_observations,
    detect_and_log_nan,
    safe_exp,
    check_inertia_matrix_condition,
    check_physical_limits,
    sanitize_observation_dict,
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


def exp_func(x, gain, offset, use_safe_exp=False):
    """Exponential reward shaping function with optional safe mode.
    
    Args:
        x: Input tensor
        gain: Gain parameter
        offset: Offset parameter
        use_safe_exp: If True, use safe_exp to prevent overflow (default: False for backward compatibility)
    
    Returns:
        Exponential reward
    """
    exp_input = -gain * x**2 / offset
    if use_safe_exp:
        return safe_exp(exp_input, max_input=20.0)
    else:
        return torch.exp(exp_input)


@dataclass
class PayloadConfig:
    payload_mass: float
    offsets: Sequence[Sequence[float]]
    release_start: int
    release_interval: int
    warning_steps: int
    payload_mass_range: Optional[Sequence[float]] = None
    randomize_payload_mass: bool = False
    randomize_offsets_on_plane: bool = False
    offset_plane_radial_jitter: float = 0.0
    offset_plane_z_jitter: float = 0.0
    offset_plane_r_max: float = 0.0
    offset_plane_z_max: float = 0.0
    release_start_range: Optional[Sequence[int]] = None
    release_interval_range: Optional[Sequence[int]] = None
    randomize_release: bool = False
    log_release_events: bool = False
    force_offset_torque_scale: float = 0.0


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

        self.payload_mass = float(payload_cfg.payload_mass)
        self.payload_mass_range = payload_cfg.payload_mass_range
        self.randomize_payload_mass = bool(payload_cfg.randomize_payload_mass)
        self.base_offsets = torch.as_tensor(
            np.array(payload_cfg.offsets, dtype=np.float32), device=self.device
        )
        self.num_payloads = self.base_offsets.shape[0]
        self.offsets = (
            self.base_offsets.unsqueeze(0).expand(self.num_envs, -1, -1).clone()
        )
        self.randomize_offsets_on_plane = bool(payload_cfg.randomize_offsets_on_plane)
        self.offset_plane_radial_jitter = float(payload_cfg.offset_plane_radial_jitter)
        self.offset_plane_z_jitter = float(payload_cfg.offset_plane_z_jitter)
        self.offset_plane_r_max = float(payload_cfg.offset_plane_r_max)
        self.offset_plane_z_max = float(payload_cfg.offset_plane_z_max)

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
        self.force_offset_torque_scale = float(payload_cfg.force_offset_torque_scale)

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
        self.payload_mass_per_env = torch.full(
            (self.num_envs,), self.payload_mass, device=self.device
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
        env_ids = env_ids.to(self.device).long()
        self.attached_mask[env_ids] = True
        self.step_counter[env_ids] = 0
        self._assign_random_release_start(env_ids)
        self._assign_random_release_order(env_ids)
        self.release_cursor[env_ids] = 0
        self.last_release_index[env_ids] = -1
        self.last_release_mass[env_ids] = 0.0
        self.just_released_flag[env_ids] = False
        self._sample_payload_mass(env_ids)
        self._randomize_offsets_on_plane(env_ids)
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
        self.last_release_mass[env_id] = self.payload_mass_per_env[env_id]
        self.just_released_flag[env_id] = True
        if self.log_release_events:
            logger.info(
                "[ReleaseEvent] env=%d payload=%d offset=%s step=%d",
                env_id,
                payload_id,
                self.offsets[env_id, payload_id].tolist(),
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


    def _sample_payload_mass(self, env_ids: torch.Tensor):
        if not self.randomize_payload_mass or not self.payload_mass_range:
            self.payload_mass_per_env[env_ids] = self.payload_mass
            return
        low, high = self.payload_mass_range
        if low > high:
            low, high = high, low
        low = float(max(0.0, low))
        high = float(max(0.0, high))
        if low == high:
            self.payload_mass_per_env[env_ids] = low
            return
        samples = torch.rand((env_ids.shape[0],), device=self.device) * (high - low) + low
        self.payload_mass_per_env[env_ids] = samples

    def _randomize_offsets_on_plane(self, env_ids: torch.Tensor):
        if not self.randomize_offsets_on_plane:
            self.offsets[env_ids] = self.base_offsets
            return
        if self.offset_plane_radial_jitter <= 0.0 and self.offset_plane_z_jitter <= 0.0:
            self.offsets[env_ids] = self.base_offsets
            return

        base = self.base_offsets
        xy = base[:, 0:2]
        r0 = torch.norm(xy, dim=1, keepdim=True).clamp(min=1e-6)
        dir_xy = xy / r0
        z0 = base[:, 2:3]
        r_max = self.offset_plane_r_max
        z_max = self.offset_plane_z_max

        for env_id in env_ids.long().tolist():
            # 同一架飞机的所有载荷共享相同的距离偏移（只有方向不同）
            radial_delta = (
                (torch.rand((1, 1), device=self.device) * 2.0 - 1.0)
                * self.offset_plane_radial_jitter
            )
            z_delta = (
                (torch.rand((1, 1), device=self.device) * 2.0 - 1.0)
                * self.offset_plane_z_jitter
            )
            # 广播到所有 num_payloads 个载荷
            r = (r0 + radial_delta).clamp(min=1e-4)
            if r_max > 0.0:
                r = torch.clamp(r, max=r_max)
            z = z0 + z_delta
            if z_max > 0.0:
                z = torch.clamp(z, min=-z_max, max=z_max)
            xy_new = dir_xy * r
            self.offsets[env_id] = torch.cat([xy_new, z], dim=1)

    def _update_mass_properties(self, env_ids: torch.Tensor):
        env_id_list = env_ids.long().tolist()
        for env_id in env_id_list:
            attached = self.attached_mask[env_id]
            payload_count = int(attached.sum().item())
            payload_mass_each = float(self.payload_mass_per_env[env_id].item())
            payload_mass_sum = payload_count * payload_mass_each
            self.current_payload_mass[env_id] = payload_mass_sum

            base_mass = float(self.base_mass[env_id].item())
            total_mass = base_mass + payload_mass_sum
            inertia_np = self.base_inertia[env_id].cpu().numpy().copy()
            weighted_offset = np.zeros(3, dtype=np.float32)
            for idx, attached_flag in enumerate(attached.tolist()):
                if not attached_flag:
                    continue
                offset = self.offsets[env_id, idx].cpu().numpy()
                inertia_np += point_mass_inertia(payload_mass_each, offset)
                weighted_offset += payload_mass_each * offset

            if total_mass > 0.0:
                self.com_offset_body[env_id] = torch.as_tensor(
                    weighted_offset / total_mass, device=self.device, dtype=torch.float32
                )
            else:
                self.com_offset_body[env_id] = 0.0

            props = self.actor_props[env_id]
            props[0].mass = total_mass
            # NaN Prevention: Check and regularize inertia matrix if ill-conditioned
            inertia_np = check_inertia_matrix_condition(
                inertia_np, 
                max_condition=1000.0, 
                regularization=1e-6
            )
            props[0].inertia = _np_to_mat33(inertia_np)
            self.gym.set_actor_rigid_body_properties(
                self.env_handles[env_id], self.robot_handles[env_id], props, recomputeInertia=False
            )

    def compute_body_torque(self, orientations: torch.Tensor) -> torch.Tensor:
        """
        计算因质心(COM)偏移而在机体坐标系中产生的扰动扭矩。
        扭矩 τ = r_com × F_gravity，其中所有向量都在机体坐标系中表示。
        """
        # 1. 计算总质量和附加载荷质量
        payload_mass_each = self.payload_mass_per_env
        payload_mass = self.attached_mask.sum(dim=1) * payload_mass_each
        total_mass = self.base_mass + payload_mass

        # 2. 在机体坐标系中计算质心偏移 (r_com_b)
        # r_com_b = (Σ m_i * r_i) / M_total, 其中 r_i 是载荷的偏移量
        weighted_offsets = (self.attached_mask.float().unsqueeze(-1) * self.offsets).sum(dim=1)
        weighted_offsets *= payload_mass_each.unsqueeze(-1)

        com_offset_body = torch.zeros((self.num_envs, 3), device=self.device)
        valid_mass = total_mass > 1e-6
        com_offset_body[valid_mass] = weighted_offsets[valid_mass] / total_mass[valid_mass].unsqueeze(-1)

        # 3. 在世界坐标系中计算重力 (F_g_w)
        gravity_force_world = self.gravity.unsqueeze(0) * total_mass.unsqueeze(-1)

        # 4. 将重力从世界坐标系转换到机体坐标系 (F_g_b = R_bw * F_g_w)
        gravity_force_body = quat_rotate_inverse(orientations, gravity_force_world)

        # 5. 在机体坐标系中计算扭矩 (τ_b = r_com_b × F_g_b)
        return torch.cross(com_offset_body, gravity_force_body, dim=1)

    def compute_force_offset_torque(self, total_force_body: torch.Tensor) -> torch.Tensor:
        """
        近似补偿 COM 偏移带来的力矩：tau_eq = - r_com × F_total (机体坐标系)。
        total_force_body: (N, 3) 机体系下的外部合力（不含重力）。
        """
        return -self.force_offset_torque_scale * torch.cross(
            self.com_offset_body, total_force_body, dim=1
        )

    def compute_total_force_body(self) -> torch.Tensor:
        """
        严格版本：将每个刚体局部坐标系的力转换到机体坐标系后求和。
        """
        global_dict = self.env_manager.IGE_env.global_tensor_dict
        robot_force = global_dict.get("robot_force_tensor", None)
        if robot_force is None:
            return torch.zeros((self.num_envs, 3), device=self.device)

        num_rb_robot = self.env_manager.IGE_env.num_rigid_bodies_robot
        num_rb_env = self.env_manager.IGE_env.num_rigid_bodies_per_env
        if not num_rb_robot or not num_rb_env:
            return robot_force.sum(dim=1)

        rb_state = global_dict.get("rigid_body_state_tensor", None)
        if rb_state is None:
            return robot_force.sum(dim=1)

        rb_state = rb_state.reshape(self.num_envs, num_rb_env, -1)
        robot_rb_state = rb_state[:, :num_rb_robot, :]
        link_quat = robot_rb_state[:, :, 3:7]
        root_quat = robot_rb_state[:, 0, 3:7]
        force_local = robot_force[:, :num_rb_robot, :]

        link_quat_flat = link_quat.reshape(-1, 4)
        force_local_flat = force_local.reshape(-1, 3)
        force_world_flat = quat_rotate(link_quat_flat, force_local_flat)
        force_world = force_world_flat.reshape(self.num_envs, num_rb_robot, 3)

        root_quat_expanded = root_quat.unsqueeze(1).expand(-1, num_rb_robot, -1)
        force_body_flat = quat_rotate_inverse(
            root_quat_expanded.reshape(-1, 4), force_world.reshape(-1, 3)
        )
        force_body = force_body_flat.reshape(self.num_envs, num_rb_robot, 3)
        return force_body.sum(dim=1)

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
        self.use_omniscient_gains = bool(getattr(self.task_config, "use_omniscient_gains", True))
        self.dagger_frac = float(getattr(self.task_config, "dagger_frac", 0.0))
        self._dagger_init_frac = self.dagger_frac
        self._dagger_updates = 0
        self.imitation_err_threshold = float(getattr(self.task_config, "imitation_err_threshold", 1e9))

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
        reward_cfg = self.task_config.reward_parameters
        self._imitation_weight_start = float(
            reward_cfg.get("imitation_weight_start", reward_cfg.get("imitation_weight", 0.0))
        )
        self._imitation_weight_end = float(
            reward_cfg.get("imitation_weight_end", self._imitation_weight_start)
        )
        self._imitation_weight_decay_reward = float(
            reward_cfg.get("imitation_weight_decay_reward", 0.0)
        )
        self._imitation_weight_decay_span = float(
            reward_cfg.get("imitation_weight_decay_span", 0.0)
        )
        self._imitation_reward_ema_alpha = float(
            reward_cfg.get("imitation_reward_ema_alpha", 0.05)
        )
        self._imitation_weight_current = self._imitation_weight_start
        self._imitation_reward_ema = None
        self._imitation_schedule_enabled = (
            self._imitation_weight_decay_reward > 0.0
            and self._imitation_weight_start != self._imitation_weight_end
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
        
        # --- Omniscient Teacher Setup ---
        # Initialize Shadow Controller (LeePositionController) using same logic as robot
        # We use a separate instance to run "What-If" scenarios with Optimal Gains
        self.shadow_controller, _ = controller_registry.make_controller(
            "lee_position_control",
            self.task_config.num_envs,
            self.device
        )
        # Access the Real Controller (Fixed Gains)
        self.real_controller = self.sim_env.robot_manager.robot.controller
        
        # Initialize Shadow Controller Tensors (Needs global dictionary)
        # Assuming sim_env.IGE_env exposes global_tensor_dict
        self.shadow_controller.init_tensors(self.sim_env.IGE_env.global_tensor_dict)
        
        # Config for Gain Schedule will be accessed in _update_teacher_residual

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
        # 补偿限制：优先从 task_config 读取，fallback 到 controller config
        self.comp_thrust_limit = getattr(
            self.task_config, "compensation_thrust_limit",
            getattr(lee_controller_with_comp_config.control, "compensation_thrust_limit", 0.5)
        )
        torque_limits = getattr(
            self.task_config, "compensation_torque_limits",
            getattr(lee_controller_with_comp_config.control, "compensation_torque_limits", [0.5, 0.5, 0.1])
        )
        self.comp_torque_limits = torch.as_tensor(torque_limits, device=self.device)
        # 教师模式：特权向量维度由配置文件中的 privileged_observation_space_dim 定义
        # 简化版本: 7 维 [mass, com_x, com_y, com_z, released_mass, warning_flag, just_released]

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

        obs_cfg = getattr(self.task_config, "observation_parameters", None) or {}
        if not isinstance(obs_cfg, dict):
            obs_cfg = {}
        self.obs_include_payload_mass = bool(obs_cfg.get("include_payload_mass", True))
        self.obs_include_payload_com = bool(obs_cfg.get("include_payload_com", True))
        self.obs_include_last_release_mass = bool(
            obs_cfg.get("include_last_release_mass", True)
        )
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
            "privileged_obs": torch.zeros(
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
            payload_mass_range=self.task_config.payload_parameters.get("payload_mass_range"),
            randomize_payload_mass=self.task_config.payload_parameters.get(
                "randomize_payload_mass", False
            ),
            randomize_offsets_on_plane=self.task_config.payload_parameters.get(
                "randomize_offsets_on_plane", False
            ),
            offset_plane_radial_jitter=self.task_config.payload_parameters.get(
                "offset_plane_radial_jitter", 0.0
            ),
            offset_plane_z_jitter=self.task_config.payload_parameters.get(
                "offset_plane_z_jitter", 0.0
            ),
            offset_plane_r_max=self.task_config.payload_parameters.get("offset_plane_r_max", 0.0),
            offset_plane_z_max=self.task_config.payload_parameters.get("offset_plane_z_max", 0.0),
            release_start_range=self.task_config.payload_parameters.get("release_start_range"),
            release_interval_range=self.task_config.payload_parameters.get("release_interval_range"),
            randomize_release=self.task_config.payload_parameters.get("randomize_release", True),
            log_release_events=self.task_config.payload_parameters.get("log_release_events", False),
            force_offset_torque_scale=self.task_config.payload_parameters.get(
                "force_offset_torque_scale", 0.0
            ),
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

        # --- Extended Physical Parameter Randomization ---
        self.randomize_motor_tc = rand_cfg.get("randomize_motor_time_constant", False)
        self.motor_tc_range = rand_cfg.get("motor_time_constant_range", [0.05, 0.05])
        
        self.randomize_motor_thrust_k = rand_cfg.get("randomize_motor_thrust_constant", False)
        self.motor_thrust_k_scale = rand_cfg.get("motor_thrust_constant_range_scale", [1.0, 1.0])
        
        self.randomize_drag = rand_cfg.get("randomize_drag_coefficients", False)
        self.lin_drag_range = rand_cfg.get("lin_drag_coeff_range", [0.0, 0.0])
        self.ang_drag_range = rand_cfg.get("ang_drag_coeff_range", [0.0, 0.0])
        
        self.randomize_ext_dist = rand_cfg.get("randomize_external_disturbance", False)
        self.ext_force_range = rand_cfg.get("external_force_range", [0.0, 0.0])
        self.ext_torque_range = rand_cfg.get("external_torque_range", [0.0, 0.0])
        
        self.ext_forces = torch.zeros((self.sim_env.num_envs, 3), device=self.device)
        self.ext_torques = torch.zeros((self.sim_env.num_envs, 3), device=self.device)

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
        
        # NaN prevention: configuration for numerical stability
        self.reward_clip_min = float(getattr(self.task_config, "reward_clip_min", -100.0))
        self.reward_clip_max = float(getattr(self.task_config, "reward_clip_max", 100.0))
        self.obs_clip_limits = {
            "robot_position": (-50.0, 50.0),
            "robot_linvel": (-50.0, 50.0),
            "robot_body_linvel": (-50.0, 50.0),
            "robot_angvel": (-20.0, 20.0),
            "robot_body_angvel": (-20.0, 20.0),
        }
        self.physical_velocity_limit = 1000000.0 # Force disable limit
        self.physical_angvel_limit = 1000000.0   # Force disable limit
        self.nan_detection_interval = int(getattr(self.task_config, "nan_detection_interval", 100))
        self.nan_count = 0
        self.physical_violation_count = 0
        
        self._init_rollout_logger()

    def _patch_pre_physics_step(self):
        robot_manager = self.sim_env.robot_manager
        orig_pre_physics_step = robot_manager.pre_physics_step
        payload_manager = self.payload_manager
        sim_env = self.sim_env

        def patched_pre_physics_step(actions, _orig=orig_pre_physics_step):
            _orig(actions)
            # Apply External Disturbance (Wind) - Add to root link (index 0)
            # Use 'payload_manager.env_manager' to access robot tensors directly or use the captured scope if valid
            # SimEnv -> RobotManager -> Robot
            robot = sim_env.robot_manager.robot
            robot.robot_force_tensors[:, 0, :] += self.ext_forces
            robot.robot_torque_tensors[:, 0, :] += self.ext_torques

            orientations = sim_env.IGE_env.global_tensor_dict["robot_orientation"]
            body_torque = payload_manager.compute_body_torque(orientations)
            if payload_manager.force_offset_torque_scale != 0.0:
                total_force_body = payload_manager.compute_total_force_body()
                body_torque = body_torque + payload_manager.compute_force_offset_torque(
                    total_force_body
                )
            robot.robot_torque_tensors[:, 0, :] += body_torque

        robot_manager.pre_physics_step = patched_pre_physics_step

    def close(self):
        self._flush_rollout_logger()
        if hasattr(self, "sim_builder") and self.sim_builder is not None:
            self.sim_builder.delete_env()

    def reset(self):
        self.infos = {}
        self.payload_manager.reset()
        self.sim_env.reset()
        self._randomize_ext_params(None)  # Randomize custom physics params AFTER sim reset
        self.prev_pos_dist[:] = 0.0
        self._refresh_env_state(env_ids=None, reset_payload_manager=False)
        return self.get_return_tuple()

    def reset_idx(self, env_ids):
        env_tensor = self._get_env_tensor(env_ids)
        self.payload_manager.reset(env_ids=env_tensor)
        self.sim_env.reset_idx(env_ids)
        self._randomize_ext_params(env_tensor)  # Randomize custom physics params AFTER sim reset
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
            
            # 始终使用策略动作（无 DAgger 混合）
            self.actions = clamped_policy_actions
            
            # 计算模仿误差用于记录
            policy_imitation_err = torch.norm(
                 clamped_policy_actions - self.teacher_residual, dim=1
            ).mean().item()
            self._last_policy_imitation_err = policy_imitation_err

        self.controller_actions[:, 0:3] = self.target_position
        self.controller_actions[:, 3] = 0.0
        clamped_actions = torch.clamp(self.actions, -1.0, 1.0)
        # 基础奖励/补偿常开，不再使用窗口掩码
        reward_params = self.task_config.reward_parameters



        # 动作映射 (3维 -> 8维控制器):
        # action[0] -> controller[4] thrust 补偿
        # action[1] -> controller[5] roll 力矩补偿
        # action[2] -> controller[6] pitch 力矩补偿
        # controller[7] yaw 力矩补偿固定为 0 (不再由策略控制)
        self.controller_actions[:, 4] = clamped_actions[:, 0]  # thrust
        self.controller_actions[:, 5] = clamped_actions[:, 1]  # roll
        self.controller_actions[:, 6] = clamped_actions[:, 2]  # pitch
        self.controller_actions[:, 7] = 0.0  # yaw 固定为 0

        self.sim_env.step(actions=self.controller_actions)
        
        # 1. Enhanced NaN detection with logging (Original data check)
        # This MUST be done before sanitization to correctly identify crashed environments
        _reset_on_nonfinite(self.obs_dict, self.device)

        # 2. Check for physical limit violations (extreme velocities)
        violation_mask = check_physical_limits(
            self.obs_dict,
            velocity_limit=self.physical_velocity_limit,
            angvel_limit=self.physical_angvel_limit,
        )
        if violation_mask.any():
            self.physical_violation_count += violation_mask.sum().item()
            # Mark as crashes to trigger reset
            self.obs_dict["crashes"][violation_mask] = True
        
        # 3. NaN Prevention: Sanitize observations before use in network/reward
        # Now we replace NaNs with zeros to protect the network, but since we already
        # marked them as crashes above, they will receive crash penalties.
        # 3. NaN Prevention: Sanitize observations before use
        # Store sanitized obs in a temporary variable (or swap), but keep raw link for Sim
        # IMPORTANT: We swap self.obs_dict to sanitized version for calculation, 
        # but must restore it to raw version before next sim step!
        self._raw_obs_dict_link = self.obs_dict
        self.obs_dict = sanitize_observation_dict(
            self.obs_dict, 
            clip_limits=self.obs_clip_limits,
            normalize_quaternions=True
        )
        
        # Periodic NaN monitoring for debugging
        if self.counter % self.nan_detection_interval == 0:
            for key in ["robot_position", "robot_orientation", "robot_linvel", "robot_body_angvel"]:
                if key in self.obs_dict:
                    # Note: These will likely be clean now due to sanitize above, but useful for logs
                    if detect_and_log_nan(self.obs_dict[key], f"obs/{key}", self.counter, logger):
                        self.nan_count += 1

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
        
        # NaN Prevention: 奖励裁剪已禁用 - 允许奖励自然增长以便训练进步
        # self.rewards = clip_rewards(self.rewards, min_r=self.reward_clip_min, max_r=self.reward_clip_max)



        # 记录位置相关奖励分量，便于 TB 观察
        dist = torch.norm(pos_error_body, dim=1)
        pos_reward = exp_func(dist, 3.0, 8.0) + exp_func(dist, 2.0, 4.0)
        dist_reward = (20 - dist) / 40.0
        ups = quat_axis(self.obs_dict["robot_orientation"], 2)
        tiltage = torch.abs(1 - ups[..., 2])
        up_reward = 0.2 / (0.1 + tiltage * tiltage)
        spinnage = torch.norm(self.obs_dict["robot_body_angvel"], dim=1)
        ang_vel_reward = (1.0 / (1.0 + spinnage * spinnage)) * 3

        self.rewards += self._compute_payload_penalties(clamped_actions)

        # 模仿专家残差（仅 Teacher 模式生效）
        imitation_w = float(reward_params.get("imitation_weight", 0.0))
        if self.teacher_mode and self._imitation_schedule_enabled:
            reward_mean = float(self.rewards.mean().item())
            imitation_w = self._update_imitation_weight(reward_mean)
        if self.teacher_mode and imitation_w > 0.0:
            # 分离推力和力矩的模仿权重
            w_thrust = float(reward_params.get("imitation_weight_thrust", imitation_w))
            w_torque = float(reward_params.get("imitation_weight_torque", imitation_w))
            
            # 推力模仿惩罚 (action[0])
            thrust_err = (clamped_actions[:, 0] - self.teacher_residual[:, 0]) ** 2
            # 力矩模仿惩罚 (action[1:3] = roll, pitch，无 yaw)
            torque_err = torch.mean((clamped_actions[:, 1:3] - self.teacher_residual[:, 1:3]) ** 2, dim=1)
            
            # 分别加权
            imit_penalty = w_thrust * thrust_err + w_torque * torque_err
            self.rewards -= imit_penalty



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
            if self.teacher_mode and "privileged_obs" in self.task_obs:
                priv_norm = torch.norm(self.task_obs["privileged_obs"], dim=1).mean().item()
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
                imit_err_per_env = torch.norm(clamped_actions - self.teacher_residual, dim=1)
                imit_err = imit_err_per_env.mean().item()
                self.tb_writer.add_scalar("imitation/err", imit_err, self.counter)
                self.tb_writer.add_scalar("imitation/policy_err_raw", self._last_policy_imitation_err, self.counter)
                self.tb_writer.add_scalar("imitation/err_for_decay", self._last_decay_imitation_err, self.counter)
                
                # 分场景 err 统计
                mass = self.payload_manager.current_payload_mass
                mass_median = mass.median()
                
                # 高质量 vs 低质量
                high_mass_mask = mass > mass_median
                low_mass_mask = ~high_mass_mask
                if high_mass_mask.sum() > 10:
                    err_high_mass = imit_err_per_env[high_mass_mask].mean().item()
                    self.tb_writer.add_scalar("imitation/err_high_mass", err_high_mass, self.counter)
                if low_mass_mask.sum() > 10:
                    err_low_mass = imit_err_per_env[low_mass_mask].mean().item()
                    self.tb_writer.add_scalar("imitation/err_low_mass", err_low_mass, self.counter)
                
                # 刚释放 vs 稳定期
                just_released = getattr(self.payload_manager, 'just_released_mask', None)
                if just_released is not None:
                    released_mask = just_released.bool()
                    stable_mask = ~released_mask
                    if released_mask.sum() > 5:
                        err_released = imit_err_per_env[released_mask].mean().item()
                        self.tb_writer.add_scalar("imitation/err_just_released", err_released, self.counter)
                    if stable_mask.sum() > 10:
                        err_stable = imit_err_per_env[stable_mask].mean().item()
                        self.tb_writer.add_scalar("imitation/err_stable", err_stable, self.counter)
                # Mass-Thrust 相关性监控：检查策略是否学会根据质量调整推力
                # 正相关 = 质量大时输出大推力（正确）
                # 负相关 = 质量大时输出小推力（错误）
                # 接近0 = 没学到关系
                mass = self.payload_manager.current_payload_mass
                policy_thrust = clamped_actions[:, 0]
                teacher_thrust = self.teacher_residual[:, 0]
                
                # 计算皮尔逊相关系数
                def pearson_corr(x, y):
                    x_mean = x.mean()
                    y_mean = y.mean()
                    x_std = x.std() + 1e-8
                    y_std = y.std() + 1e-8
                    return ((x - x_mean) * (y - y_mean)).mean() / (x_std * y_std)
                
                # 只有当质量分布有足够方差时才计算相关性
                mass_std = mass.std().item()
                if mass_std > 0.005:  # 质量标准差 > 5g 时才计算
                    corr_policy = pearson_corr(mass, policy_thrust).item()
                    corr_teacher = pearson_corr(mass, teacher_thrust).item()
                    
                    # 使用 EMA 平滑
                    ema_alpha = 0.1
                    if not hasattr(self, '_ema_corr_policy'):
                        self._ema_corr_policy = corr_policy
                        self._ema_corr_teacher = corr_teacher
                    else:
                        self._ema_corr_policy = ema_alpha * corr_policy + (1 - ema_alpha) * self._ema_corr_policy
                        self._ema_corr_teacher = ema_alpha * corr_teacher + (1 - ema_alpha) * self._ema_corr_teacher
                    
                    self.tb_writer.add_scalar("correlation/mass_vs_policy_thrust", self._ema_corr_policy, self.counter)
                    self.tb_writer.add_scalar("correlation/mass_vs_teacher_thrust", self._ema_corr_teacher, self.counter)
                    self.tb_writer.add_scalar("correlation/mass_std", mass_std, self.counter)
                    # 相关性比值：policy/teacher，理想值=1
                    if abs(self._ema_corr_teacher) > 0.1:
                        corr_ratio = self._ema_corr_policy / self._ema_corr_teacher
                        self.tb_writer.add_scalar("correlation/policy_teacher_ratio", corr_ratio, self.counter)
        # 记录/打印当前 dagger_frac，便于逐“epoch”（tb_log_interval）观察
        if self.tb_writer is not None:
             self.tb_writer.flush()
        else:
            if self.tb_writer is not None:
                self.tb_writer.flush()
        # Capture return tuple using sanitized state
        ret_tuple = self.get_return_tuple()
        
        # RESTORE raw link before simulation/reset logic (Critical for next step!)
        self.obs_dict = self._raw_obs_dict_link
        
        return ret_tuple

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



        # 加速度惩罚：远离目标的加速度始终重罚，朝向目标的加速度随距离变近惩罚加重
        accel_away_coef = reward_cfg.get("accel_penalty_away_coef", 0.0)
        accel_toward_coef = reward_cfg.get("accel_penalty_toward_coef", 0.0)
        accel_penalty = torch.zeros(self.sim_env.num_envs, device=self.device)

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



        pos_penalty_coef = reward_cfg.get("position_error_penalty_coef", 0.0)
        pos_penalty = -pos_penalty_coef * torch.norm(pos_error, dim=1)
        z_penalty_coef = reward_cfg.get("z_error_penalty_coef", 0.0)
        z_penalty = -z_penalty_coef * torch.abs(pos_error[:, 2])














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



        raw_components = {
            "attitude": _mean_detached(attitude_term),
            "position": _mean_detached(pos_penalty),
            "z_position": _mean_detached(z_penalty),

            "velocity": _mean_detached(velocity_penalty),
            "angular_velocity": _mean_detached(ang_penalty),
            "smooth": _mean_detached(smooth_penalty),




            "acceleration": _mean_detached(accel_penalty),
        }
        # 过滤掉恒为零的条目，避免空白 TB 图
        self._last_reward_components = {
            k: v for k, v in raw_components.items() if abs(v) > 1e-9
        }

        return reward_window_mask_float * (
            attitude_term

            + accel_penalty
            + velocity_penalty
            + ang_penalty
            + smooth_penalty


            + pos_penalty
            + z_penalty


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
        # [0-8] 旋转矩阵 (9)
        # [9-11] 机体角速度 (3)  
        # [12-15] 4位附着掩码 (4)
        # [16] 释放预警标志 (0 或 1)
        # [17-19] 上一时刻动作 (3): thrust, roll, pitch
        
        # 旋转矩阵
        rot_mat = quat_to_rotation_matrix(self.obs_dict["robot_orientation"]).reshape(
            self.sim_env.num_envs, 9
        )
        self.task_obs["observations"][:, 0:9] = rot_mat
        
        # 机体角速度
        self.task_obs["observations"][:, 9:12] = self.obs_dict["robot_body_angvel"]
        
        # 4位附着掩码 (恢复原始格式)
        payload_obs = self.payload_manager.get_observation_features()
        attached = payload_obs["attached_mask"]  # [N, 4], 1=attached, 0=released
        self.task_obs["observations"][:, 12:16] = attached
        
        # 释放预警标志
        self.task_obs["observations"][:, 16] = payload_obs["warning_flag"]
        
        # 上一时刻动作 (3维: thrust, roll, pitch)
        self.task_obs["observations"][:, 17:20] = self.prev_actions

        # [Masking] Zero out Basic components based on config
        obs_params = self.task_config.observation_parameters
        obs_tensor = self.task_obs["observations"]
        
        if not obs_params.get("include_base_rot", True):
            obs_tensor[:, 0:9] = 0.0
        if not obs_params.get("include_base_angvel", True):
            obs_tensor[:, 9:12] = 0.0
        if not obs_params.get("include_base_attached", True):
            obs_tensor[:, 12:16] = 0.0
        if not obs_params.get("include_base_warning", True):
            obs_tensor[:, 16] = 0.0
        if not obs_params.get("include_base_prev_action", True):
            obs_tensor[:, 17:20] = 0.0

        self._apply_observation_noise()

        if not self.teacher_mode:
            return {}
        else:
            # 18-dimensional extended privileged observation
            # Manually normalize to [0, 1] or [-1, 1] to avoid RunningObsNorm issues
            priv_dim = self.task_config.privileged_observation_space_dim
            priv_vec = torch.zeros((self.sim_env.num_envs, priv_dim), device=self.device)
            
            # --- Basic Privileged Info (0-7) ---
            if priv_dim > 0:
                # 0: payload mass (0~1.6 kg → 0~1, 4×0.4kg max)
                max_mass = 1.6  # 4 payloads × 0.4 kg each
                priv_vec[:, 0] = payload_obs["payload_mass"] / max_mass
            if priv_dim > 3:
                # 1-3: COM offset (-0.4~0.4 m → -1~1)
                max_offset = 0.4
                priv_vec[:, 1:4] = payload_obs["com_offset"] / max_offset
            if priv_dim > 6:
                # 4-6: true inertia diag
                # Use _compute_true_inertia() instead of nominal base_inertia
                true_inertia = self._compute_true_inertia()  # [N, 3, 3]
                true_inertia_diag = torch.diagonal(true_inertia, dim1=1, dim2=2)  # [N, 3]
                typical_inertia = 0.008  # kg·m² (matches new base_link inertia)
                priv_vec[:, 4:7] = true_inertia_diag / typical_inertia

            # --- Extended Privileged Info (7-17) ---
            if priv_dim > 7:
                robot = self.sim_env.robot_manager.robot
                model = robot.control_allocator.motor_model
                
                # 7: Motor Thrust Constant Scale (Assuming isotropic)
                base_k = (model.cfg.motor_thrust_constant_min + model.cfg.motor_thrust_constant_max) / 2.0
                if hasattr(model, 'motor_thrust_constant'):
                    current_k = model.motor_thrust_constant.mean(dim=1)
                    priv_vec[:, 7] = current_k / base_k
                else:
                    priv_vec[:, 7] = 1.0

            if priv_dim > 8:
                # 8: Motor Time Constant
                # Normalize by 0.1 (max reasonable value)
                priv_vec[:, 8] = model.motor_time_constants_increasing.mean(dim=1) / 0.1

            if priv_dim > 11:
                # 9-11: Linear Drag Coefficients
                # Normalize by 0.2 (max range)
                priv_vec[:, 9:12] = robot.body_vel_linear_damping_coefficient / 0.2

            if priv_dim > 14:
                # 12-14: Angular Drag Coefficients
                # Normalize by 0.05 (max range)
                priv_vec[:, 12:15] = robot.angvel_linear_damping_coefficient / 0.05

            if priv_dim > 17:
                # 15-17: External Force (Wind)
                # Normalize by 0.2 (max range)
                priv_vec[:, 15:18] = self.ext_forces / 0.2

            # [Masking] Zero out components based on config
            obs_params = self.task_config.observation_parameters
            if priv_dim > 0 and not obs_params.get("include_priv_mass", True):
                priv_vec[:, 0] = 0.0
            if priv_dim > 3 and not obs_params.get("include_priv_com", True):
                priv_vec[:, 1:4] = 0.0
            if priv_dim > 6 and not obs_params.get("include_priv_inertia", True):
                priv_vec[:, 4:7] = 0.0
            
            if priv_dim > 7 and not obs_params.get("include_priv_motor_thrust", True):
                priv_vec[:, 7] = 0.0
            if priv_dim > 8 and not obs_params.get("include_priv_motor_tc", True):
                priv_vec[:, 8] = 0.0
            if priv_dim > 11 and not obs_params.get("include_priv_drag_lin", True):
                priv_vec[:, 9:12] = 0.0
            if priv_dim > 14 and not obs_params.get("include_priv_drag_ang", True):
                priv_vec[:, 12:15] = 0.0
            if priv_dim > 17 and not obs_params.get("include_priv_disturbance", True):
                priv_vec[:, 15:18] = 0.0

            self.task_obs["privileged_obs"] = priv_vec

        self.task_obs["rewards"] = self.rewards
        self.task_obs["terminations"] = self.terminations
        self.task_obs["truncations"] = self.truncations

    def update_target_position(self, target_pos: torch.Tensor, env_ids: Optional[torch.Tensor] = None):
        """
        External setter for target position to decouple training scripts from internal state.
        Args:
            target_pos: (N, 3) or (1, 3) target tensor
            env_ids: Optional subset of envs to update. If None, updates all.
        """
        if not torch.is_tensor(target_pos):
            target_pos = torch.as_tensor(target_pos, device=self.device, dtype=torch.float32)
        else:
            if target_pos.device != self.device:
                target_pos = target_pos.to(self.device)
            if target_pos.dtype != torch.float32:
                target_pos = target_pos.float()
        if target_pos.ndim == 1:
            target_pos = target_pos.unsqueeze(0)
        if target_pos.shape[-1] != 3:
            raise ValueError(f"target_pos must have last dim=3, got shape {tuple(target_pos.shape)}")
        if env_ids is None:
            self.target_position[:] = target_pos
        else:
            env_ids = self._get_env_tensor(env_ids)
            if target_pos.shape[0] == 1 and env_ids.numel() > 1:
                target_pos = target_pos.expand(env_ids.numel(), -1)
            self.target_position[env_ids] = target_pos

    def get_task_observations(self):
        """
        Getter for task observations to decouple training scripts from direct dict access.
        """
        return self.task_obs

    def _get_env_tensor(self, env_ids=None):
        if env_ids is None:
            return torch.arange(self.sim_env.num_envs, device=self.device, dtype=torch.long)
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(self.device).long()
        return torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

    def _update_imitation_weight(self, reward_mean: float) -> float:
        if not self._imitation_schedule_enabled:
            return self._imitation_weight_start
        if self._imitation_reward_ema is None:
            self._imitation_reward_ema = reward_mean
        else:
            alpha = self._imitation_reward_ema_alpha
            self._imitation_reward_ema = (
                (1.0 - alpha) * self._imitation_reward_ema + alpha * reward_mean
            )

        span = self._imitation_weight_decay_span
        if span <= 0.0:
            progress = 1.0 if self._imitation_reward_ema >= self._imitation_weight_decay_reward else 0.0
        else:
            start = self._imitation_weight_decay_reward - 0.5 * span
            end = self._imitation_weight_decay_reward + 0.5 * span
            if end <= start:
                progress = 1.0 if self._imitation_reward_ema >= end else 0.0
            else:
                progress = (self._imitation_reward_ema - start) / (end - start)
                progress = float(max(0.0, min(1.0, progress)))

        self._imitation_weight_current = (
            self._imitation_weight_start
            + (self._imitation_weight_end - self._imitation_weight_start) * progress
        )
        return self._imitation_weight_current

    def _update_teacher_residual(self):
        """Compute teacher residual: Physical Compensation + Omniscient PID Correction."""
        
        # 1. Physical Torque Compensation (Feedforward Gravity/Inertia)
        orientations = self.obs_dict["robot_orientation"]
        tau_payload = self.payload_manager.compute_body_torque(orientations)
        tau_force = torch.zeros_like(tau_payload)
        if self.payload_manager.force_offset_torque_scale != 0.0:
            total_force_body = self.payload_manager.compute_total_force_body()
            tau_force = self.payload_manager.compute_force_offset_torque(total_force_body)
        
        # Gyroscopic Inertia Compensation
        angvel = self.obs_dict["robot_body_angvel"]
        I_true = self._compute_true_inertia()
        I_nom = self.payload_manager.base_inertia_nominal
        Iw_true = torch.bmm(I_true, angvel.unsqueeze(-1)).squeeze(-1)
        Iw_nom = torch.bmm(I_nom, angvel.unsqueeze(-1)).squeeze(-1)
        gyro_true = torch.cross(angvel, Iw_true, dim=1)
        gyro_nom = torch.cross(angvel, Iw_nom, dim=1)
        tau_inertia = gyro_true - gyro_nom

        total_tau_phys = -(tau_payload + tau_force) + tau_inertia
        
        if self.use_omniscient_gains:
            # --- Part 2: Bias-Free Inverse Dynamics Compensation ---
            
            # A. Access Real Controller Output (Fixed Gains)
            # We need to run the controller to get the baseline torque
            # Note: The controller update has likely already run or we run it now explicitly
            # to be safe, let's run it on the current state.
            
            # Prepare Input Command
            # Expect [x, y, z, yaw, vx, vy, vz, yaw_rate] for LeePositionController
            # controller_actions has [x, y, z, yaw] (subset for PPO?) or does it generally have 4?
            # If PPO outputs 4, we assume velocities are 0.
            
            real_input = torch.zeros((self.sim_env.num_envs, 8), device=self.device)
            real_input[:, 0:4] = self.controller_actions[:, 0:4]
            # Velocities (cols 4-7) remain 0.0
            
            # Get Baseline Wrench (Force + Torque)
            # Note: We must ensure this doesn't update internal state if it's stateful (Lee is stateless per step)
            wrench_fix = self.real_controller.update(real_input)
            
            torque_fix = wrench_fix[:, 3:6] # [N, 3] in Body Frame
            thrust_fix = wrench_fix[:, 2]   # [N] along Z
            
            # B. Deconstruct Baseline Torque to get Feedback Component
            # tau_total = tau_fb + tau_gyro_nom
            # We want to scale tau_fb.
            # tau_fb = tau_total - tau_gyro_nom
            # (Note: We calculated gyro_nom in Part 1)
            tau_fb_nom = torque_fix - gyro_nom
            
            # C. Inertia Scaling (The "Analytic Gain Schedule")
            # tau_fb_req = J_true * J_nom^-1 * tau_fb_nom
            # J_nom is diagonal, so J_nom^-1 is 1/diag
            angle_vel = self.obs_dict["robot_body_angvel"]
            
            # J_nom_inv: [N, 3] or [3] if constant. 
            # I_nom is [N, 3, 3]. It's diagonal.
            I_nom_diag = torch.diagonal(I_nom, dim1=1, dim2=2)
            I_nom_inv = 1.0 / (I_nom_diag + 1e-6) # Avoid div by zero, shape [N, 3]
            
            # Calculate J_true * J_nom^-1
            # Since J_true is full matrix and J_nom is diagonal:
            # (J_true * J_nom^-1)_ij = J_true_ik * (1/J_nom_kj) -> J_true * diag(1/J_nom)
            # We can do this via broadcasting or bmm with diagonal matrix
            # BMM approach:
            I_nom_inv_mat = torch.diag_embed(I_nom_inv) # [N, 3, 3]
            Inertia_Scale = torch.bmm(I_true, I_nom_inv_mat) # [N, 3, 3]
            
            # Apply Scaling to Feedback Torque
            tau_fb_req = torch.bmm(Inertia_Scale, tau_fb_nom.unsqueeze(-1)).squeeze(-1) # [N, 3]
            
            # D. Drag Compensation (Damping)
            # F_drag = -D_v * v (Linear) -> compensated by adding +D_v * v
            # Tau_drag = -D_w * w (Angular) -> compensated by adding +D_w * w
            
            robot = self.sim_env.robot_manager.robot
            lin_vel = self.obs_dict["robot_body_linvel"]
            
            # Linear Drag Force (Body Frame)
            # coeffs are usually positive, force is -coeff * vel
            # To cancel, we need +coeff * vel
            lin_drag_coeffs = robot.body_vel_linear_damping_coefficient # [N, 3]
            force_drag_body = lin_drag_coeffs * lin_vel
            # We only can compensate thrust (Z-axis force)
            thrust_drag_comp = force_drag_body[:, 2] 
            
            # Angular Drag Torque (Body Frame)
            ang_drag_coeffs = robot.angvel_linear_damping_coefficient # [N, 3]
            torque_drag_comp = ang_drag_coeffs * angle_vel
            
            # E. External Disturbance Cancellation
            # To cancel F_ext, add -F_ext
            # ext_forces are in World Frame? Typically yes.
            # Need to rotate to Body Frame for thrust comp.
            
            # Rotate F_ext (World) to Body
            # R^T * F_ext
            from aerial_gym.utils.math import quat_rotate_inverse
            f_ext_body = quat_rotate_inverse(orientations, self.ext_forces)
            # Cancel Z component
            thrust_ext_comp = -f_ext_body[:, 2]
            
            # External Torque (Body Frame)
            # To cancel, add -tau_ext
            torque_ext_comp = -self.ext_torques
            
            # F. Total Required Wrench
            # Torque = Scaled_FB + Gyro_True + Drag + Ext_Cancel + Offset_Cancel
            # (Note: Offset_Cancel is handled in Part 1 as tau_payload/force)
            # Wait, Part 1 calculated `total_tau_phys = -(tau_payload + tau_force) + tau_inertia`
            # where `tau_inertia = gyro_true - gyro_nom`
            # And `residual = ... + total_tau_phys`
            # Let's align carefully.
            
            # Target Torque for Teacher:
            # tau_target = tau_fb_req + gyro_true + torque_drag_comp + torque_ext_comp - (tau_payload + tau_force)
            
            # The Residual we want to add to `torque_fix` is:
            # Res = tau_target - torque_fix
            # Res = (tau_fb_req - tau_fb_nom) + (gyro_true - gyro_nom) + drag + ext - payload
            #      |_______________________|   |____________________|  
            #             Scaling Diff               Gyro Diff
            
            # We already have `total_tau_phys` which includes (Gyro Diff - Payload - ForceOffset)
            # So we just need to add: Scaling Diff + Drag + Ext
            
            tau_scaling_diff = tau_fb_req - tau_fb_nom
            
            # Combine all torque residuals
            total_torque_residual = (
                total_tau_phys         # (Gyro_True - Gyro_Nom) - Payload - ForceOffset
                + tau_scaling_diff     # (J_true/J_nom * FB - FB)
                + torque_drag_comp     # + D_w * w
                + torque_ext_comp      # - tau_ext
            )
            
            # Combine thrust residuals
            # We have mass diff from Phys Comp (gravity) which is added later?
            # Let's check downstream...
            # The function returns `self.teacher_residual`
            # Currently it expects [thrust_res_norm, torque_res_norm]
            
            # Calculate Thrust Residual (Force)
            # Base logic handles (M_true - M_nom) * g via `phys_thrust_action` later?
            # Let's look at lines 1583+ in original code.
            # Usually `phys_thrust_action` compensates for mass difference.
            # We need to add Drag and Ext Force to that.
            
            # Convert Forces to Normalized Actions
            thrust_limit = self.task_config.compensation_thrust_limit
            torque_limits = torch.tensor(self.task_config.compensation_torque_limits, device=self.device)
            
            # Extract Force Residuals
            
            # --- Thrust Inertia Scaling ---
            # Compensate for sluggish acceleration: F_req = (M_true / M_nom) * F_nom
            # Residual = (M_true/M_nom - 1) * F_nom
            # F_nom is `thrust_fix` (approx)
            
            # Use base_mass [N] as nominal mass
            m_nom = self.payload_manager.base_mass
            m_true = m_nom + self.payload_manager.current_payload_mass
            # Ensure safe division (though base_mass should be > 0)
            mass_ratio = m_true / (m_nom + 1e-6)
            
            # Isolate Dynamic Thrust (remove hover part) to avoid double-counting gravity
            # thrust_fix is Total Thrust (Acceleration + Gravity)
            # We already compensate Gravity Diff in Part 3.
            # We only want to scale the Acceleration part: F_acc = F_total - F_hover
            
            g = 9.81
            thrust_hover_nom = m_nom * g
            thrust_dynamic = thrust_fix.squeeze(-1) - thrust_hover_nom
            
            # Scale only the dynamic part
            thrust_scaling_comp = (mass_ratio - 1.0) * thrust_dynamic
            
            d_thrust_N = thrust_drag_comp + thrust_ext_comp + thrust_scaling_comp
            
            d_thrust_N = thrust_drag_comp + thrust_ext_comp + thrust_scaling_comp
            
            # (Note: Mass diff for Gravity is handled by logic downstream, but mass ratio helps acceleration)
            
            pid_thrust_action = d_thrust_N / thrust_limit
            pid_torque_action = total_torque_residual / torque_limits
            
        else:
            # Fallback (Should not happen if use_omniscient_gains is True)
            pid_thrust_action = torch.zeros_like(self.controller_actions[:, 0])
            pid_torque_action = torch.zeros_like(self.controller_actions[:, 1:4])
            
        # 3. Combine with Mass Compensation (Standard)
        # Calculate gravity compensation for mass difference (Payloads only)
        # We need the extra force required to hold the payload.
        force_z = self.payload_manager.current_payload_mass * 9.81
        phys_thrust_action = force_z / self.task_config.compensation_thrust_limit
        
        # Final Residual
        # Note: phys_thrust_action accounts for Gravity Diff
        # pid_thrust_action accounts for Drag + Ext Force
        # pid_torque_action accounts for All Torque Diffs (Scaling + Gyro + Drag + Ext + Offset)
        
        self.teacher_residual[:, 0] = phys_thrust_action + pid_thrust_action
        # Only assign Roll and Pitch torques (indices 1, 2)
        # pid_torque_action is 3D [Roll, Pitch, Yaw]
        # teacher_residual is 3D [Thrust, Roll, Pitch] (Action Space = 3)
        self.teacher_residual[:, 1:3] = pid_torque_action[:, 0:2]
        
        # Clamp
        self.teacher_residual = torch.clamp(self.teacher_residual, -1.0, 1.0)
            
        self._last_tau_payload = total_tau_phys.detach()


    def _compute_true_inertia(self):
        """Recompute true inertia tensor from base + attached payloads."""
        inertia = self.payload_manager.base_inertia_nominal.clone()
        attached = self.payload_manager.attached_mask.float()
        offsets = self.payload_manager.offsets
        payload_mass_each = self.payload_manager.payload_mass_per_env
        eye = torch.eye(3, device=self.device).unsqueeze(0)
        for idx in range(self.payload_manager.num_payloads):
            offset = offsets[:, idx, :]
            r_sq = torch.sum(offset * offset, dim=1)
            outer = offset.unsqueeze(2) * offset.unsqueeze(1)
            point_I = payload_mass_each.view(-1, 1, 1) * (
                r_sq.view(-1, 1, 1) * eye - outer
            )
            mask = attached[:, idx].view(-1, 1, 1)
            inertia += point_I * mask
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
    def _randomize_ext_params(self, env_ids):
        if env_ids is None:
            env_ids = torch.arange(self.sim_env.num_envs, device=self.device, dtype=torch.long)
        num_envs = len(env_ids)
        
        # 1. External Disturbance (Wind)
        if self.randomize_ext_dist:
            # Force
            f_low, f_high = self.ext_force_range
            f_mag = torch.rand(num_envs, device=self.device) * (f_high - f_low) + f_low
            f_dir = torch.randn((num_envs, 3), device=self.device)
            f_dir = f_dir / (torch.norm(f_dir, dim=1, keepdim=True) + 1e-6)
            self.ext_forces[env_ids] = f_dir * f_mag.unsqueeze(1)
            # Torque
            t_low, t_high = self.ext_torque_range
            t_mag = torch.rand(num_envs, device=self.device) * (t_high - t_low) + t_low
            t_dir = torch.randn((num_envs, 3), device=self.device)
            t_dir = t_dir / (torch.norm(t_dir, dim=1, keepdim=True) + 1e-6)
            self.ext_torques[env_ids] = t_dir * t_mag.unsqueeze(1)
        else:
            self.ext_forces[env_ids] = 0.0
            self.ext_torques[env_ids] = 0.0

        robot = self.sim_env.robot_manager.robot

        # 2. Drag Coefficients
        if self.randomize_drag:
            # Linear Drag
            ld_low, ld_high = self.lin_drag_range
            lin_drag = torch.rand(num_envs, device=self.device) * (ld_high - ld_low) + ld_low
            robot.body_vel_linear_damping_coefficient[env_ids] = lin_drag.unsqueeze(1)
            # Angular Drag
            ad_low, ad_high = self.ang_drag_range
            ang_drag = torch.rand(num_envs, device=self.device) * (ad_high - ad_low) + ad_low
            robot.angvel_linear_damping_coefficient[env_ids] = ang_drag.unsqueeze(1)

        # 3. Motor Parameters
        model = robot.control_allocator.motor_model
        # Motor Time Constant
        if self.randomize_motor_tc:
            tc_low, tc_high = self.motor_tc_range
            tc = torch.rand(num_envs, device=self.device) * (tc_high - tc_low) + tc_low
            model.motor_time_constants_increasing[env_ids] = tc.unsqueeze(1)
            model.motor_time_constants_decreasing[env_ids] = tc.unsqueeze(1)
            
        # Motor Thrust Constant (Scale nominal)
        if self.randomize_motor_thrust_k and hasattr(model, 'motor_thrust_constant'):
            k_scale_low, k_scale_high = self.motor_thrust_k_scale
            scale = torch.rand(num_envs, device=self.device) * (k_scale_high - k_scale_low) + k_scale_low
            # Use average of min/max as nominal base
            base_k = (model.cfg.motor_thrust_constant_min + model.cfg.motor_thrust_constant_max) / 2.0
            model.motor_thrust_constant[env_ids] = base_k * scale.unsqueeze(1)

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
        # logger.info(
        #     "[DebugReset %d] env_ids=%s targets=%s release_info=%s",
        #     self._debug_reset_count,
        #     env_ids,
        #     target_samples,
        #     release_info,
        # )
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
        # 设置初始位置为目标位置 (0, 0, 0)
        single_state[0:3] = torch.tensor([0.0, 0.0, 0.0], device=self.device)
        single_state[6] = 1.0
        if env_ids is None:
            vec_root[:, 0, :] = single_state
        else:
            vec_root[env_ids.long(), 0, :] = single_state
        gym.set_actor_root_state_tensor(
            sim, gymtorch.unwrap_tensor(self.sim_env.IGE_env.unfolded_vec_root_tensor)
        )


# exp_func 已在文件开头定义（line 71），支持 use_safe_exp 参数


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
    # NaN Prevention: Use safe_exp in exp_func calls
    pos_reward = exp_func(dist, 3.0, 8.0, use_safe_exp=True) + exp_func(dist, 2.0, 4.0, use_safe_exp=True)
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



    total_reward[:] = torch.where(
        crashes > 0.0, parameter_dict["crash_penalty"] * torch.ones_like(total_reward), total_reward
    )
    return total_reward, crashes


def _reset_on_nonfinite(obs_dict, device):
    keys = (
        "robot_position",
        "robot_orientation",
        "robot_body_linvel",
        "robot_body_angvel",
    )
    bad_envs = None
    for key in keys:
        tensor = obs_dict.get(key)
        if tensor is None:
            continue
        invalid = ~torch.isfinite(tensor)
        if not invalid.any():
            continue
        invalid_envs = invalid.reshape(invalid.shape[0], -1).any(dim=1)
        bad_envs = invalid_envs if bad_envs is None else (bad_envs | invalid_envs)
    if bad_envs is None:
        return
    if bad_envs.any():
        crashes = obs_dict.get("crashes")
        if crashes is not None:
            crashes[bad_envs] = True
