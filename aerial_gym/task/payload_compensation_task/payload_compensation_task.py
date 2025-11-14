from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import torch
from isaacgym import gymapi, gymtorch

from aerial_gym.task.base_task import BaseTask
from aerial_gym.sim.sim_builder import SimBuilder
from aerial_gym.utils.logging import CustomLogger
from aerial_gym.utils.math import quat_apply_inverse, quat_axis, quat_rotate_inverse

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
        self.step_counter[env_id] = 0
        if attach_row.any():
            self.next_release_step[env_id] = self._sample_interval()
        else:
            self.next_release_step[env_id] = torch.iinfo(torch.int64).max

        self._update_mass_properties(torch.tensor([env_id], device=self.device, dtype=torch.long))

    def _assign_random_release_order(self, env_ids: torch.Tensor):
        for env_id in env_ids.long().tolist():
            self.release_orders[env_id] = torch.randperm(
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
        if bounds is None:
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
        )
        self.payload_manager = PayloadManager(self.sim_env, payload_cfg)
        self.payload_manager.reset()
        self._patch_pre_physics_step()
        self._initialize_vehicle_state()

        self.counter = 0

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
        return self.get_return_tuple()

    def reset_idx(self, env_ids):
        self.target_position[:, 0:3] = 0.0
        env_tensor = torch.as_tensor(env_ids, device=self.device)
        self.payload_manager.reset(env_ids=env_tensor)
        self.sim_env.reset_idx(env_ids)
        self._initialize_vehicle_state(env_ids=env_tensor)

    def render(self):
        return None

    def step(self, actions):
        self.counter += 1
        self.prev_actions[:] = self.actions
        self.actions = actions

        self.payload_manager.step()

        self.controller_actions[:, 0:3] = self.target_position
        self.controller_actions[:, 3] = 0.0
        self.controller_actions[:, 4:] = torch.clamp(self.actions, -1.0, 1.0)

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
        )
        self.rewards[:] = base_rewards
        self.rewards += self._compute_payload_penalties()

        self.truncations[:] = torch.where(
            self.sim_env.sim_steps > self.task_config.episode_len_steps, 1, 0
        )
        self.sim_env.post_reward_calculation_step()

        self.infos = {
            "last_release_index": self.payload_manager.last_release_index.clone().detach().cpu(),
            "just_released": self.payload_manager.just_released_flag.clone().detach().cpu(),
        }
        return self.get_return_tuple()

    def _compute_payload_penalties(self):
        reward_cfg = self.task_config.reward_parameters
        euler = self.obs_dict["robot_euler_angles"]
        roll_pitch_error = torch.norm(euler[:, 0:2], dim=1)
        base_penalty = -reward_cfg["attitude_penalty_coef"] * roll_pitch_error

        release_multiplier = torch.ones_like(base_penalty)
        release_multiplier = torch.where(
            self.payload_manager.just_released_flag,
            release_multiplier * reward_cfg["release_attitude_boost"],
            release_multiplier,
        )
        attitude_term = base_penalty * release_multiplier

        comp_penalty = -reward_cfg["comp_torque_penalty_coef"] * torch.sum(
            self.actions**2, dim=1
        )

        return attitude_term + comp_penalty

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
        self.task_obs["observations"][:, 0:3] = (
            self.target_position - self.obs_dict["robot_position"]
        )
        self.task_obs["observations"][:, 3:7] = self.obs_dict["robot_orientation"]
        self.task_obs["observations"][:, 7:10] = self.obs_dict["robot_body_linvel"]
        self.task_obs["observations"][:, 10:13] = self.obs_dict["robot_body_angvel"]

        payload_obs = self.payload_manager.get_observation_features()
        idx = 13
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

        self.task_obs["rewards"] = self.rewards
        self.task_obs["terminations"] = self.terminations
        self.task_obs["truncations"] = self.truncations

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

    crashes[:] = torch.where(dist > 8.0, torch.ones_like(crashes), crashes)
    total_reward[:] = torch.where(
        crashes > 0.0, parameter_dict["crash_penalty"] * torch.ones_like(total_reward), total_reward
    )
    return total_reward, crashes
