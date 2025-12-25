from dataclasses import dataclass
from typing import Optional, Sequence

import atexit
import csv
import json
import os

import numpy as np
import torch
from isaacgym import gymapi, gymtorch

from aerial_gym.task.base_task import BaseTask
from aerial_gym.sim.sim_builder import SimBuilder
from aerial_gym.utils.logging import CustomLogger
from aerial_gym.utils.math import (
    quat_apply_inverse,
    quat_axis,
    quat_from_euler_xyz_tensor,
    quat_mul,
)
from gym.spaces import Dict, Box

logger = CustomLogger("payload_estimation_task")


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
    payload_mass_range: Optional[Sequence[float]] = None
    randomize_payload_mass: bool = False
    payload_count_range: Optional[Sequence[int]] = None
    randomize_payload_count: bool = False
    randomize_offsets_on_plane: bool = False
    offset_plane_radial_jitter: float = 0.0
    offset_plane_z_jitter: float = 0.0


class PayloadRandomizer:
    """Randomize payload mass/count and update rigid body mass/inertia."""

    def __init__(self, env_manager, payload_cfg: PayloadConfig):
        self.env_manager = env_manager
        self.device = env_manager.device
        self.num_envs = env_manager.num_envs
        self.gym = env_manager.IGE_env.gym
        self.sim = env_manager.IGE_env.sim
        self.env_handles = env_manager.IGE_env.env_handles
        self.robot_handles = env_manager.robot_manager.robot_handles

        self.cfg = payload_cfg
        self.payload_mass = float(payload_cfg.payload_mass)
        self.payload_mass_range = payload_cfg.payload_mass_range
        self.randomize_payload_mass = bool(payload_cfg.randomize_payload_mass)
        self.payload_count_range = payload_cfg.payload_count_range
        self.randomize_payload_count = bool(payload_cfg.randomize_payload_count)

        self.base_offsets = torch.as_tensor(
            np.array(payload_cfg.offsets, dtype=np.float32), device=self.device
        )
        self.num_payloads = self.base_offsets.shape[0]
        self.offsets = self.base_offsets.unsqueeze(0).expand(self.num_envs, -1, -1).clone()
        self.randomize_offsets_on_plane = bool(payload_cfg.randomize_offsets_on_plane)
        self.offset_plane_radial_jitter = float(payload_cfg.offset_plane_radial_jitter)
        self.offset_plane_z_jitter = float(payload_cfg.offset_plane_z_jitter)

        self.attached_mask = torch.ones(
            (self.num_envs, self.num_payloads), dtype=torch.bool, device=self.device
        )
        self.payload_mass_per_env = torch.full(
            (self.num_envs,), self.payload_mass, device=self.device
        )
        self.current_payload_mass = torch.zeros(self.num_envs, device=self.device)
        self.com_offset_body = torch.zeros((self.num_envs, 3), device=self.device)

        self.actor_props = []
        self.base_mass = torch.zeros(self.num_envs, device=self.device)
        self.base_inertia = torch.zeros((self.num_envs, 3, 3), device=self.device)
        self._cache_rigid_body_props()

    def _cache_rigid_body_props(self):
        for env_id in range(self.num_envs):
            env_handle = self.env_handles[env_id]
            robot_handle = self.robot_handles[env_id]
            props = self.gym.get_actor_rigid_body_properties(env_handle, robot_handle)
            self.actor_props.append(props)
            base_prop = props[0]
            self.base_mass[env_id] = base_prop.mass
            self.base_inertia[env_id] = torch.from_numpy(_mat33_to_np(base_prop.inertia))

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

    def _normalized_count_range(self):
        if not self.randomize_payload_count or not self.payload_count_range:
            return None
        low, high = self.payload_count_range
        low = int(max(0, min(low, self.num_payloads)))
        high = int(max(0, min(high, self.num_payloads)))
        if low > high:
            low, high = high, low
        return low, high

    def _sample_payload_mask(self, env_ids: torch.Tensor):
        count_range = self._normalized_count_range()
        if count_range is None:
            self.attached_mask[env_ids] = True
            return
        low, high = count_range
        for env_id in env_ids.long().tolist():
            count = int(torch.randint(low, high + 1, (1,), device=self.device).item())
            mask = torch.zeros(self.num_payloads, dtype=torch.bool, device=self.device)
            if count > 0:
                perm = torch.randperm(self.num_payloads, device=self.device)
                mask[perm[:count]] = True
            self.attached_mask[env_id] = mask

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

        for env_id in env_ids.long().tolist():
            radial_delta = (
                (torch.rand((self.num_payloads, 1), device=self.device) * 2.0 - 1.0)
                * self.offset_plane_radial_jitter
            )
            z_delta = (
                (torch.rand((self.num_payloads, 1), device=self.device) * 2.0 - 1.0)
                * self.offset_plane_z_jitter
            )
            r = (r0 + radial_delta).clamp(min=1e-4)
            z = z0 + z_delta
            xy_new = dir_xy * r
            self.offsets[env_id] = torch.cat([xy_new, z], dim=1)

    def reset(self, env_ids: Optional[torch.Tensor] = None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = env_ids.to(self.device).long()
        if env_ids.numel() == 0:
            return
        self._sample_payload_mass(env_ids)
        self._sample_payload_mask(env_ids)
        self._randomize_offsets_on_plane(env_ids)
        self._update_mass_properties(env_ids)

    def _update_mass_properties(self, env_ids: torch.Tensor):
        env_id_list = env_ids.long().tolist()
        for env_id in env_id_list:
            attached = self.attached_mask[env_id]
            payload_count = int(attached.sum().item())
            payload_mass = float(self.payload_mass_per_env[env_id].item())
            payload_mass_sum = payload_count * payload_mass
            self.current_payload_mass[env_id] = payload_mass_sum

            base_mass = float(self.base_mass[env_id].item())
            total_mass = base_mass + payload_mass_sum
            inertia_np = self.base_inertia[env_id].cpu().numpy().copy()
            weighted_offset = np.zeros(3, dtype=np.float32)

            for idx, attached_flag in enumerate(attached.tolist()):
                if not attached_flag:
                    continue
                offset = self.offsets[env_id, idx].cpu().numpy()
                inertia_np += point_mass_inertia(payload_mass, offset)
                weighted_offset += payload_mass * offset

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
                self.env_handles[env_id],
                self.robot_handles[env_id],
                props,
                recomputeInertia=False,
            )


class PayloadEstimationTask(BaseTask):
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
        reward_params = getattr(self.task_config, "reward_parameters", None)
        if isinstance(reward_params, dict):
            for key, value in reward_params.items():
                reward_params[key] = torch.tensor(value, device=self.device)

        logger.info("Building environment for payload estimation task.")
        logger.info(
            "\nSim Name: {},\nEnv Name: {},\nRobot Name: {}, \nController Name: {}".format(
                self.task_config.sim_name,
                self.task_config.env_name,
                self.task_config.robot_name,
                self.task_config.controller_name,
            )
        )

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
        self.counter = 0

        self.target_position = torch.zeros(
            (self.sim_env.num_envs, 3), device=self.device, requires_grad=False
        )
        self.target_yaw = torch.zeros(
            (self.sim_env.num_envs,), device=self.device, requires_grad=False
        )

        self.obs_dict = self.sim_env.get_obs()
        self.terminations = self.obs_dict["crashes"]
        self.truncations = self.obs_dict["truncations"]
        self.rewards = torch.zeros(self.truncations.shape[0], device=self.device)

        self.observation_space_dim = self.task_config.observation_space_dim
        self.action_space_dim = self.task_config.action_space_dim

        self.observation_space = Dict(
            {
                "observations": Box(
                    low=-1.0,
                    high=1.0,
                    shape=(self.observation_space_dim,),
                    dtype=np.float32,
                )
            }
        )
        self.action_space = Box(
            low=-1.0,
            high=1.0,
            shape=(self.action_space_dim,),
            dtype=np.float32,
        )

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
            payload_mass_range=self.task_config.payload_parameters.get("payload_mass_range"),
            randomize_payload_mass=self.task_config.payload_parameters.get(
                "randomize_payload_mass", False
            ),
            payload_count_range=self.task_config.payload_parameters.get("payload_count_range"),
            randomize_payload_count=self.task_config.payload_parameters.get(
                "randomize_payload_count", False
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
        )
        self.payload_manager = PayloadRandomizer(self.sim_env, payload_cfg)
        self.payload_manager.reset()

        rand_cfg = getattr(self.task_config, "randomization_parameters", None) or {}
        if not isinstance(rand_cfg, dict):
            rand_cfg = {}
        self.initial_position_noise = torch.tensor(
            rand_cfg.get("initial_position_noise", [0.0, 0.0, 0.0]), device=self.device
        )
        self.initial_orientation_noise = torch.tensor(
            rand_cfg.get("initial_orientation_noise_deg", [0.0, 0.0, 0.0]),
            device=self.device,
        )
        self.initial_orientation_noise = self.initial_orientation_noise * np.pi / 180.0
        self.target_position_range = rand_cfg.get("target_position_range")
        self.current_target_range_tensor = (
            torch.tensor(self.target_position_range, device=self.device, dtype=torch.float32)
            if self.target_position_range is not None
            else None
        )

        self._init_excitation_trajectory()
        self._init_data_logger()
        self._initialize_vehicle_state()

    def _init_data_logger(self):
        cfg = getattr(self.task_config, "data_collection_parameters", None) or {}
        if not isinstance(cfg, dict) or not cfg.get("enable", False):
            self._log_file = None
            self._log_writer = None
            return
        path = cfg.get("log_path", "logs/payload_estimation_data.csv")
        log_dir = os.path.dirname(path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        self._log_all_envs = bool(cfg.get("log_all_envs", False))
        self._log_env_id = int(cfg.get("log_env_id", 0))
        self._log_interval = int(cfg.get("log_interval", 1))
        self._log_max_steps = int(cfg.get("max_steps", 0))
        self._log_flush_every = int(cfg.get("flush_every", 200))
        self._log_rows = 0
        self._log_episode_id = np.zeros(self.sim_env.num_envs, dtype=np.int64)
        self._log_last_episode_step = np.full(self.sim_env.num_envs, -1, dtype=np.int64)
        self._log_file = open(path, "w", newline="")
        self._log_writer = csv.writer(self._log_file)
        self._log_num_motors = 0
        self._log_use_rps = False
        robot = getattr(self.sim_env.robot_manager, "robot", None)
        if robot is not None:
            allocator = getattr(robot, "control_allocator", None)
            if allocator is not None:
                self._log_num_motors = int(getattr(allocator.cfg, "num_motors", 0))
                self._log_use_rps = bool(getattr(allocator.motor_model.cfg, "use_rps", False))

        header = [
            "row_type",
            "nominal_json",
            "env_id",
            "step",
            "episode_id",
            "episode_step",
            "time",
            "dt",
            "episode_progress",
            "target_x",
            "target_y",
            "target_z",
            "target_yaw",
            "pos_x",
            "pos_y",
            "pos_z",
            "quat_x",
            "quat_y",
            "quat_z",
            "quat_w",
            "euler_roll",
            "euler_pitch",
            "euler_yaw",
            "linvel_x",
            "linvel_y",
            "linvel_z",
            "angvel_x",
            "angvel_y",
            "angvel_z",
            "body_linvel_x",
            "body_linvel_y",
            "body_linvel_z",
            "body_angvel_x",
            "body_angvel_y",
            "body_angvel_z",
            "vehicle_quat_x",
            "vehicle_quat_y",
            "vehicle_quat_z",
            "vehicle_quat_w",
            "vehicle_linvel_x",
            "vehicle_linvel_y",
            "vehicle_linvel_z",
            "imu_acc_x",
            "imu_acc_y",
            "imu_acc_z",
            "imu_gyro_x",
            "imu_gyro_y",
            "imu_gyro_z",
            "action_x",
            "action_y",
            "action_z",
            "action_yaw",
            "command_mode",
        ]
        if self._log_num_motors > 0:
            header.extend([f"motor_thrust_{i}" for i in range(self._log_num_motors)])
            if self._log_use_rps:
                header.extend([f"motor_rps_{i}" for i in range(self._log_num_motors)])
                header.extend([f"motor_rpm_{i}" for i in range(self._log_num_motors)])
        header.extend(
            [
                "label_payload_mass",
                "label_payload_mass_each",
                "label_payload_count",
                "label_total_mass",
                "label_com_offset_x",
                "label_com_offset_y",
                "label_com_offset_z",
                "label_inertia_xx",
                "label_inertia_xy",
                "label_inertia_xz",
                "label_inertia_yy",
                "label_inertia_yz",
                "label_inertia_zz",
            ]
        )
        for idx in range(self.payload_manager.num_payloads):
            header.extend(
                [
                    f"label_payload_attached_{idx}",
                    f"label_offset_{idx}_x",
                    f"label_offset_{idx}_y",
                    f"label_offset_{idx}_z",
                ]
            )
        self._log_header_len = len(header)
        self._log_writer.writerow(header)
        self._write_nominal_info()
        logger.info("[DataLog] enabled path=%s (env=%d)", path, self._log_env_id)
        atexit.register(self._close_data_logger)

    def _write_nominal_info(self):
        if self._log_writer is None:
            return
        env_id = 0
        base_mass = float(self.payload_manager.base_mass[env_id].item())
        base_inertia = self.payload_manager.base_inertia[env_id].cpu().numpy().tolist()
        gravity = self.sim_env.IGE_env.global_tensor_dict["gravity"][0].detach().cpu().numpy().tolist()
        sim_dt = float(self.sim_env.sim_config.sim.dt)
        offsets = self.payload_manager.base_offsets.detach().cpu().numpy().tolist()

        robot = getattr(self.sim_env.robot_manager, "robot", None)
        allocator = getattr(robot, "control_allocator", None) if robot is not None else None
        motor_cfg = None
        alloc_cfg = None
        if allocator is not None:
            alloc_cfg = allocator.cfg
            motor_cfg = allocator.motor_model.cfg

        imu_cfg = None
        if robot is not None and getattr(robot.cfg.sensor_config, "enable_imu", False):
            imu_cfg = robot.cfg.sensor_config.imu_config

        nominal = {
            "task_name": "payload_estimation_task",
            "sim_name": self.task_config.sim_name,
            "env_name": self.task_config.env_name,
            "robot_name": self.task_config.robot_name,
            "controller_name": self.task_config.controller_name,
            "episode_len_steps": int(self.task_config.episode_len_steps),
            "dt": sim_dt,
            "gravity": gravity,
            "base_mass": base_mass,
            "base_inertia": base_inertia,
            "payload_offsets": offsets,
            "payload_mass_default": float(self.payload_manager.payload_mass),
            "payload_mass_range": self.payload_manager.payload_mass_range,
            "payload_count_range": self.payload_manager.payload_count_range,
            "randomize_payload_mass": self.payload_manager.randomize_payload_mass,
            "randomize_payload_count": self.payload_manager.randomize_payload_count,
            "randomize_offsets_on_plane": self.payload_manager.randomize_offsets_on_plane,
            "offset_plane_radial_jitter": self.payload_manager.offset_plane_radial_jitter,
            "offset_plane_z_jitter": self.payload_manager.offset_plane_z_jitter,
            "target_position_range": self.target_position_range,
            "trajectory_parameters": self._trajectory_nominal_info,
        }
        if alloc_cfg is not None:
            nominal["allocation_matrix"] = alloc_cfg.allocation_matrix
            nominal["motor_directions"] = alloc_cfg.motor_directions
            nominal["force_application_level"] = alloc_cfg.force_application_level
        if motor_cfg is not None:
            nominal["motor_model"] = {
                "use_rps": bool(getattr(motor_cfg, "use_rps", False)),
                "thrust_to_torque_ratio": float(motor_cfg.thrust_to_torque_ratio),
                "motor_thrust_constant_min": float(motor_cfg.motor_thrust_constant_min),
                "motor_thrust_constant_max": float(motor_cfg.motor_thrust_constant_max),
                "motor_time_constant_increasing_min": float(motor_cfg.motor_time_constant_increasing_min),
                "motor_time_constant_increasing_max": float(motor_cfg.motor_time_constant_increasing_max),
                "motor_time_constant_decreasing_min": float(motor_cfg.motor_time_constant_decreasing_min),
                "motor_time_constant_decreasing_max": float(motor_cfg.motor_time_constant_decreasing_max),
                "max_thrust": float(motor_cfg.max_thrust),
                "min_thrust": float(motor_cfg.min_thrust),
                "max_thrust_rate": float(motor_cfg.max_thrust_rate),
                "use_discrete_approximation": bool(motor_cfg.use_discrete_approximation),
            }
        if imu_cfg is not None:
            nominal["imu_config"] = {
                "gravity_compensation": bool(getattr(imu_cfg, "gravity_compensation", False)),
                "world_frame": bool(getattr(imu_cfg, "world_frame", True)),
                "bias_std": list(getattr(imu_cfg, "bias_std", [])),
                "imu_noise_std": list(getattr(imu_cfg, "imu_noise_std", [])),
                "max_measurement_value": list(getattr(imu_cfg, "max_measurement_value", [])),
                "max_bias_init_value": list(getattr(imu_cfg, "max_bias_init_value", [])),
                "min_euler_rotation_deg": list(getattr(imu_cfg, "min_euler_rotation_deg", [])),
                "max_euler_rotation_deg": list(getattr(imu_cfg, "max_euler_rotation_deg", [])),
            }

        nominal_json = json.dumps(nominal, separators=(",", ":"), ensure_ascii=True)
        row = ["nominal", nominal_json] + [""] * (self._log_header_len - 2)
        self._log_writer.writerow(row)

    def _close_data_logger(self):
        if self._log_file is None:
            return
        self._log_file.flush()
        self._log_file.close()
        self._log_file = None
        self._log_writer = None

    def _log_step_data(self):
        if self._log_writer is None:
            return
        if self._log_interval > 1 and (self.counter % self._log_interval) != 0:
            return
        if self._log_max_steps > 0 and self._log_rows >= self._log_max_steps:
            return
        if self._log_all_envs:
            env_ids = range(self.sim_env.num_envs)
        else:
            env_id = self._log_env_id
            if env_id < 0 or env_id >= self.sim_env.num_envs:
                return
            env_ids = [env_id]
        dt_tensor = self.obs_dict.get("dt")
        if torch.is_tensor(dt_tensor):
            dt = float(dt_tensor.item())
        elif isinstance(dt_tensor, (float, int)):
            dt = float(dt_tensor)
        else:
            dt = 0.0
        sim_time = float(self.counter) * dt
        target_np = self.target_position.detach().cpu().numpy()
        target_yaw_np = self.target_yaw.detach().cpu().numpy()
        pos_np = self.obs_dict["robot_position"].detach().cpu().numpy()
        quat_np = self.obs_dict["robot_orientation"].detach().cpu().numpy()
        euler_np = self.obs_dict["robot_euler_angles"].detach().cpu().numpy()
        linvel_np = self.obs_dict["robot_linvel"].detach().cpu().numpy()
        angvel_np = self.obs_dict["robot_angvel"].detach().cpu().numpy()
        body_linvel_np = self.obs_dict["robot_body_linvel"].detach().cpu().numpy()
        body_angvel_np = self.obs_dict["robot_body_angvel"].detach().cpu().numpy()
        vehicle_quat_np = self.obs_dict["robot_vehicle_orientation"].detach().cpu().numpy()
        vehicle_linvel_np = self.obs_dict["robot_vehicle_linvel"].detach().cpu().numpy()

        imu = self.obs_dict.get("imu_measurement")
        if imu is None:
            imu_np = np.full((self.sim_env.num_envs, 6), np.nan, dtype=np.float32)
        else:
            imu_np = imu.detach().cpu().numpy().astype(np.float32)
        actions_np = self.actions.detach().cpu().numpy().astype(np.float32)

        motor_thrust_np = None
        motor_rps_np = None
        motor_rpm_np = None
        if self._log_num_motors > 0:
            allocator = getattr(self.sim_env.robot_manager.robot, "control_allocator", None)
            motor_model = getattr(allocator, "motor_model", None) if allocator is not None else None
            if motor_model is not None:
                thrust = motor_model.current_motor_thrust
                motor_thrust_np = thrust.detach().cpu().numpy().astype(np.float32)
                if self._log_use_rps:
                    kf = motor_model.motor_thrust_constant
                    rps = torch.sqrt(torch.clamp(thrust, min=0.0) / kf)
                    motor_rps_np = rps.detach().cpu().numpy().astype(np.float32)
                    motor_rpm_np = motor_rps_np * 60.0
            else:
                motor_thrust_np = np.full(
                    (self.sim_env.num_envs, self._log_num_motors), np.nan, dtype=np.float32
                )
                if self._log_use_rps:
                    motor_rps_np = np.full_like(motor_thrust_np, np.nan)
                    motor_rpm_np = np.full_like(motor_thrust_np, np.nan)

        payload_mass_np = self.payload_manager.current_payload_mass.detach().cpu().numpy()
        payload_mass_each_np = self.payload_manager.payload_mass_per_env.detach().cpu().numpy()
        attached_np = self.payload_manager.attached_mask.detach().cpu().numpy().astype(bool)
        offsets_np = self.payload_manager.offsets.detach().cpu().numpy()
        com_offset_np = self.payload_manager.com_offset_body.detach().cpu().numpy()
        base_mass_np = self.payload_manager.base_mass.detach().cpu().numpy()
        base_inertia_np = self.payload_manager.base_inertia.detach().cpu().numpy()

        for env_id in env_ids:
            if self._log_max_steps > 0 and self._log_rows >= self._log_max_steps:
                break
            episode_step = int(self.sim_env.sim_steps[env_id].item())
            if episode_step < self._log_last_episode_step[env_id]:
                self._log_episode_id[env_id] += 1
            self._log_last_episode_step[env_id] = episode_step
            episode_id = int(self._log_episode_id[env_id])
            episode_progress = float(episode_step) / float(
                max(1, self.task_config.episode_len_steps)
            )

            payload_mass = float(payload_mass_np[env_id])
            payload_mass_each = float(payload_mass_each_np[env_id])
            payload_count = int(attached_np[env_id].sum())
            target = target_np[env_id]
            target_yaw = float(target_yaw_np[env_id])
            pos = pos_np[env_id]
            quat = quat_np[env_id]
            euler = euler_np[env_id]
            linvel = linvel_np[env_id]
            angvel = angvel_np[env_id]
            body_linvel = body_linvel_np[env_id]
            body_angvel = body_angvel_np[env_id]
            vehicle_quat = vehicle_quat_np[env_id]
            vehicle_linvel = vehicle_linvel_np[env_id]
            imu_vals = imu_np[env_id].tolist()
            actions = actions_np[env_id]

            row = [
                "data",
                "",
                int(env_id),
                int(self.counter),
                int(episode_id),
                int(episode_step),
                sim_time,
                dt,
                episode_progress,
                float(target[0]),
                float(target[1]),
                float(target[2]),
                target_yaw,
                float(pos[0]),
                float(pos[1]),
                float(pos[2]),
                float(quat[0]),
                float(quat[1]),
                float(quat[2]),
                float(quat[3]),
                float(euler[0]),
                float(euler[1]),
                float(euler[2]),
                float(linvel[0]),
                float(linvel[1]),
                float(linvel[2]),
                float(angvel[0]),
                float(angvel[1]),
                float(angvel[2]),
                float(body_linvel[0]),
                float(body_linvel[1]),
                float(body_linvel[2]),
                float(body_angvel[0]),
                float(body_angvel[1]),
                float(body_angvel[2]),
                float(vehicle_quat[0]),
                float(vehicle_quat[1]),
                float(vehicle_quat[2]),
                float(vehicle_quat[3]),
                float(vehicle_linvel[0]),
                float(vehicle_linvel[1]),
                float(vehicle_linvel[2]),
            ]
            row.extend(imu_vals)
            row.extend(actions.tolist())
            row.append(1 if self.use_excitation_trajectory else 0)
            if motor_thrust_np is not None:
                row.extend(motor_thrust_np[env_id].tolist())
            if motor_rps_np is not None:
                row.extend(motor_rps_np[env_id].tolist())
                row.extend(motor_rpm_np[env_id].tolist())
            total_mass = float(payload_mass + base_mass_np[env_id])
            com_offset = com_offset_np[env_id]
            inertia_np = base_inertia_np[env_id].copy()
            attached = attached_np[env_id]
            offsets = offsets_np[env_id]
            for offset in offsets[attached]:
                inertia_np += point_mass_inertia(payload_mass_each, offset)
            row.extend(
                [
                    payload_mass,
                    payload_mass_each,
                    payload_count,
                    total_mass,
                    float(com_offset[0]),
                    float(com_offset[1]),
                    float(com_offset[2]),
                    float(inertia_np[0, 0]),
                    float(inertia_np[0, 1]),
                    float(inertia_np[0, 2]),
                    float(inertia_np[1, 1]),
                    float(inertia_np[1, 2]),
                    float(inertia_np[2, 2]),
                ]
            )
            for attached_flag, offset in zip(attached.tolist(), offsets):
                row.extend(
                    [
                        int(attached_flag),
                        float(offset[0]),
                        float(offset[1]),
                        float(offset[2]),
                    ]
                )
            self._log_writer.writerow(row)
            self._log_rows += 1
            if self._log_flush_every > 0 and (self._log_rows % self._log_flush_every) == 0:
                self._log_file.flush()

    def close(self):
        self._close_data_logger()
        if hasattr(self, "sim_builder") and self.sim_builder is not None:
            self.sim_builder.delete_env()
        elif hasattr(self, "sim_env") and self.sim_env is not None:
            delete_fn = getattr(self.sim_env, "delete_env", None)
            if callable(delete_fn):
                delete_fn()

    def reset(self):
        self.infos = {}
        self.payload_manager.reset()
        self.sim_env.reset()
        self._refresh_env_state(env_ids=None)
        return self.get_return_tuple()

    def reset_idx(self, env_ids):
        env_tensor = self._get_env_tensor(env_ids)
        self.payload_manager.reset(env_ids=env_tensor)
        self.sim_env.reset_idx(env_ids)
        self._refresh_env_state(env_ids=env_tensor)

    def _get_env_tensor(self, env_ids=None):
        if env_ids is None:
            return torch.arange(self.sim_env.num_envs, device=self.device, dtype=torch.long)
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(self.device).long()
        return torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

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

    def _normalize_range(self, value, default_min, default_max):
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            low = float(value[0])
            high = float(value[1])
        elif isinstance(value, (int, float)):
            low = float(value)
            high = float(value)
        else:
            low = float(default_min)
            high = float(default_max)
        if low > high:
            low, high = high, low
        return low, high

    def _as_vec3(self, value, default):
        if isinstance(value, (int, float)):
            return [float(value)] * 3
        if isinstance(value, (list, tuple)) and len(value) == 3:
            return [float(value[0]), float(value[1]), float(value[2])]
        if isinstance(value, (list, tuple)) and len(value) == 1:
            return [float(value[0])] * 3
        return [float(default)] * 3

    def _init_excitation_trajectory(self):
        cfg = getattr(self.task_config, "trajectory_parameters", None) or {}
        if not isinstance(cfg, dict):
            cfg = {}
        self.use_excitation_trajectory = bool(cfg.get("enable", False))
        self.trajectory_mode = cfg.get("mode", "sine")
        position_range = cfg.get("position_range")
        if position_range is None:
            position_range = self.target_position_range
        self._trajectory_range = position_range
        if position_range is not None:
            self._trajectory_range_tensor = torch.tensor(
                position_range, device=self.device, dtype=torch.float32
            )
        else:
            self._trajectory_range_tensor = None

        amp_scale = self._as_vec3(cfg.get("amplitude_scale", [0.6, 0.6, 0.4]), 0.5)
        self._trajectory_amp_scale = torch.tensor(
            amp_scale, device=self.device, dtype=torch.float32
        ).clamp(min=0.0, max=1.0)
        self._trajectory_cycles_min, self._trajectory_cycles_max = self._normalize_range(
            cfg.get("cycles_range", [1.0, 2.5]), 1.0, 2.5
        )
        yaw_center_range = cfg.get("yaw_center_deg_range", [0.0, 0.0])
        self._trajectory_yaw_center_deg_min, self._trajectory_yaw_center_deg_max = (
            self._normalize_range(yaw_center_range, 0.0, 0.0)
        )
        self._trajectory_yaw_amp_rad = float(np.deg2rad(cfg.get("yaw_amplitude_deg", 0.0)))
        self._trajectory_yaw_cycles_min, self._trajectory_yaw_cycles_max = self._normalize_range(
            cfg.get("yaw_cycles_range", [0.5, 1.5]), 0.5, 1.5
        )

        num_envs = self.sim_env.num_envs
        self._traj_center = torch.zeros((num_envs, 3), device=self.device)
        self._traj_amp = torch.zeros((num_envs, 3), device=self.device)
        self._traj_phase = torch.zeros((num_envs, 3), device=self.device)
        self._traj_cycles = torch.zeros((num_envs, 3), device=self.device)
        self._traj_yaw_center = torch.zeros((num_envs,), device=self.device)
        self._traj_yaw_amp = torch.zeros((num_envs,), device=self.device)
        self._traj_yaw_phase = torch.zeros((num_envs,), device=self.device)
        self._traj_yaw_cycles = torch.zeros((num_envs,), device=self.device)

        self._trajectory_nominal_info = {
            "enable": self.use_excitation_trajectory,
            "mode": self.trajectory_mode,
            "position_range": self._trajectory_range,
            "amplitude_scale": amp_scale,
            "cycles_range": [self._trajectory_cycles_min, self._trajectory_cycles_max],
            "yaw_center_deg_range": [
                self._trajectory_yaw_center_deg_min,
                self._trajectory_yaw_center_deg_max,
            ],
            "yaw_amplitude_deg": float(np.rad2deg(self._trajectory_yaw_amp_rad)),
            "yaw_cycles_range": [
                self._trajectory_yaw_cycles_min,
                self._trajectory_yaw_cycles_max,
            ],
        }

    def _sample_excitation_trajectory(self, env_ids=None):
        if not self.use_excitation_trajectory:
            return
        if self._trajectory_range_tensor is None:
            return
        env_tensor = self._get_env_tensor(env_ids)
        if env_tensor.numel() == 0:
            return

        lows = self._trajectory_range_tensor[:, 0]
        highs = self._trajectory_range_tensor[:, 1]
        span = (highs - lows).clamp(min=1e-6)
        center = torch.rand((env_tensor.shape[0], 3), device=self.device) * span + lows
        margin = torch.minimum(center - lows, highs - center).clamp(min=0.0)
        amp = margin * self._trajectory_amp_scale

        cycles_span = self._trajectory_cycles_max - self._trajectory_cycles_min
        cycles = (
            torch.rand((env_tensor.shape[0], 3), device=self.device) * cycles_span
            + self._trajectory_cycles_min
        )
        phase = torch.rand((env_tensor.shape[0], 3), device=self.device) * (2.0 * np.pi)

        self._traj_center[env_tensor] = center
        self._traj_amp[env_tensor] = amp
        self._traj_cycles[env_tensor] = cycles
        self._traj_phase[env_tensor] = phase

        yaw_center_span = self._trajectory_yaw_center_deg_max - self._trajectory_yaw_center_deg_min
        yaw_center_deg = (
            torch.rand((env_tensor.shape[0],), device=self.device) * yaw_center_span
            + self._trajectory_yaw_center_deg_min
        )
        yaw_center = torch.deg2rad(yaw_center_deg)
        if self._trajectory_yaw_amp_rad > 0.0:
            yaw_amp = (
                torch.rand((env_tensor.shape[0],), device=self.device)
                * self._trajectory_yaw_amp_rad
            )
        else:
            yaw_amp = torch.zeros((env_tensor.shape[0],), device=self.device)
        yaw_cycles_span = self._trajectory_yaw_cycles_max - self._trajectory_yaw_cycles_min
        yaw_cycles = (
            torch.rand((env_tensor.shape[0],), device=self.device) * yaw_cycles_span
            + self._trajectory_yaw_cycles_min
        )
        yaw_phase = torch.rand((env_tensor.shape[0],), device=self.device) * (2.0 * np.pi)

        self._traj_yaw_center[env_tensor] = yaw_center
        self._traj_yaw_amp[env_tensor] = yaw_amp
        self._traj_yaw_cycles[env_tensor] = yaw_cycles
        self._traj_yaw_phase[env_tensor] = yaw_phase

    def _update_target_from_trajectory(self, env_ids=None):
        if not self.use_excitation_trajectory:
            return
        if self._trajectory_range_tensor is None:
            return
        env_tensor = self._get_env_tensor(env_ids)
        if env_tensor.numel() == 0:
            return
        steps = self.sim_env.sim_steps[env_tensor].float()
        denom = float(max(1, self.task_config.episode_len_steps))
        t = torch.clamp(steps / denom, 0.0, 1.0)
        angles = (
            2.0 * np.pi * self._traj_cycles[env_tensor] * t.unsqueeze(1)
            + self._traj_phase[env_tensor]
        )
        pos = self._traj_center[env_tensor] + self._traj_amp[env_tensor] * torch.sin(angles)
        lows = self._trajectory_range_tensor[:, 0]
        highs = self._trajectory_range_tensor[:, 1]
        pos = torch.max(torch.min(pos, highs), lows)
        self.target_position[env_tensor] = pos

        yaw_angles = (
            2.0 * np.pi * self._traj_yaw_cycles[env_tensor] * t
            + self._traj_yaw_phase[env_tensor]
        )
        yaw = self._traj_yaw_center[env_tensor] + self._traj_yaw_amp[env_tensor] * torch.sin(
            yaw_angles
        )
        self.target_yaw[env_tensor] = yaw

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

    def _refresh_env_state(self, env_ids=None):
        env_tensor = None if env_ids is None else self._get_env_tensor(env_ids)
        if env_tensor is not None and env_tensor.numel() == 0:
            return
        if env_tensor is None:
            self.target_position[:, 0:3] = 0.0
            self.target_yaw[:] = 0.0
        else:
            self.target_position[env_tensor.long(), 0:3] = 0.0
            self.target_yaw[env_tensor.long()] = 0.0
        self._initialize_vehicle_state(env_ids=env_tensor)
        if self.use_excitation_trajectory:
            self._sample_excitation_trajectory(env_ids=env_tensor)
            self._update_target_from_trajectory(env_ids=env_tensor)
        else:
            self._randomize_target_positions(env_ids=env_tensor)
        self._apply_initial_state_noise(env_ids=env_tensor)

    def render(self):
        return None

    def step(self, actions):
        if not torch.is_tensor(actions):
            actions = torch.as_tensor(actions, device=self.device, dtype=torch.float32)
        else:
            if actions.device != self.device:
                actions = actions.to(self.device)
            if actions.dtype != torch.float32:
                actions = actions.float()
        self.counter += 1
        self.prev_actions[:] = self.actions
        if self.use_excitation_trajectory:
            self._update_target_from_trajectory()
            self.actions[:, 0:3] = self.target_position
            self.actions[:, 3] = self.target_yaw
        else:
            self.actions[:] = actions
            self.target_yaw[:] = self.actions[:, 3]

        self.sim_env.step(actions=self.actions)
        self.rewards[:], self.terminations[:] = self.compute_rewards_and_crashes(self.obs_dict)

        if self.task_config.return_state_before_reset:
            return_tuple = self.get_return_tuple()

        self.truncations[:] = torch.where(
            self.sim_env.sim_steps > self.task_config.episode_len_steps, 1, 0
        )
        self._log_step_data()
        envs_to_reset = self.sim_env.post_reward_calculation_step()
        if torch.is_tensor(envs_to_reset) and envs_to_reset.numel() > 0:
            self.payload_manager.reset(envs_to_reset)
            self._refresh_env_state(env_ids=envs_to_reset)
        self.infos = {}

        if not self.task_config.return_state_before_reset:
            return_tuple = self.get_return_tuple()
        return return_tuple

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
        idx = 0
        self.task_obs["observations"][:, idx : idx + 3] = (
            self.target_position - self.obs_dict["robot_position"]
        )
        idx += 3
        imu = self.obs_dict.get("imu_measurement")
        if imu is not None:
            self.task_obs["observations"][:, idx : idx + 6] = imu
        else:
            self.task_obs["observations"][:, idx : idx + 6] = 0.0
        idx += 6
        self.task_obs["observations"][:, idx : idx + 3] = self.obs_dict["robot_body_linvel"]
        idx += 3
        self.task_obs["observations"][:, idx : idx + 3] = self.obs_dict["robot_body_angvel"]
        self.task_obs["rewards"] = self.rewards
        self.task_obs["terminations"] = self.terminations
        self.task_obs["truncations"] = self.truncations

    def compute_rewards_and_crashes(self, obs_dict):
        robot_position = obs_dict["robot_position"]
        target_position = self.target_position
        robot_linvel = obs_dict["robot_linvel"]
        robot_vehicle_orientation = obs_dict["robot_vehicle_orientation"]
        robot_orientation = obs_dict["robot_orientation"]

        target_orientation = torch.zeros_like(robot_orientation, device=self.device)
        target_orientation[:, 3] = 1.0

        angular_velocity = obs_dict["robot_body_angvel"]
        root_quats = obs_dict["robot_orientation"]

        pos_error_vehicle_frame = quat_apply_inverse(
            robot_vehicle_orientation, (target_position - robot_position)
        )

        return compute_reward(
            pos_error_vehicle_frame,
            robot_linvel,
            root_quats,
            angular_velocity,
            obs_dict["crashes"],
            1.0,
            self.actions,
            self.prev_actions,
            self.task_config.reward_parameters,
        )


@torch.jit.script
def exp_func(x, gain, exp):
    # type: (Tensor, float, float) -> Tensor
    return gain * torch.exp(-exp * x * x)


@torch.jit.script
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
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor, Tensor, Dict[str, Tensor]) -> Tuple[Tensor, Tensor]
    dist = torch.norm(pos_error, dim=1)
    pos_reward = exp_func(dist, 3.0, 8.0) + exp_func(dist, 2.0, 4.0)
    dist_reward = (20 - dist) / 40.0

    ups = quat_axis(robot_quats, 2)
    tiltage = torch.abs(1 - ups[..., 2])
    up_reward = 0.2 / (0.1 + tiltage * tiltage)

    spinnage = torch.norm(robot_angvels, dim=1)
    ang_vel_reward = (1.0 / (1.0 + spinnage * spinnage)) * 3

    total_reward = pos_reward + dist_reward + pos_reward * (up_reward + ang_vel_reward)
    total_reward[:] = curriculum_level_multiplier * total_reward

    crashes[:] = torch.where(dist > 8.0, torch.ones_like(crashes), crashes)
    total_reward[:] = torch.where(
        crashes > 0.0, -20 * torch.ones_like(total_reward), total_reward
    )
    return total_reward, crashes
