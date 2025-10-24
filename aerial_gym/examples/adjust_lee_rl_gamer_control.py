"""Tune Lee controller gains with rl-games PPO and multi-env Isaac Gym."""

import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import sys

# 确保 PyTorch 在 Isaac Gym 之后导入，避免 gymdeps 检查失败
from isaacgym import gymapi  # noqa: F401

import gym
from gym import spaces
import numpy as np
import torch

# 兼容旧版依赖（如 networkx）仍引用 NumPy 早期别名的情况
if not hasattr(np, "int"):  # pragma: no cover - 仅针对旧依赖补丁
    np.int = int  # type: ignore[attr-defined]
if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]

from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import AlgoObserver
from rl_games.torch_runner import Runner

from aerial_gym.sim.sim_builder import SimBuilder
from aerial_gym.examples.my_position_control import adjust_motor_thrust_limits
from aerial_gym.utils.math import (
    quat_rotate,
    quat_rotate_inverse,
    get_euler_xyz_tensor,
    ssa,
)

from tqdm import tqdm

gym.logger.set_level(gym.logger.ERROR)


# ---------------------------- 常量与配置 ---------------------------- #

# -- 基础环境配置 --
SIM_NAME = "base_sim"
ENV_NAME = "empty_env"
ROBOT_NAME = "base_quadrotor"
CONTROLLER_NAME = "lee_position_control"

# -- 强化学习训练默认值 --
DEFAULT_EVAL_HORIZON = 500     # rollout 步数，越大评估越长
DEFAULT_TOTAL_TIMESTEPS = 320   # 估算训练总步数，用于推断 max_epochs
DEFAULT_NUM_WORKERS = 1        # rl-games 并行 actor 数量
DEFAULT_SIM_NUM_ENVS = 2048       # Isaac Gym 中的并行无人机数量
DEFAULT_MAX_EPOCHS = 256        # rl-games 训练的最大 epoch 数

# -- 仿真物理/控制参数 --
DEFAULT_THRUST_MARGIN = 2.5     # 推力裕度，决定最大推力倍数
DEFAULT_HEADLESS = False         # True=关闭查看器，False=开启可视化
DEFAULT_USE_WARP = True         # Warp 物理加速开关

# -- 训练日志与检查点 --
DEFAULT_EXPERIMENT_NAME = "lee_gain_tune"
DEFAULT_HISTORY_PATH = "rlg_eval_history.jsonl"
DEFAULT_METRIC_LOG_PATH = "rlg_metric_log.jsonl"
DEFAULT_CHECKPOINT_DIR = "rlg_checkpoints"

# -- 任务场景配置 --
RELEASE_PLAN = [
    (500, "rear_right"),
    (1000, "front_left"),
    (1500, "front_right"),
    (2000, "rear_left"),
]

PARAM_SPECS = [
    ("K_pos_xyz", 3, [0.5, 5.0]),
    ("K_vel_xyz", 3, [0.3, 5.0]),
    ("K_rot_xyz", 3, [0.2, 3.0]),
    ("K_angvel_xyz", 3, [0.05, 0.5]),
]

LOSS_WEIGHTS = {
    "pos_rmse": 1.0,
    "att_rmse": 0.6,
    "force_mean": 0.2,
    "final_error": 1.5,
}

# 训练监控提示：
# - Policy 表现：reward = -loss，best_payload 会记录 pos/att RMSE、力均值、最终误差，趋势向好代表收敛。
# - 损失曲线：losses/a_loss、losses/c_loss、losses/entropy 写入 metric_log，可配合 plot_loss_curves() 评估训练。
# - 方差与稳定性：history_path 记录每次评估指标，可离线分析 reward 波动或绘制箱线图。
# - 策略评估：best_lee_gains_rlg.json 与 checkpoint_dir 中的权重可用于重放、恢复意外中断的训练。

# 子机参数（质量 / 偏移）
PAYLOAD_LAYOUT = [
    ("front_left", 1.0, torch.tensor([0.16, 0.16, -0.05])),
    ("front_right", 1.0, torch.tensor([0.16, -0.16, -0.05])),
    ("rear_left", 1.0, torch.tensor([-0.16, 0.16, -0.05])),
    ("rear_right", 1.0, torch.tensor([-0.16, -0.16, -0.05])),
]


def _ensure_parent_dir(path: Optional[str]) -> None:
    if not path:
        return
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


# ---------------------------- 数据结构 ---------------------------- #

@dataclass
class Payload:
    name: str
    mass: float
    offset: torch.Tensor


class MetricRecorder:
    """Append scalar metrics to a JSONL file for post-run analysis."""

    def __init__(self, path: Optional[str]):
        self.path = path
        if self.path:
            _ensure_parent_dir(self.path)
            # 创建/清空文件头部，防止旧内容混淆
            with open(self.path, "w", encoding="utf-8") as f:
                f.write("")

    def log(self, tag: str, value: float, step: Optional[int] = None, walltime: Optional[float] = None):
        if self.path is None or value is None:
            return
        entry = {
            "tag": tag,
            "value": float(value),
            "step": int(step) if step is not None else None,
            "walltime": float(walltime) if walltime is not None else None,
            "timestamp": time.time(),
        }
        with open(self.path, "a", encoding="utf-8") as fp:
            fp.write(json.dumps(entry, ensure_ascii=False) + "\n")


class ResultsTracker:
    """记录评估指标，并在出现更优策略时保存。"""

    def __init__(self, best_path: str, history_path: Optional[str] = None):
        self.best_path = best_path
        self.history_path = history_path
        _ensure_parent_dir(self.best_path)
        if self.history_path:
            _ensure_parent_dir(self.history_path)
        self.best_payload: Optional[Dict[str, object]] = None

    def update(self, gains: np.ndarray, metrics: Dict[str, float]):
        entry = {
            "reward": float(-metrics["loss"]),
            "gains": gains.astype(float).tolist(),
            "timestamp": time.time(),
            **{k: float(v) for k, v in metrics.items()},
        }
        is_best = self.best_payload is None or entry["reward"] > self.best_payload["reward"]
        entry["is_best"] = is_best
        self._append_history(entry)
        if is_best:
            self.best_payload = entry
            try:
                with open(self.best_path, "w", encoding="utf-8") as fp:
                    json.dump(entry, fp, indent=2, ensure_ascii=False)
            except OSError as exc:
                print(f"[adjust_lee_rl_gamer_control] 写入 {self.best_path} 失败: {exc}")

    def _append_history(self, payload: Dict[str, object]):
        if not self.history_path:
            return
        try:
            with open(self.history_path, "a", encoding="utf-8") as fp:
                fp.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except OSError as exc:
            print(f"[adjust_lee_rl_gamer_control] 写入 {self.history_path} 失败: {exc}")


class ProgressObserver(AlgoObserver):
    """
    自定义 observer：使用 tqdm 展示训练进度，并记录损失曲线以便会后绘图。
    """

    def __init__(self, total_epochs: int, tracker: ResultsTracker, metric_recorder: Optional[MetricRecorder] = None):
        super().__init__()
        self.requested_total = int(total_epochs) if total_epochs and total_epochs > 0 else None
        self.tracker = tracker
        self.metric_recorder = metric_recorder
        self.pbar = None
        self.algo = None
        self.loss_history = {}
        self._wrapped_writer = False
        self._original_add_scalar = None
        self._last_epoch = 0
        self._last_logged_best_epoch = 0

    def before_init(self, base_name, config, experiment_name):
        if self.requested_total is None:
            max_epochs = config.get("max_epochs", -1)
            if isinstance(max_epochs, int) and max_epochs > 0:
                self.requested_total = max_epochs

    def after_init(self, algo):
        self.algo = algo
        algo_max_epochs = getattr(algo, "max_epochs", -1)
        total = None
        if isinstance(algo_max_epochs, int) and algo_max_epochs > 0:
            total = algo_max_epochs
        elif self.requested_total:
            total = self.requested_total

        self.pbar = tqdm(total=total, desc="rl-games training", unit="epoch")
        self._wrap_writer()

    def _wrap_writer(self):
        if self._wrapped_writer or self.algo is None:
            return
        writer = getattr(self.algo, "writer", None)
        if writer is None or not hasattr(writer, "add_scalar"):
            return

        original_add_scalar = writer.add_scalar

        def add_scalar(tag, scalar_value, *args, **kwargs):
            global_step = None
            walltime = None
            if args:
                global_step = args[0]
            if len(args) > 1:
                walltime = args[1]
            global_step = kwargs.pop("global_step", global_step)
            walltime = kwargs.pop("walltime", walltime)
            kwargs.pop("new_style", None)
            kwargs.pop("double_precision", None)
            value = scalar_value
            if hasattr(value, "item"):
                try:
                    value = value.item()
                except Exception:
                    pass
            try:
                value_f = float(value)
            except Exception:
                value_f = None

            if value_f is not None and tag in ("losses/a_loss", "losses/c_loss", "losses/entropy"):
                epoch = getattr(self.algo, "epoch_num", 0)
                self.loss_history.setdefault(tag, []).append((epoch, value_f))

            call_kwargs = {}
            if global_step is not None:
                call_kwargs["global_step"] = global_step
            if walltime is not None:
                call_kwargs["walltime"] = walltime
            call_kwargs.update(kwargs)

            try:
                result = original_add_scalar(tag, value, **call_kwargs)
            except TypeError:
                result = original_add_scalar(tag, value)

            if self.metric_recorder and value_f is not None:
                step_int = None
                if global_step is not None:
                    try:
                        step_int = int(global_step)
                    except Exception:
                        step_int = None
                self.metric_recorder.log(tag, value_f, step_int, walltime)
            return result

        writer.add_scalar = add_scalar
        self._original_add_scalar = original_add_scalar
        self._wrapped_writer = True

    def after_steps(self):
        pass

    def after_print_stats(self, frame, epoch_num, total_time):
        if self.pbar is None:
            return
        if epoch_num <= self._last_epoch:
            return

        delta = epoch_num - self._last_epoch
        self.pbar.update(delta)
        self._last_epoch = epoch_num

        postfix = {}
        if self.loss_history.get("losses/a_loss"):
            postfix["actor_loss"] = f"{self.loss_history['losses/a_loss'][-1][1]:.4f}"
        if self.loss_history.get("losses/c_loss"):
            postfix["critic_loss"] = f"{self.loss_history['losses/c_loss'][-1][1]:.4f}"
        if self.loss_history.get("losses/entropy"):
            postfix["entropy"] = f"{self.loss_history['losses/entropy'][-1][1]:.4f}"

        if self.tracker and self.tracker.best_payload:
            postfix["best_reward"] = f"{self.tracker.best_payload['reward']:.4f}"
            self._log_best_metrics(epoch_num)

        if postfix:
            self.pbar.set_postfix(postfix)

    def close(self):
        if getattr(self, "pbar", None) is not None:
            self.pbar.close()
            self.pbar = None
        if self._wrapped_writer and self.algo and getattr(self.algo, "writer", None):
            self.algo.writer.add_scalar = self._original_add_scalar
            self._wrapped_writer = False
            self._original_add_scalar = None

    def _log_best_metrics(self, epoch_num: int):
        if not self.tracker or not self.tracker.best_payload:
            return
        if epoch_num <= self._last_logged_best_epoch:
            return
        writer = getattr(self.algo, "writer", None)
        if writer is None:
            return
        best = self.tracker.best_payload
        writer.add_scalar("custom/best_reward", best["reward"], epoch_num)
        writer.add_scalar("custom/best_pos_rmse", best["pos_rmse"], epoch_num)
        writer.add_scalar("custom/best_final_error", best["final_error"], epoch_num)
        if self.metric_recorder:
            self.metric_recorder.log("custom/best_reward", best["reward"], epoch_num, None)
            self.metric_recorder.log("custom/best_pos_rmse", best["pos_rmse"], epoch_num, None)
            self.metric_recorder.log("custom/best_final_error", best["final_error"], epoch_num, None)
        self._last_logged_best_epoch = epoch_num

    def get_loss_history(self) -> Dict[str, List[Tuple[int, float]]]:
        return {tag: list(values) for tag, values in self.loss_history.items()}

    def __del__(self):
        self.close()


# ---------------------------- 负载管理器 ---------------------------- #

class SimplePayloadManager:
    """
    与 adjust_lee_control 中一致：负责更新总质量 / 惯量，并在释放时施加冲击。
    """

    def __init__(self, env_manager, payloads: List[Payload], device: torch.device):
        self.env_manager = env_manager
        self.device = device
        self.payloads: Dict[str, Payload] = {p.name: p for p in payloads}
        self.attached = {p.name: True for p in payloads}

        self.gym = env_manager.IGE_env.gym
        self.sim = env_manager.IGE_env.sim
        self.env_handles = list(env_manager.IGE_env.env_handles)
        self.robot_handles = list(env_manager.robot_manager.robot_handles)

        self.body_props_per_env = [
            self.gym.get_actor_rigid_body_properties(env_handle, robot_handle)
            for env_handle, robot_handle in zip(self.env_handles, self.robot_handles)
        ]

        base_prop = self.body_props_per_env[0][0]
        self.base_mass = base_prop.mass
        self.base_com = torch.tensor(
            [base_prop.com.x, base_prop.com.y, base_prop.com.z],
            device=self.device,
            dtype=torch.float32,
        )
        self.base_inertia = torch.tensor(
            [
                [base_prop.inertia.x.x, base_prop.inertia.x.y, base_prop.inertia.x.z],
                [base_prop.inertia.y.x, base_prop.inertia.y.y, base_prop.inertia.y.z],
                [base_prop.inertia.z.x, base_prop.inertia.z.y, base_prop.inertia.z.z],
            ],
            device=self.device,
            dtype=torch.float32,
        )

        self.pending_impulses: List[Dict[str, torch.Tensor]] = []
        self.apply_to_sim()

    def _compute_total_properties(self):
        total_mass = torch.tensor(self.base_mass, device=self.device)
        numerator_com = self.base_mass * self.base_com.clone()

        for name, payload in self.payloads.items():
            if not self.attached[name]:
                continue
            total_mass += payload.mass
            numerator_com += payload.mass * payload.offset.to(self.device)

        combined_com = torch.zeros(3, device=self.device)
        if total_mass.item() > 1e-6:
            combined_com[:] = numerator_com / total_mass

        identity = torch.eye(3, device=self.device)
        inertia_total = self.base_inertia.clone()
        d_base = self.base_com - combined_com
        inertia_total += self.base_mass * (
            torch.dot(d_base, d_base) * identity - torch.outer(d_base, d_base)
        )

        for name, payload in self.payloads.items():
            if not self.attached[name]:
                continue
            r = payload.offset.to(self.device) - combined_com
            inertia_total += payload.mass * (torch.dot(r, r) * identity - torch.outer(r, r))

        return total_mass, inertia_total, combined_com

    def apply_to_sim(self):
        total_mass, inertia_total, combined_com = self._compute_total_properties()

        inertia_np = inertia_total.detach().cpu().numpy()
        for env_handle, robot_handle, props in zip(
            self.env_handles, self.robot_handles, self.body_props_per_env
        ):
            base_prop = props[0]
            base_prop.mass = float(total_mass.item())
            base_prop.com = torch_to_vec3(combined_com)
            base_prop.inertia = tensor33_to_mat33(inertia_np)
            self.gym.set_actor_rigid_body_properties(
                env_handle, robot_handle, props, recomputeInertia=False
            )

        env = self.env_manager
        env.robot_manager.robot_mass = float(total_mass.item())
        env.robot_manager.robot_masses.fill_(float(total_mass.item()))
        env.IGE_env.global_tensor_dict["robot_mass"].fill_(float(total_mass.item()))

        inertia_tensor = torch.tensor(inertia_np, device=self.device, dtype=torch.float32)
        env.robot_manager.robot_inertia = inertia_tensor
        env.robot_manager.robot_inertias[:] = inertia_tensor
        env.IGE_env.global_tensor_dict["robot_inertia"][:] = inertia_tensor

        controller_mass = torch.full((env.num_envs, 1), float(total_mass.item()), device=self.device)
        env.robot_manager.robot.controller.mass = controller_mass

        adjust_motor_thrust_limits(env_manager=self.env_manager, margin=DEFAULT_THRUST_MARGIN)

    def release(self, name: str):
        if name not in self.payloads or not self.attached[name]:
            return

        payload = self.payloads[name]
        prev_mass = self.current_mass()
        offset = payload.offset.to(self.device)
        self.attached[name] = False
        self.apply_to_sim()

        new_mass = self.current_mass()
        root_state = self.env_manager.IGE_env.global_tensor_dict["vec_root_tensor"]
        base_states = root_state[:, 0] if root_state.ndim == 3 else root_state
        if new_mass > 1e-5:
            scale = prev_mass / new_mass
            base_states[:, 7:10] *= scale
        self.env_manager.IGE_env.write_to_sim()

        gravity_vec = self.env_manager.IGE_env.global_tensor_dict["gravity"][0].to(self.device)
        torque_impulse = -payload.mass * torch.cross(offset, gravity_vec)
        if torch.linalg.norm(torque_impulse).item() > 1e-6:
            self.pending_impulses.append(
                {"torque": torque_impulse, "steps": torch.tensor(45, device=self.device)}
            )

    def consume_impulse(self) -> torch.Tensor:
        if not self.pending_impulses:
            return torch.zeros(3, device=self.device)

        total = torch.zeros(3, device=self.device)
        remaining: List[Dict[str, torch.Tensor]] = []
        for item in self.pending_impulses:
            total += item["torque"]
            steps_left = item["steps"] - 1
            if steps_left.item() > 0:
                item["steps"] = steps_left
                remaining.append(item)
        self.pending_impulses = remaining
        return total

    def current_mass(self) -> float:
        env = self.env_manager
        return float(env.robot_manager.robot_mass)

    def reset(self):
        self.attached = {name: True for name in self.payloads}
        self.pending_impulses.clear()
        self.apply_to_sim()


# ---------------------------- 工具函数 ---------------------------- #

def torch_to_vec3(t: torch.Tensor) -> gymapi.Vec3:
    return gymapi.Vec3(float(t[0].item()), float(t[1].item()), float(t[2].item()))


def tensor33_to_mat33(m: np.ndarray) -> gymapi.Mat33:
    mat = gymapi.Mat33()
    mat.x = gymapi.Vec3(*m[0])
    mat.y = gymapi.Vec3(*m[1])
    mat.z = gymapi.Vec3(*m[2])
    return mat


# ---------------------------- 单环境定义 ---------------------------- #

class LeeGainTuningEnv(gym.Env):
    metadata = {"render.modes": []}

    def __init__(
        self,
        device: str = None,
        eval_horizon: int = DEFAULT_EVAL_HORIZON,
        release_plan: List[Tuple[int, str]] = None,
        tracker: ResultsTracker = None,
        **kwargs,
    ):
        super().__init__()
        device_override = kwargs.pop("device", None)
        self.headless = bool(kwargs.pop("headless", DEFAULT_HEADLESS))
        self.use_warp = bool(kwargs.pop("use_warp", DEFAULT_USE_WARP))
        sim_num_envs = kwargs.pop("sim_num_envs", None)
        if sim_num_envs is None:
            sim_num_envs = kwargs.pop("num_envs", 1)
        self.sim_num_envs = max(1, int(sim_num_envs))
        if device_override is not None:
            device = device_override
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if isinstance(device, torch.device):
            device = device.type + (f":{device.index}" if device.index is not None else "")
        self.device_str = device
        self.device = torch.device(self.device_str)
        self.eval_horizon = eval_horizon
        self.release_plan = release_plan or []
        self.tracker = tracker

        self.param_count = sum(spec[1] for spec in PARAM_SPECS)
        self.action_space = spaces.Box(-np.ones(self.param_count), np.ones(self.param_count), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf * np.ones(1, dtype=np.float32),
            high=np.inf * np.ones(1, dtype=np.float32),
        )

        self.sim_builder = SimBuilder()
        self.env_manager = self.sim_builder.build_env(
            sim_name=SIM_NAME,
            env_name=ENV_NAME,
            robot_name=ROBOT_NAME,
            controller_name=CONTROLLER_NAME,
            args=None,
            device=self.device_str,
            num_envs=self.sim_num_envs,
            headless=self.headless,
            use_warp=self.use_warp,
        )
        payloads = [Payload(name=n, mass=m, offset=offset) for n, m, offset in PAYLOAD_LAYOUT]
        self.payload_manager = SimplePayloadManager(self.env_manager, payloads, self.device)
        self.controller = self.env_manager.robot_manager.robot.controller
        num_actions = self.env_manager.robot_manager.robot.num_actions
        self.target_actions = torch.zeros(
            (self.env_manager.num_envs, num_actions), device=self.device, dtype=torch.float32
        )
        self.release_lookup = {step: name for step, name in self.release_plan}
        self._closed = False

    def reset(self):
        self.env_manager.reset()
        self.payload_manager.reset()
        self.env_manager.reset_tensors()
        self.target_actions.zero_()
        return np.zeros(1, dtype=np.float32)

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if not np.isfinite(action).all():
            action = np.nan_to_num(action, nan=0.0)
        action = np.clip(action, -0.95, 0.95)

        gains = self._denormalize(action)
        metrics = self._evaluate(gains)
        if not np.all(np.isfinite(list(metrics.values()))):
            metrics = {
                "pos_rmse": 1e3,
                "att_rmse": 1e3,
                "force_mean": 1e3,
                "final_error": 1e3,
                "loss": 1e6,
            }

        if self.tracker is not None:
            self.tracker.update(gains, metrics)
        reward = -metrics["loss"]
        obs = np.zeros(1, dtype=np.float32)
        done = True
        return obs, float(reward), done, metrics

    # -------------------- 评估逻辑（复用 adjust_lee_control） -------------------- #

    def _evaluate(self, gains: np.ndarray) -> Dict[str, float]:
        self._apply_gains(self.controller, gains)
        self.env_manager.reset()
        self.payload_manager.reset()
        self.env_manager.reset_tensors()
        self.target_actions.zero_()
        return self._rollout()

    def _apply_gains(self, controller, gains: np.ndarray):
        ptr = 0
        def _assign(dst, value_row):
            rows = dst.shape[0]
            copied = value_row.expand(rows, -1).clone()
            dst.copy_(copied)

        for name, dim, bounds in PARAM_SPECS:
            segment = gains[ptr : ptr + dim]
            ptr += dim
            low, high = bounds
            actual = ((segment + 1.0) * 0.5) * (high - low) + low
            tensor = torch.tensor(actual, device=self.device, dtype=torch.float32).view(1, -1)
            if name.startswith("K_pos"):
                _assign(controller.K_pos_tensor_current, tensor)
                _assign(controller.K_pos_tensor_min, tensor)
                _assign(controller.K_pos_tensor_max, tensor)
            elif name.startswith("K_vel"):
                _assign(controller.K_linvel_tensor_current, tensor)
                _assign(controller.K_linvel_tensor_min, tensor)
                _assign(controller.K_linvel_tensor_max, tensor)
            elif name.startswith("K_rot"):
                _assign(controller.K_rot_tensor_current, tensor)
                _assign(controller.K_rot_tensor_min, tensor)
                _assign(controller.K_rot_tensor_max, tensor)
            elif name.startswith("K_angvel"):
                _assign(controller.K_angvel_tensor_current, tensor)
                _assign(controller.K_angvel_tensor_min, tensor)
                _assign(controller.K_angvel_tensor_max, tensor)

    def _rollout(self) -> Dict[str, float]:
        env_manager = self.env_manager
        payload_manager = self.payload_manager
        release_lookup = self.release_lookup
        crash_detected = False

        pos_errors = []
        att_errors = []
        control_effort = []

        for step in range(self.eval_horizon):
            obs = env_manager.get_obs()
            orientations = obs["robot_orientation"]

            if step in release_lookup:
                payload_manager.release(release_lookup[step])

            env_manager.reset_tensors()

            torque_world = payload_manager.consume_impulse()
            if torch.linalg.norm(torque_world).item() > 0.0:
                torque_body = quat_rotate_inverse(
                    orientations,
                    torque_world.unsqueeze(0).expand(env_manager.num_envs, -1),
                )
                env_manager.IGE_env.global_tensor_dict["robot_torque_tensor"][:, 0, :] += torque_body

            env_manager.step(actions=self.target_actions)
            reset_envs = env_manager.post_reward_calculation_step()
            reset_count = int(reset_envs.numel()) if isinstance(reset_envs, torch.Tensor) else len(reset_envs)
            if reset_count > 0:
                crash_detected = True
                break

            obs_after = env_manager.get_obs()
            pos = obs_after["robot_position"]
            att = ssa(get_euler_xyz_tensor(obs_after["robot_orientation"]))
            forces = env_manager.IGE_env.global_tensor_dict["robot_force_tensor"]

            if (
                not torch.isfinite(pos).all()
                or not torch.isfinite(att).all()
                or not torch.isfinite(forces).all()
            ):
                crash_detected = True
                break

            pos_errors.append(torch.norm(pos, dim=1).mean())
            att_errors.append(torch.norm(att, dim=1).mean())
            control_effort.append(torch.norm(forces, dim=-1).mean())

        if crash_detected or not pos_errors:
            return {
                "pos_rmse": 1e3,
                "att_rmse": 1e3,
                "force_mean": 1e3,
                "final_error": 1e3,
                "loss": 1e6,
            }

        pos_rmse = torch.stack(pos_errors).mean().item()
        att_rmse = torch.stack(att_errors).mean().item()
        force_mean = torch.stack(control_effort).mean().item()
        final_error = pos_errors[-1].item()

        loss = (
            LOSS_WEIGHTS["pos_rmse"] * pos_rmse
            + LOSS_WEIGHTS["att_rmse"] * att_rmse
            + LOSS_WEIGHTS["force_mean"] * force_mean
            + LOSS_WEIGHTS["final_error"] * final_error
        )

        return {
            "pos_rmse": pos_rmse,
            "att_rmse": att_rmse,
            "force_mean": force_mean,
            "final_error": final_error,
            "loss": loss,
        }

    def _denormalize(self, action: np.ndarray) -> np.ndarray:
        gains = []
        ptr = 0
        for _, dim, bounds in PARAM_SPECS:
            segment = action[ptr : ptr + dim]
            ptr += dim
            low, high = bounds
            gains.append(((segment + 1.0) * 0.5) * (high - low) + low)
        return np.concatenate(gains, dtype=np.float32)

    def close(self):
        if getattr(self, "_closed", False):
            return
        self._closed = True
        try:
            self.sim_builder.delete_env()
        except Exception:
            pass
        self.env_manager = None
        self.payload_manager = None

    def __del__(self):
        self.close()


# ---------------------------- 自定义 VecEnv ---------------------------- #

class LeeGainVecEnv(vecenv.IVecEnv):
    def __init__(self, config_name, num_actors, **kwargs):
        creator = env_configurations.configurations[config_name]["env_creator"]
        self.envs = [creator(**kwargs) for _ in range(num_actors)]
        self.num_actors = num_actors

        sample_env = self.envs[0]
        self.action_space = sample_env.action_space
        self.observation_space = sample_env.observation_space

    def step(self, actions):
        actions = np.asarray(actions)
        obs_list = []
        reward_list = []
        done_list = []
        info_list = []

        for env, act in zip(self.envs, actions):
            obs, reward, done, info = env.step(act)
            if done:
                obs_reset = env.reset()
                obs_list.append(obs_reset)
            else:
                obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)

        obs_arr = np.stack(obs_list).astype(np.float32)
        reward_arr = np.asarray(reward_list, dtype=np.float32)
        done_arr = np.asarray(done_list, dtype=np.uint8)
        return obs_arr, reward_arr, done_arr, info_list

    def reset(self):
        obs = [env.reset() for env in self.envs]
        return np.stack(obs).astype(np.float32)

    def reset_done(self):
        return self.reset()

    def get_number_of_agents(self):
        return 1

    def get_env_info(self):
        return {"action_space": self.action_space, "observation_space": self.observation_space}

    def close(self):
        for env in self.envs:
            if hasattr(env, "close"):
                env.close()

    def __del__(self):
        self.close()


# 注册环境
def register_env(tracker: ResultsTracker, eval_horizon: int, release_plan):
    env_configurations.register(
        "lee_gain_tune",
        {
            "env_creator": lambda **kwargs: LeeGainTuningEnv(
                tracker=tracker,
                eval_horizon=eval_horizon,
                release_plan=release_plan,
                **kwargs,
            ),
            "vecenv_type": "LEE-GAIN-VEC",
        },
    )

    vecenv.register(
        "LEE-GAIN-VEC",
        lambda config_name, num_actors, **kwargs: LeeGainVecEnv(config_name, num_actors, **kwargs),
    )


# ---------------------------- 可视化 ---------------------------- #

def plot_loss_curves(loss_history: Dict[str, List[Tuple[int, float]]]):
    if not loss_history:
        print("训练未产生可用损失记录，跳过绘图。")
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("未检测到 matplotlib，无法绘制损失曲线。可运行 `pip install matplotlib` 安装后重试。")
        return

    tag_to_label = {
        "losses/a_loss": "actor loss",
        "losses/c_loss": "critic loss",
        "losses/entropy": "entropy",
    }

    plt.figure(figsize=(8, 5))
    for tag, series in sorted(loss_history.items()):
        if not series:
            continue
        epochs = [epoch for epoch, _ in series]
        values = [value for _, value in series]
        plt.plot(epochs, values, label=tag_to_label.get(tag, tag))

    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("rl-games Training Loss Curves")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    try:
        plt.show()
    except Exception as exc:
        output_path = "rl_games_loss_curve.png"
        plt.savefig(output_path)
        print(f"无法直接显示图像，已将曲线保存到 {output_path}: {exc}")


def plot_eval_history(history_path: Optional[str]):
    if not history_path or not os.path.exists(history_path):
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("未检测到 matplotlib，无法绘制评估历史。")
        return

    records = []
    try:
        with open(history_path, "r", encoding="utf-8") as fp:
            for line in fp:
                entry = line.strip()
                if not entry:
                    continue
                try:
                    records.append(json.loads(entry))
                except json.JSONDecodeError:
                    continue
    except OSError as exc:
        print(f"读取评估历史失败: {exc}")
        return

    if not records:
        print("评估历史为空，跳过绘图。")
        return

    indices = list(range(1, len(records) + 1))
    metric_keys = ["reward", "pos_rmse", "att_rmse", "force_mean", "final_error"]
    metric_keys = [k for k in metric_keys if all(k in r for r in records)]
    if not metric_keys:
        return

    fig, axes = plt.subplots(len(metric_keys), 1, figsize=(8, 2.5 * len(metric_keys)), sharex=True)
    if len(metric_keys) == 1:
        axes = [axes]
    for ax, key in zip(axes, metric_keys):
        values = [float(r[key]) for r in records]
        ax.plot(indices, values, marker="o", label=key)
        ax.set_ylabel(key)
        ax.grid(alpha=0.3)
        ax.legend()
    axes[-1].set_xlabel("Evaluation index")
    fig.tight_layout()
    try:
        plt.show()
    except Exception as exc:
        output_path = "rl_games_eval_history.png"
        fig.savefig(output_path)
        print(f"无法直接显示评估曲线，已保存到 {output_path}: {exc}")


# ---------------------------- RL Games 配置 ---------------------------- #

def create_config(num_workers: int, max_epochs: int) -> Dict:
    num_actors = max(num_workers, 1)
    horizon = max(32, num_actors * 8)  # 保证足够的 rollout 长度
    batch_size = num_actors * horizon
    minibatch = max(32, batch_size // 4)
    minibatch = min(minibatch, batch_size)
    while batch_size % minibatch != 0 and minibatch > num_actors:
        minibatch -= num_actors
    if batch_size % minibatch != 0:
        minibatch = num_actors

    config = {
        "params": {
            "seed": 42,
            "algo": {"name": "a2c_continuous"},
            "model": {"name": "continuous_a2c_logstd"},
            "network": {
                "name": "actor_critic",
                "separate": False,
                "space": {
                    "continuous": {
                        "mu_activation": "None",
                        "sigma_activation": "None",
                        "mu_init": {"name": "default"},
                        "sigma_init": {"name": "const_initializer", "val": -0.5},
                        "fixed_sigma": False,
                    }
                },
                "mlp": {
                    "units": [128, 64],
                    "activation": "elu",
                    "initializer": {"name": "default", "scale": 2},
                },
            },
            "config": {
                "name": "lee_gain_tune_run",
                "env_name": "lee_gain_tune",
                "env_config": {},
                "num_actors": num_actors,
                "horizon_length": horizon,
                "batch_size": batch_size,
                "minibatch_size": minibatch,
                "mini_epochs": 4,
                "ppo": True,
                "gamma": 0.99,
                "tau": 0.95,
                "learning_rate": 1e-4,
                "lr_schedule": None,
                "grad_norm": 1.0,
                "entropy_coef": 0.001,
                "critic_coef": 2.0,
                "clip_value": False,
                "e_clip": 0.2,
                "truncate_grads": True,
                "normalize_advantage": True,
                "normalize_input": False,
                "normalize_value": True,
                "bounds_loss_coef": 0.0001,
                "max_epochs": max_epochs,
                "save_best_after": 10,
                "score_to_win": 1e9,
                "player": {"render": False, "deterministic": True},
                "reward_shaper": {"scale_value": 1.0},
                "print_stats": False,
            },
        }
    }
    return config


# ---------------------------- 主流程 ---------------------------- #

def parse_args():
    parser = argparse.ArgumentParser(description="使用 rl-games 调优 Lee 控制器增益")
    parser.add_argument("--device", default=None, help="设备字符串，如 cuda:0 或 cpu")
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS, help="rl-games 并行 actor 数量")
    parser.add_argument("--num-envs", type=int, default=None, help=argparse.SUPPRESS)  # 兼容旧脚本参数
    parser.add_argument("--sim-envs", type=int, default=DEFAULT_SIM_NUM_ENVS, help="Isaac Gym 并行无人机数量")
    parser.add_argument("--total-timesteps", type=int, default=DEFAULT_TOTAL_TIMESTEPS, help="训练总步数（用于估算 max_epochs）")
    parser.add_argument("--eval-horizon", type=int, default=DEFAULT_EVAL_HORIZON, help="单次评估的仿真步数")
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=DEFAULT_MAX_EPOCHS,
        help="rl-games 的最大 epoch 数（覆盖 total-timesteps 推算）",
    )
    parser.add_argument("--best-path", default="best_lee_gains_rlg.json", help="最佳结果保存路径")
    parser.add_argument(
        "--no-headless",
        action="store_true",
        help=f"取消 headless 以打开查看器窗口（默认{'开启 headless' if DEFAULT_HEADLESS else '关闭 headless'}）",
    )
    parser.add_argument(
        "--disable-warp",
        action="store_true",
        help=f"禁用 Warp 加速（默认{'启用' if DEFAULT_USE_WARP else '禁用'}）",
    )
    parser.add_argument("--history-path", default=DEFAULT_HISTORY_PATH, help="评估指标历史输出 (JSONL)")
    parser.add_argument("--metric-log", default=DEFAULT_METRIC_LOG_PATH, help="训练损失/指标日志 (JSONL)")
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT_DIR, help="rl-games 检查点保存目录")
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME, help="rl-games experiment_name")
    parser.add_argument("--load-checkpoint", default=None, help="恢复训练时载入的 checkpoint 路径")
    return parser.parse_args()


def main():
    args = parse_args()

    device_str = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    tracker = ResultsTracker(args.best_path, history_path=args.history_path)
    metric_recorder = MetricRecorder(args.metric_log)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    num_workers = args.num_workers
    if args.num_envs is not None:
        num_workers = args.num_envs
    num_workers = max(1, int(num_workers))
    sim_envs = max(1, int(args.sim_envs))
    headless = DEFAULT_HEADLESS
    if args.no_headless:
        headless = False
    use_warp = DEFAULT_USE_WARP and not args.disable_warp

    experiment_name = args.experiment_name or DEFAULT_EXPERIMENT_NAME

    register_env(tracker, args.eval_horizon, RELEASE_PLAN)

    derived_epochs = max(32, args.total_timesteps // num_workers)
    if args.max_epochs and args.max_epochs > 0:
        max_epochs = args.max_epochs
    else:
        max_epochs = max(DEFAULT_MAX_EPOCHS, derived_epochs)
    config = create_config(num_workers, max_epochs)
    env_cfg = config["params"]["config"]["env_config"]
    env_cfg["device"] = device_str
    env_cfg["sim_num_envs"] = sim_envs
    env_cfg["headless"] = headless
    env_cfg["use_warp"] = use_warp
    cfg = config["params"]["config"]
    cfg["experiment_name"] = experiment_name
    cfg["full_experiment_name"] = experiment_name
    cfg["train_dir"] = args.checkpoint_dir

    observer = ProgressObserver(max_epochs, tracker, metric_recorder=metric_recorder)
    runner = Runner(algo_observer=observer)

    try:
        runner.load(config)
        runner.reset()
        run_args = {"train": True}
        if args.load_checkpoint:
            run_args["checkpoint"] = args.load_checkpoint
        # --- 开始重定向 ---
        # 保存原始的 stdout
        original_stdout = sys.stdout 
        try:
            # 打开 "空设备" (在 Windows 上是 'nul', 
            # 在 Linux/macOS 上是 '/dev/null')
            f = open(os.devnull, 'w')
            # 将 stdout 重定向到 "空设备"
            sys.stdout = f
            
            # 运行训练，此时所有 print() 都被丢弃
            runner.run(run_args)
            
        finally:
            # 恢复 stdout，以便后续的 print() (如 '最佳增益') 可以正常工作
            sys.stdout = original_stdout
            f.close()
        # --- 结束重定向 ---

    finally:
        observer.close()

    if tracker.best_payload:
        print("\n========== 最佳增益（rl-games） ==========")
        print(json.dumps(tracker.best_payload, indent=2))
    else:
        print("\n未找到有效的增益结果，请检查训练是否成功。")

    loss_history = observer.get_loss_history()
    plot_loss_curves(loss_history)
    plot_eval_history(args.history_path)


if __name__ == "__main__":
    main()
