"""
adjust_lee_control.py
=====================

利用强化学习（PPO）搜索 Lee 控制器的增益参数。

运行前准备
----------
1. 建议在 headless 模式下运行，并且关闭所有可视化输出（本脚本不会启动 viewer）。
2. 请先安装 Stable-Baselines3 以及其依赖：
   pip install stable-baselines3[extra]
3. 如果在 GPU 上训练，请确认 CUDA 可用并设置好 `CUDA_VISIBLE_DEVICES`。
4. 本脚本默认使用 12 维连续动作（Kp/Kv/Kr/Ko 的 xyz 分量），
   可根据需要在 TrainingSettings.param_specs 中调整上下界或添加新的待优化参数。
5. 如需观察画面，可启动脚本时附加 `--viewer` 关闭 headless 并打开 Isaac Gym viewer。

注意事项
--------
- 训练本质上仍然是黑盒优化，时间开销取决于仿真耗时与并行环境数量。
- 为了方便复现，我们使用 DummyVecEnv（单环境）；若要并行，可在 TrainingSettings.num_env_workers 中调大并使用 SubprocVecEnv。
- 每次评估都会重新构建仿真环境，避免状态污染，但也意味着训练较耗时。
"""

import argparse
import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "float"):
    np.float = float

# NOTE: 为避免 “PyTorch 先于 Isaac Gym 导入” 的报错，先导入 isaacgym 模块，再导入 torch。  # CODEx
from isaacgym import gymapi  # noqa: F401  (仅为触发正确的导入顺序)

import gym
import torch
from gym import spaces
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from aerial_gym.sim.sim_builder import SimBuilder
from aerial_gym.examples.my_position_control import adjust_motor_thrust_limits  # CODEx
from aerial_gym.utils.math import (
    quat_rotate,
    quat_rotate_inverse,
    get_euler_xyz_tensor,
    ssa,
)

# ------------------------------- 调参常量区 ----------------------------------

SIM_NAME = "base_sim"
ENV_NAME = "empty_env"
ROBOT_NAME = "base_quadrotor"
CONTROLLER_NAME = "lee_position_control"


@dataclass
class TrainingSettings:
    """集中管理主要训练超参数，修改此处即可批量调整实验设置。

    - eval_horizon: 单次评估的仿真步数，越大越稳，耗时越久。
    - total_timesteps: PPO 训练的总步数，决定训练时长。
    - num_env_workers: DummyVecEnv 并行环境数量，增大可加速但占显存。
    - thrust_margin: 调整电机最大推力的裕量，匹配质量变化需求。
    - best_result_path: 保存最优增益结果的文件路径。
    - release_plan: 子机释放时间表，单位为仿真步。
    - param_specs: 各增益的搜索上下界。
    - loss_weights: 损失函数中各指标的权重。
    """

    eval_horizon: int = 2500  # 单次评估的仿真步数
    total_timesteps: int = 320
    num_env_workers: int = 1024
    thrust_margin: float = 2.5
    best_result_path: str = "best_lee_gains.json"
    release_plan: List[Tuple[int, str]] = field(
        default_factory=lambda: [
            (500, "rear_right"),
            (1000, "front_left"),
            (1500, "front_right"),
            (2000, "rear_left"),
        ]
    )
    param_specs: List[Tuple[str, int, List[float]]] = field(
        default_factory=lambda: [
            ("K_pos_xyz", 3, [0.5, 3.0]),
            ("K_vel_xyz", 3, [0.3, 4.0]),
            ("K_rot_xyz", 3, [0.2, 1.0]),
            ("K_angvel_xyz", 3, [0.05, 0.5]),
        ]
    )
    loss_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "pos_rmse": 1.0,
            "att_rmse": 0.6,
            "force_mean": 0.2,
            "final_error": 1.5,
        }
    )
    ppo_n_steps: int = 64  # 每次 rollout 的步数，越大越平滑
    ppo_batch_size: int = 256  # PPO 训练 batch 大小
    ppo_verbose: int = 1  # Stable-Baselines3 的日志打印级别
    tensorboard_log: str = None  # 若需要写 TensorBoard 日志，可设置目录路径
    metrics_plot_path: str = "ppo_metrics.png"  # 训练结束后保存误差曲线图的路径


CONFIG = TrainingSettings()


class ResultsTracker:
    """记录并保存当前最佳增益与指标。"""

    def __init__(self, path: str):
        self.path = path
        self.best_payload = None

    def update(self, gains: np.ndarray, metrics: Dict[str, float]):
        reward = -metrics["loss"]
        if self.best_payload is None or reward > self.best_payload.get("reward", -np.inf):
            payload = {
                "reward": float(reward),
                "gains": gains.astype(float).tolist(),
                "timestamp": time.time(),
                **{k: float(v) for k, v in metrics.items()},
            }
            self.best_payload = payload
            try:
                with open(CONFIG.best_result_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2)
            except OSError as err:
                print(f"[adjust_lee_control] 写入 {CONFIG.best_result_path} 失败: {err}")


class MetricsRecorder(BaseCallback):
    """Record metrics during PPO training and save convergence plots."""

    def __init__(self, plot_path: str):
        super().__init__()
        self.plot_path = plot_path
        self.history = {
            "pos_rmse": {"x": [], "y": []},
            "att_rmse": {"x": [], "y": []},
            "force_mean": {"x": [], "y": []},
            "final_error": {"x": [], "y": []},
            "loss": {"x": [], "y": []},
        }

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        if not infos:
            return True
        for info in infos:
            if not info:
                continue
            for key, store in self.history.items():
                val = info.get(key)
                if val is None or not isinstance(val, (int, float)):
                    continue
                if np.isfinite(val):
                    store["x"].append(self.num_timesteps)
                    store["y"].append(float(val))
                    self.logger.record(f"metrics/{key}", float(val))
        return True

    def _on_training_end(self) -> None:
        if not any(store["y"] for store in self.history.values()):
            return
        plt.figure(figsize=(8, 5))
        for key, store in self.history.items():
            if store["y"]:
                plt.plot(store["x"], store["y"], label=key)
        plt.xlabel("Timesteps")
        plt.ylabel("Metric value")
        plt.title("PPO Evaluation Metrics")
        plt.grid(True, alpha=0.3)
        plt.legend()
        try:
            plt.savefig(self.plot_path, dpi=150)
            print(f"[adjust_lee_control] metrics plot saved to {self.plot_path}")
        except OSError as err:
            print(f"[adjust_lee_control] failed to save metrics plot: {err}")
        finally:
            plt.close()


# 子机布置（质量/偏移/半径，仅用于动态更新惯量；若在 URDF 中已有模型，可保持一致）
PAYLOAD_LAYOUT = [
    ("front_left", 1.0, torch.tensor([0.16, 0.16, -0.05])),
    ("front_right", 1.0, torch.tensor([0.16, -0.16, -0.05])),
    ("rear_left", 1.0, torch.tensor([-0.16, 0.16, -0.05])),
    ("rear_right", 1.0, torch.tensor([-0.16, -0.16, -0.05])),
]


@dataclass
class Payload:
    name: str
    mass: float
    offset: torch.Tensor


class SimplePayloadManager:
    """
    无视图版本的负载管理器：负责更新质量/惯量，并在释放时施加扭矩冲击。
    """

    def __init__(self, env_manager, payloads: List[Payload], device: torch.device):
        self.env_manager = env_manager
        self.device = device
        self.payloads: Dict[str, Payload] = {p.name: p for p in payloads}
        self.attached = {p.name: True for p in payloads}

        self.gym = env_manager.IGE_env.gym
        self.sim = env_manager.IGE_env.sim
        self.env_handle = env_manager.IGE_env.env_handles[0]
        self.robot_handle = env_manager.robot_manager.robot_handles[0]

        # 原始属性
        props = self.gym.get_actor_rigid_body_properties(self.env_handle, self.robot_handle)
        base_prop = props[0]
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

        self.body_props = props
        self.pending_impulses: List[Dict[str, torch.Tensor]] = []

        self.mass_history: List[float] = []
        self.com_history: List[torch.Tensor] = []

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

        base_prop = self.body_props[0]
        base_prop.mass = float(total_mass.item())
        base_prop.com = torch_to_vec3(combined_com)
        inertia_np = inertia_total.detach().cpu().numpy()
        base_prop.inertia = tensor33_to_mat33(inertia_np)

        self.gym.set_actor_rigid_body_properties(
            self.env_handle, self.robot_handle, self.body_props, recomputeInertia=False
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

        adjust_motor_thrust_limits(env_manager=self.env_manager, margin=CONFIG.thrust_margin)

        self.mass_history.append(float(total_mass.item()))
        self.com_history.append(combined_com.detach().cpu())

    def release(self, name: str):
        if name not in self.payloads:
            return
        if not self.attached[name]:
            return

        payload = self.payloads[name]
        prev_mass = self.current_mass()
        offset = payload.offset.to(self.device)
        self.attached[name] = False
        self.apply_to_sim()

        new_mass = self.current_mass()
        root_state = self.env_manager.IGE_env.global_tensor_dict["vec_root_tensor"]
        base_state = root_state[0, 0]
        if new_mass > 1e-5:
            scale = prev_mass / new_mass
            base_state[7:10] *= scale

        gravitational = self.env_manager.IGE_env.global_tensor_dict["gravity"][0].to(self.device)
        torque_impulse = -payload.mass * torch.cross(offset, gravitational)
        if torch.linalg.norm(torque_impulse).item() > 1e-6:
            self.pending_impulses.append(
                {"torque": torque_impulse, "steps": torch.tensor(45, device=self.device)}
            )

    def consume_impulse(self) -> torch.Tensor:
        if not self.pending_impulses:
            return torch.zeros(3, device=self.device)

        total = torch.zeros(3, device=self.device)
        remain: List[Dict[str, torch.Tensor]] = []

        for item in self.pending_impulses:
            total += item["torque"]
            steps_left = item["steps"] - 1
            if steps_left.item() > 0:
                item["steps"] = steps_left
                remain.append(item)
        self.pending_impulses = remain
        return total

    def current_mass(self) -> float:
        return self.mass_history[-1] if self.mass_history else self.base_mass

    def reset(self):
        self.attached = {name: True for name in self.payloads}
        self.pending_impulses.clear()
        self.mass_history.clear()
        self.com_history.clear()
        self.apply_to_sim()


def torch_to_vec3(t: torch.Tensor) -> gymapi.Vec3:
    return gymapi.Vec3(float(t[0].item()), float(t[1].item()), float(t[2].item()))


def tensor33_to_mat33(m: np.ndarray) -> gymapi.Mat33:
    mat = gymapi.Mat33()
    mat.x = gymapi.Vec3(*m[0])
    mat.y = gymapi.Vec3(*m[1])
    mat.z = gymapi.Vec3(*m[2])
    return mat


class LeeGainTuningEnv(gym.Env):
    metadata = {"render.modes": []}

    def __init__(self, device: str = None, eval_horizon: int = None, headless: bool = True):
        super().__init__()
        device_str = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        if isinstance(device_str, torch.device):
            device_str = device_str.type + (
                f":{device_str.index}" if device_str.index is not None else ""
            )
        self.device_str = device_str
        self.device = torch.device(device_str)
        self.eval_horizon = eval_horizon if eval_horizon is not None else CONFIG.eval_horizon
        self.headless = headless

        self.param_specs = CONFIG.param_specs
        self.loss_weights = CONFIG.loss_weights
        self.release_plan = CONFIG.release_plan
        self.param_count = sum(spec[1] for spec in self.param_specs)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.param_count,), dtype=np.float32)
        # 这里简单返回最近一次评估的三个指标
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)

        self.sim_builder = SimBuilder()
        self.current_obs = np.zeros(3, dtype=np.float32)
        self.best_result: Dict[str, float] = {"reward": -np.inf}

    def reset(self):
        return self.current_obs.copy()

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        if not np.isfinite(action).all():  # CODEx: 务必把非法动作修正为数值
            action = np.nan_to_num(action, nan=0.0)
        action = np.clip(action, -0.95, 0.95)

        gains = self._denormalize(action)
        metrics = self._evaluate(gains)
        # 评估中如果发生异常（返回 NaN/Inf），直接给一个巨大损失  # CODEx
        if not np.isfinite(list(metrics.values())).all():
            metrics = {
                "pos_rmse": 1e3,
                "att_rmse": 1e3,
                "force_mean": 1e3,
                "final_error": 1e3,
                "loss": 1e6,
            }
        reward = -metrics["loss"]

        self.current_obs = np.array(
            [metrics["pos_rmse"], metrics["att_rmse"], metrics["force_mean"]],
            dtype=np.float32,
        )
        done = True
        info = metrics

        if reward > self.best_result.get("reward", -np.inf):
            payload = {
                "reward": float(reward),
                "gains": gains.astype(float).tolist(),
                "timestamp": time.time(),
                **{k: float(v) for k, v in metrics.items()},
            }
            self.best_result = payload
            try:
                with open(CONFIG.best_result_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2)
            except OSError as err:
                print(
                    f"[adjust_lee_control] 写入 {CONFIG.best_result_path} 失败: {err}"
                )

        return self.current_obs.copy(), reward, done, info

    # ------------------------------------------------------------------ #
    # 评估流程：构建仿真 -> 设置增益 -> 运行 -> 收集指标 -> 销毁仿真
    # ------------------------------------------------------------------ #
    def _evaluate(self, gains: np.ndarray) -> Dict[str, float]:
        env_manager = None
        payload_manager = None
        try:
            env_manager = self.sim_builder.build_env(
                sim_name=SIM_NAME,
                env_name=ENV_NAME,
                robot_name=ROBOT_NAME,
                controller_name=CONTROLLER_NAME,
                args=None,
                device=self.device_str,
                num_envs=1,
                headless=self.headless,
                use_warp=True,
            )

            controller = env_manager.robot_manager.robot.controller
            self._apply_gains(controller, gains)

            payloads = [Payload(name=n, mass=m, offset=offset) for n, m, offset in PAYLOAD_LAYOUT]
            payload_manager = SimplePayloadManager(env_manager, payloads, self.device)

            metrics = self._rollout(env_manager, payload_manager)

            # 收尾
            self._close_env(env_manager)
            env_manager = None
            return metrics
        finally:
            if env_manager is not None:
                self._close_env(env_manager)
            torch.cuda.empty_cache()

    def _close_env(self, env_manager):
        try:
            env_manager.IGE_env.gym.destroy_sim(env_manager.IGE_env.sim)
        except Exception:
            pass

    def _apply_gains(self, controller, gains: np.ndarray):
        ptr = 0
        for name, dim, bounds in self.param_specs:
            segment = gains[ptr : ptr + dim]
            ptr += dim
            low, high = bounds
            actual = ((segment + 1.0) * 0.5) * (high - low) + low
            tensor = torch.tensor(actual, device=self.device, dtype=torch.float32).unsqueeze(0)

            if name.startswith("K_pos"):
                controller.K_pos_tensor_current[:] = tensor
                controller.K_pos_tensor_min[:] = tensor
                controller.K_pos_tensor_max[:] = tensor
            elif name.startswith("K_vel"):
                controller.K_linvel_tensor_current[:] = tensor
                controller.K_linvel_tensor_min[:] = tensor
                controller.K_linvel_tensor_max[:] = tensor
            elif name.startswith("K_rot"):
                controller.K_rot_tensor_current[:] = tensor
                controller.K_rot_tensor_min[:] = tensor
                controller.K_rot_tensor_max[:] = tensor
            elif name.startswith("K_angvel"):
                controller.K_angvel_tensor_current[:] = tensor
                controller.K_angvel_tensor_min[:] = tensor
                controller.K_angvel_tensor_max[:] = tensor

    def _rollout(self, env_manager, payload_manager: SimplePayloadManager) -> Dict[str, float]:
        env_manager.reset()

        target = torch.tensor([[0.0, 0.0, 0.0, 0.0]], device=self.device)
        release_lookup = {step: name for step, name in self.release_plan}

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

            env_manager.step(actions=target)

            obs_after = env_manager.get_obs()
            pos = obs_after["robot_position"]
            att = ssa(get_euler_xyz_tensor(obs_after["robot_orientation"]))
            forces = env_manager.IGE_env.global_tensor_dict["robot_force_tensor"]

            if (
                not torch.isfinite(pos).all()
                or not torch.isfinite(att).all()
                or not torch.isfinite(forces).all()
            ):
                return {
                    "pos_rmse": 1e3,
                    "att_rmse": 1e3,
                    "force_mean": 1e3,
                    "final_error": 1e3,
                    "loss": 1e6,
                }

            pos_errors.append(torch.norm(pos, dim=1).mean())
            att_errors.append(torch.norm(att, dim=1).mean())
            control_effort.append(torch.norm(forces, dim=-1).mean())

        pos_rmse = torch.stack(pos_errors).mean().item()
        att_rmse = torch.stack(att_errors).mean().item()
        force_mean = torch.stack(control_effort).mean().item()
        final_error = pos_errors[-1].item()

        lw = self.loss_weights
        loss = (
            lw["pos_rmse"] * pos_rmse
            + lw["att_rmse"] * att_rmse
            + lw["force_mean"] * force_mean
            + lw["final_error"] * final_error
        )

        return {
            "pos_rmse": pos_rmse,
            "att_rmse": att_rmse,
            "force_mean": force_mean,
            "final_error": final_error,
            "loss": loss,
        }

    def _denormalize(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32)
        assert action.shape[0] == self.param_count
        return action


def make_env(device: str, headless: bool = True):
    def _init():
        return LeeGainTuningEnv(device=device, headless=headless)

    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None, help="用于仿真和训练的设备，如 cuda:0 或 cpu")
    parser.add_argument("--timesteps", type=int, default=CONFIG.total_timesteps, help="训练步数")
    parser.add_argument("--ppo-n-steps", type=int, default=CONFIG.ppo_n_steps, help="PPO 每次 rollout 的步数 n_steps")
    parser.add_argument("--ppo-batch-size", type=int, default=CONFIG.ppo_batch_size, help="PPO 训练 batch_size")
    parser.add_argument("--ppo-verbose", type=int, default=CONFIG.ppo_verbose, help="PPO verbose 日志等级")
    parser.add_argument("--tensorboard-log", default=CONFIG.tensorboard_log, help="TensorBoard 日志目录 (可选)")
    parser.add_argument("--metrics-plot", default=CONFIG.metrics_plot_path, help="保存误差曲线图的路径")
    parser.add_argument("--viewer", action="store_true", help="关闭 headless，打开 Isaac Gym viewer")
    args = parser.parse_args()

    headless = not args.viewer
    env_fns = [make_env(args.device, headless=headless) for _ in range(CONFIG.num_env_workers)]
    vec_env = DummyVecEnv(env_fns)

    model = PPO(
        "MlpPolicy",
        vec_env,
        n_steps=args.ppo_n_steps,
        batch_size=args.ppo_batch_size,
        verbose=args.ppo_verbose,
        tensorboard_log=args.tensorboard_log,
        device=args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
    )

    total_timesteps = args.timesteps
    metrics_callback = MetricsRecorder(args.metrics_plot)
    model.learn(total_timesteps=total_timesteps, callback=metrics_callback, progress_bar=True)

    best = vec_env.envs[0].best_result
    print("\n================ 最优结果 ================")
    print(f"Reward: {best['reward']:.4f}")
    print(
        "Gains vector (按 TrainingSettings.param_specs 顺序识别):",
        np.asarray(best["gains"], dtype=np.float32),
    )
    print(
        f"pos_rmse={best['pos_rmse']:.4f}, att_rmse={best['att_rmse']:.4f}, "
        f"force_mean={best['force_mean']:.4f}, final_error={best['final_error']:.4f}"
    )


if __name__ == "__main__":
    main()
