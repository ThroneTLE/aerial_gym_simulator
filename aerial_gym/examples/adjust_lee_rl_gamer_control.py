"""
adjust_lee_rl_gamer_control.py
===============================

使用 `rl_games` 框架对 Lee 控制器的增益进行黑盒优化，相比单环境 + SB3 的方案，
这里借助 rl-games 的高吞吐 PPO 实现更高效的并行评估。

脚本要点
--------
* 复用 `adjust_lee_control.py` 里同样的仿真评估逻辑，但将其包装为单步 Gym 环境。
* 通过自定义 `vecenv` 将多个评估环境并行运行，充分利用 GPU（RTX 4060）。
* 使用 rl-games 的 Runner 直接加载配置字典并启动训练，训练过程中不断把最佳增益写入 JSON。

运行方式
--------
```bash
python -m aerial_gym.examples.adjust_lee_rl_gamer_control \
    --device cuda:0 \
    --num-envs 4 \
    --total-timesteps 256 \
    --eval-horizon 600 \
    --max-epochs 128 \
    --best-path best_lee_gains_rlg.json
```

训练完成后可在 `best_lee_gains_rlg.json` 中查看当前最优增益以及各项指标。
"""

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

# 确保 PyTorch 在 Isaac Gym 之后导入，避免 gymdeps 检查失败
from isaacgym import gymapi  # noqa: F401

import gym
from gym import spaces
import numpy as np
import torch

from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner

from aerial_gym.sim.sim_builder import SimBuilder
from aerial_gym.examples.my_position_control import adjust_motor_thrust_limits
from aerial_gym.utils.math import (
    quat_rotate,
    quat_rotate_inverse,
    get_euler_xyz_tensor,
    ssa,
)


# ---------------------------- 常量与配置 ---------------------------- #

SIM_NAME = "base_sim"
ENV_NAME = "empty_env"
ROBOT_NAME = "base_quadrotor"
CONTROLLER_NAME = "lee_position_control"

# 默认评估步数 / RL 总步数，可通过命令行覆盖
DEFAULT_EVAL_HORIZON = 600
DEFAULT_TOTAL_TIMESTEPS = 160
DEFAULT_NUM_ENV_WORKERS = 4

DEFAULT_THRUST_MARGIN = 2.5

# 子机释放时间表（仿真步） # CODEx
RELEASE_PLAN = [
    (500, "rear_right"),
    (1000, "front_left"),
    (1500, "front_right"),
    (2000, "rear_left"),
]

# 增益范围（与 adjust_lee_control 中保持一致，但适当收窄）
PARAM_SPECS = [
    ("K_pos_xyz", 3, [0.5, 3.0]),
    ("K_vel_xyz", 3, [0.3, 4.0]),
    ("K_rot_xyz", 3, [0.2, 1.0]),
    ("K_angvel_xyz", 3, [0.05, 0.5]),
]

LOSS_WEIGHTS = {
    "pos_rmse": 1.0,
    "att_rmse": 0.6,
    "force_mean": 0.2,
    "final_error": 1.5,
}

# 子机参数（质量 / 偏移）
PAYLOAD_LAYOUT = [
    ("front_left", 1.0, torch.tensor([0.16, 0.16, -0.05])),
    ("front_right", 1.0, torch.tensor([0.16, -0.16, -0.05])),
    ("rear_left", 1.0, torch.tensor([-0.16, 0.16, -0.05])),
    ("rear_right", 1.0, torch.tensor([-0.16, -0.16, -0.05])),
]


# ---------------------------- 数据结构 ---------------------------- #

@dataclass
class Payload:
    name: str
    mass: float
    offset: torch.Tensor


class ResultsTracker:
    """
    记录并保存当前最佳增益与指标。
    """

    def __init__(self, path: str):
        self.path = path
        self.best_payload = None

    def update(self, gains: np.ndarray, metrics: Dict[str, float]):
        reward = -metrics["loss"]
        if self.best_payload is None or reward > self.best_payload["reward"]:
            payload = {
                "reward": float(reward),
                "gains": gains.astype(float).tolist(),
                "timestamp": time.time(),
                **{k: float(v) for k, v in metrics.items()},
            }
            self.best_payload = payload
            try:
                with open(self.path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2)
            except OSError as exc:
                print(f"[adjust_lee_rl_gamer_control] 写入 {self.path} 失败: {exc}")


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
        self.env_handle = env_manager.IGE_env.env_handles[0]
        self.robot_handle = env_manager.robot_manager.robot_handles[0]

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
        base_state = root_state[0, 0]
        if new_mass > 1e-5:
            scale = prev_mass / new_mass
            base_state[7:10] *= scale

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

    def reset(self):
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

        self.tracker.update(gains, metrics)
        reward = -metrics["loss"]
        obs = np.zeros(1, dtype=np.float32)
        done = True
        return obs, float(reward), done, metrics

    # -------------------- 评估逻辑（复用 adjust_lee_control） -------------------- #

    def _evaluate(self, gains: np.ndarray) -> Dict[str, float]:
        env_manager = None
        try:
            env_manager = self.sim_builder.build_env(
                sim_name=SIM_NAME,
                env_name=ENV_NAME,
                robot_name=ROBOT_NAME,
                controller_name=CONTROLLER_NAME,
                args=None,
                device=self.device_str,
                num_envs=1,
                headless=True,
                use_warp=True,
            )

            controller = env_manager.robot_manager.robot.controller
            self._apply_gains(controller, gains)

            payloads = [
                Payload(name=n, mass=m, offset=offset) for n, m, offset in PAYLOAD_LAYOUT
            ]
            payload_manager = SimplePayloadManager(env_manager, payloads, self.device)

            metrics = self._rollout(env_manager, payload_manager)
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
        for name, dim, bounds in PARAM_SPECS:
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


# ---------------------------- RL Games 配置 ---------------------------- #

def create_config(num_envs: int, max_epochs: int) -> Dict:
    num_actors = max(num_envs, 1)
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
                        "sigma_init": {"name": "const_initializer", "val": -2.0},
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
                "entropy_coef": 0.0,
                "critic_coef": 2.0,
                "clip_value": False,
                "e_clip": 0.2,
                "truncate_grads": True,
                "normalize_advantage": True,
                "normalize_input": False,
                "normalize_value": False,
                "bounds_loss_coef": 0.0001,
                "max_epochs": max_epochs,
                "save_best_after": 10,
                "score_to_win": 1e9,
                "player": {"render": False, "deterministic": True},
                "reward_shaper": {"scale_value": 1.0},
            },
        }
    }
    return config


# ---------------------------- 主流程 ---------------------------- #

def parse_args():
    parser = argparse.ArgumentParser(description="使用 rl-games 调优 Lee 控制器增益")
    parser.add_argument("--device", default=None, help="设备字符串，如 cuda:0 或 cpu")
    parser.add_argument("--num-envs", type=int, default=DEFAULT_NUM_ENV_WORKERS, help="并行环境数量")
    parser.add_argument("--total-timesteps", type=int, default=DEFAULT_TOTAL_TIMESTEPS, help="训练总步数（用于估算 max_epochs）")
    parser.add_argument("--eval-horizon", type=int, default=DEFAULT_EVAL_HORIZON, help="单次评估的仿真步数")
    parser.add_argument("--max-epochs", type=int, default=128, help="rl-games 的最大 epoch 数（覆盖 total-timesteps 推算）")
    parser.add_argument("--best-path", default="best_lee_gains_rlg.json", help="最佳结果保存路径")
    return parser.parse_args()


def main():
    args = parse_args()

    device_str = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    tracker = ResultsTracker(args.best_path)

    register_env(tracker, args.eval_horizon, RELEASE_PLAN)

    max_epochs = args.max_epochs or max(32, args.total_timesteps // max(args.num_envs, 1))
    config = create_config(args.num_envs, max_epochs)
    config["params"]["config"]["env_config"]["device"] = device_str

    runner = Runner()
    runner.load(config)
    runner.reset()

    runner.run({"train": True})

    if tracker.best_payload:
        print("\n========== 最佳增益（rl-games） ==========")
        print(json.dumps(tracker.best_payload, indent=2))
    else:
        print("\n未找到有效的增益结果，请检查训练是否成功。")


if __name__ == "__main__":
    main()
