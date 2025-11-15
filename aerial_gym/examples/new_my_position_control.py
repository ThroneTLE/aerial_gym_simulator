import argparse
import os
import sys
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import yaml

from aerial_gym.registry.task_registry import task_registry
from aerial_gym.utils.math import get_euler_xyz_tensor

import torch

DEFAULT_CKPT = (
    "aerial_gym/rl_training/rl_games/runs/payload_full_rl_test_15-12-17-21/nn/payload_full_rl_test.pth"
)

TASK_BY_ACTION_DIM = {
    3: "payload_compensation_task",
    4: "payload_compensation_task_full_rl",
}
ENV_ACTION_DIM = {name: dim for dim, name in TASK_BY_ACTION_DIM.items()}

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False


class PolicyNetwork(torch.nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_units: Sequence[int]):
        super().__init__()
        layers: List[torch.nn.Module] = []
        in_dim = obs_dim
        for units in hidden_units:
            layers.append(torch.nn.Linear(in_dim, units))
            layers.append(torch.nn.ELU())
            in_dim = units
        self.actor_mlp = torch.nn.Sequential(*layers)
        self.mu = torch.nn.Linear(in_dim, action_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.actor_mlp(obs)
        return self.mu(x)


def _str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("yes", "true", "t", "1"):
        return True
    if value in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError(f"无法解析布尔值: {value}")


def parse_args():
    parser = argparse.ArgumentParser(description="Payload compensation policy rollout.")
    parser.add_argument("--num_envs", type=int, default=1, help="并行环境数量（建议 1 用于绘图）")
    parser.add_argument("--steps", type=int, default=2500, help="仿真步数")
    parser.add_argument(
        "--headless",
        type=_str2bool,
        nargs="?",
        const=True,
        default=False,
        help="是否关闭可视化窗口（True/False），默认 False",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CKPT,
        help="RL-Games 训练生成的 .pth 模型路径",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="aerial_gym/rl_training/rl_games/ppo_aerial_quad.yaml",
        help="训练使用的 YAML 配置（用于读取网络结构）",
    )
    return parser.parse_args()


def load_training_config(config_path: str):
    # Explicit UTF-8 prevents yaml from choking on Chinese comments when locale defaults to ASCII
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_checkpoint(checkpoint_path: str):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model_state = ckpt.get("model", {})
    action_tensor = model_state.get("a2c_network.mu.weight")
    if action_tensor is None:
        raise RuntimeError("Checkpoint 缺少 a2c_network.mu.weight，无法推断动作维度。")
    action_dim = action_tensor.shape[0]
    return ckpt, action_dim


def build_policy(
    cfg: dict,
    checkpoint_data: dict,
    obs_dim: int,
    action_dim: int,
    device: torch.device,
):
    hidden_units = cfg.get("params", {}).get("network", {}).get("mlp", {}).get("units", [256, 128, 64])

    policy = PolicyNetwork(obs_dim, action_dim, hidden_units).to(device)
    model_state = checkpoint_data["model"]

    actor_state = {}
    for key, value in model_state.items():
        if key.startswith("a2c_network.actor_mlp") or key.startswith("a2c_network.mu"):
            new_key = key.replace("a2c_network.", "")
            actor_state[new_key] = value

    missing = set(policy.state_dict().keys()) - set(actor_state.keys())
    if missing:
        raise RuntimeError(f"Checkpoint缺少以下权重: {missing}")
    policy.load_state_dict(actor_state)
    policy.eval()
    return policy


def _add_axis_sliders(fig, axes):
    fig.subplots_adjust(bottom=0.25)
    slider_color = "#24a8a8"
    ax_xscale = fig.add_axes([0.15, 0.1, 0.7, 0.03], facecolor=slider_color)
    ax_yscale = fig.add_axes([0.15, 0.05, 0.7, 0.03], facecolor=slider_color)
    slider_x = Slider(ax_xscale, "X轴缩放", 0.2, 5.0, valinit=1.0)
    slider_y = Slider(ax_yscale, "Y轴缩放", 0.2, 5.0, valinit=1.0)

    base_limits = {axis: {"x": axis.get_xlim(), "y": axis.get_ylim()} for axis in axes}

    def _apply_scale(_):
        x_scale = slider_x.val
        y_scale = slider_y.val

        for axis in axes:
            base_xlim = base_limits[axis]["x"]
            base_ylim = base_limits[axis]["y"]

            x_mid = 0.5 * (base_xlim[0] + base_xlim[1])
            x_span = (base_xlim[1] - base_xlim[0]) / x_scale
            axis.set_xlim(x_mid - 0.5 * x_span, x_mid + 0.5 * x_span)

            y_mid = 0.5 * (base_ylim[0] + base_ylim[1])
            y_span = (base_ylim[1] - base_ylim[0]) / y_scale
            axis.set_ylim(y_mid - 0.5 * y_span, y_mid + 0.5 * y_span)

        fig.canvas.draw_idle()

    slider_x.on_changed(_apply_scale)
    slider_y.on_changed(_apply_scale)
    return slider_x, slider_y


def plot_results(z_history, euler_history, release_history):
    if not z_history:
        print("无可绘制数据。")
        return

    steps = np.arange(len(z_history))
    eulers = np.unwrap(np.array(euler_history), axis=0)
    eulers_deg = np.rad2deg(eulers)
    eulers_deg -= np.round(eulers_deg[0] / 360.0) * 360.0

    fig, (ax_z, ax_euler) = plt.subplots(2, 1, figsize=(10, 8))

    ax_z.plot(steps, z_history, label="Z 轴高度")
    ax_z.set_xlabel("步数")
    ax_z.set_ylabel("Z 轴高度 (米)")
    ax_z.set_title("无人机 Z 轴位置曲线")
    ax_z.grid(True)
    ax_z.legend()

    for release_step, payload_idx in release_history:
        ax_z.axvline(release_step, color="r", linestyle="--", alpha=0.6)
        ax_z.text(
            release_step,
            ax_z.get_ylim()[1],
            f"释放 {payload_idx}",
            color="r",
            fontsize=9,
            verticalalignment="top",
            horizontalalignment="center",
            rotation=90,
        )

    ax_euler.plot(steps, eulers_deg[:, 0], label="Roll (°)")
    ax_euler.plot(steps, eulers_deg[:, 1], label="Pitch (°)")
    ax_euler.plot(steps, eulers_deg[:, 2], label="Yaw (°)")
    ax_euler.set_xlabel("步数")
    ax_euler.set_ylabel("角度 (°)")
    ax_euler.set_title("无人机姿态角 (XYZ)")
    ax_euler.grid(True)
    ax_euler.legend()

    _add_axis_sliders(fig, [ax_z, ax_euler])
    plt.show()


def _resolve_env_name(cfg_env_name: str, checkpoint_action_dim: int) -> str:
    cfg_expected_dim = ENV_ACTION_DIM.get(cfg_env_name)
    if cfg_env_name and cfg_expected_dim == checkpoint_action_dim:
        return cfg_env_name

    guessed_env = TASK_BY_ACTION_DIM.get(checkpoint_action_dim)
    if guessed_env:
        if cfg_env_name and cfg_env_name != guessed_env:
            print(
                f"注意: YAML env_name={cfg_env_name} 与 checkpoint 推断值 {guessed_env} 不一致，使用后者"
            )
        return guessed_env

    if cfg_env_name:
        return cfg_env_name

    return "payload_compensation_task"


def main():
    args = parse_args()
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"找不到 checkpoint: {args.checkpoint}")

    cfg = load_training_config(args.config) or {}
    checkpoint_data, checkpoint_action_dim = load_checkpoint(args.checkpoint)

    cfg_env_name = cfg.get("params", {}).get("config", {}).get("env_name")
    env_name = _resolve_env_name(cfg_env_name, checkpoint_action_dim)

    original_argv = sys.argv
    sys.argv = [sys.argv[0]]
    try:
        task = task_registry.make_task(
            env_name,
            num_envs=args.num_envs,
            headless=args.headless,
        )
    finally:
        sys.argv = original_argv

    obs_dim = task.task_config.observation_space_dim
    action_dim = task.task_config.action_space_dim
    if action_dim != checkpoint_action_dim:
        task.close()
        raise RuntimeError(
            f"任务 {env_name} 的动作维度 {action_dim} 与 checkpoint 的 {checkpoint_action_dim} 不匹配，"
            "请检查 config 的 env_name 设置。"
        )
    device = torch.device(task.device)

    policy = build_policy(cfg, checkpoint_data, obs_dim, action_dim, device)

    print(
        f"启动 {env_name}：envs={task.sim_env.num_envs}, "
        f"action_dim={action_dim}\n使用模型: {args.checkpoint}"
    )

    task_obs, rewards, terms, truncs, infos = task.reset()

    z_history: List[float] = []
    euler_history: List[np.ndarray] = []
    release_history: List[Tuple[int, int]] = []

    with torch.no_grad():
        for step in range(args.steps):
            obs_tensor = torch.as_tensor(
                task.task_obs["observations"], device=device, dtype=torch.float32
            )
            actions = torch.clamp(policy(obs_tensor), -1.0, 1.0)
            task_obs, rewards, terms, truncs, infos = task.step(actions)

            env_id = 0
            pos = task.obs_dict["robot_position"][env_id].detach().cpu().numpy()
            quat = task.obs_dict["robot_orientation"][env_id : env_id + 1]
            euler = get_euler_xyz_tensor(quat)[0].detach().cpu().numpy()

            z_history.append(pos[2])
            euler_history.append(euler)

            if task.payload_manager.just_released_flag[env_id]:
                payload_idx = int(task.payload_manager.last_release_index[env_id].item())
                release_history.append((step, payload_idx))

    try:
        task.close()
    except AttributeError as exc:
        print(f"警告: task.close() 失败 ({exc})，跳过显式销毁。")
    finally:
        del task
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print("仿真结束，开始绘图...")
    plot_results(z_history, euler_history, release_history)


if __name__ == "__main__":
    main()
