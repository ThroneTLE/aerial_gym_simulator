import argparse
import os
import sys
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import yaml

from aerial_gym.registry.task_registry import task_registry
from aerial_gym.utils.math import get_euler_xyz_tensor

import torch
from rl_games.algos_torch.moving_mean_std import GeneralizedMovingStats
"""
conda run --no-capture-output -n aerialgym python aerial_gym/examples/new_my_position_control.py   --num_envs 1 --steps 2000 --headless False   --checkpoint runs/teacher_residual_stage1_06-06-48-09/nn/teacher_residual_stage1.pth

"""
DEFAULT_CKPT = (
    "runs/teacher_residual_stage1_27-17-36-18/nn/teacher_residual_stage1.pth"  # 修改为你的默认模型路径
)

# Demo 默认使用训练 YAML 指定的任务；仅在缺少配置时退回补偿任务。
DEFAULT_ENV_NAME = "payload_compensation_task_teacher"

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False


class PolicyNetwork(torch.nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_units: Sequence[int],
        rnn_cfg: dict,
        num_envs: int,
        device: torch.device,
    ):
        super().__init__()
        layers: List[torch.nn.Module] = []
        in_dim = obs_dim
        for units in hidden_units:
            layers.append(torch.nn.Linear(in_dim, units))
            layers.append(torch.nn.ELU())
            in_dim = units
        if layers:
            self.actor_mlp = torch.nn.Sequential(*layers)
            mlp_out_dim = in_dim
        else:
            self.actor_mlp = torch.nn.Identity()
            mlp_out_dim = obs_dim

        rnn_name = (rnn_cfg or {}).get("name", "").lower()
        self.uses_rnn = rnn_name == "gru" and rnn_cfg.get("units", 0) > 0
        self.before_mlp = bool((rnn_cfg or {}).get("before_mlp", False)) if self.uses_rnn else False
        self.layer_norm_after_rnn = bool((rnn_cfg or {}).get("layer_norm", False)) if self.uses_rnn else False
        self.hidden_state: Optional[torch.Tensor] = None
        self.num_envs = num_envs

        if self.uses_rnn:
            hidden_size = int(rnn_cfg.get("units", mlp_out_dim))
            self.rnn = torch.nn.GRU(
                input_size=obs_dim if self.before_mlp else mlp_out_dim,
                hidden_size=hidden_size,
                num_layers=int(rnn_cfg.get("layers", 1)),
                batch_first=True,
            )
            if self.layer_norm_after_rnn:
                self.layer_norm = torch.nn.LayerNorm(hidden_size)
            self.final_feature_dim = mlp_out_dim if self.before_mlp else hidden_size
        else:
            self.final_feature_dim = mlp_out_dim

        self.mu = torch.nn.Linear(self.final_feature_dim, action_dim)
        self.to(device)
        self.reset_hidden_state(device=device)

    def reset_hidden_state(self, env_ids=None, device=None):
        if not self.uses_rnn:
            return
        dev = device or (self.hidden_state.device if self.hidden_state is not None else next(self.parameters()).device)
        if self.hidden_state is None or self.hidden_state.shape[1] != self.num_envs or self.hidden_state.device != dev:
            self.hidden_state = torch.zeros(self.rnn.num_layers, self.num_envs, self.rnn.hidden_size, device=dev)
            return
        if env_ids is None:
            self.hidden_state.zero_()
        else:
            idx = torch.as_tensor(env_ids, device=self.hidden_state.device, dtype=torch.long)
            if idx.numel() > 0:
                self.hidden_state[:, idx, :] = 0.0

    def _ensure_hidden(self, batch_size: int, device: torch.device):
        if not self.uses_rnn:
            return
        if self.hidden_state is None or self.hidden_state.shape[1] != batch_size or self.hidden_state.device != device:
            self.hidden_state = torch.zeros(self.rnn.num_layers, batch_size, self.rnn.hidden_size, device=device)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = obs
        if self.uses_rnn and self.before_mlp:
            self._ensure_hidden(x.shape[0], x.device)
            rnn_out, self.hidden_state = self.rnn(x.unsqueeze(1), self.hidden_state)
            self.hidden_state = self.hidden_state.detach()
            features = rnn_out.squeeze(1)
            if self.layer_norm_after_rnn:
                features = self.layer_norm(features)
            x = self.actor_mlp(features)
        else:
            x = self.actor_mlp(x)
            if self.uses_rnn:
                self._ensure_hidden(x.shape[0], x.device)
                rnn_out, self.hidden_state = self.rnn(x.unsqueeze(1), self.hidden_state)
                self.hidden_state = self.hidden_state.detach()
                x = rnn_out.squeeze(1)
                if self.layer_norm_after_rnn:
                    x = self.layer_norm(x)
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
    parser.add_argument("--steps", type=int, default=1500, help="仿真步数")
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


def load_obs_rms(checkpoint_data: dict, obs_dim: int, device: torch.device):
    """加载训练时的 running mean/std，并在推理时复用，避免分布偏移导致抖动。"""
    rms_state = checkpoint_data.get("running_mean_std")
    if rms_state is None:
        return None
    rms = GeneralizedMovingStats(obs_dim, impl="mean_std")
    # 兼容旧格式（running_mean/running_var/count）与新格式（step/mean/sqrs）
    if {"running_mean", "running_var", "count"} <= set(rms_state.keys()):
        # running_var 是方差，sqrs = var + mean^2；count 记到 step 里
        mean = rms_state["running_mean"]
        var = rms_state["running_var"]
        count = rms_state.get("count", torch.tensor([1.0]))
        sqrs = var + mean * mean
        compat_state = {
            "step": torch.as_tensor(count, dtype=torch.int32).view(1),
            "mean": mean,
            "sqrs": sqrs,
        }
        rms.load_state_dict(compat_state, strict=False)
    else:
        try:
            rms.load_state_dict(rms_state, strict=False)
        except RuntimeError:
            # 若形状不符，直接跳过归一化
            return None
    rms.to(device)
    rms.eval()
    return rms


def _log_rms_stats(rms: Optional[GeneralizedMovingStats]):
    if rms is None:
        print("obs_rms: None（未找到训练时的 running_mean_std）")
        return
    with torch.no_grad():
        mean = rms.mean.detach().cpu()
        sqrs = rms.sqrs.detach().cpu()
        step = int(rms.step.item()) if hasattr(rms, "step") else -1
        var = torch.clamp_min(sqrs - mean * mean, 0.0)
        std = torch.sqrt(var)
        print(
            "obs_rms: loaded, "
            f"step={step}, "
            f"mean[min,max]=({mean.min():.4f}, {mean.max():.4f}), "
            f"std[min,max]=({std.min():.4f}, {std.max():.4f})"
        )


def build_policy(
    cfg: dict,
    checkpoint_data: dict,
    obs_dim: int,
    action_dim: int,
    device: torch.device,
    num_envs: int,
):
    network_cfg = cfg.get("params", {}).get("network", {})
    hidden_units = network_cfg.get("mlp", {}).get("units", [256, 128, 64])
    rnn_cfg = network_cfg.get("rnn", {})

    policy = PolicyNetwork(obs_dim, action_dim, hidden_units, rnn_cfg, num_envs, device)
    model_state = checkpoint_data["model"]

    actor_state = {}
    for key, value in model_state.items():
        if not key.startswith("a2c_network."):
            continue
        new_key = key.replace("a2c_network.", "", 1)
        if new_key.startswith("rnn.rnn."):
            new_key = new_key.replace("rnn.rnn.", "rnn.", 1)
        if new_key.startswith(("actor_mlp", "mu", "rnn", "layer_norm")):
            actor_state[new_key] = value

    missing = set(policy.state_dict().keys()) - set(actor_state.keys())
    if missing:
        raise RuntimeError(f"Checkpoint缺少以下权重: {missing}")
    policy.load_state_dict(actor_state)
    policy.eval()
    policy.reset_hidden_state()
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


def plot_results(
    z_history,
    euler_history,
    release_history,
    policy_actions,
    teacher_actions=None,
    pos_history=None,
    target_history=None,
    ideal_circle=None,
):
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

    # XY 轨迹与平移距离
    if pos_history:
        pos_arr = np.vstack(pos_history)
        xy = pos_arr[:, :2]
        r = np.linalg.norm(xy, axis=1)
        fig_xy, (ax_xy, ax_r) = plt.subplots(1, 2, figsize=(13, 5))
        ax_xy.plot(xy[:, 0], xy[:, 1], label="轨迹", color="C0")
        ax_xy.scatter([0], [0], color="k", s=30, marker="x", label="原点")
        if ideal_circle:
            cx, cy, radius = ideal_circle
            theta = np.linspace(0, 2 * np.pi, 200)
            ax_xy.plot(cx + radius * np.cos(theta), cy + radius * np.sin(theta), "--", color="C1", alpha=0.6, label="理想圆轨迹")
        for release_step, _ in release_history:
            if 0 <= release_step < len(xy):
                ax_xy.scatter(xy[release_step, 0], xy[release_step, 1], color="r", s=25, marker="o", alpha=0.6)
        ax_xy.set_xlabel("X (m)")
        ax_xy.set_ylabel("Y (m)")
        ax_xy.set_title("XY 平面轨迹")
        ax_xy.axis("equal")
        ax_xy.grid(True)
        ax_xy.legend()

        ax_r.plot(steps, r, label="平移距离 |XY|", color="C2")
        for release_step, _ in release_history:
            ax_r.axvline(release_step, color="r", linestyle=":", alpha=0.5)
        ax_r.set_xlabel("步数")
        ax_r.set_ylabel("距离 (m)")
        ax_r.set_title("XY 平移距离随时间")
        ax_r.grid(True)
        ax_r.legend()
        # 评估跟踪误差：位置相对目标点
        if target_history is not None:
            tgt_arr = np.vstack(target_history)[: len(xy)]
            err_xy = xy - tgt_arr[:, :2]
            err_norm = np.linalg.norm(err_xy, axis=1)
            fig_err, ax_err = plt.subplots(1, 1, figsize=(10, 3))
            ax_err.plot(steps[: len(err_norm)], err_norm, color="C3", label="|pos-target|")
            for release_step, _ in release_history:
                ax_err.axvline(release_step, color="r", linestyle=":", alpha=0.4)
            ax_err.set_xlabel("步数")
            ax_err.set_ylabel("跟踪误差 (m)")
            ax_err.set_title("XY 跟踪误差")
            ax_err.grid(True)
            ax_err.legend()
        # 若无目标历史但有理想圆，保留到理想圆的径向误差
        elif ideal_circle:
            cx, cy, radius = ideal_circle
            radial = np.linalg.norm(xy - np.array([cx, cy]), axis=1)
            radial_err = radial - radius
            fig_err, ax_err = plt.subplots(1, 1, figsize=(10, 3))
            ax_err.plot(steps, radial_err, color="C3")
            for release_step, _ in release_history:
                ax_err.axvline(release_step, color="r", linestyle=":", alpha=0.4)
            ax_err.set_xlabel("步数")
            ax_err.set_ylabel("径向误差 (m)")
            ax_err.set_title("圆轨迹径向误差")
            ax_err.grid(True)

    # 补偿动作对比：策略输出 vs 教师残差（如有）
    if policy_actions:
        act_arr = np.vstack(policy_actions)
        action_dim = act_arr.shape[1]
        teacher_arr = np.vstack(teacher_actions) if teacher_actions is not None and len(teacher_actions) == len(policy_actions) else None
        fig_act, axes = plt.subplots(action_dim, 1, figsize=(10, max(4, 2 * action_dim)), sharex=True)
        if action_dim == 1:
            axes = [axes]
        labels = ["thrust"] + [f"torque_{i}" for i in range(1, action_dim)]
        for i, ax in enumerate(axes):
            ax.plot(steps, act_arr[:, i], label="policy", color="C0")
            if teacher_arr is not None:
                ax.plot(steps, teacher_arr[:, i], label="teacher", color="C1", linestyle="--")
            for release_step, _ in release_history:
                ax.axvline(release_step, color="r", linestyle=":", alpha=0.5)
            ax.set_ylabel(labels[i])
            ax.grid(True)
            ax.legend()
        axes[-1].set_xlabel("步数")
        fig_act.suptitle("残差补偿对比（策略 vs 教师）")

    save_path = os.environ.get("AERIAL_SAVE_PLOT")
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        if policy_actions:
            fig_act.savefig(os.path.splitext(save_path)[0] + "_actions.png", dpi=150, bbox_inches="tight")
        print(f"已保存绘图到 {save_path}")
    else:
        plt.show()


def _resolve_env_name(cfg_env_name: Optional[str]) -> str:
    if cfg_env_name:
        return cfg_env_name
    return DEFAULT_ENV_NAME


def main():
    args = parse_args()
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"找不到 checkpoint: {args.checkpoint}")

    cfg = load_training_config(args.config) or {}
    checkpoint_data, checkpoint_action_dim = load_checkpoint(args.checkpoint)
    obs_rms = None

    # 演示强制使用教师任务，以匹配教师 checkpoint 的 obs 维度
    env_name = DEFAULT_ENV_NAME

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

    # 尝试加载训练时的 obs running mean/std，用于推理归一化，避免分布漂移
    obs_rms = load_obs_rms(checkpoint_data, obs_dim, device)
    _log_rms_stats(obs_rms)

    policy = build_policy(
        cfg, checkpoint_data, obs_dim, action_dim, device, task.sim_env.num_envs
    )

    print(
        f"启动 {env_name}：envs={task.sim_env.num_envs}, "
        f"action_dim={action_dim}\n使用模型: {args.checkpoint}"
    )

    task_obs, rewards, terms, truncs, infos = task.reset()
    policy.reset_hidden_state()

    z_history: List[float] = []
    euler_history: List[np.ndarray] = []
    pos_history: List[np.ndarray] = []
    target_history: List[np.ndarray] = []
    release_history: List[Tuple[int, int]] = []
    policy_actions: List[np.ndarray] = []
    teacher_actions: List[np.ndarray] = []

    with torch.no_grad():
        for step in range(args.steps):
            obs_tensor = torch.as_tensor(
                task.task_obs["observations"], device=device, dtype=torch.float32
            )
            if obs_rms is not None:
                # 仅归一化 base obs；privileged obs 由策略内部处理/固定归一化
                obs_tensor = obs_rms(obs_tensor, denorm=False)
            actions = torch.clamp(policy(obs_tensor), -1.0, 1.0)
            task_obs, rewards, terms, truncs, infos = task.step(actions)

            if policy.uses_rnn:
                done_tensor = torch.as_tensor(terms, device=device).bool()
                trunc_tensor = torch.as_tensor(truncs, device=device).bool()
                reset_envs = torch.nonzero(done_tensor | trunc_tensor, as_tuple=False).squeeze(-1)
                if reset_envs.numel() > 0:
                    policy.reset_hidden_state(env_ids=reset_envs.tolist())

            env_id = 0
            pos = task.obs_dict["robot_position"][env_id].detach().cpu().numpy()
            tgt = task.target_position[env_id].detach().cpu().numpy()
            quat = task.obs_dict["robot_orientation"][env_id : env_id + 1]
            euler = get_euler_xyz_tensor(quat)[0].detach().cpu().numpy()

            pos_history.append(pos.copy())
            target_history.append(tgt.copy())
            z_history.append(pos[2])
            euler_history.append(euler)
            policy_actions.append(actions[env_id].detach().cpu().numpy())
            if hasattr(task, "teacher_residual"):
                # 优先使用 infos 中已经汇总好的教师“总输出”（PD+补偿）
                if isinstance(infos, dict) and "teacher_actions" in infos:
                    teacher_actions.append(
                        torch.as_tensor(infos["teacher_actions"][env_id])
                        .detach()
                        .cpu()
                        .numpy()
                    )
                else:
                    teacher_total = torch.clamp(
                        task.base_pd_norm[env_id] + task.teacher_residual[env_id],
                        -1.0,
                        1.0,
                    )
                    teacher_actions.append(teacher_total.detach().cpu().numpy())

            if task.payload_manager.just_released_flag[env_id]:
                payload_idx = int(task.payload_manager.last_release_index[env_id].item())
                release_history.append((step, payload_idx))

    # 估计理想圆轨迹参数（基于任务配置，若有）
    ideal_circle = None
    traj_cfg = getattr(task.task_config, "trajectory_parameters", None) or {}
    if str(traj_cfg.get("type", "")).lower() == "circle":
        cx, cy, _ = traj_cfg.get("center", [0.0, 0.0, 0.0])
        radius = float(traj_cfg.get("radius", 0.0))
        ideal_circle = (cx, cy, radius)

    try:
        task.close()
    except AttributeError as exc:
        print(f"警告: task.close() 失败 ({exc})，跳过显式销毁。")
    finally:
        del task
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print("仿真结束，开始绘图...")
    plot_results(
        z_history,
        euler_history,
        release_history,
        policy_actions,
        teacher_actions if teacher_actions else None,
        pos_history=pos_history,
        target_history=target_history,
        ideal_circle=ideal_circle,
    )


if __name__ == "__main__":
    main()
