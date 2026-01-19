import argparse
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

import yaml

from aerial_gym.registry.task_registry import task_registry
from aerial_gym.utils.math import get_euler_xyz_tensor

# Ensure privileged network is registered before building the model.
from aerial_gym.rl_training.rl_games.nn import privileged_actor_critic  # noqa: F401
from rl_games.algos_torch import model_builder
import torch
'''
默认行为：运行 new_my_position_control2.py 时，默认仅展示多环境聚合统计图（状态稳定性、策略行为分析等），不再弹出大量的单环境波形图。
详细模式：若需要查看单环境细节（XY轨迹、力矩角度关系等），请使用参数 --show_details。
bash
python aerial_gym/examples/new_my_position_control2.py --show_details
'''
DEFAULT_ENV_NAME = "payload_compensation_task_teacher"
DEFAULT_CONFIG = "aerial_gym/rl_training/rl_games/ppo_aerial_quad_aux.yaml"
DEFAULT_CKPT = "runs/teacher_aux_fixed_imitation_19-21-52-29/nn/last_teacher_aux_fixed_imitation_ep_135_rew_15005.909.pth"
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False


def _str2bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("yes", "true", "t", "1"):
        return True
    if value in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError(f"Unable to parse bool: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run privileged RL-Games checkpoint with task observations."
    )
    parser.add_argument("--num_envs", type=int, default=1024, help="Number of parallel envs.")
    parser.add_argument("--steps", type=int, default=1500, help="Number of simulation steps.")
    parser.add_argument(
        "--headless",
        type=_str2bool,
        nargs="?",
        const=False,
        default=False,
        help="Headless mode (True/False).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CKPT,
        help="Path to RL-Games .pth checkpoint.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG,
        help="Training YAML config path.",
    )
    parser.add_argument(
        "--env_name",
        type=str,
        default=None,
        help="Override env_name (default: config env_name or payload_compensation_task_teacher).",
    )
    parser.add_argument(
        "--deterministic",
        type=_str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Use mean action instead of sampling.",
    )
    parser.add_argument(
        "--early_plot",
        type=_str2bool,
        nargs="?",
        const=True,
        default=True,
        help="记录并绘制提前结束回合的随机化分布",
    )
    parser.add_argument(
        "--show_plot",
        type=_str2bool,
        nargs="?",
        const=True,
        default=True,
        help="运行结束后弹出图像窗口",
    )
    parser.add_argument(
        "--save_plot",
        type=_str2bool,
        nargs="?",
        const=True,
        default=False,
        help="保存图像到 AERIAL_SAVE_PLOT 指定路径",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="保存图像到指定路径（优先于 AERIAL_SAVE_PLOT）",
    )

    parser.add_argument(
        "--show_details",
        type=_str2bool,
        nargs="?",
        const=True,
        default=False,
        help="是否显示单环境详细分析图（默认多环境时不显示）",
    )
    return parser.parse_args()


def load_training_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_checkpoint(checkpoint_path: str) -> Tuple[Dict[str, Any], int]:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model_state = ckpt.get("model", {})
    action_tensor = model_state.get("a2c_network.mu.weight")
    if action_tensor is None:
        raise RuntimeError("Checkpoint missing a2c_network.mu.weight, cannot infer action_dim.")
    action_dim = int(action_tensor.shape[0])
    return ckpt, action_dim


def build_model(
    cfg: Dict[str, Any],
    obs_dim: int,
    action_dim: int,
    num_envs: int,
    device: torch.device,
):
    params = cfg.get("params", {})
    builder = model_builder.ModelBuilder()
    model = builder.load(params)
    model_cfg = {
        "actions_num": action_dim,
        "input_shape": (obs_dim,),
        "num_seqs": num_envs,
        "value_size": 1,
        "normalize_value": bool(params.get("config", {}).get("normalize_value", False)),
        "normalize_input": bool(params.get("config", {}).get("normalize_input", False)),
    }
    network = model.build(model_cfg)
    network.to(device)
    network.eval()
    return network


def _to_device(states, device: torch.device):
    if states is None:
        return None
    if isinstance(states, (list, tuple)):
        return tuple(state.to(device) for state in states)
    return states.to(device)


def _reset_rnn_states(states, env_ids: torch.Tensor):
    if states is None:
        return None
    if env_ids is None or env_ids.numel() == 0:
        return states
    if not torch.is_tensor(env_ids):
        env_ids = torch.as_tensor(env_ids, device=states[0].device if isinstance(states, (list, tuple)) else states.device)
    if env_ids.numel() == 0:
        return states

    def _zero_state(state):
        if state is None:
            return state
        if state.dim() >= 2:
            state[:, env_ids, :] = 0.0
        else:
            state.zero_()
        return state

    if isinstance(states, (list, tuple)):
        return tuple(_zero_state(state) for state in states)
    return _zero_state(states)


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


def plot_multi_env_results(
    all_z_history: List[np.ndarray],
    all_euler_history: List[np.ndarray],
    all_policy_actions: List[np.ndarray],
    all_teacher_actions: Optional[List[np.ndarray]] = None,
    release_history: Optional[List[Tuple[int, int]]] = None,
    save_path: Optional[str] = None,
):
    """
    绘制多环境的平均值曲线（带标准差阴影）。
    
    Args:
        all_z_history: List of [num_envs] arrays, one per step
        all_euler_history: List of [num_envs, 3] arrays
        all_policy_actions: List of [num_envs, action_dim] arrays
        all_teacher_actions: Optional list of [num_envs, action_dim] arrays
        release_history: List of (step, payload_idx) tuples (from env_id=0)
        save_path: Optional path to save figures
    """
    if not all_z_history:
        return
    
    release_history = release_history or []
    num_steps = len(all_z_history)
    steps = np.arange(num_steps)
    
    # Stack to [steps, num_envs, ...]
    z_arr = np.stack(all_z_history, axis=0)  # [steps, num_envs]
    euler_arr = np.stack(all_euler_history, axis=0)  # [steps, num_envs, 3]
    action_arr = np.stack(all_policy_actions, axis=0)  # [steps, num_envs, action_dim]
    
    # 计算平均值和标准差
    z_mean = z_arr.mean(axis=1)
    z_std = z_arr.std(axis=1)
    
    # ====== 修复角度统计问题 ======
    # 问题：不同环境的初始朝向不同，直接平均没有意义
    # 解决：统计相对于初始角度的变化量 (delta)
    euler_initial = euler_arr[0:1, :, :]  # [1, num_envs, 3] 第0步的角度
    euler_delta = euler_arr - euler_initial  # [steps, num_envs, 3] 相对变化
    
    # 对 delta 进行 unwrap（处理 ±π 跳变）
    euler_delta_unwrapped = np.zeros_like(euler_delta)
    for env_idx in range(euler_delta.shape[1]):
        for angle_idx in range(3):
            euler_delta_unwrapped[:, env_idx, angle_idx] = np.unwrap(euler_delta[:, env_idx, angle_idx])
    
    # 现在可以安全地计算平均变化和标准差
    euler_delta_mean = euler_delta_unwrapped.mean(axis=1)  # [steps, 3]
    euler_delta_std = euler_delta_unwrapped.std(axis=1)
    
    action_mean = action_arr.mean(axis=1)  # [steps, action_dim]
    action_std = action_arr.std(axis=1)
    
    # Convert euler delta to degrees
    euler_delta_mean_deg = np.rad2deg(euler_delta_mean)
    euler_delta_std_deg = np.rad2deg(euler_delta_std)
    
    num_envs = z_arr.shape[1]
    
    # ====== Combined Figure 1: State & Stability (Z, Euler Delta, Angular MAE) ======
    fig_state, axes_state = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    ax_z, ax_euler, ax_mae = axes_state
    fig_state.suptitle(f"飞行状态与稳定性分析 (N={num_envs})", fontsize=16, fontweight='bold')
    
    # --- Subplot 1: Z-Height ---
    ax_z.plot(steps, z_mean, label=f"Z 高度 (平均)", color='C0', linewidth=2)
    ax_z.fill_between(steps, z_mean - z_std, z_mean + z_std, alpha=0.3, color='C0', label="±1σ")
    ax_z.set_ylabel("Z 轴高度 (m)")
    ax_z.set_title("1. 无人机 Z 轴高度")
    ax_z.grid(True, alpha=0.3)
    ax_z.legend(loc='upper right')
    
    for release_step, _ in release_history:
        ax_z.axvline(release_step, color='r', linestyle='--', alpha=0.6)

    # --- Subplot 2: Relative Attitude Change ---
    colors_euler = ['C1', 'C2', 'C3']
    labels_euler = ['ΔRoll', 'ΔPitch', 'ΔYaw']
    for i, (color, label) in enumerate(zip(colors_euler, labels_euler)):
        ax_euler.plot(steps, euler_delta_mean_deg[:, i], label=f"{label} (°)", color=color, linewidth=1.5)
        ax_euler.fill_between(
            steps, 
            euler_delta_mean_deg[:, i] - euler_delta_std_deg[:, i],
            euler_delta_mean_deg[:, i] + euler_delta_std_deg[:, i],
            alpha=0.2, color=color
        )
    ax_euler.set_ylabel("角度变化 (°)")
    ax_euler.set_title("2. 姿态角变化 (相对于初始时刻)")
    ax_euler.grid(True, alpha=0.3)
    ax_euler.legend(loc='upper right')
    ax_euler.axhline(0, color='gray', linestyle=':', alpha=0.5)
    
    for release_step, _ in release_history:
        ax_euler.axvline(release_step, color='r', linestyle='--', alpha=0.6)

    # --- Subplot 3: Absolute Angular MAE ---
    # 计算相对于初始角度的绝对偏差平均值: Mean(|Angle - Initial|)
    euler_abs_delta_mean_deg = np.rad2deg(np.abs(euler_delta_unwrapped).mean(axis=1)) # [steps, 3]
    euler_abs_delta_std_deg = np.rad2deg(np.abs(euler_delta_unwrapped).std(axis=1))
    
    mae_colors = ['C1', 'C2', 'C3'] # Delta Roll, Pitch, Yaw
    mae_labels = ['|ΔRoll|', '|ΔPitch|', '|ΔYaw|']
    
    for i in range(3):
        ax_mae.plot(steps, euler_abs_delta_mean_deg[:, i], label=f'{mae_labels[i]} (Avg)', color=mae_colors[i], linewidth=1.5)
        ax_mae.fill_between(steps, 
                           euler_abs_delta_mean_deg[:, i] - euler_abs_delta_std_deg[:, i],
                           euler_abs_delta_mean_deg[:, i] + euler_abs_delta_std_deg[:, i],
                           color=mae_colors[i], alpha=0.1)

    ax_mae.set_ylabel("绝对偏差 (°)")
    ax_mae.set_xlabel("步数")
    ax_mae.set_title("3. 平均绝对角度漂移 (稳定性指标)")
    ax_mae.grid(True, alpha=0.3)
    ax_mae.legend(loc='upper right')
    
    for release_step, _ in release_history:
        ax_mae.axvline(release_step, color='r', linestyle='--', alpha=0.5)

    fig_state.tight_layout()


    # ====== Combined Figure 2: Policy Analysis (Imitation Error, Correlation) ======
    action_dim = action_mean.shape[1]
    has_teacher = all_teacher_actions is not None and len(all_teacher_actions) == num_steps
    
    if has_teacher:
        # 布局：上部分为 Imitation Error (N行), 下部分为 Correlation (1行)
        # 使用 GridSpec 或直接 subplots 调整
        total_rows = action_dim + 1
        fig_analysis, axes_analysis = plt.subplots(total_rows, 1, figsize=(12, 3.5 * total_rows), sharex=True)
        fig_analysis.suptitle(f"策略行为分析 (Policy Analysis)", fontsize=16, fontweight='bold')
        
        # --- Part A: Imitation Error ---
        teacher_arr = np.stack(all_teacher_actions, axis=0)
        imitation_error = np.abs(action_arr - teacher_arr)
        imitation_error_mean = imitation_error.mean(axis=1)
        imitation_error_std = imitation_error.std(axis=1)
        
        action_labels = ['Thrust 误差 (%)'] + [f'Torque {i} 误差 (%)' for i in range(1, action_dim)]
        
        for i in range(action_dim):
            ax = axes_analysis[i]
            # Convert to percentage
            mean_pct = imitation_error_mean[:, i] * 100.0
            std_pct = imitation_error_std[:, i] * 100.0
            
            ax.plot(steps, mean_pct, label='平均误差', color='C3', linewidth=1.5)
            ax.fill_between(
                steps,
                mean_pct - std_pct,
                mean_pct + std_pct,
                alpha=0.3, color='C3'
            )
            overall_mae = mean_pct.mean()
            ax.axhline(overall_mae, color='C3', linestyle='--', alpha=0.5)
            ax.text(num_steps * 0.02, overall_mae * 1.1, f'MAE={overall_mae:.2f}%', fontsize=9, color='C3')
            
            ax.set_ylabel(action_labels[i])
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right')
            for release_step, _ in release_history:
                ax.axvline(release_step, color='r', linestyle=':', alpha=0.4)
                
        # --- Part B: Correlation ---
        ax_corr = axes_analysis[-1]
        
        # Calculate Correlation
        correlation_per_step = np.zeros((num_steps, action_dim))
        for t in range(num_steps):
            for d in range(action_dim):
                p = action_arr[t, :, d]
                te = teacher_arr[t, :, d]
                if np.std(te) > 1e-6:
                    corr = np.corrcoef(p, te)[0, 1]
                    correlation_per_step[t, d] = corr ** 2
                else:
                    correlation_per_step[t, d] = 1.0
                    
        corr_labels = ['Thrust', 'Torque 1', 'Torque 2'][:action_dim]
        # Correlation colors
        c_colors = ['C0', 'C1', 'C2']
        for i in range(action_dim):
            ax_corr.plot(steps, correlation_per_step[:, i], label=f'{corr_labels[i]} R²', 
                        color=c_colors[i], linewidth=1.5, alpha=0.8)
            avg_r2 = correlation_per_step[:, i].mean()
            ax_corr.axhline(avg_r2, color=c_colors[i], linestyle=':', alpha=0.3)
        
        ax_corr.set_ylabel("R² (相关性)")
        ax_corr.set_xlabel("步数")
        ax_corr.set_ylim(-0.1, 1.1)
        ax_corr.axhline(1.0, color='gray', linestyle='--', alpha=0.3)
        ax_corr.grid(True, alpha=0.3)
        ax_corr.legend(loc='lower right')
        
        for release_step, _ in release_history:
            ax_corr.axvline(release_step, color='r', linestyle=':', alpha=0.4)
            
        fig_analysis.tight_layout()
    else:
        print("[Warning] No teacher actions, skipping Policy Analysis plot.")

    # ====== 保存图像 ======
    if save_path:
        base_path = os.path.splitext(save_path)[0]
        fig_state.savefig(f"{base_path}_state_stability.png", dpi=150, bbox_inches="tight")
        if has_teacher:
            fig_analysis.savefig(f"{base_path}_policy_analysis.png", dpi=150, bbox_inches="tight")
        print(f"[合并图表] 已保存到 {base_path}_*.png")


def plot_results(
    z_history,
    euler_history,
    release_history,
    policy_actions,
    teacher_actions=None,
    pos_history=None,
    target_history=None,
    ideal_circle=None,
    save_path: Optional[str] = None,
    obs_history=None,
):
    if not z_history:
        print("无可绘制数据。")
        return
        
    if obs_history is not None:
        _plot_observation_analysis(obs_history, save_path)

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
            ax_xy.plot(
                cx + radius * np.cos(theta),
                cy + radius * np.sin(theta),
                "--",
                color="C1",
                alpha=0.6,
                label="理想圆轨迹",
            )
        for release_step, _ in release_history:
            if 0 <= release_step < len(xy):
                ax_xy.scatter(
                    xy[release_step, 0],
                    xy[release_step, 1],
                    color="r",
                    s=25,
                    marker="o",
                    alpha=0.6,
                )
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
        teacher_arr = (
            np.vstack(teacher_actions)
            if teacher_actions is not None and len(teacher_actions) == len(policy_actions)
            else None
        )
        fig_act, axes = plt.subplots(
            action_dim, 1, figsize=(10, max(4, 2 * action_dim)), sharex=True
        )
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
        
        # 新增：力矩-角度关系图 (Roll 和 Pitch 分开)
        if action_dim >= 3 and euler_history:
            # 使用已经 unwrap 处理过的角度数据
            roll_deg = eulers_deg[:, 0]
            pitch_deg = eulers_deg[:, 1]
            
            # 策略的力矩输出 (dim 1 = roll torque, dim 2 = pitch torque)
            policy_roll_torque = act_arr[:, 1]
            policy_pitch_torque = act_arr[:, 2]
            
            # 教师的力矩输出（如有）
            teacher_roll_torque = teacher_arr[:, 1] if teacher_arr is not None else None
            teacher_pitch_torque = teacher_arr[:, 2] if teacher_arr is not None else None
            
            fig_torque_angle, (ax_roll, ax_pitch) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            
            # --- Roll ---
            ax_roll_angle = ax_roll
            ax_roll_torque = ax_roll.twinx()
            
            # 角度（左轴）
            line_roll_angle, = ax_roll_angle.plot(steps, roll_deg, color='C0', linewidth=1.5, label='Roll 角度 (°)')
            ax_roll_angle.set_ylabel('Roll 角度 (°)', color='C0')
            ax_roll_angle.tick_params(axis='y', labelcolor='C0')
            ax_roll_angle.axhline(0, color='C0', linestyle=':', alpha=0.3)
            
            # 力矩（右轴）
            line_roll_policy, = ax_roll_torque.plot(steps, policy_roll_torque, color='C1', linewidth=1.2, label='Policy Roll 力矩')
            if teacher_roll_torque is not None:
                line_roll_teacher, = ax_roll_torque.plot(steps, teacher_roll_torque, color='C2', linestyle='--', linewidth=1.2, label='Teacher Roll 力矩')
            ax_roll_torque.set_ylabel('Roll 力矩 (归一化)', color='C1')
            ax_roll_torque.tick_params(axis='y', labelcolor='C1')
            ax_roll_torque.axhline(0, color='C1', linestyle=':', alpha=0.3)
            
            # 标注释放点
            for release_step, payload_idx in release_history:
                ax_roll.axvline(release_step, color='r', linestyle=':', alpha=0.5)
            
            # 图例
            lines = [line_roll_angle, line_roll_policy]
            labels = [line_roll_angle.get_label(), line_roll_policy.get_label()]
            if teacher_roll_torque is not None:
                lines.append(line_roll_teacher)
                labels.append(line_roll_teacher.get_label())
            ax_roll.legend(lines, labels, loc='upper right')
            ax_roll.set_title('Roll: 角度 vs 力矩补偿\n(正力矩 → 正角度变化)')
            ax_roll.grid(True, alpha=0.3)
            
            # --- Pitch ---
            ax_pitch_angle = ax_pitch
            ax_pitch_torque = ax_pitch.twinx()
            
            # 角度（左轴）
            line_pitch_angle, = ax_pitch_angle.plot(steps, pitch_deg, color='C3', linewidth=1.5, label='Pitch 角度 (°)')
            ax_pitch_angle.set_ylabel('Pitch 角度 (°)', color='C3')
            ax_pitch_angle.tick_params(axis='y', labelcolor='C3')
            ax_pitch_angle.axhline(0, color='C3', linestyle=':', alpha=0.3)
            
            # 力矩（右轴）
            line_pitch_policy, = ax_pitch_torque.plot(steps, policy_pitch_torque, color='C4', linewidth=1.2, label='Policy Pitch 力矩')
            if teacher_pitch_torque is not None:
                line_pitch_teacher, = ax_pitch_torque.plot(steps, teacher_pitch_torque, color='C5', linestyle='--', linewidth=1.2, label='Teacher Pitch 力矩')
            ax_pitch_torque.set_ylabel('Pitch 力矩 (归一化)', color='C4')
            ax_pitch_torque.tick_params(axis='y', labelcolor='C4')
            ax_pitch_torque.axhline(0, color='C4', linestyle=':', alpha=0.3)
            
            # 标注释放点
            for release_step, payload_idx in release_history:
                ax_pitch.axvline(release_step, color='r', linestyle=':', alpha=0.5)
            
            # 图例
            lines = [line_pitch_angle, line_pitch_policy]
            labels = [line_pitch_angle.get_label(), line_pitch_policy.get_label()]
            if teacher_pitch_torque is not None:
                lines.append(line_pitch_teacher)
                labels.append(line_pitch_teacher.get_label())
            ax_pitch.legend(lines, labels, loc='upper right')
            ax_pitch.set_title('Pitch: 角度 vs 力矩补偿\n(正力矩 → 正角度变化)')
            ax_pitch.grid(True, alpha=0.3)
            
            ax_pitch.set_xlabel('步数')
            fig_torque_angle.suptitle('力矩输出与姿态角关系', fontsize=14, fontweight='bold')
            fig_torque_angle.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        if policy_actions:
            fig_act.savefig(
                os.path.splitext(save_path)[0] + "_actions.png",
                dpi=150,
                bbox_inches="tight",
            )
        print(f"已保存绘图到 {save_path}")


def _capture_episode_meta(task, env_ids: torch.Tensor) -> List[Dict[str, Any]]:
    payload_manager = task.payload_manager
    metas: List[Dict[str, Any]] = []
    for env_id in env_ids.long().tolist():
        offsets = payload_manager.offsets[env_id].detach().cpu().numpy()
        com_offset = payload_manager.com_offset_body[env_id].detach().cpu().numpy()
        target_pos = task.target_position[env_id].detach().cpu().numpy()
        offset_r = np.linalg.norm(offsets[:, :2], axis=1)
        offset_z = offsets[:, 2]
        metas.append(
            {
                "env_id": int(env_id),
                "payload_mass": float(payload_manager.payload_mass_per_env[env_id].item()),
                "payload_mass_total": float(payload_manager.current_payload_mass[env_id].item()),
                "com_offset": com_offset,
                "com_offset_norm": float(np.linalg.norm(com_offset)),
                "offsets": offsets,
                "offset_r_max": float(np.max(offset_r)) if offset_r.size > 0 else 0.0,
                "offset_z_abs_max": float(np.max(np.abs(offset_z))) if offset_z.size > 0 else 0.0,
                "release_start_step": int(payload_manager.next_release_step[env_id].item()),
                "release_order": payload_manager.release_orders[env_id]
                .detach()
                .cpu()
                .numpy()
                .tolist(),
                "target_xy_radius": float(np.linalg.norm(target_pos[:2])),
                "target_z": float(target_pos[2]),
            }
        )
    return metas


def plot_early_episode_stats(
    records: List[Dict[str, Any]],
    episode_len_limit: int,
    save_path: Optional[str] = None,
) -> None:
    if not records:
        print("未发现提前结束的回合。")
        return

    payload_mass = np.array([r["payload_mass"] for r in records], dtype=np.float32)
    com_norm = np.array([r["com_offset_norm"] for r in records], dtype=np.float32)
    offset_r_max = np.array([r["offset_r_max"] for r in records], dtype=np.float32)
    offset_z_abs_max = np.array([r["offset_z_abs_max"] for r in records], dtype=np.float32)
    release_start = np.array([r["release_start_step"] for r in records], dtype=np.float32)
    target_xy = np.array([r["target_xy_radius"] for r in records], dtype=np.float32)
    steps = np.array([r["steps"] for r in records], dtype=np.float32)
    rewards = np.array([r["reward"] for r in records], dtype=np.float32)
    reasons = [r["reason"] for r in records]

    crash_count = sum(1 for r in reasons if r == "crash")
    timeout_count = sum(1 for r in reasons if r == "timeout")

    print(
        f"[EarlyEpisodes] count={len(records)} / total<{episode_len_limit} steps, "
        f"crash={crash_count}, timeout={timeout_count}"
    )
    print(
        f"payload_mass range: {payload_mass.min():.4f} ~ {payload_mass.max():.4f}, "
        f"com_norm range: {com_norm.min():.4f} ~ {com_norm.max():.4f}, "
        f"release_start range: {release_start.min():.0f} ~ {release_start.max():.0f}"
    )

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    ax = axes.ravel()

    ax[0].hist(payload_mass, bins=20, color="C0", alpha=0.75)
    ax[0].set_title("提前结束：payload_mass 分布")
    ax[0].set_xlabel("payload_mass")
    ax[0].set_ylabel("count")

    ax[1].hist(com_norm, bins=20, color="C1", alpha=0.75)
    ax[1].set_title("提前结束：COM 偏移范数")
    ax[1].set_xlabel("|r_com|")
    ax[1].set_ylabel("count")

    ax[2].hist(release_start, bins=20, color="C2", alpha=0.75)
    ax[2].set_title("提前结束：release_start_step")
    ax[2].set_xlabel("release_start_step")
    ax[2].set_ylabel("count")

    ax[3].scatter(payload_mass, steps, s=18, alpha=0.7, color="C3")
    ax[3].set_title("steps vs payload_mass")
    ax[3].set_xlabel("payload_mass")
    ax[3].set_ylabel("steps")

    ax[4].scatter(com_norm, steps, s=18, alpha=0.7, color="C4")
    ax[4].set_title("steps vs |r_com|")
    ax[4].set_xlabel("|r_com|")
    ax[4].set_ylabel("steps")

    ax[5].scatter(offset_r_max, steps, s=18, alpha=0.7, color="C5", label="offset_r_max")
    ax[5].scatter(offset_z_abs_max, steps, s=18, alpha=0.7, color="C6", label="|offset_z|max")
    ax[5].set_title("steps vs offset 范围")
    ax[5].set_xlabel("offset range (m)")
    ax[5].set_ylabel("steps")
    ax[5].legend()

    fig.suptitle("提前结束回合的随机化参数分布与关联")

    if save_path:
        fig.savefig(
            os.path.splitext(save_path)[0] + "_early_stats.png",
            dpi=150,
            bbox_inches="tight",
        )


def main() -> None:
    args = parse_args()
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    cfg = load_training_config(args.config) or {}
    env_name = (
        args.env_name
        or cfg.get("params", {}).get("config", {}).get("env_name", None)
        or DEFAULT_ENV_NAME
    )

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
    priv_dim = task.task_config.privileged_observation_space_dim
    cfg_priv_dim = cfg.get("params", {}).get("network", {}).get("privileged_dim")
    if cfg_priv_dim is not None and int(cfg_priv_dim) != int(priv_dim):
        print(f"Warning: task priv_dim={priv_dim}, cfg privileged_dim={cfg_priv_dim}")

    checkpoint, checkpoint_action_dim = load_checkpoint(args.checkpoint)
    action_dim = task.task_config.action_space_dim
    if action_dim != checkpoint_action_dim:
        task.close()
        raise RuntimeError(
            f"Action dim mismatch: task={action_dim}, checkpoint={checkpoint_action_dim}"
        )

    device = torch.device(task.device)
    
    # Auto-detect normalize_input from checkpoint
    has_rms = any(k.startswith("running_mean_std") for k in checkpoint["model"].keys())
    
    if "params" not in cfg: cfg["params"] = {}
    if "config" not in cfg["params"]: cfg["params"]["config"] = {}
    
    if has_rms:
        print("Checkpoint indicates normalize_input=True. Forcing config to match.")
        cfg["params"]["config"]["normalize_input"] = True
    else:
        print("Checkpoint indicates normalize_input=False. Forcing config to match.")
        cfg["params"]["config"]["normalize_input"] = False

    model = build_model(cfg, obs_dim, action_dim, task.sim_env.num_envs, device)
    model.load_state_dict(checkpoint["model"], strict=True)
    if getattr(model, "normalize_input", False) and "running_mean_std" in checkpoint:
        model.running_mean_std.load_state_dict(checkpoint["running_mean_std"])

    task.reset()
    rnn_states = _to_device(model.get_default_rnn_state(), device)
    
    # ====== 多环境数据收集（用于计算平均值和标准差） ======
    # 每个 list 的元素是一个 step 的所有环境数据 [num_envs, ...]
    all_z_history: List[np.ndarray] = []        # [num_envs]
    all_euler_history: List[np.ndarray] = []    # [num_envs, 3]
    all_pos_history: List[np.ndarray] = []      # [num_envs, 3]
    all_policy_actions: List[np.ndarray] = []   # [num_envs, action_dim]
    all_teacher_actions: List[np.ndarray] = []  # [num_envs, action_dim]
    
    # 单环境历史（保留向后兼容性）
    z_history: List[float] = []
    euler_history: List[np.ndarray] = []
    pos_history: List[np.ndarray] = []
    target_history: List[np.ndarray] = []
    release_history: List[Tuple[int, int]] = []
    policy_actions: List[np.ndarray] = []
    teacher_actions: List[np.ndarray] = []
    obs_history: List[np.ndarray] = []  # Record full observation vector
    
    early_enabled = bool(args.early_plot)
    episode_len_limit = int(getattr(task.task_config, "episode_len_steps", args.steps)) + 1
    episode_steps = torch.zeros(task.sim_env.num_envs, device=device, dtype=torch.long)
    episode_rewards = torch.zeros(task.sim_env.num_envs, device=device, dtype=torch.float32)
    episode_has_meta = torch.zeros(task.sim_env.num_envs, device=device, dtype=torch.bool)
    episode_meta: List[Optional[Dict[str, Any]]] = [None] * task.sim_env.num_envs
    early_records: List[Dict[str, Any]] = []
    total_episodes = 0
    num_envs = task.sim_env.num_envs

    print(
        f"Running {env_name}: envs={num_envs}, obs_dim={obs_dim}, "
        f"priv_dim={priv_dim}, action_dim={action_dim}"
    )
    print(f"[多环境模式] 将收集所有 {num_envs} 个环境的数据并计算平均值±标准差")
    
    with torch.no_grad():
        for step in range(args.steps):
            if early_enabled:
                missing_meta = torch.nonzero(~episode_has_meta, as_tuple=False).squeeze(-1)
                if missing_meta.numel() > 0:
                    metas = _capture_episode_meta(task, missing_meta)
                    for env_idx, meta in zip(missing_meta.long().tolist(), metas):
                        episode_meta[env_idx] = meta
                    episode_has_meta[missing_meta] = True

            obs = torch.as_tensor(
                task.task_obs["observations"], device=device, dtype=torch.float32
            )
            priv = task.task_obs.get("privileged_obs", None)
            if priv is not None:
                priv = torch.as_tensor(priv, device=device, dtype=torch.float32)
            input_dict = {
                "is_train": False,
                "prev_actions": None,
                "obs": obs,
                "privileged_obs": priv,
                "rnn_states": rnn_states,
                "seq_length": 1,
            }
            result = model(input_dict)
            action = result["mus"] if args.deterministic else result["actions"]
            action = torch.clamp(action, -1.0, 1.0)
            task_obs, rewards, terms, truncs, infos = task.step(action)
            rnn_states = _to_device(result.get("rnn_states", None), device)
            done_envs = torch.nonzero(terms | truncs, as_tuple=False).squeeze(-1)
            rnn_states = _reset_rnn_states(rnn_states, done_envs)
            if early_enabled:
                episode_steps += 1
                episode_rewards += rewards.detach()
                if done_envs.numel() > 0:
                    total_episodes += int(done_envs.numel())
                    for env_id in done_envs.long().tolist():
                        ep_steps = int(episode_steps[env_id].item())
                        ep_reward = float(episode_rewards[env_id].item())
                        reason = "crash" if bool(terms[env_id].item()) else "timeout"
                        if ep_steps < episode_len_limit:
                            meta = episode_meta[env_id] or {}
                            record = {
                                **meta,
                                "steps": ep_steps,
                                "reward": ep_reward,
                                "reason": reason,
                            }
                            early_records.append(record)
                        episode_steps[env_id] = 0
                        episode_rewards[env_id] = 0.0
                        episode_has_meta[env_id] = False
                        episode_meta[env_id] = None

            # ====== 收集所有环境的数据 ======
            all_pos = task.obs_dict["robot_position"].detach().cpu().numpy()  # [num_envs, 3]
            all_quat = task.obs_dict["robot_orientation"]  # [num_envs, 4]
            all_euler = get_euler_xyz_tensor(all_quat).detach().cpu().numpy()  # [num_envs, 3]
            all_actions = action.detach().cpu().numpy()  # [num_envs, action_dim]
            
            all_z_history.append(all_pos[:, 2].copy())
            all_euler_history.append(all_euler.copy())
            all_pos_history.append(all_pos.copy())
            all_policy_actions.append(all_actions.copy())
            
            # Teacher actions (所有环境)
            if isinstance(infos, dict) and "teacher_actions" in infos:
                teacher_act = torch.as_tensor(infos["teacher_actions"]).detach().cpu().numpy()
                all_teacher_actions.append(teacher_act.copy())
            elif hasattr(task, "teacher_residual"):
                teacher_act = task.teacher_residual.detach().cpu().numpy()
                all_teacher_actions.append(teacher_act.copy())
            
            # ====== 单环境历史（env_id=0，保持兼容） ======
            env_id = 0
            pos = all_pos[env_id]
            tgt = task.target_position[env_id].detach().cpu().numpy()
            euler = all_euler[env_id]

            pos_history.append(pos.copy())
            target_history.append(tgt.copy())
            z_history.append(pos[2])
            euler_history.append(euler)
            policy_actions.append(all_actions[env_id])
            obs_history.append(obs[env_id].detach().cpu().numpy())

            if len(all_teacher_actions) > 0:
                teacher_actions.append(all_teacher_actions[-1][env_id])

            if task.payload_manager.just_released_flag[env_id]:
                payload_idx = int(task.payload_manager.last_release_index[env_id].item())
                release_history.append((step, payload_idx))

    ideal_circle = None
    traj_cfg = getattr(task.task_config, "trajectory_parameters", None) or {}
    if str(traj_cfg.get("type", "")).lower() == "circle":
        cx, cy, _ = traj_cfg.get("center", [0.0, 0.0, 0.0])
        radius = float(traj_cfg.get("radius", 0.0))
        ideal_circle = (cx, cy, radius)

    save_path = None
    if args.save_plot:
        save_path = args.save_path or os.environ.get("AERIAL_SAVE_PLOT")
        if save_path is None:
            print("Warning: save_plot=True 但未设置 AERIAL_SAVE_PLOT 或 --save_path")
    
    # ====== 绘制多环境平均值图 ======
    if all_z_history:
        plot_multi_env_results(
            all_z_history=all_z_history,
            all_euler_history=all_euler_history,
            all_policy_actions=all_policy_actions,
            all_teacher_actions=all_teacher_actions if all_teacher_actions else None,
            release_history=release_history,
            save_path=save_path,
        )
    
    # 单环境详细图（保持原有逻辑）
    # 单环境详细图
    # 逻辑调整：只有在 num_envs=1 或者用户强制 --show_details 时才显示
    if num_envs == 1 or args.show_details:
        plot_results(
            z_history,
            euler_history,
            release_history,
            policy_actions,
            teacher_actions=teacher_actions,
            pos_history=pos_history,
            target_history=target_history,
            ideal_circle=ideal_circle,
            save_path=save_path,
            obs_history=obs_history,
        )
    else:
        print("[Info] 多环境模式下默认隐藏单环境详细波形图 (使用 --show_details 开启)")

    if early_enabled:
        plot_early_episode_stats(
            early_records,
            episode_len_limit,
            save_path=save_path,
        )
    if args.show_plot:
        plt.show()
    else:
        plt.close("all")

    try:
        task.close()
    except AttributeError as exc:
        print(f"Warning: task.close() failed ({exc}), skipping explicit cleanup.")

def _plot_observation_analysis(obs_history, save_path=None):
    if not obs_history:
        return
    obs_arr = np.array(obs_history)
    steps = np.arange(len(obs_arr))
    
    fig, axes = plt.subplots(5, 1, figsize=(12, 18), sharex=True)
    
    # 1. Rotation Matrix (0:9)
    # Plot diagonal elements to check for deviation from 1.0
    axes[0].plot(steps, obs_arr[:, 0], label="r00", alpha=0.5)
    axes[0].plot(steps, obs_arr[:, 4], label="r11", alpha=0.5)
    axes[0].plot(steps, obs_arr[:, 8], label="r22", alpha=0.5)
    # Plot key off-diagonal elements
    axes[0].plot(steps, obs_arr[:, 2], label="r02 (Pitch)", color='red', alpha=0.8)
    axes[0].plot(steps, obs_arr[:, 6], label="r20", color='red', alpha=0.4, linestyle="--")
    axes[0].plot(steps, obs_arr[:, 5], label="r12 (Roll)", color='blue', alpha=0.8)
    axes[0].plot(steps, obs_arr[:, 7], label="r21", color='blue', alpha=0.4, linestyle="--")
    
    axes[0].set_title("Rotation Matrix (r02=Pitch-like, r12=Roll-like)")
    axes[0].legend(loc="upper right", ncol=4, fontsize=8)
    axes[0].grid(True, alpha=0.3)
    
    # 2. Angular Velocity (9:12)
    axes[1].plot(steps, obs_arr[:, 9], label="wx", color='C0')
    axes[1].plot(steps, obs_arr[:, 10], label="wy", color='C1')
    axes[1].plot(steps, obs_arr[:, 11], label="wz", color='C2')
    axes[1].set_title("Angular Velocity (Body-Frame)")
    axes[1].set_ylabel("rad/s")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)
    
    # 3. Payload Masks (12:16)
    if obs_arr.shape[1] > 15:
        axes[2].step(steps, obs_arr[:, 12], label="M0 (Front R)", where='post')
        axes[2].step(steps, obs_arr[:, 13], label="M1 (Front L)", where='post')
        axes[2].step(steps, obs_arr[:, 14], label="M2 (Rear L)", where='post')
        axes[2].step(steps, obs_arr[:, 15], label="M3 (Rear R)", where='post')
        axes[2].set_title("Payload Attachment Masks (1=Attached, 0=Released)")
        axes[2].set_ylim(-0.1, 1.1)
        axes[2].legend(loc="upper right", ncol=4)
        axes[2].grid(True, alpha=0.3)

    # 4. Previous Actions (17:20) & Warning Flag (16)
    if obs_arr.shape[1] > 19:
        ax3 = axes[3]
        ax3_r = ax3.twinx()
        
        # Actions on left axis
        ax3.plot(steps, obs_arr[:, 17], label="prev_thrust", color='C4', alpha=0.8)
        ax3.plot(steps, obs_arr[:, 18], label="prev_roll", color='C5', alpha=0.8)
        ax3.plot(steps, obs_arr[:, 19], label="prev_pitch", color='C6', alpha=0.8)
        ax3.set_ylabel("Action (-1 ~ 1)")
        
        # Warning flag on right axis
        line_warn, = ax3_r.plot(steps, obs_arr[:, 16], label="warning", color="red", alpha=0.3, linewidth=2)
        ax3_r.set_ylabel("Release Warning", color="red")
        ax3_r.tick_params(axis='y', labelcolor="red")
        ax3_r.set_ylim(-0.1, 1.1)
        
        ax3.set_title("Previous Actions & Release Warning")
        ax3.legend(loc="upper left")
        ax3.grid(True, alpha=0.3)

    # 5. Extra Observation Info (20:22)
    if obs_arr.shape[1] > 21:
        axes[4].plot(steps, obs_arr[:, 20], label="Next Release ID", color='purple')
        axes[4].plot(steps, obs_arr[:, 21], label="Next Release Mass", color='orange')
        axes[4].set_title("Additional Meta-Info (Next Release Prediction)")
        axes[4].legend(loc="upper right")
        axes[4].grid(True, alpha=0.3)
    else:
        axes[4].text(0.5, 0.5, "No additional observation dimensions (20-21)", 
                   ha='center', va='center', transform=axes[4].transAxes)

    fig.suptitle("Observation Space Analysis", fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        base, ext = os.path.splitext(save_path)
        fig.savefig(f"{base}_obs_analysis{ext}", dpi=150)
    pass


if __name__ == "__main__":
    main()
