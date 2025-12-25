"""Quick sanity check for teacher residual: apply expert residual as action and log basic stats."""
import argparse
import os
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np

from aerial_gym.config.task_config import payload_compensation_task_teacher_config as teacher_cfg
from aerial_gym.task.payload_compensation_task.payload_compensation_task import PayloadCompensationTask
from aerial_gym.utils.math import get_euler_xyz_tensor

import torch

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False


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


def run_demo(steps=2000, headless=False, device="cuda:0"):
    cfg = teacher_cfg.task_config
    cfg.headless = headless
    cfg.device = device
    env = PayloadCompensationTask(cfg)
    obs, *_ = env.reset()
    positions = []
    z_history: List[float] = []
    euler_history: List[np.ndarray] = []
    pos_history: List[np.ndarray] = []
    target_history: List[np.ndarray] = []
    release_history: List[Tuple[int, int]] = []
    policy_actions: List[np.ndarray] = []
    teacher_actions: List[np.ndarray] = []
    max_residual = torch.zeros(env.task_config.action_space_dim, device=device)
    for step in range(steps):
        # 在 Teacher 模式下，env 内部会更新 teacher_residual；直接用它作为动作输入
        if hasattr(env, "teacher_residual"):
            actions = env.teacher_residual.clone()
            # 跟踪残差各通道的绝对最大值（归一化到 [-1,1]）
            max_residual = torch.maximum(max_residual, torch.max(torch.abs(actions), dim=0).values)
        else:
            actions = torch.zeros((env.task_config.num_envs, env.task_config.action_space_dim), device=device)
        obs, rewards, terms, truncs, infos = env.step(actions)
        pos = env.obs_dict["robot_position"][0].detach().cpu().numpy()
        tgt = env.target_position[0].detach().cpu().numpy()
        quat = env.obs_dict["robot_orientation"][0:1]
        euler = get_euler_xyz_tensor(quat)[0].detach().cpu().numpy()

        positions.append(pos.copy())
        pos_history.append(pos.copy())
        target_history.append(tgt.copy())
        z_history.append(pos[2])
        euler_history.append(euler)
        policy_actions.append(actions[0].detach().cpu().numpy())
        if hasattr(env, "teacher_residual"):
            teacher_actions.append(env.teacher_residual[0].detach().cpu().numpy())

        if env.payload_manager.just_released_flag[0]:
            payload_idx = int(env.payload_manager.last_release_index[0].item())
            release_history.append((step, payload_idx))

    positions = np.array(positions)
    drift = np.linalg.norm(positions - positions[0], axis=1)
    print(f"Max drift over {steps} steps (env0): {drift.max():.4f} m")
    # 将归一化残差还原到物理量级（力/力矩），便于了解补偿需求峰值
    comp_thrust_limit = float(env.comp_thrust_limit)
    comp_torque_limits = env.comp_torque_limits.detach().cpu().numpy()
    max_residual_np = max_residual.detach().cpu().numpy()
    max_thrust_comp = max_residual_np[0] * comp_thrust_limit
    max_torque_comp = max_residual_np[1:] * comp_torque_limits
    print(
        "Max teacher residual (|.|, normalized): "
        f"thrust={max_residual_np[0]:.3f}, torque={max_residual_np[1:]}",
    )
    print(
        "Approx physical compensation peak: "
        f"thrust={max_thrust_comp:.4f} (same unit as comp_thrust_limit), "
        f"torque={max_torque_comp} (same unit as comp_torque_limits)",
    )

    ideal_circle: Optional[Tuple[float, float, float]] = None
    traj_cfg = getattr(env.task_config, "trajectory_parameters", None) or {}
    if str(traj_cfg.get("type", "")).lower() == "circle":
        cx, cy, _ = traj_cfg.get("center", [0.0, 0.0, 0.0])
        radius = float(traj_cfg.get("radius", 0.0))
        ideal_circle = (cx, cy, radius)

    plot_results(
        z_history,
        euler_history,
        release_history,
        policy_actions,
        teacher_actions=teacher_actions,
        pos_history=pos_history,
        target_history=target_history,
        ideal_circle=ideal_circle,
    )

    try:
        env.close()
    except AttributeError as exc:
        print(f"Warning: env.close() failed ({exc}), skipping explicit cleanup.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument(
        "--headless",
        type=lambda x: str(x).lower() in ("1", "true", "yes"),
        default=False,
        help="是否无头运行，传 True/False",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    run_demo(steps=args.steps, headless=args.headless, device=args.device)
