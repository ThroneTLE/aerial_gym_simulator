"""Quick sanity check for teacher residual: apply expert residual as action and plot 3D tracking."""
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


from aerial_gym.config.task_config import payload_compensation_task_teacher_config as teacher_cfg
from aerial_gym.task.payload_compensation_task.payload_compensation_task import PayloadCompensationTask

import torch


def _snapshot_payload_plan(env):
    if not hasattr(env, "payload_manager"):
        return None
    pm = env.payload_manager
    return {
        "release_orders": pm.release_orders.detach().clone(),
        "next_release_step": pm.next_release_step.detach().clone(),
    }


def _restore_payload_plan(env, plan):
    if plan is None or not hasattr(env, "payload_manager"):
        return
    pm = env.payload_manager
    if not pm.enable_payload:
        return
    pm.release_orders[:] = plan["release_orders"]
    pm.next_release_step[:] = plan["next_release_step"]
    pm.release_cursor[:] = 0
    pm.step_counter[:] = 0
    pm.release_warning_flag[:] = False
    pm.just_released_flag[:] = False
    pm.last_release_index[:] = -1
    pm.last_release_mass[:] = 0.0
    pm.attached_mask[:] = True
    pm.current_payload_mass[:] = pm.num_payloads * pm.payload_mass
    pm.com_offset_body[:] = 0.0
    pm._update_mass_properties(torch.arange(pm.num_envs, device=pm.device))


def _run_rollout(env, steps, target_seq_tensor, target_vel_tensor, use_expert):
    positions = []
    targets = []
    errors = []
    max_residual = torch.zeros(env.task_config.action_space_dim, device=env.device)
    for step in range(steps):
        if target_seq_tensor is not None:
            env.target_position[:] = target_seq_tensor[step]
        if target_vel_tensor is not None:
            env.target_velocity[:] = target_vel_tensor[step]
            controller = env.sim_env.robot_manager.robot.controller
            if getattr(controller, "use_velocity_feedforward", False):
                controller.ff_velocity[:] = env.target_velocity
        if use_expert and hasattr(env, "_update_teacher_residual"):
            env._update_teacher_residual()
            actions = env.teacher_residual.clone()
            max_residual = torch.maximum(
                max_residual, torch.max(torch.abs(actions), dim=0).values
            )
        else:
            actions = torch.zeros(
                (env.task_config.num_envs, env.task_config.action_space_dim), device=env.device
            )
        obs, rewards, terms, truncs, infos = env.step(actions)
        pos = env.obs_dict["robot_position"][0].detach().cpu().numpy()
        positions.append(pos)
        tgt = env.target_position[0].detach().cpu().numpy()
        targets.append(tgt)
        errors.append(pos - tgt)
    return {
        "positions": np.array(positions),
        "targets": np.array(targets),
        "errors": np.array(errors),
        "max_residual": max_residual,
    }


def run_demo(steps=2000, headless=False, device="cuda:0", mode="pd"):
    cfg = teacher_cfg.task_config
    cfg.headless = headless
    cfg.device = device
    if hasattr(cfg, "trajectory_parameters"):
        cfg.trajectory_parameters["enable"] = True
    env = PayloadCompensationTask(cfg)
    env.reset()
    payload_plan = _snapshot_payload_plan(env)
    target_seq = None
    if getattr(env, "trajectory_enabled", False) and getattr(env, "trajectory_buffer", None) is not None:
        target_seq = env.trajectory_buffer[0].detach().clone()
    if target_seq is None:
        target_seq = torch.zeros((steps, 3), device=env.device)
    elif target_seq.shape[0] < steps:
        pad = steps - target_seq.shape[0]
        target_seq = torch.cat([target_seq, target_seq[-1:].repeat(pad, 1)], dim=0)
    target_seq_tensor = target_seq.to(env.device)
    dt = float(getattr(getattr(env.sim_env, "IGE_env", None), "global_tensor_dict", {}).get("dt", 0.01))
    target_vel = torch.zeros_like(target_seq_tensor)
    if target_seq_tensor.shape[0] > 1:
        target_vel[1:] = (target_seq_tensor[1:] - target_seq_tensor[:-1]) / max(dt, 1e-6)
    env.trajectory_enabled = False

    pd_rollout = None
    expert_rollout = None
    if mode in ("pd", "both"):
        pd_rollout = _run_rollout(env, steps, target_seq_tensor, target_vel, use_expert=False)
    if mode in ("expert", "both"):
        env.reset()
        _restore_payload_plan(env, payload_plan)
        env.trajectory_enabled = False
        expert_rollout = _run_rollout(env, steps, target_seq_tensor, target_vel, use_expert=True)
    if expert_rollout is not None:
        if hasattr(env, "payload_manager"):
            pm = env.payload_manager
            print(
                "[DebugPayload] enable=%s mass=%.6f attached=%s current_mass=%.6f"
                % (
                    pm.enable_payload,
                    float(pm.payload_mass),
                    pm.attached_mask[0].detach().cpu().numpy().tolist(),
                    float(pm.current_payload_mass[0].item()),
                )
            )
        if hasattr(env, "teacher_residual"):
            print(
                "[DebugResidual] sample=%s"
                % (env.teacher_residual[0].detach().cpu().numpy().tolist(),)
            )
    env.close()

    if expert_rollout is not None:
        positions = expert_rollout["positions"]
        targets = expert_rollout["targets"]
        errors = expert_rollout["errors"]
        max_residual = expert_rollout["max_residual"]
    else:
        positions = pd_rollout["positions"]
        targets = pd_rollout["targets"]
        errors = pd_rollout["errors"]
        max_residual = torch.zeros(env.task_config.action_space_dim, device=device)
    drift = np.linalg.norm(positions - positions[0], axis=1)
    print(f"Max drift over {steps} steps (env0): {drift.max():.4f} m")
    # 将归一化残差还原到物理量级（力/力矩），便于了解补偿需求峰值
    if expert_rollout is not None:
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

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    label = "Expert" if expert_rollout is not None else "PD"
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], label=label, color="C0")
    ax.scatter(
        positions[0, 0],
        positions[0, 1],
        positions[0, 2],
        label="Start",
        color="#2ca02c",
        s=60,
        edgecolor="k",
        linewidth=0.5,
    )
    ax.scatter(
        positions[-1, 0],
        positions[-1, 1],
        positions[-1, 2],
        label="End",
        color="#d62728",
        s=70,
        marker="^",
        edgecolor="k",
        linewidth=0.5,
    )
    if targets is not None and len(targets) == len(positions):
        ax.plot(
            targets[:, 0],
            targets[:, 1],
            targets[:, 2],
            label="Target",
            color="C1",
            linestyle="--",
            alpha=0.7,
        )
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Expert Trajectory Tracking (3D)")
    ax.legend()

    if errors is not None:
        fig_err = plt.figure(figsize=(9, 4))
        ax_err = fig_err.add_subplot(111)
        err_norm = np.linalg.norm(errors, axis=1)
        if pd_rollout is not None:
            pd_err_norm = np.linalg.norm(pd_rollout["errors"], axis=1)
            ax_err.plot(pd_err_norm, color="C2", label="PD |pos - target|")
        if expert_rollout is not None:
            ax_err.plot(err_norm, color="C3", label="Expert |pos - target|")
        ax_err.set_xlabel("Step")
        ax_err.set_ylabel("Position Error (m)")
        ax_err.set_title("Tracking Error Comparison" if mode == "both" else "Tracking Error")
        ax_err.grid(True, alpha=0.3)
        ax_err.legend()

    save_path = os.environ.get("AERIAL_SAVE_PLOT")
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig_suffix = "_expert_3d.png" if expert_rollout is not None else "_pd_3d.png"
        fig.savefig(
            os.path.splitext(save_path)[0] + fig_suffix,
            dpi=150,
            bbox_inches="tight",
        )
        if errors is not None:
            err_suffix = "_compare_error.png" if mode == "both" else "_error.png"
            fig_err.savefig(
                os.path.splitext(save_path)[0] + err_suffix,
                dpi=150,
                bbox_inches="tight",
            )
        print(f"已保存绘图到 {save_path}")
    else:
        plt.show()


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
    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["pd", "expert", "both"],
        help="选择只跑 PD / 只跑专家 / 两者对比",
    )
    #python aerial_gym/examples/teacher_expert_demo.py --mode pd
    #python aerial_gym/examples/teacher_expert_demo.py --mode expert
    #python aerial_gym/examples/teacher_expert_demo.py --mode both
    args = parser.parse_args()
    run_demo(steps=args.steps, headless=args.headless, device=args.device, mode=args.mode)
