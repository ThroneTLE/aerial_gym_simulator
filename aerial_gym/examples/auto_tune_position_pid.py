"""
Random search tuner for position-loop PID gains (P, D, I) on the current trajectory.
"""
import argparse
import os
import numpy as np
#python aerial_gym/examples/auto_tune_position_pid.py --trials 50 --steps 2000 --device cuda:0
#python aerial_gym/examples/auto_tune_position_pid.py --trials 2048 --steps 3000 --num_envs 1024
#python aerial_gym/examples/auto_tune_position_pid.py --trials 2048 --steps 3000 --num_envs 1024 --weight_std 0.8 --weight_diff 0.4 --range_ratio 0.5


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


def _set_position_gains(ctrl, k_p, k_d, k_i):
    ctrl.K_pos_tensor_current.copy_(k_p)
    ctrl.K_linvel_tensor_current.copy_(k_d)
    if hasattr(ctrl, "K_pos_int_tensor"):
        ctrl.K_pos_int_tensor = k_i.clone()
    if hasattr(ctrl, "position_error_integral"):
        ctrl.position_error_integral.zero_()


def _rollout_batch(
    env,
    target_seq,
    target_vel,
    steps,
    early_stop_ratio,
    warmup_steps,
    best_score,
    weight_std,
    weight_diff,
    active_count,
):
    device = env.device
    num_envs = env.task_config.num_envs
    err_sum = torch.zeros(num_envs, device=device)
    err_sq_sum = torch.zeros(num_envs, device=device)
    diff_sum = torch.zeros(num_envs, device=device)
    steps_count = torch.zeros(num_envs, device=device)
    active = torch.zeros(num_envs, device=device, dtype=torch.bool)
    if active_count > 0:
        active[:active_count] = True
    prev_err = torch.zeros(num_envs, device=device)

    for step in range(steps):
        env.target_position[:] = target_seq[step]
        env.target_velocity[:] = target_vel[step]
        controller = env.sim_env.robot_manager.robot.controller
        if getattr(controller, "use_velocity_feedforward", False):
            controller.ff_velocity[:] = env.target_velocity
        actions = torch.zeros(
            (env.task_config.num_envs, env.task_config.action_space_dim), device=device
        )
        env.step(actions)
        pos = env.obs_dict["robot_position"]
        err = torch.norm(pos - env.target_position, dim=1)
        diff = torch.abs(err - prev_err)
        prev_err = err

        if active.any():
            err_sum = err_sum + err * active.float()
            err_sq_sum = err_sq_sum + (err * err) * active.float()
            diff_sum = diff_sum + diff * active.float()
            steps_count = steps_count + active.float()

        if best_score is not None and step + 1 >= warmup_steps:
            mean_err = torch.where(
                steps_count > 0, err_sum / torch.clamp_min(steps_count, 1.0), err_sum
            )
            std_err = torch.zeros_like(mean_err)
            valid = steps_count > 0
            if valid.any():
                var = err_sq_sum[valid] / steps_count[valid] - mean_err[valid] ** 2
                std_err[valid] = torch.sqrt(torch.clamp_min(var, 0.0))
            mean_diff = torch.where(
                steps_count > 0, diff_sum / torch.clamp_min(steps_count, 1.0), diff_sum
            )
            score = mean_err + weight_std * std_err + weight_diff * mean_diff
            active = torch.where(
                score <= best_score * (1.0 + early_stop_ratio),
                active,
                torch.zeros_like(active),
            )
            if not active.any():
                break

    mean_err = torch.where(
        steps_count > 0, err_sum / torch.clamp_min(steps_count, 1.0), err_sum
    )
    var = torch.zeros_like(mean_err)
    valid = steps_count > 0
    if valid.any():
        var[valid] = err_sq_sum[valid] / steps_count[valid] - mean_err[valid] ** 2
    std_err = torch.sqrt(torch.clamp_min(var, 0.0))
    mean_diff = torch.where(
        steps_count > 0, diff_sum / torch.clamp_min(steps_count, 1.0), diff_sum
    )
    score = mean_err + weight_std * std_err + weight_diff * mean_diff
    score = torch.where(steps_count > 0, score, torch.full_like(score, float("inf")))
    return score


def run_search(
    trials=50,
    steps=2000,
    seed=1,
    early_stop_ratio=0.2,
    warmup_steps=200,
    num_envs=1024,
    headless=True,
    device="cuda:0",
    range_ratio=0.2,
    weight_std=0.5,
    weight_diff=0.2,
):
    cfg = teacher_cfg.task_config
    cfg.headless = headless
    cfg.device = device
    cfg.num_envs = num_envs
    cfg.teacher_mode = False
    if hasattr(cfg, "payload_parameters"):
        cfg.payload_parameters["randomize_release"] = False
    if hasattr(cfg, "trajectory_parameters"):
        cfg.trajectory_parameters["enable"] = True

    env = PayloadCompensationTask(cfg)
    torch.manual_seed(seed)
    np.random.seed(seed)
    env.reset()
    env.trajectory_enabled = True

    target_seq = env.trajectory_buffer[0].detach().clone()
    if target_seq.shape[0] < steps:
        pad = steps - target_seq.shape[0]
        target_seq = torch.cat([target_seq, target_seq[-1:].repeat(pad, 1)], dim=0)
    target_seq = target_seq.to(env.device)
    dt = float(getattr(getattr(env.sim_env, "IGE_env", None), "global_tensor_dict", {}).get("dt", 0.01))
    target_vel = torch.zeros_like(target_seq)
    if target_seq.shape[0] > 1:
        target_vel[1:] = (target_seq[1:] - target_seq[:-1]) / max(dt, 1e-6)
    payload_plan = _snapshot_payload_plan(env)

    ctrl = env.sim_env.robot_manager.robot.controller
    base_kp = ctrl.K_pos_tensor_current[0].detach().cpu().numpy()
    base_kd = ctrl.K_linvel_tensor_current[0].detach().cpu().numpy()
    base_ki = (
        ctrl.K_pos_int_tensor[0].detach().cpu().numpy()
        if hasattr(ctrl, "K_pos_int_tensor")
        else np.zeros(3, dtype=np.float32)
    )

    low_kp, high_kp = (1.0 - range_ratio) * base_kp, (1.0 + range_ratio) * base_kp
    low_kd, high_kd = (1.0 - range_ratio) * base_kd, (1.0 + range_ratio) * base_kd
    low_ki, high_ki = (1.0 - range_ratio) * base_ki, (1.0 + range_ratio) * base_ki

    best_score = None
    best_params = None
    total_trials = trials
    batch_size = env.task_config.num_envs
    num_batches = int(np.ceil(total_trials / batch_size))

    for batch in range(num_batches):
        start = batch * batch_size
        end = min(total_trials, start + batch_size)
        active_count = end - start
        if active_count <= 0:
            break

        torch.manual_seed(seed + batch + 1)
        np.random.seed(seed + batch + 1)
        env.reset()
        _restore_payload_plan(env, payload_plan)
        env.trajectory_enabled = False

        k_p_np = np.random.uniform(low_kp, high_kp, size=(batch_size, 3)).astype(np.float32)
        k_d_np = np.random.uniform(low_kd, high_kd, size=(batch_size, 3)).astype(np.float32)
        k_i_np = np.random.uniform(low_ki, high_ki, size=(batch_size, 3)).astype(np.float32)

        k_p = torch.tensor(k_p_np, device=env.device)
        k_d = torch.tensor(k_d_np, device=env.device)
        k_i = torch.tensor(k_i_np, device=env.device)
        _set_position_gains(ctrl, k_p, k_d, k_i)

        scores = _rollout_batch(
            env,
            target_seq,
            target_vel,
            steps,
            early_stop_ratio=early_stop_ratio,
            warmup_steps=warmup_steps,
            best_score=best_score,
            weight_std=weight_std,
            weight_diff=weight_diff,
            active_count=active_count,
        )
        scores = scores[:active_count].detach().cpu().numpy()
        best_idx = int(np.argmin(scores))
        batch_best = float(scores[best_idx])
        if best_score is None or batch_best < best_score:
            best_score = batch_best
            best_params = {
                "K_pos": k_p_np[best_idx],
                "K_vel": k_d_np[best_idx],
                "K_int": k_i_np[best_idx],
            }
        print(
            f"[batch {batch + 1}/{num_batches}] "
            f"best_batch={batch_best:.4f} best_all={best_score:.4f}"
        )

    env.close()
    if best_params is None:
        print("No valid trials finished (early stopped).")
        return
    print("Best gains:")
    print(f"K_pos={best_params['K_pos']}")
    print(f"K_vel={best_params['K_vel']}")
    print(f"K_int={best_params['K_int']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--early_stop_ratio", type=float, default=0.3)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--num_envs", type=int, default=1024)
    parser.add_argument("--headless", type=lambda x: str(x).lower() in ("1", "true", "yes"), default=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--range_ratio", type=float, default=0.2)
    parser.add_argument("--weight_std", type=float, default=0.5)
    parser.add_argument("--weight_diff", type=float, default=0.2)
    args = parser.parse_args()
    run_search(
        trials=args.trials,
        steps=args.steps,
        seed=args.seed,
        early_stop_ratio=args.early_stop_ratio,
        warmup_steps=args.warmup_steps,
        num_envs=args.num_envs,
        headless=args.headless,
        device=args.device,
        range_ratio=args.range_ratio,
        weight_std=args.weight_std,
        weight_diff=args.weight_diff,
    )
