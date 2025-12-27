#!/usr/bin/env python3
import argparse
import os

import numpy as np


from aerial_gym.registry.task_registry import task_registry
import torch

def _update_stats(mean, m2, count, batch):
    batch = batch.reshape(-1, batch.shape[-1]).float()
    batch_count = batch.shape[0]
    batch_mean = batch.mean(dim=0)
    batch_m2 = ((batch - batch_mean) ** 2).sum(dim=0)
    if count == 0:
        return batch_mean, batch_m2, batch_count
    delta = batch_mean - mean
    count_new = count + batch_count
    mean_new = mean + delta * (batch_count / count_new)
    m2_new = m2 + batch_m2 + (delta * delta) * (count * batch_count / count_new)
    return mean_new, m2_new, count_new


def _collect_stats(task, steps, warmup):
    obs, *_ = task.reset()
    base = obs["observations"]
    priv = obs.get("priviliged_obs")
    if priv is None:
        raise RuntimeError("No privileged_obs in task output; use teacher task config.")

    base_mean = torch.zeros(base.shape[-1], device=base.device)
    base_m2 = torch.zeros_like(base_mean)
    base_count = 0
    priv_mean = torch.zeros(priv.shape[-1], device=priv.device)
    priv_m2 = torch.zeros_like(priv_mean)
    priv_count = 0

    for _ in range(warmup):
        actions = torch.rand(
            (task.sim_env.num_envs, task.task_config.action_space_dim),
            device=task.device,
        ) * 2.0 - 1.0
        task.step(actions)

    for _ in range(steps):
        actions = torch.rand(
            (task.sim_env.num_envs, task.task_config.action_space_dim),
            device=task.device,
        ) * 2.0 - 1.0
        obs, *_ = task.step(actions)
        base = obs["observations"]
        priv = obs.get("priviliged_obs")
        if priv is None:
            raise RuntimeError("privileged_obs missing during rollout.")
        base_mean, base_m2, base_count = _update_stats(base_mean, base_m2, base_count, base)
        priv_mean, priv_m2, priv_count = _update_stats(priv_mean, priv_m2, priv_count, priv)

    base_var = base_m2 / max(base_count, 1)
    priv_var = priv_m2 / max(priv_count, 1)
    return base_mean, torch.sqrt(base_var), priv_mean, torch.sqrt(priv_var)


def main():
    parser = argparse.ArgumentParser(description="Resample fixed_stats.npz for AerialGym tasks.")
    parser.add_argument("--task", default="payload_compensation_task_teacher")
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--render", action="store_false", dest="headless")
    parser.add_argument("--use_warp", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--min_std", type=float, default=1e-3)
    parser.add_argument("--out", default=os.path.join(os.getcwd(), "fixed_stats.npz"))
    args = parser.parse_args()

    torch.set_grad_enabled(False)

    task = task_registry.make_task(
        args.task,
        seed=args.seed,
        num_envs=args.num_envs,
        headless=args.headless,
        use_warp=args.use_warp,
    )
    try:
        base_mean, base_std, priv_mean, priv_std = _collect_stats(task, args.steps, args.warmup)
    finally:
        task.close()

    min_std = float(args.min_std)
    base_std = torch.clamp(base_std, min=min_std)
    priv_std = torch.clamp(priv_std, min=min_std)

    np.savez(
        args.out,
        base_mean=base_mean.cpu().numpy(),
        base_std=base_std.cpu().numpy(),
        priv_mean=priv_mean.cpu().numpy(),
        priv_std=priv_std.cpu().numpy(),
    )
    print(f"Saved fixed stats to {args.out}")
    print(f"base_mean shape {tuple(base_mean.shape)} base_std min {float(base_std.min().item()):.6f}")
    print(f"priv_mean shape {tuple(priv_mean.shape)} priv_std min {float(priv_std.min().item()):.6f}")


if __name__ == "__main__":
    main()
