"""
对一批 checkpoint 离线评估 imitation 误差（策略动作与 teacher 残差的 L2 距离）。

默认使用 payload_compensation_task_teacher 环境，复用训练时的 obs RMS 和网络结构。
示例：
  conda run -n aerialgym python tools/eval_imitation_err_offline.py \
    --run_dir runs/teacher_residual_stage1_10-02-56-40/nn \
    --config aerial_gym/rl_training/rl_games/ppo_aerial_quad.yaml \
    --steps 512 --num_envs 64
"""

import argparse
import glob
import os
from typing import List

import isaacgym  # ensure imported before torch to avoid import order issues
import torch

from aerial_gym.examples.new_my_position_control import (
    build_policy,
    load_checkpoint,
    load_obs_rms,
    load_training_config,
)
from aerial_gym.registry.task_registry import task_registry


def _str2bool(val):
    if isinstance(val, bool):
        return val
    val = val.lower()
    return val in ("1", "true", "t", "yes")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate imitation error for checkpoints.")
    parser.add_argument("--run_dir", required=True, help="包含若干 .pth 的目录，例如 runs/xxx/nn")
    parser.add_argument("--config", default="aerial_gym/rl_training/rl_games/ppo_aerial_quad.yaml")
    parser.add_argument("--env_name", default="payload_compensation_task_teacher")
    parser.add_argument("--steps", type=int, default=512, help="每个 ckpt 评估的步数")
    parser.add_argument("--num_envs", type=int, default=64, help="并行环境数量")
    parser.add_argument("--headless", type=_str2bool, default=True)
    return parser.parse_args()


def evaluate_ckpt(ckpt_path: str, cfg: dict, task, steps: int):
    checkpoint_data, checkpoint_action_dim = load_checkpoint(ckpt_path)

    obs_dim = task.task_config.observation_space_dim
    action_dim = task.task_config.action_space_dim
    if action_dim != checkpoint_action_dim:
        raise RuntimeError(f"{ckpt_path} 动作维度 {checkpoint_action_dim} 与任务 {action_dim} 不匹配")
    device = torch.device(task.device)

    obs_rms = load_obs_rms(checkpoint_data, obs_dim, device)
    policy = build_policy(cfg, checkpoint_data, obs_dim, action_dim, device, task.sim_env.num_envs)
    policy.eval()
    task.reset()
    policy.reset_hidden_state()

    total_err = 0.0
    count = 0
    with torch.no_grad():
        for _ in range(steps):
            obs_tensor = torch.as_tensor(task.task_obs["observations"], device=device, dtype=torch.float32)
            if obs_rms is not None:
                obs_tensor = obs_rms(obs_tensor, denorm=False)
            actions = torch.clamp(policy(obs_tensor), -1.0, 1.0)
            task_obs, rewards, terms, truncs, infos = task.step(actions)

            if policy.uses_rnn:
                done_tensor = torch.as_tensor(terms, device=device).bool()
                trunc_tensor = torch.as_tensor(truncs, device=device).bool()
                reset_envs = torch.nonzero(done_tensor | trunc_tensor, as_tuple=False).squeeze(-1)
                if reset_envs.numel() > 0:
                    policy.reset_hidden_state(env_ids=reset_envs.tolist())

            if hasattr(task, "teacher_residual"):
                teacher = task.teacher_residual
                err = torch.norm(actions - teacher, dim=1).mean().item()
                total_err += err
                count += 1
    return total_err / max(count, 1)


def main():
    args = parse_args()
    cfg = load_training_config(args.config) or {}
    ckpt_paths: List[str] = sorted(glob.glob(os.path.join(args.run_dir, "*.pth")))
    if not ckpt_paths:
        raise FileNotFoundError(f"在 {args.run_dir} 未找到 .pth 文件")

    # 只创建一次环境，重复复位
    task = task_registry.make_task(args.env_name, num_envs=args.num_envs, headless=args.headless)
    results = []
    for path in ckpt_paths:
        try:
            mean_err = evaluate_ckpt(path, cfg, task, args.steps)
            results.append((mean_err, path))
            print(f"{path}: imitation_err_mean={mean_err:.4f}")
        except Exception as exc:
            print(f"{path}: 评估失败 {exc}")

    try:
        task.close()
    except Exception:
        pass

    if results:
        best = min(results, key=lambda x: x[0])
        print(f"\n最小 imitation_err: {best[0]:.4f} -> {best[1]}")


if __name__ == "__main__":
    main()
