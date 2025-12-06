"""Quick sanity check for teacher residual: apply expert residual as action and log basic stats."""
import argparse
import numpy as np


from aerial_gym.config.task_config import payload_compensation_task_teacher_config as teacher_cfg
from aerial_gym.task.payload_compensation_task.payload_compensation_task import PayloadCompensationTask

import torch
def run_demo(steps=2000, headless=False, device="cuda:0"):
    cfg = teacher_cfg.task_config
    cfg.headless = headless
    cfg.device = device
    env = PayloadCompensationTask(cfg)
    obs, *_ = env.reset()
    positions = []
    for _ in range(steps):
        # 在 Teacher 模式下，env 内部会更新 teacher_residual；直接用它作为动作输入
        if hasattr(env, "teacher_residual"):
            actions = env.teacher_residual.clone()
        else:
            actions = torch.zeros((env.task_config.num_envs, env.task_config.action_space_dim), device=device)
        obs, rewards, terms, truncs, infos = env.step(actions)
        positions.append(env.obs_dict["robot_position"][0].detach().cpu().numpy())
    env.close()
    positions = np.array(positions)
    drift = np.linalg.norm(positions - positions[0], axis=1)
    print(f"Max drift over {steps} steps (env0): {drift.max():.4f} m")


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
