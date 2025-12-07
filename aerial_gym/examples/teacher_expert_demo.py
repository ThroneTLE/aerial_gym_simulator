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
    max_residual = torch.zeros(env.task_config.action_space_dim, device=device)
    for _ in range(steps):
        # 在 Teacher 模式下，env 内部会更新 teacher_residual；直接用它作为动作输入
        if hasattr(env, "teacher_residual"):
            actions = env.teacher_residual.clone()
            # 跟踪残差各通道的绝对最大值（归一化到 [-1,1]）
            max_residual = torch.maximum(max_residual, torch.max(torch.abs(actions), dim=0).values)
        else:
            actions = torch.zeros((env.task_config.num_envs, env.task_config.action_space_dim), device=device)
        obs, rewards, terms, truncs, infos = env.step(actions)
        positions.append(env.obs_dict["robot_position"][0].detach().cpu().numpy())
    env.close()
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
