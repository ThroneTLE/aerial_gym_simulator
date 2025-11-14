import argparse

from aerial_gym.registry.task_registry import task_registry
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Payload compensation mini simulation.")
    parser.add_argument("--num_envs", type=int, default=4, help="并行环境数量")
    parser.add_argument("--steps", type=int, default=800, help="仿真步数")
    parser.add_argument("--headless", action="store_true", help="是否关闭可视化窗口")
    return parser.parse_args()


def main():
    args = parse_args()
    task = task_registry.make_task(
        "payload_compensation_task",
        num_envs=args.num_envs,
        headless=args.headless,
    )

    obs, rewards, terms, truncs, infos = task.reset()
    device = task.device
    actions = torch.zeros(
        (task.sim_env.num_envs, task.task_config.action_space_dim), device=device
    )

    print(
        f"启动 payload_compensation_task：envs={task.sim_env.num_envs}, "
        f"action_dim={task.task_config.action_space_dim}"
    )

    for step in range(args.steps):
        if step % 200 == 0:
            actions = 0.15 * torch.randn_like(actions)

        obs, rewards, terms, truncs, infos = task.step(actions)

        released_mask = infos["just_released"].bool()
        if released_mask.any():
            env_ids = torch.nonzero(released_mask, as_tuple=False).squeeze(-1).tolist()
            payload_ids = infos["last_release_index"][released_mask].tolist()
            reward_val = rewards.mean().item()
            for env_id, payload_id in zip(env_ids, payload_ids):
                print(
                    f"[Step {step}] env {env_id} 释放挂点 {payload_id} | 平均奖励: {reward_val:.3f}"
                )

    task.close()
    print("仿真结束。")


if __name__ == "__main__":
    main()
