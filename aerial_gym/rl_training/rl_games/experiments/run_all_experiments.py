#!/usr/bin/env python3
"""
批量训练不同 MLP 配置的实验脚本。
用法：
    python run_all_experiments.py                    # 顺序训练所有配置
    python run_all_experiments.py --configs small large  # 只训练指定配置
    python run_all_experiments.py --max_epochs 2000  # 限制最大 epoch
"""
import argparse
import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime

# 实验配置
EXPERIMENTS = {
    "small": {
        "yaml": "mlp_small.yaml",
        "mlp": "[128, 128]",
        "desc": "小型 MLP",
    },
    "medium": {
        "yaml": "mlp_medium.yaml",
        "mlp": "[256, 256, 128]",
        "desc": "中型 MLP（基准）",
    },
    "large": {
        "yaml": "mlp_large.yaml",
        "mlp": "[512, 256, 128]",
        "desc": "大型 MLP",
    },
    "xlarge": {
        "yaml": "mlp_xlarge.yaml",
        "mlp": "[512, 512, 256]",
        "desc": "超大型 MLP",
    },
}

SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parents[2]  # aerial_gym_simulator
PYTHON = sys.executable


def run_training(config_name: str, max_epochs: int = None, num_envs: int = 2048):
    """运行单个配置的训练"""
    exp = EXPERIMENTS.get(config_name)
    if not exp:
        print(f"[ERROR] 未知配置: {config_name}")
        return False
    
    yaml_path = SCRIPT_DIR / exp["yaml"]
    if not yaml_path.exists():
        print(f"[ERROR] 配置文件不存在: {yaml_path}")
        return False
    
    # 生成实验名称（带时间戳）
    timestamp = datetime.now().strftime("%m%d_%H%M")
    experiment_name = f"exp_{config_name}_{timestamp}"
    
    print("=" * 60)
    print(f"[实验] {exp['desc']}")
    print(f"       MLP: {exp['mlp']}")
    print(f"       配置: {yaml_path.name}")
    print(f"       名称: {experiment_name}")
    print("=" * 60)
    
    cmd = [
        PYTHON, "-m", "aerial_gym.rl_training.rl_games.runner",
        "--train",
        "--file", str(yaml_path),
        "--task", "payload_compensation_task_teacher",
        "--experiment_name", experiment_name,
        "--num_envs", str(num_envs),
        "--headless", "True",
    ]
    
    if max_epochs:
        # 需要修改 yaml 或通过命令行传递
        print(f"[INFO] max_epochs 参数需要在 yaml 中设置")
    
    try:
        result = subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)
        print(f"\n[OK] {config_name} 训练完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[FAIL] {config_name} 训练失败: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n[中断] {config_name} 训练被用户中断")
        return False


def main():
    parser = argparse.ArgumentParser(description="批量训练 MLP 架构实验")
    parser.add_argument(
        "--configs",
        nargs="+",
        choices=list(EXPERIMENTS.keys()) + ["all"],
        default=["all"],
        help="要训练的配置列表",
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=2048,
        help="并行环境数",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=None,
        help="最大 epoch 数（覆盖 yaml 设置）",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="列出所有可用配置",
    )
    args = parser.parse_args()
    
    if args.list:
        print("\n可用实验配置：")
        print("-" * 50)
        for name, exp in EXPERIMENTS.items():
            print(f"  {name:10s} - {exp['desc']:15s} MLP: {exp['mlp']}")
        return
    
    # 确定要运行的配置
    if "all" in args.configs:
        configs_to_run = list(EXPERIMENTS.keys())
    else:
        configs_to_run = args.configs
    
    print("\n" + "=" * 60)
    print("MLP 架构对比实验")
    print("=" * 60)
    print(f"配置列表: {configs_to_run}")
    print(f"环境数量: {args.num_envs}")
    print()
    
    results = {}
    for config in configs_to_run:
        success = run_training(config, max_epochs=args.max_epochs, num_envs=args.num_envs)
        results[config] = "OK" if success else "FAIL"
    
    # 打印总结
    print("\n" + "=" * 60)
    print("实验结果总结")
    print("=" * 60)
    for config, status in results.items():
        exp = EXPERIMENTS[config]
        status_str = "✅" if status == "OK" else "❌"
        print(f"  {status_str} {config:10s} ({exp['mlp']})")
    print()


if __name__ == "__main__":
    main()
