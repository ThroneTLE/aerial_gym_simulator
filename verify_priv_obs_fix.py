#!/usr/bin/env python3
"""
快速验证：privileged_obs 是否被正确收集到 experience_buffer
不需要完整训练，只跑几步看日志即可
"""

import sys
import isaacgym
import torch

# 先导入 rl_games 相关
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner

# 注册网络和环境
from aerial_gym.rl_training.rl_games.nn import privileged_actor_critic
from aerial_gym.rl_training.rl_games.a2c_aux_loss import A2CAgentWithAuxLoss
from aerial_gym.registry.task_registry import task_registry
from aerial_gym.rl_training.rl_games import runner as aerial_runner

import yaml

def main():
    # 用最小配置测试
    config_path = "aerial_gym/rl_training/rl_games/ppo_aerial_quad_aux.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 最小化环境数和 epoch
    config['params']['config']['num_actors'] = 8
    config['params']['config']['env_config']['num_envs'] = 8
    config['params']['config']['max_epochs'] = 1
    config['params']['config']['horizon_length'] = 8  # 很短的 horizon
    config['params']['config']['env_config']['headless'] = True
    
    sys.argv = [sys.argv[0]]  # 清空参数
    
    runner = Runner()
    runner.algo_factory.register_builder(
        'a2c_continuous_aux',
        lambda **kwargs: A2CAgentWithAuxLoss(**kwargs)
    )
    runner.load(config)
    
    # 获取 agent
    agent = runner.algo_factory.create(
        runner.algo_name,
        base_name='run',
        params=runner.params
    )
    
    print("\n" + "=" * 80)
    print("验证 privileged_obs 收集")
    print("=" * 80)
    
    # 检查 experience_buffer 中是否有 privileged_obs
    agent.init_tensors()
    
    if 'privileged_obs' in agent.experience_buffer.tensor_dict:
        shape = agent.experience_buffer.tensor_dict['privileged_obs'].shape
        print(f"✅ privileged_obs 已在 experience_buffer 中注册！")
        print(f"   形状: {shape}")
    else:
        print("❌ privileged_obs 未注册到 experience_buffer")
        return
    
    if 'privileged_obs' in agent.tensor_list:
        print(f"✅ privileged_obs 已在 tensor_list 中，会被包含到 batch_dict")
    else:
        print("❌ privileged_obs 不在 tensor_list 中")
        return
    
    # 模拟一次 play_steps 看看是否正确填充
    print("\n模拟运行 play_steps...")
    agent.obs = agent.env_reset()
    batch_dict = agent.play_steps()
    
    if 'privileged_obs' in batch_dict:
        priv = batch_dict['privileged_obs']
        print(f"✅ privileged_obs 已在 batch_dict 中！")
        print(f"   形状: {priv.shape}")
        print(f"   非零元素比例: {(priv != 0).float().mean().item():.2%}")
        print(f"   均值: {priv.mean():.4f}")
        print(f"   标准差: {priv.std():.4f}")
        
        if priv.std() > 0.001:
            print("✅ 数据有变化，说明正确收集了不同环境的 privileged_obs！")
        else:
            print("⚠️ 数据方差太小，可能有问题")
    else:
        print("❌ privileged_obs 不在 batch_dict 中")
    
    print("\n" + "=" * 80)
    print("验证完成！如果看到所有 ✅，说明修复成功")
    print("=" * 80)

if __name__ == "__main__":
    main()
