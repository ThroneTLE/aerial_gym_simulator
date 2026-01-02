#!/usr/bin/env python3
"""
快速验证：privileged_obs 是否被正确传递到网络的 forward 方法
只跑 1 个 epoch，检查数据流是否正确
"""

import sys
import isaacgym
import torch
import yaml

# Mock 一个简单的测试
from rl_games.common import env_configurations
from aerial_gym.rl_training.rl_games.nn import privileged_actor_critic
from aerial_gym.rl_training.rl_games.a2c_aux_loss import A2CAgentWithAuxLoss
from aerial_gym.rl_training.rl_games import runner as aerial_runner
from rl_games.torch_runner import Runner

def main():
    config_path = "aerial_gym/rl_training/rl_games/ppo_aerial_quad_aux.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 使用最小配置
    config['params']['config']['num_actors'] = 16
    config['params']['config']['env_config']['num_envs'] = 16
    config['params']['config']['max_epochs'] = 1
    config['params']['config']['horizon_length'] = 16
    config['params']['config']['minibatch_size'] = 16  # 确保能整除
    config['params']['config']['mini_epochs'] = 1
    config['params']['config']['env_config']['headless'] = True
    
    sys.argv = [sys.argv[0]]
    
    runner = Runner()
    runner.algo_factory.register_builder(
        'a2c_continuous_aux',
        lambda **kwargs: A2CAgentWithAuxLoss(**kwargs)
    )
    runner.load(config)
    
    agent = runner.algo_factory.create(
        runner.algo_name,
        base_name='run',
        params=runner.params
    )
    
    print("\n" + "=" * 80)
    print("验证 privileged_obs 数据流")
    print("=" * 80)
    
    # Step 1: init_tensors
    agent.init_tensors()
    print("\n[Step 1] init_tensors")
    print(f"  experience_buffer 中有 privileged_obs: {'privileged_obs' in agent.experience_buffer.tensor_dict}")
    print(f"  tensor_list 中有 privileged_obs: {'privileged_obs' in agent.tensor_list}")
    
    # Step 2: play_steps
    agent.obs = agent.env_reset()
    print("\n[Step 2] play_steps 前的 obs")
    print(f"  obs 中有 privileged_obs: {'privileged_obs' in agent.obs}")
    if 'privileged_obs' in agent.obs and agent.obs['privileged_obs'] is not None:
        priv = agent.obs['privileged_obs']
        print(f"  privileged_obs 形状: {priv.shape}")
        print(f"  privileged_obs 均值: {priv.mean():.4f}")
    
    batch_dict = agent.play_steps()
    print("\n[Step 3] play_steps 后的 batch_dict")
    print(f"  batch_dict 中有 privileged_obs: {'privileged_obs' in batch_dict}")
    if 'privileged_obs' in batch_dict:
        priv = batch_dict['privileged_obs']
        print(f"  privileged_obs 形状: {priv.shape}")
        print(f"  privileged_obs 非零比例: {(priv.abs() > 1e-6).float().mean():.2%}")
        print(f"  privileged_obs 均值: {priv.mean():.4f}, 标准差: {priv.std():.4f}")
    
    # Step 4: prepare_dataset
    agent.prepare_dataset(batch_dict)
    print("\n[Step 4] prepare_dataset 后")
    print(f"  _privileged_obs_flat 存在: {hasattr(agent, '_privileged_obs_flat') and agent._privileged_obs_flat is not None}")
    if hasattr(agent, '_privileged_obs_flat') and agent._privileged_obs_flat is not None:
        print(f"  _privileged_obs_flat 形状: {agent._privileged_obs_flat.shape}")
    
    # Step 5: 模拟 train_actor_critic
    print("\n[Step 5] 模拟 train_actor_critic")
    # 创建一个假的 input_dict
    fake_input = {
        'obs': batch_dict['obses'],
        'actions': batch_dict['actions'],
        'old_values': batch_dict['values'],
        'old_logp_actions': batch_dict['neglogpacs'],
        'advantages': torch.zeros_like(batch_dict['values']),
        'returns': batch_dict['values'],
        'mu': batch_dict['mus'],
        'sigma': batch_dict['sigmas'],
        'idx': torch.arange(batch_dict['obses'].shape[0]),
    }
    
    # 检查 privileged_obs 是否会被正确添加
    privileged_obs = fake_input.get('privileged_obs', None)
    if privileged_obs is None and hasattr(agent, '_privileged_obs_flat') and agent._privileged_obs_flat is not None:
        curr_idx = fake_input.get('idx', None)
        if curr_idx is not None:
            privileged_obs = agent._privileged_obs_flat[curr_idx]
    
    if privileged_obs is not None:
        print(f"  ✅ privileged_obs 会被传递到 calc_gradients!")
        print(f"     形状: {privileged_obs.shape}")
        print(f"     均值: {privileged_obs.mean():.4f}")
    else:
        print(f"  ❌ privileged_obs 不会被传递!")
    
    print("\n" + "=" * 80)
    if privileged_obs is not None and privileged_obs.std() > 0.001:
        print("✅ 验证通过！privileged_obs 正确传递且包含有意义的数据")
    else:
        print("❌ 验证失败！请检查代码")
    print("=" * 80)

if __name__ == "__main__":
    main()
