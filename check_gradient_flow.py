#!/usr/bin/env python3
"""
Gradient Flow Diagnostic:
Check if gradients from the policy loss flow back to the privileged encoder.
Uses the same model loading pattern as new_my_position_control2.py.
"""

import isaacgym
import torch
import yaml
from rl_games.algos_torch import model_builder as mb

from aerial_gym.rl_training.rl_games.nn import privileged_actor_critic

def main():
    print("=" * 80)
    print("Gradient Flow Diagnostic")
    print("=" * 80)
    
    # 1. Load config
    config_path = "aerial_gym/rl_training/rl_games/ppo_aerial_quad_aux.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    network_config = config['params']['network']
    network_params = network_config.get('params', {})
    
    obs_dim = network_config['space']['vector'][0]
    priv_dim = network_params.get('privileged_dim', 7)
    action_dim = 3
    
    # 2. Build network using the pattern from new_my_position_control2.py
    model = mb.ModelBuilder()
    model.load({
        "network_builder": "A2CBuilder",
        "name": network_config['name'],
    })
    
    model_class = mb.NetworkBuilder
    # Access our custom builder
    from rl_games.algos_torch.model_builder import NetworkBuilder
    
    # Directly instantiate our custom network
    builder = privileged_actor_critic.PrivilegedA2CBuilder()
    
    # Create the network
    network = builder.build(network_config['name'], **{
        'params': {
            **network_params,
            'input_shape': (obs_dim,),
            'num_seqs': 1,
            'value_size': 1,
            'actions_num': action_dim,
        }
    })
    network.to('cuda:0')
    network.train()
    
    print(f"\nNetwork built. Obs dim: {obs_dim}, Priv dim: {priv_dim}, Action dim: {action_dim}")
    print(f"Network type: {type(network)}")
    
    # 3. Create synthetic input
    batch_size = 64
    base_obs = torch.randn(batch_size, obs_dim, device='cuda:0', requires_grad=True)
    priv_obs = torch.randn(batch_size, priv_dim, device='cuda:0', requires_grad=True)
    
    obs_dict = {
        'obs': base_obs,
        'privileged_obs': priv_obs
    }
    
    # 4. Forward pass
    output = network(obs_dict)
    
    # Try different output formats
    if hasattr(output, '__iter__') and 'mus' in output:
        action_mean = output['mus']
    elif isinstance(output, dict):
        action_mean = output.get('mus') or output.get('actions_mean') or list(output.values())[0]
    else:
        action_mean = output[0] if isinstance(output, tuple) else output
    
    print(f"\nAction output shape: {action_mean.shape}")
    
    # 5. Create a mock loss (simulate BC loss: minimize distance to target)
    target_action = torch.randn_like(action_mean) * 0.5
    loss = torch.mean((action_mean - target_action) ** 2)
    
    print(f"Mock Loss: {loss.item():.4f}")
    
    # 6. Backward pass
    loss.backward()
    
    # 7. Check gradients
    print("\n" + "=" * 80)
    print("Gradient Check:")
    print("=" * 80)
    
    # Check gradient on priv_obs (input)
    if priv_obs.grad is not None:
        grad_priv_mag = priv_obs.grad.abs().mean().item()
        print(f"[INPUT] priv_obs gradient magnitude: {grad_priv_mag:.6f}")
        if grad_priv_mag > 1e-6:
            print("   ✅ Gradients flow through privileged input path.")
        else:
            print("   ❌ Gradients are effectively zero for privileged inputs!")
    else:
        print("   ❌ priv_obs.grad is None! No gradient flow at all.")
    
    # Check base_obs gradient for comparison
    if base_obs.grad is not None:
        grad_base_mag = base_obs.grad.abs().mean().item()
        print(f"[INPUT] base_obs gradient magnitude: {grad_base_mag:.6f}")
    
    # Check priv_encoder weights
    priv_encoder_found = False
    for name, module in network.named_modules():
        if 'priv' in name.lower() and hasattr(module, 'weight'):
            priv_encoder_found = True
            for pname, param in module.named_parameters():
                if param.grad is not None:
                    grad_mag = param.grad.abs().mean().item()
                    print(f"[PRIV_ENCODER] {name}.{pname}: grad mean = {grad_mag:.6f}")
                    if grad_mag < 1e-7:
                        print(f"   ⚠️ Very small gradient")
                else:
                    print(f"[PRIV_ENCODER] {name}.{pname}: grad = None")
    
    if not priv_encoder_found:
        print("[PRIV_ENCODER] Not found by name. Checking all parameters:")
        for name, param in network.named_parameters():
            if param.grad is not None:
                grad_mag = param.grad.abs().mean().item()
                print(f"  {name}: grad mean = {grad_mag:.6f}")
            else:
                print(f"  {name}: grad = None")
    
    print("\n" + "=" * 80)
    print("Diagnostic Complete.")
    print("=" * 80)

if __name__ == "__main__":
    main()
