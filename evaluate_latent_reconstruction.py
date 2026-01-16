
import os
from aerial_gym.utils.logging import CustomLogger
import yaml
from aerial_gym.config.task_config.payload_compensation_task_teacher_config import task_config
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from rl_games.torch_runner import Runner

import torch
from aerial_gym.rl_training.rl_games.a2c_aux_loss import A2CAgentWithAuxLoss


from rl_games.common import env_configurations, vecenv
from aerial_gym.registry.task_registry import task_registry
from aerial_gym.rl_training.rl_games.runner import AERIALRLGPUEnv

# Register Environment
env_configurations.register(
    "payload_compensation_task_teacher",
    {
        "env_creator": lambda **kwargs: task_registry.make_task(
            "payload_compensation_task_teacher", **kwargs
        ),
        "vecenv_type": "AERIAL-RLGPU",
    },
)

# Register VecEnv
vecenv.register(
    "AERIAL-RLGPU",
    lambda config_name, num_actors, **kwargs: AERIALRLGPUEnv(config_name, num_actors, **kwargs),
)

def evaluate_reconstruction():
    logger = CustomLogger("LatentEval")
    
    # 1. Load Configuration
    config_path = "aerial_gym/rl_training/rl_games/ppo_aerial_quad_aux.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Force evaluation mode
    config['params']['config']['minibatch_size'] = config['params']['config']['num_actors']  # full batch
    config['params']['train_dir'] = 'runs'
    
    # 2. Initialize Runner
    runner = Runner()
    runner.load(config)
    
    # 3. Register and Create Agent
    runner.algo_factory.register_builder('a2c_continuous_aux', lambda **kwargs: A2CAgentWithAuxLoss(**kwargs))
    # Inject default observer as A2CBase expects it in config
    if 'features' not in config['params']['config']:
        config['params']['config']['features'] = {}
    config['params']['config']['features']['observer'] = runner.algo_observer
    config['params']['features'] = {'observer': runner.algo_observer} # Keep this just in case
    agent = runner.algo_factory.create('a2c_continuous_aux', base_name='run', params=config['params'])
    
    # 4. Load Checkpoint
    latest_ckpt = "runs/teacher_aux_fixed_imitation_16-19-28-52/nn/last_teacher_aux_fixed_imitation_ep_86_rew_5256.954.pth"
    if not os.path.exists(latest_ckpt):
        latest_ckpt = os.path.abspath(latest_ckpt)
        
    logger.info(f"Loading checkpoint: {latest_ckpt}")
    agent.restore(latest_ckpt)
    
    # 5. Initialize Envs
    # Player usually initializes env in __init__, so we just get it
    env = agent.vec_env
    # We must reset to get initial observation
    obs = env.reset()
    
    # 4. Run Evaluation Episode
    env = agent.vec_env
    obs = env.reset()
    
    # Get network
    if hasattr(agent.model, 'a2c_network'):
        network = agent.model.a2c_network
    else:
        network = agent.model.network
        
    if network.priv_encoder is None or network.aux_decoder is None:
        logger.error("Network does not have privileged encoder or aux decoder!")
        return

    # Data collection
    num_steps = 2500
    privileged_dim = network.priv_dim
    
    # Storage
    true_priv_data = [] # [steps, envs, dim]
    recon_priv_data = [] # [steps, envs, dim]
    
    logger.info(f"Running evaluation for {num_steps} steps...")
    
    agent.model.eval()
    with torch.no_grad():
        for i in range(num_steps):
            # Extract privileged obs
            if isinstance(obs, dict):
                priv_obs = obs.get('privileged_obs')
            else:
                # Assuming last N dims if simple tensor
                priv_obs = obs[:, -privileged_dim:]
            
            if priv_obs is None:
                logger.error("Could not find privileged_obs in observations!")
                break
                
            # 1. Normalize (as done in forward)
            priv_norm = network.priv_norm(priv_obs)
            
            # 2. Encode
            priv_embed = network.priv_encoder(priv_norm)
            
            # 3. Decode
            priv_recon = network.aux_decoder(priv_embed)
            
            true_priv_data.append(priv_norm.cpu().numpy())
            recon_priv_data.append(priv_recon.cpu().numpy())
            
            # Step environment
            # Step environment
            # Bypass agent.model and use network directly
            res = network(obs) 
            # res is (mu, logstd, value, states) for A2C continuous
            actions = res[0]
            # Ensure actions are detached/cpu for env if needed (env.step usually checks)
            # But getting from network keeps them on device.
            # Convert to right shape/type? env.step expects tensor usually.
            
            obs, rewards, dones, infos = env.step(actions)
            
    # Conver to arrays
    true_priv_data = np.array(true_priv_data) # [steps, envs, dim]
    recon_priv_data = np.array(recon_priv_data)
    
    # 5. Calculate Metrics & Plot
    # Select first environment for plotting
    env_idx = 0 
    
    feature_names = [
        "Payload Mass", "COM X", "COM Y", "COM Z", 
        "Release Mass", "Warning Flag", "Just Released"
    ]
    
    fig, axes = plt.subplots(privileged_dim, 1, figsize=(10, 2 * privileged_dim))
    steps_axis = np.arange(num_steps)
    
    logger.info("Evaluation Metrics (Avg over all envs):")
    
    for dim in range(privileged_dim):
        true_traj = true_priv_data[:, env_idx, dim]
        recon_traj = recon_priv_data[:, env_idx, dim]
        
        # Calculate R2 and MAE over all data (not just plotted env)
        all_true = true_priv_data[:, :, dim].flatten()
        all_recon = recon_priv_data[:, :, dim].flatten()
        
        mae = np.mean(np.abs(all_true - all_recon))
        mse = np.mean((all_true - all_recon)**2)
        var = np.var(all_true)
        r2 = 1.0 - (mse / (var + 1e-8))
        
        logger.info(f"  Dim {dim} ({feature_names[dim]}): R2 = {r2:.4f}, MAE = {mae:.4f}")
        
        ax = axes[dim] if privileged_dim > 1 else axes
        ax.plot(steps_axis, true_traj, label='Ground Truth (Norm)', color='black', linestyle='--')
        ax.plot(steps_axis, recon_traj, label='Reconstruction', color='cyan', alpha=0.8)
        ax.set_title(f"Dim {dim}: {feature_names[dim]} (R2={r2:.2f})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
    run_dir = os.path.dirname(os.path.dirname(latest_ckpt))  # e.g. runs/teacher_aux_fixed_imitation_16-16-24-01
    plot_path = os.path.join(run_dir, "latent_reconstruction_eval.png")
    plt.savefig(plot_path)
    logger.info(f"Saved evaluation plot to: {plot_path}")
    
    # Also log numeric results to a file
    with open(os.path.join(run_dir, "latent_metrics.txt"), "w") as f:
        f.write("Dimension, Name, R2, MAE\n")
        for dim in range(privileged_dim):
             all_true = true_priv_data[:, :, dim].flatten()
             all_recon = recon_priv_data[:, :, dim].flatten()
             mae = np.mean(np.abs(all_true - all_recon))
             mse = np.mean((all_true - all_recon)**2)
             var = np.var(all_true)
             r2 = 1.0 - (mse / (var + 1e-8))
             f.write(f"{dim}, {feature_names[dim]}, {r2:.6f}, {mae:.6f}\n")

if __name__ == "__main__":
    evaluate_reconstruction()
