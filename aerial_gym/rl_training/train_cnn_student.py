"""
Stage 2: CNN Student Supervised Learning Training Script.

This script trains a CNN encoder to predict the 8-dim latent variable z_t
from historical observations, supervised by the frozen teacher's privileged encoder.

Usage:
    python aerial_gym/rl_training/train_cnn_student.py \
        --teacher_checkpoint runs/teacher_residual_stage1_27-17-06-04/nn/teacher_residual_stage1.pth \
        --num_envs 1024 \
        --history_len 50 \
        --epochs 500 \
        --lr 1e-4 \
        --experiment_name cnn_student_stage2
"""

# Isaac Gym MUST be imported before torch
import isaacgym  # noqa: F401

import os
import sys
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

# Use importlib to load modules directly, avoiding rl_games/__init__.py
import importlib.util

def load_module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Get the base path
_base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load cnn_student_encoder directly
_cnn_encoder_path = os.path.join(_base_path, "rl_training/rl_games/nn/cnn_student_encoder.py")
cnn_encoder_module = load_module_from_file("cnn_student_encoder", _cnn_encoder_path)
CNNStudentEncoder = cnn_encoder_module.CNNStudentEncoder
ObsHistoryBuffer = cnn_encoder_module.ObsHistoryBuffer
create_cnn_encoder = cnn_encoder_module.create_cnn_encoder

# Load PayloadCompensationTask
from aerial_gym.task.payload_compensation_task.payload_compensation_task import (
    PayloadCompensationTask,
)

# Import yaml for config loading
import yaml


def load_training_config(config_path: str):
    """Load training YAML config."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_teacher_policy(cfg, obs_dim, action_dim, num_envs, device):
    """Build the full Teacher policy network."""
    import importlib.util
    
    # Temporarily modify sys.path to prioritize conda rl_games over local
    original_path = sys.path.copy()
    # Remove paths containing aerial_gym/rl_training/rl_games
    sys.path = [p for p in sys.path if "aerial_gym" not in p]
    
    try:
        # Import rl_games from conda installation
        from rl_games.algos_torch import model_builder
        
        # Register privileged actor critic by loading directly
        _base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        _priv_ac_path = os.path.join(_base_path, "rl_training/rl_games/nn/privileged_actor_critic.py")
        spec = importlib.util.spec_from_file_location("privileged_actor_critic", _priv_ac_path)
        priv_ac_module = importlib.util.module_from_spec(spec)
        sys.modules["aerial_gym.rl_training.rl_games.nn.privileged_actor_critic"] = priv_ac_module
        spec.loader.exec_module(priv_ac_module)
        
        params = cfg.get("params", {})
        builder = model_builder.ModelBuilder()
        model = builder.load(params)
        model_cfg = {
            "actions_num": action_dim,
            "input_shape": (obs_dim,),
            "num_seqs": num_envs,
            "value_size": 1,
            "normalize_value": bool(params.get("config", {}).get("normalize_value", False)),
            "normalize_input": bool(params.get("config", {}).get("normalize_input", False)),
        }
        network = model.build(model_cfg)
        network.to(device)
        network.eval()
        return network
    finally:
        # Restore original sys.path
        sys.path = original_path


def load_teacher_encoder(checkpoint_path: str, device: str):
    """
    Load the trained teacher model and extract the privileged encoder.
    
    Returns:
        priv_encoder: Frozen privileged encoder (41 -> 8) WITH normalization
    """
    print(f"Loading teacher checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # rl-games saves model under 'model' key
    model_state = checkpoint.get("model", checkpoint)
    
    # Extract privileged encoder weights and normalization params
    priv_encoder_state = {}
    priv_norm_mean = None
    priv_norm_std = None
    
    for key, value in model_state.items():
        if "a2c_network.priv_encoder" in key:
            # Remove prefix: a2c_network.priv_encoder.0.weight -> 0.weight
            local_key = key.replace("a2c_network.priv_encoder.", "")
            priv_encoder_state[local_key] = value
        elif "a2c_network.priv_norm.mean" in key:
            priv_norm_mean = value
        elif "a2c_network.priv_norm.std" in key:
            priv_norm_std = value
    
    if not priv_encoder_state:
        raise RuntimeError(f"No priv_encoder weights found in checkpoint")
    
    priv_dim = 41
    hidden = 128
    embed_dim = 8
    
    # Build encoder WITH normalization (same as validate script)
    class TeacherEncoder(nn.Module):
        def __init__(self, priv_dim, hidden, embed_dim, norm_mean, norm_std):
            super().__init__()
            # Fixed normalization
            if norm_mean is not None:
                self.register_buffer("norm_mean", norm_mean.float())
            else:
                self.register_buffer("norm_mean", torch.zeros(priv_dim))
            if norm_std is not None:
                self.register_buffer("norm_std", norm_std.float())
            else:
                self.register_buffer("norm_std", torch.ones(priv_dim))
            
            # Encoder layers
            self.encoder = nn.Sequential(
                nn.Linear(priv_dim, hidden),
                nn.ELU(),
                nn.Linear(hidden, hidden),
                nn.ELU(),
                nn.Linear(hidden, embed_dim),
            )
        
        def forward(self, x):
            # Apply normalization
            x = (x - self.norm_mean) / (self.norm_std + 1e-6)
            return self.encoder(x)
    
    priv_encoder = TeacherEncoder(
        priv_dim, hidden, embed_dim, priv_norm_mean, priv_norm_std
    ).to(device)
    
    # Load encoder weights
    priv_encoder.encoder.load_state_dict(priv_encoder_state)
    
    # Freeze encoder
    for param in priv_encoder.parameters():
        param.requires_grad = False
    priv_encoder.eval()
    
    print(f"Loaded teacher encoder: {priv_dim} -> {embed_dim}")
    if priv_norm_mean is not None:
        print(f"  norm_mean: {priv_norm_mean[:5].tolist()}...")
        print(f"  norm_std: {priv_norm_std[:5].tolist()}...")
    
    return priv_encoder


def create_task(num_envs: int, device: str):
    """Create the payload compensation task for data collection."""
    from aerial_gym.config.task_config.payload_compensation_task_teacher_config import (
        task_config,
    )
    
    # Override some settings for data collection
    task_config.num_envs = num_envs
    task_config.headless = True
    task_config.device = device
    
    task = PayloadCompensationTask(
        task_config=task_config,
        seed=42,
        num_envs=num_envs,
        headless=True,
        device=device,
    )
    return task


def collect_training_sample(
    task,
    history_buffer: ObsHistoryBuffer,
    teacher_encoder: nn.Module,
    device: str,
):
    """
    Collect one training sample: (obs_history, teacher_z).
    
    Returns:
        obs_history: (num_envs, obs_dim, history_len)
        teacher_z: (num_envs, latent_dim)
    """
    # Get current observation
    obs_dict = task.obs_dict
    base_obs = task.task_obs["observations"]  # (num_envs, 29)
    priv_obs = task.task_obs.get("priviliged_obs", None)  # (num_envs, 41)
    
    if priv_obs is None:
        raise RuntimeError("Task must provide privileged observations for teacher supervision")
    
    # Push current obs to history
    history_buffer.push(base_obs)
    
    # Get history for CNN input
    obs_history = history_buffer.get()
    
    # Get teacher's latent encoding (frozen)
    with torch.no_grad():
        teacher_z = teacher_encoder(priv_obs)
    
    return obs_history, teacher_z, base_obs


def train_step(
    cnn_encoder: nn.Module,
    optimizer: optim.Optimizer,
    obs_history: torch.Tensor,
    teacher_z: torch.Tensor,
    criterion: nn.Module,
):
    """
    One training step.
    
    Returns:
        loss: scalar loss value
    """
    optimizer.zero_grad()
    
    # Forward pass
    student_z = cnn_encoder(obs_history)
    
    # Compute MSE loss
    loss = criterion(student_z, teacher_z)
    
    # Backward pass
    loss.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(cnn_encoder.parameters(), max_norm=1.0)
    
    optimizer.step()
    
    return loss.item()


def compute_metrics(student_z: torch.Tensor, teacher_z: torch.Tensor):
    """Compute additional metrics for monitoring."""
    with torch.no_grad():
        mse = torch.mean((student_z - teacher_z) ** 2).item()
        mae = torch.mean(torch.abs(student_z - teacher_z)).item()
        
        # Cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(student_z, teacher_z, dim=1)
        cos_sim_mean = cos_sim.mean().item()
        
        # Per-dimension MSE
        dim_mse = torch.mean((student_z - teacher_z) ** 2, dim=0).cpu().numpy()
        
    return {
        "mse": mse,
        "mae": mae,
        "cosine_similarity": cos_sim_mean,
        "dim_mse": dim_mse,
    }


def main():
    parser = argparse.ArgumentParser(description="Train CNN Student Encoder")
    parser.add_argument(
        "--teacher_checkpoint",
        type=str,
        required=True,
        help="Path to teacher checkpoint",
    )
    parser.add_argument("--num_envs", type=int, default=1024, help="Number of parallel envs")
    parser.add_argument("--history_len", type=int, default=50, help="Observation history length")
    parser.add_argument("--latent_dim", type=int, default=8, help="Latent dimension")
    parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs")
    parser.add_argument("--steps_per_epoch", type=int, default=256, help="Steps per epoch")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    parser.add_argument("--experiment_name", type=str, default="cnn_student", help="Experiment name")
    parser.add_argument("--save_interval", type=int, default=50, help="Checkpoint save interval")
    parser.add_argument(
        "--config",
        type=str,
        default="aerial_gym/rl_training/rl_games/ppo_aerial_quad.yaml",
        help="Training YAML config path for building Teacher policy",
    )
    
    args = parser.parse_args()
    
    device = args.device
    obs_dim = 29
    
    # Create output directory
    timestamp = datetime.now().strftime("%d-%H-%M-%S")
    run_dir = f"runs/{args.experiment_name}_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(f"{run_dir}/nn", exist_ok=True)
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir=f"{run_dir}/summaries")
    
    print("=" * 60)
    print("CNN Student Encoder Training - Stage 2")
    print("=" * 60)
    print(f"Teacher checkpoint: {args.teacher_checkpoint}")
    print(f"Num envs: {args.num_envs}")
    print(f"History length: {args.history_len}")
    print(f"Latent dim: {args.latent_dim}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Output dir: {run_dir}")
    print("=" * 60)
    
    # Load teacher encoder
    teacher_encoder = load_teacher_encoder(args.teacher_checkpoint, device)
    
    # Create CNN student encoder
    cnn_encoder = create_cnn_encoder(
        obs_dim=obs_dim,
        history_len=args.history_len,
        latent_dim=args.latent_dim,
        device=device,
    )
    print(f"CNN Encoder: {cnn_encoder}")
    
    # Load Teacher policy for generating realistic flight data
    print("Loading Teacher policy for data generation...")
    cfg = load_training_config(args.config)
    action_dim = 4  # Residual actions: [thrust, torque_x, torque_y, torque_z]
    
    # Load checkpoint to get model weights
    teacher_ckpt = torch.load(args.teacher_checkpoint, map_location=device)
    teacher_policy = build_teacher_policy(cfg, obs_dim, action_dim, args.num_envs, device)
    teacher_policy.load_state_dict(teacher_ckpt["model"], strict=True)
    if getattr(teacher_policy, "normalize_input", False) and "running_mean_std" in teacher_ckpt:
        teacher_policy.running_mean_std.load_state_dict(teacher_ckpt["running_mean_std"])
    print("Teacher policy loaded successfully.")
    
    # Create task for data collection
    print("Creating simulation environment...")
    task = create_task(args.num_envs, device)
    
    # Create history buffer
    history_buffer = ObsHistoryBuffer(
        num_envs=args.num_envs,
        obs_dim=obs_dim,
        history_len=args.history_len,
        device=device,
    )
    
    # Optimizer and loss
    optimizer = optim.Adam(cnn_encoder.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    criterion = nn.MSELoss()
    
    # Initialize environment
    print("Resetting environment...")
    task.reset()
    
    # Warm up history buffer
    print(f"Warming up history buffer ({args.history_len} steps)...")
    for _ in range(args.history_len):
        # Random actions for warm-up
        actions = torch.zeros((args.num_envs, 4), device=device)
        task.step(actions)
        base_obs = task.task_obs["observations"]
        history_buffer.push(base_obs)
    
    print("Starting training...")
    global_step = 0
    best_loss = float("inf")
    
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_cos_sim = 0.0
        
        cnn_encoder.train()
        
        for step in range(args.steps_per_epoch):
            # Get observations
            obs = torch.as_tensor(
                task.task_obs["observations"], device=device, dtype=torch.float32
            )
            priv = task.task_obs.get("priviliged_obs", None)
            if priv is not None:
                priv = torch.as_tensor(priv, device=device, dtype=torch.float32)
            
            # Use Teacher policy to generate action (NOT random!)
            with torch.no_grad():
                input_dict = {
                    "is_train": False,
                    "prev_actions": None,
                    "obs": obs,
                    "privileged_obs": priv,
                    "rnn_states": None,
                    "seq_length": 1,
                }
                result = teacher_policy(input_dict)
                actions = result["mus"]  # Use deterministic action
                actions = torch.clamp(actions, -1.0, 1.0)
            
            task.step(actions)
            
            # Collect training sample
            obs_history, teacher_z, base_obs = collect_training_sample(
                task, history_buffer, teacher_encoder, device
            )
            
            # Train step
            loss = train_step(cnn_encoder, optimizer, obs_history, teacher_z, criterion)
            epoch_loss += loss
            
            # Compute metrics
            with torch.no_grad():
                student_z = cnn_encoder(obs_history)
                metrics = compute_metrics(student_z, teacher_z)
            epoch_cos_sim += metrics["cosine_similarity"]
            
            # Handle environment resets
            if task.terminations.any() or task.truncations.any():
                reset_ids = (task.terminations > 0) | (task.truncations > 0)
                reset_ids = torch.where(reset_ids)[0]
                if reset_ids.numel() > 0:
                    history_buffer.reset(reset_ids)
            
            global_step += 1
            
            # Log to TensorBoard
            if global_step % 50 == 0:
                writer.add_scalar("loss/mse", loss, global_step)
                writer.add_scalar("metrics/cosine_similarity", metrics["cosine_similarity"], global_step)
                writer.add_scalar("metrics/mae", metrics["mae"], global_step)
        
        # Epoch summary
        avg_loss = epoch_loss / args.steps_per_epoch
        avg_cos_sim = epoch_cos_sim / args.steps_per_epoch
        
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"Loss: {avg_loss:.6f} | "
            f"CosSim: {avg_cos_sim:.4f} | "
            f"LR: {current_lr:.2e}"
        )
        
        writer.add_scalar("epoch/avg_loss", avg_loss, epoch)
        writer.add_scalar("epoch/avg_cosine_similarity", avg_cos_sim, epoch)
        writer.add_scalar("epoch/learning_rate", current_lr, epoch)
        
        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0 or avg_loss < best_loss:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": cnn_encoder.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
                "cosine_similarity": avg_cos_sim,
            }
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(checkpoint, f"{run_dir}/nn/best_cnn_encoder.pth")
                print(f"  -> Saved best model (loss: {best_loss:.6f})")
            
            torch.save(checkpoint, f"{run_dir}/nn/cnn_encoder_ep{epoch + 1}.pth")
    
    # Final save
    torch.save(
        {"model_state_dict": cnn_encoder.state_dict()},
        f"{run_dir}/nn/final_cnn_encoder.pth",
    )
    
    print("=" * 60)
    print("Training complete!")
    print(f"Best loss: {best_loss:.6f}")
    print(f"Checkpoints saved to: {run_dir}/nn/")
    print("=" * 60)
    
    writer.close()
    task.close()


if __name__ == "__main__":
    main()
