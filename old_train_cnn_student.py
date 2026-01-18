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
CNNWithTemporalAttention = cnn_encoder_module.CNNWithTemporalAttention
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


def load_teacher_encoder(checkpoint_path: str, device: str, priv_dim: int = 7):
    """
    Load the trained teacher model and extract the privileged encoder.
    
    Args:
        checkpoint_path: Path to teacher checkpoint
        device: Device to load on
        priv_dim: Privileged observation dimension (default: 7 to match current config)
    
    Returns:
        priv_encoder: Frozen privileged encoder (priv_dim -> 8) WITH normalization
    """
    print(f"Loading teacher checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # rl-games saves model under 'model' key
    model_state = checkpoint.get("model", checkpoint)
    
    # Extract privileged encoder weights and normalization params
    priv_encoder_state = {}
    priv_norm_running_mean = None
    priv_norm_running_var = None
    priv_norm_count = None
    # Also check for FixedObsNorm style (mean/std)
    priv_norm_mean = None
    priv_norm_std = None
    
    for key, value in model_state.items():
        if "a2c_network.priv_encoder" in key:
            # Remove prefix: a2c_network.priv_encoder.0.weight -> 0.weight
            local_key = key.replace("a2c_network.priv_encoder.", "")
            priv_encoder_state[local_key] = value
        # RunningObsNorm uses running_mean/running_var
        elif "a2c_network.priv_norm.running_mean" in key:
            priv_norm_running_mean = value
        elif "a2c_network.priv_norm.running_var" in key:
            priv_norm_running_var = value
        elif "a2c_network.priv_norm.count" in key:
            priv_norm_count = value
        # FixedObsNorm uses mean/std
        elif "a2c_network.priv_norm.mean" in key:
            priv_norm_mean = value
        elif "a2c_network.priv_norm.std" in key:
            priv_norm_std = value
    
    if not priv_encoder_state:
        raise RuntimeError(f"No priv_encoder weights found in checkpoint")
    
    # Auto-detect priv_dim from first encoder layer weights
    first_layer_key = "0.weight"
    if first_layer_key in priv_encoder_state:
        detected_priv_dim = priv_encoder_state[first_layer_key].shape[1]
        if detected_priv_dim != priv_dim:
            print(f"  [Auto-detect] priv_dim from checkpoint: {detected_priv_dim} (overriding {priv_dim})")
            priv_dim = detected_priv_dim
    
    hidden = 128
    embed_dim = 8
    
    # Determine normalization type from checkpoint
    use_running_norm = priv_norm_running_mean is not None
    
    # Build encoder WITH normalization
    class TeacherEncoder(nn.Module):
        def __init__(self, priv_dim, hidden, embed_dim, running_mean, running_var, fixed_mean, fixed_std, use_running):
            super().__init__()
            self.use_running_norm = use_running
            self.eps = 1e-6
            self.clip_range = 10.0
            
            if use_running and running_mean is not None:
                # RunningObsNorm style
                self.register_buffer("running_mean", running_mean.float())
                self.register_buffer("running_var", running_var.float() if running_var is not None else torch.ones(priv_dim))
            elif fixed_mean is not None:
                # FixedObsNorm style
                self.register_buffer("running_mean", fixed_mean.float())
                std = fixed_std.float() if fixed_std is not None else torch.ones(priv_dim)
                self.register_buffer("running_var", std ** 2)  # Convert std to var
            else:
                # Default (no normalization stored)
                self.register_buffer("running_mean", torch.zeros(priv_dim))
                self.register_buffer("running_var", torch.ones(priv_dim))
            
            # Encoder layers
            self.encoder = nn.Sequential(
                nn.Linear(priv_dim, hidden),
                nn.ELU(),
                nn.Linear(hidden, hidden),
                nn.ELU(),
                nn.Linear(hidden, embed_dim),
            )
        
        def forward(self, x):
            # Apply normalization (same as RunningObsNorm.forward)
            std = torch.sqrt(self.running_var + self.eps)
            normalized = (x - self.running_mean) / std
            normalized = torch.clamp(normalized, -self.clip_range, self.clip_range)
            return self.encoder(normalized)
    
    priv_encoder = TeacherEncoder(
        priv_dim, hidden, embed_dim, 
        priv_norm_running_mean, priv_norm_running_var,
        priv_norm_mean, priv_norm_std,
        use_running_norm
    ).to(device)
    
    # Load encoder weights
    priv_encoder.encoder.load_state_dict(priv_encoder_state)
    
    # Freeze encoder
    for param in priv_encoder.parameters():
        param.requires_grad = False
    priv_encoder.eval()
    
    print(f"Loaded teacher encoder: {priv_dim} -> {embed_dim}")
    norm_type = "RunningObsNorm" if use_running_norm else "FixedObsNorm"
    print(f"  Normalization type: {norm_type}")
    print(f"  running_mean[:5]: {priv_encoder.running_mean[:min(5, priv_dim)].tolist()}")
    print(f"  running_var[:5]: {priv_encoder.running_var[:min(5, priv_dim)].tolist()}")
    
    return priv_encoder


def create_task(num_envs: int, device: str, preflight_crash_threshold: float = 10.0):
    """Create the payload compensation task for data collection."""
    from aerial_gym.config.task_config.payload_compensation_task_teacher_config import (
        task_config,
    )
    
    # Override some settings for data collection
    task_config.num_envs = num_envs
    task_config.headless = True
    task_config.device = device
    # Relax crash thresholds for preflight random navigation
    task_config.crash_distance_threshold = preflight_crash_threshold
    task_config.crash_tilt_threshold_deg = 60.0  # Relax from 20° for trajectory tracking
    
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
    priv_obs = task.task_obs.get("privileged_obs", None)  # (num_envs, 41)
    
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
        default="aerial_gym/rl_training/rl_games/ppo_aerial_quad_aux.yaml",
        help="Training YAML config path for building Teacher policy",
    )
    parser.add_argument(
        "--random_preflight_steps",
        type=int,
        default=1000,
        help="Number of steps with random actions before release phase (per episode)",
    )
    parser.add_argument(
        "--use_attention",
        action="store_true",
        help="Use CNN with Temporal Attention instead of standard CNN",
    )
    parser.add_argument(
        "--preflight_waypoint_range",
        type=float,
        default=0.5,
        help="Range for random waypoint targets during preflight (meters)",
    )
    parser.add_argument(
        "--preflight_waypoint_interval",
        type=int,
        default=100,
        help="Steps between waypoint changes during preflight",
    )
    
    args = parser.parse_args()
    
    device = args.device
    
    # Create task first to get dimensions from config
    print("Creating simulation environment to detect dimensions...")
    # Use relaxed crash threshold for preflight random navigation
    task = create_task(args.num_envs, device, preflight_crash_threshold=10.0)
    obs_dim = task.task_config.observation_space_dim
    priv_dim = task.task_config.privileged_observation_space_dim
    action_dim = task.task_config.action_space_dim
    print(f"Detected dimensions: obs_dim={obs_dim}, priv_dim={priv_dim}, action_dim={action_dim}")
    
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
    print(f"Obs dim: {obs_dim}, Priv dim: {priv_dim}, Action dim: {action_dim}")
    print(f"History length: {args.history_len}")
    print(f"Latent dim: {args.latent_dim}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Output dir: {run_dir}")
    print("=" * 60)
    
    # Load teacher encoder (auto-detects priv_dim from checkpoint)
    teacher_encoder = load_teacher_encoder(args.teacher_checkpoint, device, priv_dim)
    
    # Create CNN student encoder (with or without attention)
    if args.use_attention:
        cnn_encoder = CNNWithTemporalAttention(
            obs_dim=obs_dim,
            history_len=args.history_len,
            latent_dim=args.latent_dim,
        ).to(device)
        print(f"CNN Encoder (with Temporal Attention): {cnn_encoder}")
    else:
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
    
    # Load checkpoint to get model weights
    teacher_ckpt = torch.load(args.teacher_checkpoint, map_location=device)
    teacher_policy = build_teacher_policy(cfg, obs_dim, action_dim, args.num_envs, device)
    teacher_policy.load_state_dict(teacher_ckpt["model"], strict=True)
    if getattr(teacher_policy, "normalize_input", False) and "running_mean_std" in teacher_ckpt:
        teacher_policy.running_mean_std.load_state_dict(teacher_ckpt["running_mean_std"])
    print("Teacher policy loaded successfully.")
    
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
        # Zero actions for warm-up (stable hover)
        actions = torch.zeros((args.num_envs, action_dim), device=device)
        task.step(actions)
        base_obs = task.task_obs["observations"]
        history_buffer.push(base_obs)
    
    print("Starting training...")
    global_step = 0
    best_loss = float("inf")
    
    # Per-env step counters for random preflight phase
    env_step_counters = torch.zeros(args.num_envs, dtype=torch.long, device=device)
    
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_cos_sim = 0.0
        
        cnn_encoder.train()
        
        for step in range(args.steps_per_epoch):
            # Get observations
            obs = torch.as_tensor(
                task.task_obs["observations"], device=device, dtype=torch.float32
            )
            priv = task.task_obs.get("privileged_obs", None)
            if priv is not None:
                priv = torch.as_tensor(priv, device=device, dtype=torch.float32)
            
            # Decide action: random waypoint flight for preflight, Teacher for release phase
            use_preflight = env_step_counters < args.random_preflight_steps
            
            # Update random waypoints periodically during preflight
            if use_preflight.any():
                # Change waypoints every preflight_waypoint_interval steps
                should_update_waypoint = (env_step_counters % args.preflight_waypoint_interval == 0) & use_preflight
                if should_update_waypoint.any():
                    update_envs = should_update_waypoint.nonzero(as_tuple=True)[0]
                    # Generate random target positions within range
                    random_targets = (torch.rand(update_envs.shape[0], 3, device=device) * 2 - 1) * args.preflight_waypoint_range
                    random_targets[:, 2] = random_targets[:, 2].abs()  # Keep Z positive (above ground)
                    task.target_position[update_envs] = random_targets
            
            # Preflight: zero compensation action (let controller track random waypoints)
            # Release phase: use Teacher policy for compensation
            if use_preflight.all():
                # All envs in preflight: no compensation, just position tracking
                actions = torch.zeros((args.num_envs, action_dim), device=device)
            elif use_preflight.any():
                # Mixed: zero for preflight, teacher for release
                actions = torch.zeros((args.num_envs, action_dim), device=device)
                teacher_envs = (~use_preflight).nonzero(as_tuple=True)[0]
                # Reset target to origin for release phase envs
                task.target_position[teacher_envs] = 0.0
                with torch.no_grad():
                    input_dict = {
                        "is_train": False,
                        "prev_actions": None,
                        "obs": obs[teacher_envs],
                        "privileged_obs": priv[teacher_envs] if priv is not None else None,
                        "rnn_states": None,
                        "seq_length": 1,
                    }
                    result = teacher_policy(input_dict)
                    actions[teacher_envs] = torch.clamp(result["mus"], -1.0, 1.0)
            else:
                # All envs in release phase: use Teacher policy
                task.target_position[:] = 0.0  # Target is origin during release
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
                    actions = torch.clamp(result["mus"], -1.0, 1.0)
            
            env_step_counters += 1
            
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
                    env_step_counters[reset_ids] = 0  # Reset step counters for new episodes
            
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
