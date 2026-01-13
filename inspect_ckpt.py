
import torch
import os
import sys

# Checkpoint path from the user's script
ckpt_path = "runs/teacher_aux_fixed_imitation_13-15-47-12/nn/last_teacher_aux_fixed_imitation_ep_108_rew_-86.12906.pth"

if not os.path.exists(ckpt_path):
    print(f"Checkpoint not found: {ckpt_path}")
    sys.exit(1)

print(f"Loading checkpoint: {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location="cpu")

print("Keys in checkpoint:", ckpt.keys())

if "running_mean_std" in ckpt:
    rms = ckpt["running_mean_std"]
    print("\nRunningMeanStd found!")
    print("Keys in RMS:", rms.keys())
    if "count" in rms:
        print(f"Count: {rms['count']}")
    if "running_mean" in rms:
        mean = rms["running_mean"]
        print(f"Mean shape: {mean.shape}")
        print(f"Mean values: {mean}")
    if "running_var" in rms:
        var = rms["running_var"]
        print(f"Var shape: {var.shape}")
        print(f"Var values: {var}")
else:
    print("\nWARNING: 'running_mean_std' NOT found in checkpoint!")

if "model" in ckpt:
    model_state = ckpt["model"]
    print("\nModel state dict keys (first 10):", list(model_state.keys())[:10])
