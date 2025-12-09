"""
Quick checker for privileged_actor_critic checkpoints.

Usage:
    python tools/verify_priv_checkpoint.py --ckpt runs/gen_ppo_xxx/nn/gen_ppo.pth

It does NOT import isaacgym; it loads the custom builder via importlib,
builds a fresh network to inspect shapes, and inspects the checkpoint to
see whether priv_encoder weights exist and what the first MLP layer input
dim is.
"""

import argparse
import importlib.util
import os
import sys
from typing import Dict, Tuple

import torch
from rl_games.algos_torch.model_builder import ModelBuilder


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PRIV_NET_PATH = os.path.join(
    REPO_ROOT, "aerial_gym", "rl_training", "rl_games", "nn", "privileged_actor_critic.py"
)


def load_priv_builder_module():
    """Load privileged_actor_critic without importing isaacgym."""
    spec = importlib.util.spec_from_file_location("privileged_actor_critic", PRIV_NET_PATH)
    mod = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError("Failed to load privileged_actor_critic module.")
    spec.loader.exec_module(mod)
    return mod


def build_fresh_network(
    input_shape: Tuple[int, ...],
    actions_num: int,
    priv_dim: int,
    priv_embed: int,
    priv_hidden: int,
):
    """Instantiate a fresh privileged network and return its state_dict."""
    # Ensure custom builder is registered
    load_priv_builder_module()

    cfg = {
        "name": "privileged_actor_critic",
        "model": {"name": "continuous_a2c_logstd"},
        "network": {
            "name": "privileged_actor_critic",
            "mlp": {
                "units": [256, 256, 128],
                "activation": "elu",
                "initializer": {"name": "default", "scale": 2},
            },
            "privileged_dim": priv_dim,
            "privileged_embed": priv_embed,
            "privileged_hidden": priv_hidden,
            "space": {
                "continuous": {
                    "mu_activation": "None",
                    "sigma_activation": "None",
                    "mu_init": {"name": "default"},
                    "sigma_init": {"name": "const_initializer", "val": -1.0},
                    "fixed_sigma": False,
                    "min_logstd": -4.5,
                    "max_logstd": 1.0,
                }
            },
        },
        "input_shape": input_shape,
        "actions_num": actions_num,
        "value_size": 1,
        "action_space": "continuous",
        "num_actors": 1,
    }

    model = ModelBuilder().load(cfg)
    build_cfg = {k: v for k, v in cfg.items() if k not in ["name", "model", "network"]}
    net = model.build(build_cfg)
    return net.state_dict()


def load_checkpoint_state(path: str) -> Dict[str, torch.Tensor]:
    """Extract the model state_dict from a rl-games checkpoint."""
    ckpt = torch.load(path, map_location="cpu")
    sd = ckpt.get("model", ckpt)
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    return sd


def summarize_state(sd: Dict[str, torch.Tensor]) -> Dict[str, str]:
    """Collect summary stats on priv params and first layer shapes."""
    priv_keys = [k for k in sd if "priv_encoder" in k]
    mlp0_keys = [k for k in sd if k.endswith("actor_mlp.0.weight")]
    rms_keys = [k for k in sd if "running_mean_std.running_mean" in k]
    return {
        "priv_param_count": str(len(priv_keys)),
        "priv_sample_key": priv_keys[0] if priv_keys else "N/A",
        "mlp0_in": str(sd[mlp0_keys[0]].shape[1]) if mlp0_keys else "N/A",
        "mlp0_out": str(sd[mlp0_keys[0]].shape[0]) if mlp0_keys else "N/A",
        "rms_shape": str(sd[rms_keys[0]].shape) if rms_keys else "N/A",
    }


def main():
    parser = argparse.ArgumentParser(description="Verify privileged_actor_critic checkpoint.")
    parser.add_argument("--ckpt", required=True, help="Path to rl-games checkpoint .pth")
    parser.add_argument("--input-shape", type=int, nargs="+", default=[29], help="Base obs dim tuple")
    parser.add_argument("--actions", type=int, default=4, help="Action dimensions")
    parser.add_argument("--priv-dim", type=int, default=41, help="Privileged raw dim")
    parser.add_argument("--priv-embed", type=int, default=8, help="Privileged embed dim")
    parser.add_argument("--priv-hidden", type=int, default=128, help="Privileged encoder hidden size")
    args = parser.parse_args()

    # Build reference network summary
    fresh_sd = build_fresh_network(
        tuple(args.input_shape), args.actions, args.priv_dim, args.priv_embed, args.priv_hidden
    )
    fresh_info = summarize_state(fresh_sd)

    ckpt_sd = load_checkpoint_state(args.ckpt)
    ckpt_info = summarize_state(ckpt_sd)

    print("=== Expected (fresh build) ===")
    for k, v in fresh_info.items():
        print(f"{k}: {v}")
    print("\n=== Checkpoint ===")
    for k, v in ckpt_info.items():
        print(f"{k}: {v}")

    if ckpt_info["priv_param_count"] == "0":
        print("\n[WARN] Checkpoint has no priv_encoder params — likely trained with old builder.")
    else:
        print("\n[OK] Checkpoint contains priv_encoder parameters.")


if __name__ == "__main__":
    # Ensure repo root is on path for rl_games imports
    sys.path.insert(0, REPO_ROOT)
    main()
