import os
import subprocess
import json
import re

masses = [0.1, 0.2, 0.3, 0.4]
models = [
    {"name": "TeacherFull", "ckpt": "runs/teacher_aux_fixed_imitation_19-21-52-29/nn/last_teacher_aux_fixed_imitation_ep_135_rew_15005.909.pth", "type": "eval"},
    {"name": "NoPhysRand", "ckpt": "runs/ablation_no_phys_rand_20-15-58-40/nn/last_ablation_no_phys_rand_ep_184_rew_15006.649.pth", "type": "eval"},
    {"name": "PDOnly", "ckpt": None, "type": "pd"},
    {"name": "CNNStudent", "ckpt": "runs/cnn_stage2_blind_v2_20-02-24-13/nn/best_cnn_encoder.pth", "type": "cnn"}
]

results = {} # (model_name, mass) -> success_rate

def run_eval(model, mass):
    if model["type"] == "eval":
        cmd = [
            "python", "paper/eval_ablation.py",
            "--checkpoint", model["ckpt"],
            "--name", model["name"],
            "--test_mass", str(mass),
            "--test_wind", "0.0",
            "--num_envs", "256",
            "--num_steps", "500"
        ]
    elif model["type"] == "pd":
        cmd = [
            "python", "paper/eval_ablation.py",
            "--pd_only",
            "--name", "PDOnly",
            "--test_mass", str(mass),
            "--test_wind", "0.0",
            "--num_envs", "256",
            "--num_steps", "500"
        ]
    elif model["type"] == "cnn":
        cmd = [
            "python", "aerial_gym/examples/validate_cnn_stage2.py",
            "--cnn_checkpoint", model["ckpt"],
            "--teacher_checkpoint", "runs/teacher_aux_fixed_imitation_19-21-52-29/nn/last_teacher_aux_fixed_imitation_ep_135_rew_15005.909.pth",
            "--test_mass", str(mass),
            "--test_wind", "0.0",
            "--num_envs", "256",
            "--history_len", "200",
            "--steps", "500",
            "--headless", "True"
        ]
    
    print(f"Running: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = proc.communicate()
    
    # Extract success rate
    if model["type"] == "cnn":
        # Search for "Survival Rate: XX.XX%"
        match = re.search(r"Survival Rate:\s*([\d.]+)%", stdout)
        if match:
            return float(match.group(1)) / 100.0
    else:
        # Search for RESULT_JSON
        match = re.search(r"RESULT_JSON:(\{.*\})", stdout)
        if match:
            data = json.loads(match.group(1))
            return data["success_rate"]
    
    print(f"Failed to parse result for {model['name']} at {mass}kg")
    return 0.0

print("Starting Horizontal Comparison...")
print("| Mass | Teacher (Full) | NoPhysRand | PDOnly | CNN Student |")
print("| :--- | :---: | :---: | :---: | :---: |")

for mass in masses:
    row = [f"{mass} kg"]
    for model in models:
        sr = run_eval(model, mass)
        row.append(f"{sr:.2%}")
    print(f"| {' | '.join(row)} |")

print("\nEvaluation Complete.")
