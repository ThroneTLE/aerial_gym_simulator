import subprocess
import json
import re

# 扩展测试：有风 + 极限载荷，不测 PDOnly
models = [
    {"name": "TeacherFull", "ckpt": "runs/teacher_aux_fixed_imitation_19-21-52-29/nn/last_teacher_aux_fixed_imitation_ep_135_rew_15005.909.pth", "type": "eval"},
    {"name": "NoPhysRand", "ckpt": "runs/ablation_no_phys_rand_20-15-58-40/nn/last_ablation_no_phys_rand_ep_184_rew_15006.649.pth", "type": "eval"},
    {"name": "CNNStudent", "ckpt": "runs/cnn_stage2_blind_v2_20-02-24-13/nn/best_cnn_encoder.pth", "type": "cnn"}
]

def run_eval(model, mass, wind):
    if model["type"] == "eval":
        cmd = [
            "python", "paper/eval_ablation.py",
            "--checkpoint", model["ckpt"],
            "--name", model["name"],
            "--test_mass", str(mass),
            "--test_wind", str(wind),
            "--num_envs", "256",
            "--num_steps", "500"
        ]
    elif model["type"] == "cnn":
        cmd = [
            "python", "aerial_gym/examples/validate_cnn_stage2.py",
            "--cnn_checkpoint", model["ckpt"],
            "--teacher_checkpoint", "runs/teacher_aux_fixed_imitation_19-21-52-29/nn/last_teacher_aux_fixed_imitation_ep_135_rew_15005.909.pth",
            "--test_mass", str(mass),
            "--test_wind", str(wind),
            "--num_envs", "256",
            "--history_len", "200",
            "--steps", "500",
            "--headless", "True"
        ]
    
    print(f"Running: {model['name']} @ {mass}kg, {wind}N wind")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = proc.communicate()
    
    # Extract success rate
    if model["type"] == "cnn":
        match = re.search(r"Survival Rate:\s*([\d.]+)%", stdout)
        if match:
            return float(match.group(1)) / 100.0
    else:
        match = re.search(r"RESULT_JSON:(\{.*\})", stdout)
        if match:
            data = json.loads(match.group(1))
            return data["success_rate"]
    
    print(f"  Failed to parse result!")
    return 0.0

# === Test 1: 有风测试 (0.3N, 0.1-0.4kg) ===
print("=" * 60)
print("Test 1: 有风测试 (0.3N Wind, 0.1-0.4kg)")
print("=" * 60)
print("| Mass | Teacher (Full) | NoPhysRand | CNN Student |")
print("| :--- | :---: | :---: | :---: |")

for mass in [0.1, 0.2, 0.3, 0.4]:
    row = [f"{mass} kg"]
    for model in models:
        sr = run_eval(model, mass, 0.3)
        row.append(f"{sr:.2%}")
    print(f"| {' | '.join(row)} |")

# === Test 2: 极限载荷测试 (0.0N, 0.5-0.8kg) ===
print("\n" + "=" * 60)
print("Test 2: 极限载荷测试 (Still Air, 0.5-0.8kg)")
print("=" * 60)
print("| Mass | Teacher (Full) | NoPhysRand | CNN Student |")
print("| :--- | :---: | :---: | :---: |")

for mass in [0.5, 0.6, 0.7, 0.8]:
    row = [f"{mass} kg"]
    for model in models:
        sr = run_eval(model, mass, 0.0)
        row.append(f"{sr:.2%}")
    print(f"| {' | '.join(row)} |")

print("\n" + "=" * 60)
print("Evaluation Complete!")
print("=" * 60)
