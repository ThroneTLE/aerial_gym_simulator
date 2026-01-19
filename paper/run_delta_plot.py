
import os
import sys
import subprocess
import json
import argparse
import glob

def find_latest_checkpoint(pattern):
    # pattern e.g. "runs/cnn_student_*"
    dirs = glob.glob(pattern)
    if not dirs:
        return None
    # Sort by time
    dirs.sort(key=os.path.getmtime, reverse=True)
    latest_dir = dirs[0]
    # Find .pth in nn/
    nn_dir = os.path.join(latest_dir, "nn")
    if not os.path.exists(nn_dir):
        return None
    pths = glob.glob(os.path.join(nn_dir, "*.pth"))
    if not pths:
        return None
    # Prefer "best" or model name
    best = [p for p in pths if "best" in p]
    if best:
        return best[0]
    return pths[0]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default="all", choices=["mass", "wind", "all"])
    parser.add_argument("--student_ckpt", type=str, default=None)
    parser.add_argument("--teacher_ckpt", type=str, default=None)
    parser.add_argument("--num_episodes", type=int, default=50)
    parser.add_argument("--num_envs", type=int, default=100)
    args = parser.parse_args()
    
    # Auto-detect checkoints if not provided
    if args.student_ckpt is None:
        args.student_ckpt = find_latest_checkpoint("runs/cnn_student_*")
        print(f"Auto-detected Student CP: {args.student_ckpt}")
        
    if args.teacher_ckpt is None:
        # Try ablation_full first, then teacher_aux
        args.teacher_ckpt = find_latest_checkpoint("runs/ablation_full_*")
        if args.teacher_ckpt is None:
            args.teacher_ckpt = find_latest_checkpoint("runs/teacher_aux_*")
        print(f"Auto-detected Teacher CP: {args.teacher_ckpt}")
        
    if not args.student_ckpt or not args.teacher_ckpt:
        print("Error: Could not find checkpoints. Please specify manually.")
        return

    modes = ["student", "teacher", "pure_pd"]
    # modes = ["student"] # Debug
    
    results = []

    # Experiment 1: Mass Sweep
    if args.experiment in ["mass", "all"]:
        masses = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
        print(f"\n=== Running Mass Sweep: {masses} ===")
        
        for mode in modes:
            for mass in masses:
                cmd = [
                    sys.executable, "aerial_gym/rl_training/paper/eval_delta.py", # Assuming calling from root
                    "--mode", mode,
                    "--checkpoint", args.student_ckpt if mode == "student" else args.teacher_ckpt,
                    "--mass", str(mass),
                    "--num_episodes", str(args.num_episodes),
                    "--num_envs", str(args.num_envs),
                    "--headless"
                ]
                if mode == "student":
                    cmd.extend(["--teacher_checkpoint", args.teacher_ckpt])
                
                print(f"Running {mode} at mass={mass}...")
                
                # Update path to be relative to CWD if needed.
                # If running from aerial_gym_simulator root, path should be paper/eval_delta.py
                if not os.path.exists("aerial_gym/rl_training/paper/eval_delta.py"):
                     if os.path.exists("paper/eval_delta.py"):
                         cmd[1] = "paper/eval_delta.py"
                
                try:
                    result_proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
                    # Parse JSON
                    for line in result_proc.stdout.splitlines():
                        if line.startswith("RESULT_JSON:"):
                            json_str = line.replace("RESULT_JSON:", "")
                            res = json.loads(json_str)
                            res["experiment"] = "mass"
                            results.append(res)
                            print(f"  -> Success Rate: {res['success_rate']:.2f}")
                except subprocess.CalledProcessError as e:
                    print(f"  -> Error: {e}")
                    print(e.stdout)
                    print(e.stderr)

    # Experiment 2: Wind Sweep
    if args.experiment in ["wind", "all"]:
        winds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        print(f"\n=== Running Wind Sweep: {winds} ===")
        
        for mode in modes:
            for wind in winds:
                cmd = [
                    sys.executable, "paper/eval_delta.py",
                    "--mode", mode,
                    "--checkpoint", args.student_ckpt if mode == "student" else args.teacher_ckpt,
                    "--wind", str(wind),
                    "--num_episodes", str(args.num_episodes),
                    "--num_envs", str(args.num_envs),
                    "--headless"
                ]
                if mode == "student":
                    cmd.extend(["--teacher_checkpoint", args.teacher_ckpt])
                    
                print(f"Running {mode} at wind={wind}...")
                
                try:
                    result_proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
                    for line in result_proc.stdout.splitlines():
                        if line.startswith("RESULT_JSON:"):
                            json_str = line.replace("RESULT_JSON:", "")
                            res = json.loads(json_str)
                            res["experiment"] = "wind"
                            results.append(res)
                            print(f"  -> Success Rate: {res['success_rate']:.2f}")
                except subprocess.CalledProcessError as e:
                    print(f"  -> Error: {e}")
                    print(e.stdout)
                    print(e.stderr)

    # Save Results
    with open("paper/delta_plot_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved results to paper/delta_plot_results.json")

if __name__ == "__main__":
    main()
