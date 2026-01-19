
import json
import matplotlib.pyplot as plt
import numpy as np
import os

# Set style
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Liberation Sans"]
plt.rcParams["axes.unicode_minus"] = False

def plot_results(json_path):
    if not os.path.exists(json_path):
        print(f"File not found: {json_path}")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Organize data
    experiments = {}
    for entry in data:
        exp = entry.get("experiment", "mass")
        mode = entry["mode"]
        
        if exp not in experiments:
            experiments[exp] = {}
        if mode not in experiments[exp]:
            experiments[exp][mode] = {"x": [], "y": []}
            
        x_val = entry["mass"] if exp == "mass" else entry["wind"]
        y_val = entry["success_rate"]
        
        experiments[exp][mode]["x"].append(x_val)
        experiments[exp][mode]["y"].append(y_val)
    
    # Colors/Styles
    styles = {
        "student": {"label": "Ours (Latent Distillation)", "color": "red", "marker": "o"},
        "teacher": {"label": "Baseline (End-to-End RL)", "color": "blue", "marker": "s"},
        "pure_pd": {"label": "Baseline (Pure PD)", "color": "gray", "marker": "^", "linestyle": "--"}
    }
    
    # Plot
    for exp_name, modes in experiments.items():
        plt.figure(figsize=(8, 6))
        
        for mode, values in modes.items():
            # Sort by x
            xy = sorted(zip(values["x"], values["y"]))
            x = [p[0] for p in xy]
            y = [p[1] for p in xy]
            
            style = styles.get(mode, {"label": mode, "color": "black", "marker": "x"})
            plt.plot(x, y, label=style["label"], color=style.get("color"), 
                     marker=style.get("marker"), linestyle=style.get("linestyle", "-"), linewidth=2)
            
        plt.xlabel("Payload Mass (kg)" if exp_name == "mass" else "Wind Disturbance (N)", fontsize=12)
        plt.ylabel("Success Rate", fontsize=12)
        plt.title(f"Success Rate vs. {'Payload Mass' if exp_name == 'mass' else 'Wind Disturbance'}", fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=10)
        plt.ylim(-0.05, 1.05)
        
        output_file = f"paper/figures/fig_5_delta_{exp_name}.png"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {output_file}")
        # plt.show()

if __name__ == "__main__":
    plot_results("paper/delta_plot_results.json")
