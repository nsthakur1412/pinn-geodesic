import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
import csv
from scientific_framework import ScientificMLP, TEST_SUITE
from rk45_solver import get_initial_state, solve_geodesic

def plot_comparative_results():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Scalers
    with open("data/unified_scalers.pkl", "rb") as f:
        scalers = pickle.load(f)

    # 2. Setup Models
    models = {
        "Data-Only": "results/model_data-only.pt",
        "Data + IC": "results/model_data_+_ic.pt",
        "Full PINN": "results/model_full_pinn.pt"
    }
    
    loaded_models = {}
    for name, path in models.items():
        if os.path.exists(path):
            model = ScientificMLP(hidden_layers=6, neurons_per_layer=256).to(device)
            model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
            model.eval()
            loaded_models[name] = model
            print(f"Loaded {name}")
        else:
            print(f"Warning: {path} not found!")

    if not loaded_models:
        print("No models loaded. Exiting.")
        return

    # 3. Plot Trajectories (3 panels)
    plt.figure(figsize=(18, 6))
    colors = {"Data-Only": "blue", "Data + IC": "green", "Full PINN": "red"}
    
    for idx, case in enumerate(TEST_SUITE):
        plt.subplot(1, 3, idx + 1)
        
        # --- Ground Truth ---
        r0, ur0, L = case['r0'], case['ur0'], case['L']
        uphi0 = L / (r0**2)
        state = get_initial_state(r0=r0, ur0=ur0, uphi0=uphi0)
        sol = solve_geodesic(state, [0, 300], num_points=1000)
        
        gt_x = sol.y[1] * np.cos(sol.y[2])
        gt_y = sol.y[1] * np.sin(sol.y[2])
        plt.plot(gt_x, gt_y, 'k-', lw=2, label='Ground Truth (RK45)', alpha=0.8)
        
        # --- Model Predictions ---
        lam_eval = sol.t
        inp = np.zeros((len(lam_eval), 4), dtype=np.float32)
        inp[:, 0] = lam_eval / scalers['lam_scale']
        inp[:, 1] = (r0 - scalers['r0_mean']) / scalers['r0_std']
        inp[:, 2] = (ur0 - scalers['ur0_mean']) / scalers['ur0_std']
        inp[:, 3] = (L - scalers['L_mean']) / scalers['L_std']
        inp_t = torch.tensor(inp).to(device)
        
        for name, model in loaded_models.items():
            with torch.no_grad():
                _, out_phys = model(inp_t)
            pos_p = out_phys.cpu().numpy()
            r_p = pos_p[:, 1]
            phi_p = pos_p[:, 2]
            px = r_p * np.cos(phi_p)
            py = r_p * np.sin(phi_p)
            plt.plot(px, py, color=colors[name], ls='--', label=name, alpha=0.7)

        # Horizon
        circle = plt.Circle((0, 0), 2.0, color='black', alpha=0.3)
        plt.gca().add_artist(circle)
        
        plt.title(f"Trajectory: {case['name']}")
        plt.xlabel("x [M]")
        plt.ylabel("y [M]")
        plt.axis('equal')
        plt.grid(True, alpha=0.2)
        if idx == 0:
            plt.legend()

    plt.tight_layout()
    plt.savefig("plots/study_trajectory_comparison.png", dpi=200)
    print("Saved plots/study_trajectory_comparison.png")

    # 4. Plot Loss History
    csv_path = "results/comparative_study.csv"
    if os.path.exists(csv_path):
        history = []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                history.append(row)
        
        plt.figure(figsize=(10, 6))
        for name in colors.keys():
            epochs = [int(r['epoch']) for r in history if r['experiment_type'] == name]
            losses = [float(r['total_loss']) for r in history if r['experiment_type'] == name]
            if epochs:
                plt.plot(epochs, losses, label=f"{name} Total Loss", color=colors[name], marker='o', markersize=4)
        
        plt.yscale('log')
        plt.title("Training Loss Convergence")
        plt.xlabel("Epoch")
        plt.ylabel("Loss (log scale)")
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.savefig("plots/study_loss_convergence.png", dpi=200)
        print("Saved plots/study_loss_convergence.png")

        # 5. Plot Physics Metrics (Final Epoch)
        final_entries = {}
        for r in history:
            final_entries[r['experiment_type']] = r
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        exp_names = list(colors.keys())
        max_devs = [float(final_entries[n]['bound_max_dev']) if n in final_entries else 0 for n in exp_names]
        energy_drifts = [float(final_entries[n]['energy_drift']) if n in final_entries else 1e-10 for n in exp_names]
        
        # Max Deviation
        ax1.bar(exp_names, max_devs, color=[colors[n] for n in exp_names])
        ax1.set_title("Max Trajectory Deviation (Bound Case)")
        ax1.set_ylabel("Error [M]")
        
        # Energy Drift
        ax2.bar(exp_names, energy_drifts, color=[colors[n] for n in exp_names])
        ax2.set_yscale('log')
        ax2.set_title("Relative Energy Drift")
        ax2.set_ylabel("Drift (log scale)")
        
        plt.tight_layout()
        plt.savefig("plots/study_metrics_comparison.png", dpi=200)
        print("Saved plots/study_metrics_comparison.png")

if __name__ == "__main__":
    plot_comparative_results()
