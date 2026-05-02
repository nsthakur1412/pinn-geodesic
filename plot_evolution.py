import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
import csv
from scientific_framework import ScientificMLP, TEST_SUITE
from rk45_solver import get_initial_state, solve_geodesic

def plot_architecture_evolution():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Scalers
    with open("data/unified_scalers.pkl", "rb") as f:
        scalers = pickle.load(f)

    # 2. Setup Models to Compare
    # We compare the PINNs mostly
    comparison_models = {
        "Baseline (128)": {"path": "results/model_pinn_128_baseline.pt", "layers": 5, "width": 128, "res": False, "color": "orange"},
        "Refined (256-Res)": {"path": "results/model_pinn_256_res.pt", "layers": 6, "width": 256, "res": True, "color": "green"},
        "Stiff (256-HighWt)": {"path": "results/model_pinn_256_stiff.pt", "layers": 6, "width": 256, "res": True, "color": "red"}
    }
    
    loaded_models = {}
    for name, info in comparison_models.items():
        if os.path.exists(info['path']):
            model = ScientificMLP(hidden_layers=info['layers'], neurons_per_layer=info['width'], use_residual=info['res']).to(device)
            model.load_state_dict(torch.load(info['path'], map_location=device, weights_only=True))
            model.eval()
            loaded_models[name] = (model, info['color'])
            print(f"Loaded {name}")
        else:
            print(f"Warning: {info['path']} not found!")

    if not loaded_models:
        print("No models loaded. Exiting.")
        return

    # 3. Plot Trajectories (3 panels)
    plt.figure(figsize=(20, 7))
    
    for idx, case in enumerate(TEST_SUITE):
        plt.subplot(1, 3, idx + 1)
        
        # --- Ground Truth ---
        r0, ur0, L = case['r0'], case['ur0'], case['L']
        uphi0 = L / (r0**2)
        state = get_initial_state(r0=r0, ur0=ur0, uphi0=uphi0)
        sol = solve_geodesic(state, [0, 300], num_points=1000)
        
        gt_x = sol.y[1] * np.cos(sol.y[2])
        gt_y = sol.y[1] * np.sin(sol.y[2])
        plt.plot(gt_x, gt_y, 'k-', lw=2.5, label='RK45 Truth', alpha=0.9)
        
        # --- Model Predictions ---
        lam_eval = sol.t
        inp = np.zeros((len(lam_eval), 4), dtype=np.float32)
        inp[:, 0] = lam_eval / scalers['lam_scale']
        inp[:, 1] = (r0 - scalers['r0_mean']) / scalers['r0_std']
        inp[:, 2] = (ur0 - scalers['ur0_mean']) / scalers['ur0_std']
        inp[:, 3] = (L - scalers['L_mean']) / scalers['L_std']
        inp_t = torch.tensor(inp).to(device)
        
        for name, (model, color) in loaded_models.items():
            with torch.no_grad():
                _, out_phys = model(inp_t)
            pos_p = out_phys.cpu().numpy()
            px = pos_p[:, 1] * np.cos(pos_p[:, 2])
            py = pos_p[:, 1] * np.sin(pos_p[:, 2])
            plt.plot(px, py, color=color, ls='--', label=name, alpha=0.8)

        # Horizon
        circle = plt.Circle((0, 0), 2.0, color='black', alpha=0.4)
        plt.gca().add_artist(circle)
        
        plt.title(f"Evolution: {case['name']}")
        plt.xlabel("x [M]")
        plt.ylabel("y [M]")
        plt.axis('equal')
        plt.grid(True, alpha=0.15)
        if idx == 0:
            plt.legend()

    plt.tight_layout()
    plt.savefig("plots/architecture_evolution_comparison.png", dpi=200)
    print("Saved plots/architecture_evolution_comparison.png")

if __name__ == "__main__":
    plot_architecture_evolution()
