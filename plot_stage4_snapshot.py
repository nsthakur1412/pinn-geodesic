import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from scientific_framework import ScientificMLP
from rk45_solver import get_initial_state, solve_geodesic

def plot_stage4_snapshot(epoch):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = f"results/model_stage4_epoch_{epoch}.pt"
    
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found!")
        return

    # 1. Load Model
    model = ScientificMLP(hidden_layers=6, neurons_per_layer=256, use_residual=True).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # 2. Setup Plotting
    with open("data/unified_scalers.pkl", "rb") as f:
        scalers = pickle.load(f)

    EXTREME_CASES = [
        {"name": "Horizon Grazer", "r0": 3.5,  "ur0": -0.05, "L": 3.8,   "lam": 400},
        {"name": "Speed Demon",    "r0": 20.0, "ur0": -0.8,  "L": 4.5,   "lam": 200},
        {"name": "Far Voyager",    "r0": 60.0, "ur0": -0.01, "L": 10.0,  "lam": 1000},
        {"name": "Chaos Orbit",    "r0": 7.0,  "ur0": 0.05,  "L": 3.05,  "lam": 1500},
    ]

    plt.figure(figsize=(20, 15))
    
    for idx, case in enumerate(EXTREME_CASES):
        r0, ur0, L = case['r0'], case['ur0'], case['L']
        
        # Ground Truth
        uphi0 = L / (r0**2)
        state = get_initial_state(r0=r0, ur0=ur0, uphi0=uphi0)
        sol = solve_geodesic(state, [0, case['lam']], num_points=1000)
        gt_x = sol.y[1] * np.cos(sol.y[2])
        gt_y = sol.y[1] * np.sin(sol.y[2])

        # PINN Prediction
        lam_eval = sol.t
        inp = np.zeros((len(lam_eval), 4), dtype=np.float32)
        inp[:, 0] = lam_eval / scalers['lam_scale']
        inp[:, 1] = (r0 - scalers['r0_mean']) / scalers['r0_std']
        inp[:, 2] = (ur0 - scalers['ur0_mean']) / scalers['ur0_std']
        inp[:, 3] = (L - scalers['L_mean']) / scalers['L_std']
        inp_t = torch.tensor(inp).to(device)
        
        with torch.no_grad():
            _, out_phys = model(inp_t)
        pos_p = out_phys.cpu().numpy()
        px = pos_p[:, 1] * np.cos(pos_p[:, 2])
        py = pos_p[:, 1] * np.sin(pos_p[:, 2])

        plt.subplot(2, 2, idx + 1)
        plt.plot(gt_x, gt_y, 'k-', lw=3, label='RK45 (Truth)', alpha=0.3)
        plt.plot(px, py, 'r--', lw=2, label=f'PINN (Epoch {epoch})')
        
        # Horizon
        circle = plt.Circle((0, 0), 2.0, color='black', alpha=0.4)
        plt.gca().add_artist(circle)
        plt.title(f"Scenario: {case['name']}")
        plt.legend()
        plt.axis('equal')
        plt.grid(True, alpha=0.1)

    plt.suptitle(f"Stage 4 Production Model Snapshot - Epoch {epoch}", fontsize=20)
    plt.savefig(f"plots/stage4_snapshot_epoch_{epoch}.png", dpi=200)
    print(f"Saved plots/stage4_snapshot_epoch_{epoch}.png")

if __name__ == "__main__":
    plot_stage4_snapshot(500)
