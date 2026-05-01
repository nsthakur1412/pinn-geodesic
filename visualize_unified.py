import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
from unified_model import UnifiedPINN
from rk45_solver import get_initial_state, solve_geodesic

def visualize_unified_results():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Model & Scalers
    pinn = UnifiedPINN(hidden_layers=5, neurons_per_layer=128).to(device)
    if os.path.exists("results/unified_pinn.pt"):
        pinn.load_state_dict(torch.load("results/unified_pinn.pt", map_location=device, weights_only=True))
        pinn.eval()
    else:
        print("Model not found!")
        return

    with open("data/unified_scalers.pkl", "rb") as f:
        scalers = pickle.load(f)

    # 2. Define Test Scenarios
    scenarios = [
        {"name": "Bound", "r0": 9.0, "ur0": 0.0, "L": 3.4, "lam_max": 400},
        {"name": "Escape", "r0": 15.0, "ur0": -0.1, "L": 6.5, "lam_max": 300},
        {"name": "Capture", "r0": 10.0, "ur0": -0.05, "L": 2.8, "lam_max": 200}
    ]

    plt.figure(figsize=(15, 5))

    for i, sc in enumerate(scenarios):
        print(f"Generating plot for {sc['name']}...")
        
        # --- Ground Truth (RK45) ---
        uphi0 = sc['L'] / (sc['r0']**2)
        state0 = get_initial_state(r0=sc['r0'], ur0=sc['ur0'], uphi0=uphi0)
        sol = solve_geodesic(state0, [0, sc['lam_max']], num_points=500)
        
        gt_r = sol.y[1]
        gt_phi = sol.y[2]
        gt_x = gt_r * np.cos(gt_phi)
        gt_y = gt_r * np.sin(gt_phi)

        # --- PINN Prediction ---
        lam_eval = np.linspace(0, sc['lam_max'], 500)
        
        # Normalize Inputs
        lam_norm = lam_eval / scalers['lam_scale']
        r0_norm = (sc['r0'] - scalers['r0_mean']) / scalers['r0_std']
        ur0_norm = (sc['ur0'] - scalers['ur0_mean']) / scalers['ur0_std']
        L_norm = (sc['L'] - scalers['L_mean']) / scalers['L_std']
        
        # Expand constants to match lam_eval length
        inputs = np.zeros((500, 4))
        inputs[:, 0] = lam_norm
        inputs[:, 1] = r0_norm
        inputs[:, 2] = ur0_norm
        inputs[:, 3] = L_norm
        
        inputs_t = torch.tensor(inputs, dtype=torch.float32).to(device)
        with torch.no_grad():
            _, out_phys = pinn(inputs_t)
            
        pinn_r = out_phys[:, 1].cpu().numpy()
        pinn_phi = out_phys[:, 2].cpu().numpy()
        pinn_x = pinn_r * np.cos(pinn_phi)
        pinn_y = pinn_r * np.sin(pinn_phi)

        # Plotting
        plt.subplot(1, 3, i+1)
        plt.plot(gt_x, gt_y, 'k-', alpha=0.5, label='RK45 (GT)')
        plt.plot(pinn_x, pinn_y, 'r--', label='PINN')
        
        # Black Hole
        circle = plt.Circle((0, 0), 2.0, color='black', label='Horizon')
        plt.gca().add_artist(circle)
        
        plt.title(f"{sc['name']} Orbit")
        plt.xlabel('x')
        plt.ylabel('y')
        plt.axis('equal')
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('plots/unified_trajectory_comparison.png', dpi=150)
    print("Plot saved to plots/unified_trajectory_comparison.png")

if __name__ == "__main__":
    visualize_unified_results()
