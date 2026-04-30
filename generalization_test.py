import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from rk45_solver import get_initial_state, solve_geodesic
from pinn_model import GeodesicPINN

def run_generalization_test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running generalization test using device: {device}")
    
    # 1. Load trained PINN (alpha=0.5)
    with open("results/alpha_sweep.pkl", "rb") as f:
        sweep_data = pickle.load(f)
        
    alpha = 0.5
    data = sweep_data[alpha]
    pinn = GeodesicPINN().to(device)
    pinn.load_state_dict(data['model_state'])
    pinn.eval()
    lam_scale = data['lam_scale']
    
    # 2. Define test cases
    test_cases = [
        {"name": "Baseline", "r0": 8.0, "ur0": 0.0, "L": 3.2},
        {"name": "Shifted r0 (9.0)", "r0": 9.0, "ur0": 0.0, "L": 3.2},
        {"name": "Inward ur0 (-0.1)", "r0": 8.0, "ur0": -0.1, "L": 3.2},
        {"name": "Higher L (3.6)", "r0": 8.0, "ur0": 0.0, "L": 3.6},
        {"name": "Lower L (2.8)", "r0": 8.0, "ur0": 0.0, "L": 2.8},
    ]
    
    # Base lambda span
    lam_span = [0, 500]
    num_points = 500
    
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    os.makedirs("plots", exist_ok=True)
    
    print("\n" + "="*50)
    print("--- GENERALIZATION TEST DIAGNOSTICS ---")
    print("="*50)
    
    for i, case in enumerate(test_cases):
        r0 = case["r0"]
        ur0 = case["ur0"]
        L = case["L"]
        uphi0 = L / (r0**2)
        
        # RK45 Ground Truth
        state = get_initial_state(r0=r0, ur0=ur0, uphi0=uphi0)
        sol = solve_geodesic(state, lam_span, num_points=num_points)
        
        lam_true = sol.t
        r_true = sol.y[1]
        phi_true = sol.y[2]
        x_true = r_true * np.cos(phi_true)
        y_true = r_true * np.sin(phi_true)
        
        # PINN Prediction
        lam_tensor = torch.tensor(lam_true / lam_scale, dtype=torch.float32).view(-1, 1).to(device)
        with torch.no_grad():
            pred = pinn(lam_tensor).cpu().numpy()
            
        r_pred = pred[:, 1]
        phi_pred = pred[:, 2]
        x_pred = r_pred * np.cos(phi_pred)
        y_pred = r_pred * np.sin(phi_pred)
        
        # Diagnostics
        max_dev = np.max(np.sqrt((x_true - x_pred)**2 + (y_true - y_pred)**2))
        
        if max_dev < 3.0:
            verdict = "GOOD -> Similar shape"
        elif max_dev < 8.0:
            verdict = "OK   -> Rough match"
        else:
            verdict = "FAIL -> Completely wrong"
            
        print(f"Case {i+1}: {case['name']}")
        print(f"  Max Deviation: {max_dev:.2f} M")
        print(f"  Verdict:       {verdict}\n")
        
        # Plotting
        ax = axes[i]
        ax.plot(x_true, y_true, 'k-', linewidth=2, label='RK45')
        ax.plot(x_pred, y_pred, 'r--', linewidth=2, label='PINN')
        ax.set_aspect('equal')
        ax.set_title(f"{case['name']}\n(r0={r0}, ur0={ur0}, L={L})")
        
        # Plot black hole
        circle = plt.Circle((0, 0), 2.0, color='black', fill=True, alpha=0.2)
        ax.add_patch(circle)
        
        if i == 0:
            ax.legend()
            
        ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig('plots/generalization_test.png', dpi=200)
    print("Saved plot to plots/generalization_test.png")
    print("="*50)

if __name__ == "__main__":
    run_generalization_test()
