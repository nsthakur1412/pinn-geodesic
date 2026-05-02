import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
from unified_model import UnifiedPINN
from rk45_solver import get_initial_state, solve_geodesic
from physics import f, compute_conserved_quantities

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================
# GENERALIZATION TEST MATRIX (Same as unified_eval)
# ============================================================
TEST_CASES = [
    {"name": "Near ISCO Bound",    "r0": 6.5,  "ur0": 0.0,   "L": 3.46},
    {"name": "Wide Bound Orbit",   "r0": 25.0, "ur0": 0.0,   "L": 5.5},
    {"name": "Unseen Bound",       "r0": 8.5,  "ur0": 0.0,   "L": 3.4},
    {"name": "Critical Scattering","r0": 20.0, "ur0": -0.15, "L": 4.0},
    {"name": "Unseen Escape",      "r0": 18.0, "ur0": -0.12, "L": 6.0},
    {"name": "High-Energy Plunge", "r0": 10.0, "ur0": -0.2,  "L": 2.5},
]

def load_no_physics_model():
    pinn = UnifiedPINN(hidden_layers=5, neurons_per_layer=128).to(device)
    path = "results/no_physics_pinn.pt"
    
    if os.path.exists(path):
        try:
            state_dict = torch.load(path, map_location=device, weights_only=True)
            pinn.load_state_dict(state_dict)
            print(f"Successfully loaded NO-PHYSICS model from {path}")
            pinn.eval()
            return pinn
        except Exception as e:
            print(f"Error loading model: {e}")
    else:
        print(f"Error: {path} not found!")
    return None

def predict_trajectory(pinn, r0, ur0, L, lam_array):
    scalers = pinn.scalers
    N = len(lam_array)
    lam_scaled = lam_array / scalers['lam_scale']
    r0_norm = (r0 - scalers['r0_mean']) / scalers['r0_std']
    ur0_norm = (ur0 - scalers['ur0_mean']) / scalers['ur0_std']
    L_norm = (L - scalers['L_mean']) / scalers['L_std']
    
    inputs = np.zeros((N, 4), dtype=np.float32)
    inputs[:, 0] = lam_scaled
    inputs[:, 1] = r0_norm
    inputs[:, 2] = ur0_norm
    inputs[:, 3] = L_norm
    
    inputs_t = torch.tensor(inputs).to(device)
    with torch.no_grad():
        _, out_phys = pinn(inputs_t)
        out_phys = out_phys.cpu().numpy()
    return out_phys

def eval_trajectory_accuracy(pinn):
    print("\n" + "="*60)
    print("NO-PHYSICS TRAJECTORY ACCURACY (100 EPOCHS)")
    print("="*60)
    
    n = len(TEST_CASES)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 6*rows))
    axes = axes.flatten()
    
    for i, case in enumerate(TEST_CASES):
        r0, ur0, L = case['r0'], case['ur0'], case['L']
        uphi0 = L / (r0**2)
        state = get_initial_state(r0=r0, ur0=ur0, uphi0=uphi0)
        sol = solve_geodesic(state, [0, 300], num_points=500)
        
        lam_true = sol.t
        r_true, phi_true = sol.y[1], sol.y[2]
        x_true = r_true * np.cos(phi_true)
        y_true = r_true * np.sin(phi_true)
        
        out = predict_trajectory(pinn, r0, ur0, L, lam_true)
        r_pred, phi_pred = out[:, 1], out[:, 2]
        x_pred = r_pred * np.cos(phi_pred)
        y_pred = r_pred * np.sin(phi_pred)
        
        max_dev = np.max(np.sqrt((x_true - x_pred)**2 + (y_true - y_pred)**2))
        verdict = "EXCELLENT" if max_dev < 3 else ("OK" if max_dev < 10 else "FAIL")
        print(f"  [{i+1}/{n}] {case['name']:25s} | MaxDev={max_dev:.2f}M | {verdict}")
        
        ax = axes[i]
        ax.plot(x_true, y_true, 'k-', linewidth=2, label='RK45')
        ax.plot(x_pred, y_pred, 'r--', linewidth=1.5, label='No-Phys PINN')
        ax.add_patch(plt.Circle((0, 0), 2.0, color='black', fill=True, alpha=0.3))
        ax.set_aspect('equal')
        ax.set_title(f"{case['name']}\nMaxDev={max_dev:.2f}M", fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig('plots/eval_no_physics_trajectories.png', dpi=200)
    plt.close()
    print("  Saved: plots/eval_no_physics_trajectories.png")

if __name__ == "__main__":
    pinn = load_no_physics_model()
    if pinn:
        eval_trajectory_accuracy(pinn)
