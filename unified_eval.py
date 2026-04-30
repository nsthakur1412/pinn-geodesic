import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from unified_model import UnifiedPINN
from rk45_solver import get_initial_state, solve_geodesic

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate_unified_model():
    print(f"Loading Unified Parameterized PINN on {device}...")
    
    # 1. Load Model and Scalers
    pinn = UnifiedPINN(hidden_layers=5, neurons_per_layer=128).to(device)
    try:
        pinn.load_state_dict(torch.load("results/unified_pinn.pt", map_location=device))
    except FileNotFoundError:
        print("Error: Train the model first by running unified_train.py!")
        return
        
    pinn.eval()
    
    with open("data/unified_scalers.pkl", "rb") as f:
        scalers = pickle.load(f)
        
    lam_scale = scalers['lam_scale']
    r0_mean, r0_std = scalers['r0_mean'], scalers['r0_std']
    ur0_mean, ur0_std = scalers['ur0_mean'], scalers['ur0_std']
    L_mean, L_std = scalers['L_mean'], scalers['L_std']
    
    # 2. Define Unseen Test Cases
    # These ICs must fall within the bounds of the training distribution, but be exactly unseen values.
    test_cases = [
        {"name": "Unseen Bound", "r0": 8.5, "ur0": 0.0, "L": 3.4},
        {"name": "Unseen Escape", "r0": 20.0, "ur0": -0.15, "L": 6.5},
        {"name": "Unseen Capture", "r0": 12.0, "ur0": -0.1, "L": 2.8},
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    print("\nEvaluating on Unseen Initial Conditions:")
    print("="*50)
    
    for i, case in enumerate(test_cases):
        r0 = case['r0']
        ur0 = case['ur0']
        L = case['L']
        
        print(f"[{i+1}/3] {case['name']} (r0={r0}, ur0={ur0}, L={L})")
        
        # RK45 Ground Truth
        uphi0 = L / (r0**2)
        state = get_initial_state(r0=r0, ur0=ur0, uphi0=uphi0)
        sol = solve_geodesic(state, [0, 300], num_points=400)
        
        lam_true = sol.t
        r_true = sol.y[1]
        phi_true = sol.y[2]
        x_true = r_true * np.cos(phi_true)
        y_true = r_true * np.sin(phi_true)
        
        # Prepare PINN Inputs: Shape (N, 4) -> [lam_scaled, r0_norm, ur0_norm, L_norm]
        N = len(lam_true)
        lam_scaled = lam_true / lam_scale
        r0_norm = (r0 - r0_mean) / r0_std
        ur0_norm = (ur0 - ur0_mean) / ur0_std
        L_norm = (L - L_mean) / L_std
        
        inputs = np.zeros((N, 4), dtype=np.float32)
        inputs[:, 0] = lam_scaled
        inputs[:, 1] = r0_norm
        inputs[:, 2] = ur0_norm
        inputs[:, 3] = L_norm
        
        inputs_tensor = torch.tensor(inputs).to(device)
        
        # PINN Prediction
        with torch.no_grad():
            # out_phys returns the physically un-standardized values [t, r, phi] directly
            _, out_phys = pinn(inputs_tensor)
            out_phys = out_phys.cpu().numpy()
            
        r_pred = out_phys[:, 1]
        phi_pred = out_phys[:, 2]
        x_pred = r_pred * np.cos(phi_pred)
        y_pred = r_pred * np.sin(phi_pred)
        
        # Error Metric
        max_dev = np.max(np.sqrt((x_true - x_pred)**2 + (y_true - y_pred)**2))
        
        if max_dev < 3.0:
            verdict = "GOOD (Excellent qualitative match)"
        elif max_dev < 10.0:
            verdict = "OK (Matches general physics curve)"
        else:
            verdict = "FAIL (Diverged completely)"
            
        print(f"      Max Deviation: {max_dev:.2f} M")
        print(f"      Verdict:       {verdict}\n")
        
        # Plotting
        ax = axes[i]
        ax.plot(x_true, y_true, 'k-', linewidth=2.5, label='RK45 (Truth)')
        ax.plot(x_pred, y_pred, 'r--', linewidth=2.0, label='Unified PINN (Pred)')
        ax.set_aspect('equal')
        ax.set_title(f"{case['name']}\nMax Dev: {max_dev:.2f} M")
        
        # Black Hole
        circle = plt.Circle((0, 0), 2.0, color='black', fill=True, alpha=0.3)
        ax.add_patch(circle)
        
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend()
            
    print("="*50)
    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig('plots/unified_test.png', dpi=200)
    print("Saved unified model evaluation plot to plots/unified_test.png")

if __name__ == "__main__":
    evaluate_unified_model()
