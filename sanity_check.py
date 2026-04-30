import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from pinn_model import GeodesicPINN

def quick_sanity_check(pinn, sol, lam_scale, device):
    """
    Given a trained PINN and RK45 solution, generates fast diagnostics
    to verify trajectory and physics validity.
    """
    pinn.eval()
    
    # 1. Ground truth data extraction
    lam_true = sol.t
    r_true = sol.y[1]
    phi_true = sol.y[2]
    
    x_true = r_true * np.cos(phi_true)
    y_true = r_true * np.sin(phi_true)
    
    # 2. PINN Prediction
    lam_tensor = torch.tensor(lam_true / lam_scale, dtype=torch.float32).view(-1, 1).to(device)
    lam_tensor.requires_grad = True
    
    pred = pinn(lam_tensor)
    
    r_pred = pred[:, 1].detach().cpu().numpy().flatten()
    phi_pred = pred[:, 2].detach().cpu().numpy().flatten()
    
    x_pred = r_pred * np.cos(phi_pred)
    y_pred = r_pred * np.sin(phi_pred)
    
    # 3. Physics Check (Energy Conservation)
    ones = torch.ones_like(lam_tensor)
    dt_dlam = torch.autograd.grad(pred[:, 0:1], lam_tensor, grad_outputs=ones, create_graph=False)[0] / lam_scale
    
    ut = dt_dlam.detach().cpu().numpy().flatten()
    
    # E = (1 - 2/r) * ut
    f = 1.0 - 2.0 / r_pred
    E = f * ut
    
    energy_drift = np.max(E) - np.min(E)
    
    # 4. Diagnostics & Printing
    max_dev = np.max(np.sqrt((x_true - x_pred)**2 + (y_true - y_pred)**2))
    
    print("\n" + "="*30)
    print("--- SANITY CHECK SUMMARY ---")
    print("="*30)
    print(f"Max Trajectory Deviation: {max_dev:.4f} M")
    print(f"Energy Variation (Delta E): {energy_drift:.4e}")
    
    # Simple thresholding for PASS/FAIL
    # If it drifts by less than 1.0M in position and energy is strictly conserved
    if max_dev < 1.0 and energy_drift < 0.05:
        print("\nVerdict: PASS")
    else:
        print("\nVerdict: FAIL")
    print("="*30 + "\n")
    
    # 5. Plotting
    os.makedirs("plots", exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Subplot 1: Cartesian Trajectory
    ax1.plot(x_true, y_true, 'k-', linewidth=2, label='RK45 (True)')
    ax1.plot(x_pred, y_pred, 'r--', linewidth=2, label='PINN (Pred)')
    ax1.set_aspect('equal')
    ax1.set_title('Trajectory Comparison (Cartesian)')
    ax1.set_xlabel('x / M')
    ax1.set_ylabel('y / M')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot central black hole just for reference
    circle = plt.Circle((0, 0), 2.0, color='black', fill=True, alpha=0.2)
    ax1.add_patch(circle)
    
    # Subplot 2: Radius vs Lambda
    ax2.plot(lam_true, r_true, 'k-', linewidth=2, label='RK45')
    ax2.plot(lam_true, r_pred, 'r--', linewidth=2, label='PINN')
    ax2.set_title('Radius vs Lambda (Phase Check)')
    ax2.set_xlabel('Affine Parameter (λ)')
    ax2.set_ylabel('Radius (r / M)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = 'plots/quick_sanity_check.png'
    plt.savefig(plot_path, dpi=200)
    print(f"Diagnostics plot saved to {plot_path}")


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running sanity check using device: {device}")
    
    try:
        with open("data/trajectories.pkl", "rb") as f:
            datasets = pickle.load(f)
        sol = datasets['bound']
        
        with open("results/alpha_sweep.pkl", "rb") as f:
            sweep_data = pickle.load(f)
            
        # Check alpha=0.5 (Balanced Model)
        alpha_test = 0.5
        print(f"Testing Model: Alpha = {alpha_test}")
        
        data = sweep_data[alpha_test]
        pinn = GeodesicPINN().to(device)
        pinn.load_state_dict(data['model_state'])
        lam_scale = data['lam_scale']
        
        quick_sanity_check(pinn, sol, lam_scale, device)
        
    except FileNotFoundError as e:
        print(f"Error: Run experiments.py first to generate models. ({e})")
