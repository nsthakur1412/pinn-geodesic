import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
import matplotlib.pyplot as plt
from scientific_framework import ScientificMLP, compute_physics_metrics
from rk45_solver import get_initial_state, solve_geodesic

def plot_conservation_metrics():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load SOTA Model
    model_path = "results/checkpoint_latest.pt"
    model = ScientificMLP(hidden_layers=6, neurons_per_layer=256, use_residual=True).to(device)
    state = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(state['model_state_dict'])
    model.eval()
    scalers = model.scalers
    
    # Orbit: Long Duration Stable Bound
    r0, ur0, L, lam_max = 10.0, 0.0, 3.8, 1000
    uphi0 = L / (r0**2)
    state_init = get_initial_state(r0=r0, ur0=ur0, uphi0=uphi0)
    sol = solve_geodesic(state_init, [0, lam_max], num_points=2000)
    lam_eval = sol.t
    
    # Calculate True Conserved Quantities
    f0 = 1.0 - 2.0 / r0
    E_true = np.sqrt(f0 * (1.0 + (1.0/f0)*ur0**2 + (L**2/r0**2)))
    L_true = L
    H_true = -1.0
    
    # PINN Prediction
    inp = np.zeros((len(lam_eval), 4), dtype=np.float32)
    inp[:, 0] = lam_eval / scalers['lam_scale']
    inp[:, 1] = (r0 - scalers['r0_mean']) / scalers['r0_std']
    inp[:, 2] = (ur0 - scalers['ur0_mean']) / scalers['ur0_std']
    inp[:, 3] = (L - scalers['L_mean']) / scalers['L_std']
    
    inp_t = torch.tensor(inp, requires_grad=True).to(device)
    # Use the same function as training for consistency
    _, E_p, L_p, H_p, _, _, _, _, _ = compute_physics_metrics(model, inp_t)
    
    E_p = E_p.detach().cpu().numpy().flatten()
    L_p = L_p.detach().cpu().numpy().flatten()
    H_p = H_p.detach().cpu().numpy().flatten()
    
    # PLOTTING
    fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    
    # 1. Energy
    axes[0].plot(lam_eval, [E_true]*len(lam_eval), 'k-', label='Exact (Analytical)', linewidth=2)
    axes[0].plot(lam_eval, E_p, 'r--', label='PINN Predicted', linewidth=1.5)
    axes[0].set_ylabel("Energy (E)", fontsize=12)
    axes[0].set_title(f"Energy Conservation ($r_0={r0}, L={L}$)", fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. Angular Momentum
    axes[1].plot(lam_eval, [L_true]*len(lam_eval), 'k-', label='Exact (Analytical)', linewidth=2)
    axes[1].plot(lam_eval, L_p, 'g--', label='PINN Predicted', linewidth=1.5)
    axes[1].set_ylabel("Angular Momentum (L)", fontsize=12)
    axes[1].set_title("Angular Momentum Conservation", fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 3. Hamiltonian
    axes[2].plot(lam_eval, [H_true]*len(lam_eval), 'k-', label='Exact (H = -1)', linewidth=2)
    axes[2].plot(lam_eval, H_p, 'b--', label='PINN Predicted', linewidth=1.5)
    axes[2].set_ylabel("Hamiltonian (H)", fontsize=12)
    axes[2].set_xlabel("Proper Time (lambda)", fontsize=12)
    axes[2].set_title("Hamiltonian Constraint Violation", fontsize=14)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # Calculate Relative Drifts
    e_drift = np.abs(E_p - E_true).mean() / E_true
    l_drift = np.abs(L_p - L_true).mean() / L_true
    h_err = np.abs(H_p - H_true).mean()
    
    plt.suptitle(f"Long-Term Conservation Analysis (Epoch 500)\nMean Rel. Drifts: E={e_drift:.2e}, L={l_drift:.2e}, H-Err={h_err:.2e}", fontsize=16)
    plt.tight_layout()
    plt.savefig("plots/conservation_analysis_long_term.png", dpi=200)
    print("Conservation plot saved to plots/conservation_analysis_long_term.png")
    print(f"Metrics: E_drift={e_drift:.4e}, L_drift={l_drift:.4e}, H_err={h_err:.4e}")

if __name__ == "__main__":
    plot_conservation_metrics()
