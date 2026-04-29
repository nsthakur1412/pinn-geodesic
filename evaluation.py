import numpy as np
import matplotlib.pyplot as plt
import torch
import pickle
import os
from pinn_model import GeodesicPINN
from physics import compute_conserved_quantities

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def load_data(results_dir="results", data_dir="data"):
    with open(os.path.join(data_dir, "trajectories.pkl"), "rb") as f:
        datasets = pickle.load(f)
        
    with open(os.path.join(results_dir, "alpha_sweep.pkl"), "rb") as f:
        alpha_sweep = pickle.load(f)
        
    with open(os.path.join(results_dir, "extrapolation_test.pkl"), "rb") as f:
        extra_test = pickle.load(f)
        
    return datasets, alpha_sweep, extra_test

def build_pinn_from_state(state_dict):
    pinn = GeodesicPINN(hidden_layers=5, neurons_per_layer=64).to(device)
    pinn.load_state_dict(state_dict)
    pinn.eval()
    return pinn

def get_pinn_predictions(pinn, lam):
    lam_tensor = torch.tensor(lam, dtype=torch.float32).view(-1, 1).to(device)
    lam_tensor.requires_grad = True
    
    # Position
    pred = pinn(lam_tensor)
    
    # Velocity
    ones = torch.ones_like(lam_tensor)
    dt_dlam = torch.autograd.grad(pred[:, 0:1], lam_tensor, grad_outputs=ones, retain_graph=True, create_graph=False)[0]
    dr_dlam = torch.autograd.grad(pred[:, 1:2], lam_tensor, grad_outputs=ones, retain_graph=True, create_graph=False)[0]
    dphi_dlam = torch.autograd.grad(pred[:, 2:3], lam_tensor, grad_outputs=ones, create_graph=False)[0]
    
    pos = pred.detach().cpu().numpy()
    vel = torch.cat([dt_dlam, dr_dlam, dphi_dlam], dim=1).detach().cpu().numpy()
    
    return pos, vel

def plot_trajectory_comparison(sol, pinn, alpha, save_path):
    lam = sol.t
    r_rk45 = sol.y[1]
    phi_rk45 = sol.y[2]
    
    pos, _ = get_pinn_predictions(pinn, lam)
    r_pinn = pos[:, 1]
    phi_pinn = pos[:, 2]
    
    # Polar plot
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))
    ax.plot(phi_rk45, r_rk45, 'b-', label='RK45 (Ground Truth)')
    ax.plot(phi_pinn, r_pinn, 'r:', linewidth=2, label=f'PINN (alpha={alpha:.1f})')
    
    # Draw event horizon and photon sphere
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(theta, np.ones_like(theta)*2.0, 'k-', linewidth=2, label='Event Horizon (r=2M)')
    ax.plot(theta, np.ones_like(theta)*3.0, 'k--', linewidth=1, label='Photon Sphere (r=3M)')
    
    ax.set_title(f"Trajectory Comparison (alpha={alpha:.1f})")
    ax.legend()
    plt.savefig(save_path)
    plt.close()

def plot_phase_space_error(sol, pinn, alpha, save_path):
    lam = sol.t
    pos_rk45 = sol.y[0:3].T
    vel_rk45 = sol.y[3:6].T
    
    pos_pinn, vel_pinn = get_pinn_predictions(pinn, lam)
    
    err_pos = np.abs(pos_rk45 - pos_pinn)
    err_vel = np.abs(vel_rk45 - vel_pinn)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    ax1.semilogy(lam, err_pos[:, 0], label='t error')
    ax1.semilogy(lam, err_pos[:, 1], label='r error')
    ax1.semilogy(lam, err_pos[:, 2], label='phi error')
    ax1.set_ylabel("Positional Error")
    ax1.legend()
    ax1.set_title(f"Phase-Space Error (alpha={alpha:.1f})")
    ax1.grid(True)
    
    ax2.semilogy(lam, err_vel[:, 0], label='u_t error')
    ax2.semilogy(lam, err_vel[:, 1], label='u_r error')
    ax2.semilogy(lam, err_vel[:, 2], label='u_phi error')
    ax2.set_ylabel("Velocity Error")
    ax2.set_xlabel("Affine Parameter lambda")
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_conservation_deviation(sol, pinn, alpha, save_path):
    lam = sol.t
    
    pos_pinn, vel_pinn = get_pinn_predictions(pinn, lam)
    state_pinn = np.concatenate([pos_pinn, vel_pinn], axis=1)
    
    E_pinn, L_pinn, norm_pinn = compute_conserved_quantities(state_pinn)
    
    # Ground truth values from initial state of RK45
    E_rk45, L_rk45, norm_rk45 = compute_conserved_quantities(sol.y[:, 0:1].T)
    E0, L0, norm0 = E_rk45[0], L_rk45[0], norm_rk45[0]
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    
    ax1.plot(lam, E_pinn - E0, 'g-', label='PINN Energy Deviation')
    ax1.axhline(0, color='k', linestyle='--')
    ax1.set_ylabel("Delta E")
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(lam, L_pinn - L0, 'b-', label='PINN Angular Momentum Deviation')
    ax2.axhline(0, color='k', linestyle='--')
    ax2.set_ylabel("Delta L")
    ax2.legend()
    ax2.grid(True)
    
    ax3.plot(lam, norm_pinn - norm0, 'r-', label='PINN Normalization Deviation')
    ax3.axhline(0, color='k', linestyle='--')
    ax3.set_ylabel("Delta Norm")
    ax3.set_xlabel("Affine Parameter lambda")
    ax3.legend()
    ax3.grid(True)
    
    plt.suptitle(f"Conservation Laws Deviation (alpha={alpha:.1f})")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def generate_tradeoff_plot(alpha_sweep, save_path):
    alphas = sorted(list(alpha_sweep.keys()))
    
    data_losses = [alpha_sweep[a]['final_data'] for a in alphas]
    phys_losses = [alpha_sweep[a]['final_phys'] for a in alphas]
    
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(data_losses, phys_losses, c=alphas, cmap='viridis', s=100)
    plt.plot(data_losses, phys_losses, 'k--', alpha=0.5)
    
    for i, txt in enumerate(alphas):
        plt.annotate(f"{txt:.1f}", (data_losses[i], phys_losses[i]), xytext=(5, 5), textcoords='offset points')
        
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Final Data Loss (MSE)")
    plt.ylabel("Final Physics Loss (Residuals)")
    plt.title("Loss Trade-off Analysis (Pareto Curve)")
    cbar = plt.colorbar(scatter)
    cbar.set_label('Alpha (Physics Weight)')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_extrapolation_test(sol, pinn, lam_max, save_path):
    lam = sol.t
    pos_pinn, _ = get_pinn_predictions(pinn, lam)
    
    r_rk45 = sol.y[1]
    r_pinn = pos_pinn[:, 1]
    
    plt.figure(figsize=(10, 6))
    plt.plot(lam, r_rk45, 'b-', label='RK45')
    plt.plot(lam, r_pinn, 'r--', label='PINN')
    plt.axvline(x=lam_max, color='k', linestyle=':', label='End of Training Data')
    
    plt.xlabel("Affine Parameter lambda")
    plt.ylabel("Radial Coordinate r")
    plt.title("Extrapolation Test: Radial Trajectory")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def main():
    if not os.path.exists("plots"):
        os.makedirs("plots")
        
    datasets, alpha_sweep, extra_test = load_data()
    bound_sol = datasets['bound']
    
    # 1. Trade-off plot
    # Disabled: The single-seed tradeoff curve is misleading. 
    # Use pareto_analysis.py for the rigorous multi-seed Pareto front instead.
    # generate_tradeoff_plot(alpha_sweep, "plots/tradeoff_curve.png")
    
    # 2. Detailed plots for specific alphas
    for alpha in [0.0, 0.5, 1.0]:
        if alpha in alpha_sweep:
            pinn = build_pinn_from_state(alpha_sweep[alpha]['model_state'])
            
            plot_trajectory_comparison(bound_sol, pinn, alpha, f"plots/trajectory_alpha_{alpha:.1f}.png")
            plot_phase_space_error(bound_sol, pinn, alpha, f"plots/error_alpha_{alpha:.1f}.png")
            plot_conservation_deviation(bound_sol, pinn, alpha, f"plots/conservation_alpha_{alpha:.1f}.png")
            
    # 3. Extrapolation test plot
    pinn_extra = build_pinn_from_state(extra_test['model_state'])
    plot_extrapolation_test(bound_sol, pinn_extra, extra_test['lam_max'], "plots/extrapolation_test.png")
    
    print("All plots generated successfully in 'plots' directory.")

if __name__ == "__main__":
    main()
