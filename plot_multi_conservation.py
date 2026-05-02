import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scientific_framework import ScientificMLP, compute_physics_metrics
from rk45_solver import get_initial_state, solve_geodesic

def plot_multi_conservation():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Models to Compare
    # Map from Display Name to (Path, Architecture_Config)
    # We assume all use the same 6x256 Residual architecture except maybe old baselines
    model_configs = {
        "Data-Only": ("results/model_data-only.pt", {"hidden_layers": 6, "neurons_per_layer": 256, "use_residual": True}),
        "Data + IC": ("results/model_data_ic_256_res.pt", {"hidden_layers": 6, "neurons_per_layer": 256, "use_residual": True}),
        "Full PINN (S1)": ("results/model_pinn_256_res.pt", {"hidden_layers": 6, "neurons_per_layer": 256, "use_residual": True}),
        "SOTA (Stage 4)": ("results/checkpoint_latest.pt", {"hidden_layers": 6, "neurons_per_layer": 256, "use_residual": True})
    }
    
    loaded_models = {}
    for name, (path, cfg) in model_configs.items():
        if os.path.exists(path):
            try:
                model = ScientificMLP(**cfg).to(device)
                ckpt = torch.load(path, map_location=device, weights_only=False)
                # Check if it's a full checkpoint or just state_dict
                if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                    model.load_state_dict(ckpt['model_state_dict'])
                else:
                    model.load_state_dict(ckpt)
                model.eval()
                loaded_models[name] = model
                print(f"Loaded {name}")
            except Exception as e:
                print(f"Failed to load {name}: {e}")
    
    if not loaded_models:
        print("No models loaded. Exiting.")
        return

    # 2. Test Orbit (Unseen extreme case or standard bound)
    r0, ur0, L, lam_max = 8.0, 0.0, 3.8, 500
    uphi0 = L / (r0**2)
    state_init = get_initial_state(r0=r0, ur0=ur0, uphi0=uphi0)
    sol = solve_geodesic(state_init, [0, lam_max], num_points=1000)
    lam_eval = sol.t
    
    # RK45 Truth
    f0 = 1.0 - 2.0 / r0
    E_true = np.sqrt(f0 * (1.0 + (1.0/f0)*ur0**2 + (L**2/r0**2)))
    L_true = L
    H_true = -1.0
    
    # 3. Predict for each model
    results = {}
    for name, model in loaded_models.items():
        scalers = model.scalers
        inp = np.zeros((len(lam_eval), 4), dtype=np.float32)
        inp[:, 0] = lam_eval / scalers['lam_scale']
        inp[:, 1] = (r0 - scalers['r0_mean']) / scalers['r0_std']
        inp[:, 2] = (ur0 - scalers['ur0_mean']) / scalers['ur0_std']
        inp[:, 3] = (L - scalers['L_mean']) / scalers['L_std']
        
        inp_t = torch.tensor(inp, requires_grad=True).to(device)
        with torch.set_grad_enabled(True): # Need grads for physics metrics
            _, E_p, L_p, H_p, _, _, _, _, _ = compute_physics_metrics(model, inp_t)
            
        results[name] = {
            'E': E_p.detach().cpu().numpy().flatten(),
            'L': L_p.detach().cpu().numpy().flatten(),
            'H': H_p.detach().cpu().numpy().flatten()
        }

    # 4. Plotting
    fig, axes = plt.subplots(3, 1, figsize=(14, 18), sharex=True)
    colors = {"Data-Only": "blue", "Data + IC": "green", "Full PINN (S1)": "orange", "SOTA (Stage 4)": "red"}
    
    # Energy
    axes[0].plot(lam_eval, [E_true]*len(lam_eval), 'k-', label='Exact (RK45)', linewidth=2.5, alpha=0.7)
    for name, res in results.items():
        axes[0].plot(lam_eval, res['E'], color=colors.get(name, 'gray'), label=name, linewidth=1.5)
    axes[0].set_ylabel("Energy (E)", fontsize=12)
    axes[0].set_title(f"Energy Conservation Comparison ($r_0={r0}, L={L}$)", fontsize=14)
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.2)
    
    # Angular Momentum
    axes[1].plot(lam_eval, [L_true]*len(lam_eval), 'k-', label='Exact (RK45)', linewidth=2.5, alpha=0.7)
    for name, res in results.items():
        axes[1].plot(lam_eval, res['L'], color=colors.get(name, 'gray'), label=name, linewidth=1.5)
    axes[1].set_ylabel("Angular Momentum (L)", fontsize=12)
    axes[1].set_title("Angular Momentum Conservation Comparison", fontsize=14)
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.2)
    
    # Hamiltonian Violation
    axes[2].plot(lam_eval, [H_true]*len(lam_eval), 'k-', label='Exact (H = -1)', linewidth=2.5, alpha=0.7)
    for name, res in results.items():
        axes[2].plot(lam_eval, res['H'], color=colors.get(name, 'gray'), label=name, linewidth=1.5)
    axes[2].set_ylabel("Hamiltonian (H)", fontsize=12)
    axes[2].set_xlabel("Proper Time (lambda)", fontsize=12)
    axes[2].set_title("Hamiltonian Constraint Violation Comparison", fontsize=14)
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.savefig("plots/multi_config_conservation.png", dpi=200)
    print("Multi-config conservation plot saved to plots/multi_config_conservation.png")

if __name__ == "__main__":
    plot_multi_conservation()
