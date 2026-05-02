import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
import matplotlib.pyplot as plt
from scientific_framework import ScientificMLP, compute_physics_metrics
from rk45_solver import get_initial_state, solve_geodesic

def comprehensive_eval():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Models to Evaluate
    configs = [
        {"name": "Stage 4 (SOTA)", "path": "results/checkpoint_latest.pt", "type": "checkpoint"},
        {"name": "Stage 1 (Full PINN)", "path": "results/model_pinn_256_res.pt", "type": "model"},
        {"name": "Data + IC", "path": "results/model_data_ic_256_res.pt", "type": "model"},
        {"name": "Data-Only", "path": "results/model_data_256_res.pt", "type": "model"},
    ]
    
    # 2. Test Cases (Unseen & Extreme)
    test_cases = [
        {"name": "Unseen Interpolation", "r0": 12.0, "ur0": -0.05, "L": 3.5, "lam": 300},
        {"name": "Near-Horizon Dive",    "r0": 6.5,  "ur0": -0.4,  "L": 2.0, "lam": 200},
        {"name": "Extreme Speed Demon",  "r0": 25.0, "ur0": -0.9,  "L": 5.0, "lam": 150},
        {"name": "Long Duration",        "r0": 10.0, "ur0": 0.0,   "L": 3.8, "lam": 800},
    ]
    
    loaded_models = []
    for conf in configs:
        if os.path.exists(conf['path']):
            model = ScientificMLP(hidden_layers=6, neurons_per_layer=256, use_residual=True).to(device)
            state = torch.load(conf['path'], map_location=device, weights_only=False)
            if conf['type'] == "checkpoint":
                model.load_state_dict(state['model_state_dict'])
            else:
                model.load_state_dict(state)
            model.eval()
            loaded_models.append((conf['name'], model))
    
    fig, axes = plt.subplots(len(test_cases), 2, figsize=(16, 5*len(test_cases)))
    
    comparison_table = []
    
    for i, case in enumerate(test_cases):
        ax_traj = axes[i, 0]
        ax_cons = axes[i, 1]
        
        r0, ur0, L, lam_max = case['r0'], case['ur0'], case['L'], case['lam']
        uphi0 = L / (r0**2)
        state = get_initial_state(r0=r0, ur0=ur0, uphi0=uphi0)
        sol = solve_geodesic(state, [0, lam_max], num_points=1000)
        
        gt_x = sol.y[1] * np.cos(sol.y[2])
        gt_y = sol.y[1] * np.sin(sol.y[2])
        
        ax_traj.plot(gt_x, gt_y, 'k-', alpha=0.6, label='RK45 (Truth)', linewidth=2.5)
        
        case_stats = {"Case": case['name']}
        
        for name, model in loaded_models:
            scalers = model.scalers
            lam_eval = sol.t
            inp = np.zeros((len(lam_eval), 4), dtype=np.float32)
            inp[:, 0] = lam_eval / scalers['lam_scale']
            inp[:, 1] = (r0 - scalers['r0_mean']) / scalers['r0_std']
            inp[:, 2] = (ur0 - scalers['ur0_mean']) / scalers['ur0_std']
            inp[:, 3] = (L - scalers['L_mean']) / scalers['L_std']
            
            inp_t = torch.tensor(inp, requires_grad=True).to(device)
            # Physics and Conservation (Needs Gradients)
            phys_res, _, _, H, _, _, _, _, _ = compute_physics_metrics(model, inp_t)
            _, out_phys = model(inp_t)
            
            p_x = out_phys[:, 1].detach().cpu().numpy() * np.cos(out_phys[:, 2].detach().cpu().numpy())
            p_y = out_phys[:, 1].detach().cpu().numpy() * np.sin(out_phys[:, 2].detach().cpu().numpy())
            h_viol = torch.abs(H + 1.0).detach().cpu().numpy()
            
            ax_traj.plot(p_x, p_y, '--', label=name, alpha=0.8)
            ax_cons.semilogy(lam_eval, h_viol, label=name, alpha=0.8)
            
            # Store metrics for table
            dev = np.sqrt((p_x - gt_x)**2 + (p_y - gt_y)**2)
            case_stats[f"{name} Dev"] = np.mean(dev)
            case_stats[f"{name} H-Viol"] = np.mean(h_viol)

        # Plot Horizon
        circle = plt.Circle((0, 0), 2.0, color='black', alpha=0.3)
        ax_traj.add_artist(circle)
        
        ax_traj.set_title(f"Trajectory: {case['name']}")
        ax_traj.axis('equal')
        ax_traj.legend(fontsize=8)
        
        ax_cons.set_title(f"Hamiltonian Violation (|H+1|): {case['name']}")
        ax_cons.set_xlabel("Lambda")
        ax_cons.grid(True, alpha=0.3)
        ax_cons.legend(fontsize=8)
        
        comparison_table.append(case_stats)

    plt.tight_layout()
    plt.savefig("plots/comprehensive_extreme_eval.png", dpi=200)
    print("Comprehensive plot saved to plots/comprehensive_extreme_eval.png")
    
    # Print Table
    print("\nSUMMARY TABLE (Mean Absolute Error / Mean H-Violation)")
    header = f"{'Case':<20} | {'SOTA Dev':<10} | {'Data Dev':<10} | {'SOTA H-Viol':<12}"
    print(header)
    print("-" * len(header))
    for row in comparison_table:
        print(f"{row['Case']:<20} | {row['Stage 4 (SOTA) Dev']:<10.4f} | {row['Data-Only Dev']:<10.4f} | {row['Stage 4 (SOTA) H-Viol']:<12.2e}")

if __name__ == "__main__":
    comprehensive_eval()
