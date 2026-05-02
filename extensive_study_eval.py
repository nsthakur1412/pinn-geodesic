import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
import csv
import time
from scientific_framework import ScientificMLP
from rk45_solver import get_initial_state, solve_geodesic
from physics import compute_conserved_quantities

def get_pinn_velocities(model, inputs_t, lam_scale):
    """Utility to get velocities via autograd."""
    lam_t = inputs_t[:, 0:1].clone().detach().requires_grad_(True)
    other_t = inputs_t[:, 1:4].clone().detach()
    model_inp = torch.cat([lam_t, other_t], dim=1)
    
    _, out_phys = model(model_inp)
    t_p, r_p, phi_p = out_phys[:, 0:1], out_phys[:, 1:2], out_phys[:, 2:3]
    
    ones = torch.ones_like(lam_t)
    ut = torch.autograd.grad(t_p, lam_t, ones, retain_graph=True)[0] / lam_scale
    ur_v = torch.autograd.grad(r_p, lam_t, ones, retain_graph=True)[0] / lam_scale
    uphi = torch.autograd.grad(phi_p, lam_t, ones, retain_graph=False)[0] / lam_scale
    
    pos = out_phys.detach().cpu().numpy()
    vel = torch.cat([ut, ur_v, uphi], dim=1).detach().cpu().numpy()
    return pos, vel

def run_extreme_tests():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Executing 128 vs 256 Extreme Comparison on {device}...")

    # 1. Setup
    with open("data/unified_scalers.pkl", "rb") as f:
        scalers = pickle.load(f)

    models_info = {
        "PINN-128": {"path": "results/model_pinn_128_baseline.pt", "layers": 5, "width": 128, "res": False, "color": "orange"},
        "PINN-256-Ref": {"path": "results/model_pinn_256_res.pt", "layers": 6, "width": 256, "res": True, "color": "green"},
        "PINN-256-Stiff": {"path": "results/model_pinn_256_stiff.pt", "layers": 6, "width": 256, "res": True, "color": "red"}
    }
    
    models = {}
    for name, info in models_info.items():
        if os.path.exists(info['path']):
            m = ScientificMLP(hidden_layers=info['layers'], neurons_per_layer=info['width'], use_residual=info['res']).to(device)
            m.load_state_dict(torch.load(info['path'], map_location=device, weights_only=True))
            m.eval()
            models[name] = (m, info['color'])
            print(f"Loaded {name}")
    
    # 2. Extreme Scenarios
    EXTREME_CASES = [
        {"name": "Horizon Grazer", "r0": 3.5,  "ur0": -0.05, "L": 3.8,   "lam": 400},
        {"name": "Speed Demon",    "r0": 20.0, "ur0": -0.8,  "L": 4.5,   "lam": 200},
        {"name": "Far Voyager",    "r0": 60.0, "ur0": -0.01, "L": 10.0,  "lam": 1000},
        {"name": "Chaos Orbit",    "r0": 7.0,  "ur0": 0.05,  "L": 3.05,  "lam": 1500},
    ]

    from scientific_framework import compute_physics_metrics
    
    report = []
    plt.figure(figsize=(20, 15))

    for idx, case in enumerate(EXTREME_CASES):
        print(f"Testing {case['name']}...")
        r0, ur0, L = case['r0'], case['ur0'], case['L']
        
        # Ground Truth
        uphi0 = L / (r0**2)
        state = get_initial_state(r0=r0, ur0=ur0, uphi0=uphi0)
        sol = solve_geodesic(state, [0, case['lam']], num_points=1000)
        gt_x = sol.y[1] * np.cos(sol.y[2])
        gt_y = sol.y[1] * np.sin(sol.y[2])

        plt.subplot(2, 2, idx + 1)
        plt.plot(gt_x, gt_y, 'k-', lw=2, label='RK45 (Truth)', alpha=0.6)
        
        case_results = {"case": case['name']}
        
        for name, (m, color) in models.items():
            lam_eval = sol.t
            inp = np.zeros((len(lam_eval), 4), dtype=np.float32)
            inp[:, 0] = lam_eval / scalers['lam_scale']
            inp[:, 1] = (r0 - scalers['r0_mean']) / scalers['r0_std']
            inp[:, 2] = (ur0 - scalers['ur0_mean']) / scalers['ur0_std']
            inp[:, 3] = (L - scalers['L_mean']) / scalers['L_std']
            inp_t = torch.tensor(inp).to(device)
            
            # 1. Trajectory
            with torch.no_grad():
                _, out_phys = m(inp_t)
            pos_p = out_phys.cpu().numpy()
            px = pos_p[:, 1] * np.cos(pos_p[:, 2])
            py = pos_p[:, 1] * np.sin(pos_p[:, 2])
            plt.plot(px, py, color=color, ls='--', label=name, alpha=0.8)
            
            # 2. Physics Loss & Conserved Quantities
            # We use autograd here, so we need a separate call
            p_loss, E, L_p, H, _, _, _ = compute_physics_metrics(m, inp_t)
            
            E_p = E.detach().cpu().numpy()
            L_val = L_p.detach().cpu().numpy()
            H_p = H.detach().cpu().numpy()
            
            e_drift = np.mean(np.abs(E_p - E_p[0]) / (np.abs(E_p[0]) + 1e-8))
            l_drift = np.mean(np.abs(L_val - L_val[0]) / (np.abs(L_val[0]) + 1e-8))
            h_viol = np.mean(np.abs(H_p + 1.0))
            
            case_results[name] = {
                "phys_loss": p_loss.item(),
                "e_drift": e_drift,
                "l_drift": l_drift,
                "h_viol": h_viol,
                "max_dev": np.max(np.sqrt((px - gt_x)**2 + (py - gt_y)**2))
            }

        # Horizon
        circle = plt.Circle((0, 0), 2.0, color='black', alpha=0.4)
        plt.gca().add_artist(circle)
        plt.title(f"Scenario: {case['name']}")
        plt.legend(fontsize=9)
        plt.axis('equal')
        plt.grid(True, alpha=0.1)
        
        report.append(case_results)

    plt.tight_layout()
    plt.savefig("plots/extreme_128_vs_256.png", dpi=200)
    
    # 3. Build Detailed Markdown Report
    with open("results/extreme_evolution_report.md", "w") as f:
        f.write("# Extreme Evolution Comparison: 128 vs 256-Res\n\n")
        f.write("Quantifying the impact of network expansion and residual connections on physical consistency in edge cases.\n\n")
        
        metrics = [
            ("Max Traj Dev (M)", "max_dev"),
            ("Physics Res Loss", "phys_loss"),
            ("Relative Energy Drift", "e_drift"),
            ("Hamiltonian Violation", "h_viol")
        ]
        
        for r in report:
            f.write(f"### Scenario: {r['case']}\n")
            headers = ["Metric", "PINN-128", "PINN-256-Ref", "PINN-256-Stiff"]
            f.write("| " + " | ".join(headers) + " |\n")
            f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")

            for m_label, m_key in metrics:
                v128 = r['PINN-128'][m_key]
                vRef = r['PINN-256-Ref'][m_key]
                vStiff = r['PINN-256-Stiff'][m_key]
                f.write(f"| {m_label} | {v128:.2e} | {vRef:.2e} | {vStiff:.2e} |\n")
            f.write("\n")

    print("Generated results/extreme_evolution_report.md and plots/extreme_128_vs_256.png")

    print("Generated results/extensive_report.md")

if __name__ == "__main__":
    run_extreme_tests()
