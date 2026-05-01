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
# GENERALIZATION TEST MATRIX
# ============================================================
TEST_CASES = [
    {"name": "Near ISCO Bound",    "r0": 6.5,  "ur0": 0.0,   "L": 3.46},
    {"name": "Wide Bound Orbit",   "r0": 25.0, "ur0": 0.0,   "L": 5.5},
    {"name": "Unseen Bound",       "r0": 8.5,  "ur0": 0.0,   "L": 3.4},
    {"name": "Critical Scattering","r0": 20.0, "ur0": -0.15, "L": 4.0},
    {"name": "Unseen Escape",      "r0": 18.0, "ur0": -0.12, "L": 6.0},
    {"name": "High-Energy Plunge", "r0": 10.0, "ur0": -0.2,  "L": 2.5},
]


def load_model():
    pinn = UnifiedPINN(hidden_layers=5, neurons_per_layer=128).to(device)
    paths = ["checkpoints/best_model.pt", "results/unified_pinn.pt"]
    
    loaded = False
    for path in paths:
        if os.path.exists(path):
            try:
                # Use weights_only=True for safety
                state_dict = torch.load(path, map_location=device, weights_only=True)
                pinn.load_state_dict(state_dict)
                print(f"Successfully loaded model from {path}")
                loaded = True
                break
            except Exception as e:
                print(f"Warning: Could not load from {path} ({e})")
                continue
    
    if not loaded:
        print("Error: No trained model found in checkpoints/ or results/!")
        return None
        
    pinn.eval()
    return pinn


def predict_trajectory(pinn, r0, ur0, L, lam_array):
    """Run PINN inference for a given IC and lambda array."""
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
    
    return out_phys  # [t, r, phi]


def get_pinn_velocities(pinn, r0, ur0, L, lam_array):
    """Get PINN velocities via autograd for conservation analysis."""
    scalers = pinn.scalers
    N = len(lam_array)
    
    lam_scaled = lam_array / scalers['lam_scale']
    r0_norm = (r0 - scalers['r0_mean']) / scalers['r0_std']
    ur0_norm = (ur0 - scalers['ur0_mean']) / scalers['ur0_std']
    L_norm = (L - scalers['L_mean']) / scalers['L_std']
    
    inp = np.zeros((N, 4), dtype=np.float32)
    inp[:, 0] = lam_scaled
    inp[:, 1] = r0_norm
    inp[:, 2] = ur0_norm
    inp[:, 3] = L_norm
    
    lam_t = torch.tensor(inp[:, 0:1], dtype=torch.float32, device=device, requires_grad=True)
    other_t = torch.tensor(inp[:, 1:4], dtype=torch.float32, device=device)
    model_inp = torch.cat([lam_t, other_t], dim=1)
    
    _, out_phys = pinn(model_inp)
    t_p, r_p, phi_p = out_phys[:, 0:1], out_phys[:, 1:2], out_phys[:, 2:3]
    
    ones = torch.ones_like(lam_t)
    ls = scalers['lam_scale']
    ut = torch.autograd.grad(t_p, lam_t, ones, retain_graph=True)[0] / ls
    ur_v = torch.autograd.grad(r_p, lam_t, ones, retain_graph=True)[0] / ls
    uphi = torch.autograd.grad(phi_p, lam_t, ones, retain_graph=False)[0] / ls
    
    positions = out_phys.detach().cpu().numpy()
    velocities = torch.cat([ut, ur_v, uphi], dim=1).detach().cpu().numpy()
    return positions, velocities


# ============================================================
# 1. TRAJECTORY ACCURACY
# ============================================================
def eval_trajectory_accuracy(pinn):
    """Compare PINN vs RK45 trajectories with quantitative metrics."""
    print("\n" + "="*60)
    print("1. TRAJECTORY ACCURACY")
    print("="*60)
    
    n = len(TEST_CASES)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 6*rows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    results = []
    
    for i, case in enumerate(TEST_CASES):
        r0, ur0, L = case['r0'], case['ur0'], case['L']
        
        # RK45 ground truth
        uphi0 = L / (r0**2)
        state = get_initial_state(r0=r0, ur0=ur0, uphi0=uphi0)
        sol = solve_geodesic(state, [0, 300], num_points=500)
        
        lam_true = sol.t
        r_true, phi_true = sol.y[1], sol.y[2]
        x_true = r_true * np.cos(phi_true)
        y_true = r_true * np.sin(phi_true)
        
        # PINN prediction
        out = predict_trajectory(pinn, r0, ur0, L, lam_true)
        r_pred, phi_pred = out[:, 1], out[:, 2]
        x_pred = r_pred * np.cos(phi_pred)
        y_pred = r_pred * np.sin(phi_pred)
        
        # Metrics
        mae_r = np.mean(np.abs(r_true - r_pred))
        mae_phi = np.mean(np.abs(phi_true - phi_pred))
        max_dev = np.max(np.sqrt((x_true - x_pred)**2 + (y_true - y_pred)**2))
        rel_rmse = np.sqrt(np.mean((r_true - r_pred)**2) / (np.mean(r_true**2) + 1e-8))
        
        results.append({
            'name': case['name'], 'mae_r': mae_r, 'mae_phi': mae_phi,
            'max_dev': max_dev, 'rel_rmse': rel_rmse
        })
        
        verdict = "EXCELLENT" if max_dev < 3 else ("OK" if max_dev < 10 else "FAIL")
        print(f"  [{i+1}/{n}] {case['name']:25s} | MAE(r)={mae_r:.3f}M | MAE(phi)={mae_phi:.3f}rad | MaxDev={max_dev:.2f}M | RelRMSE={rel_rmse:.3%} | {verdict}")
        
        # Plot
        ax = axes[i]
        ax.plot(x_true, y_true, 'k-', linewidth=2, label='RK45')
        ax.plot(x_pred, y_pred, 'r--', linewidth=1.5, label='PINN')
        circle = plt.Circle((0, 0), 2.0, color='black', fill=True, alpha=0.3)
        ax.add_patch(circle)
        isco = plt.Circle((0, 0), 6.0, color='blue', fill=False, linestyle=':', alpha=0.4)
        ax.add_patch(isco)
        ax.set_aspect('equal')
        ax.set_title(f"{case['name']}\nMaxDev={max_dev:.2f}M ({verdict})", fontsize=10)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8)
    
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('plots/eval_trajectories.png', dpi=200)
    plt.close()
    print("  Saved: plots/eval_trajectories.png")
    return results


# ============================================================
# 2. CONSERVATION VIOLATION
# ============================================================
def eval_conservation(pinn):
    """Track E, L, and Hamiltonian constraint violation over proper time."""
    print("\n" + "="*60)
    print("2. CONSERVATION VIOLATION")
    print("="*60)
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    # Use 3 representative cases
    cases = [TEST_CASES[0], TEST_CASES[2], TEST_CASES[5]]
    
    for col, case in enumerate(cases[:2]):
        r0, ur0, L = case['r0'], case['ur0'], case['L']
        uphi0 = L / (r0**2)
        state = get_initial_state(r0=r0, ur0=ur0, uphi0=uphi0)
        sol = solve_geodesic(state, [0, 300], num_points=500)
        lam = sol.t
        
        # RK45 conserved quantities
        E_rk, L_rk, H_rk = compute_conserved_quantities(sol.y.T)
        
        # PINN conserved quantities
        pos, vel = get_pinn_velocities(pinn, r0, ur0, L, lam)
        full_state = np.column_stack([pos, vel])  # [t, r, phi, ut, ur, uphi]
        E_pinn, L_pinn, H_pinn = compute_conserved_quantities(full_state)
        
        E0, L0 = E_rk[0], L_rk[0]
        
        dE_rk = np.abs(E_rk - E0) / (np.abs(E0) + 1e-8)
        dE_pinn = np.abs(E_pinn - E0) / (np.abs(E0) + 1e-8)
        dL_rk = np.abs(L_rk - L0) / (np.abs(L0) + 1e-8)
        dL_pinn = np.abs(L_pinn - L0) / (np.abs(L0) + 1e-8)
        
        axes[0, col].semilogy(lam, dE_rk + 1e-16, 'k-', label='RK45')
        axes[0, col].semilogy(lam, dE_pinn + 1e-16, 'r--', label='PINN')
        axes[0, col].set_ylabel('|ΔE/E₀|')
        axes[0, col].set_title(f"{case['name']} - Energy")
        axes[0, col].legend(fontsize=8)
        axes[0, col].grid(True, alpha=0.3)
        
        axes[1, col].semilogy(lam, dL_rk + 1e-16, 'k-', label='RK45')
        axes[1, col].semilogy(lam, dL_pinn + 1e-16, 'r--', label='PINN')
        axes[1, col].set_ylabel('|ΔL/L₀|')
        axes[1, col].set_title(f"{case['name']} - Ang. Momentum")
        axes[1, col].legend(fontsize=8)
        axes[1, col].grid(True, alpha=0.3)
        
        axes[2, col].semilogy(lam, np.abs(H_rk + 1) + 1e-16, 'k-', label='RK45')
        axes[2, col].semilogy(lam, np.abs(H_pinn + 1) + 1e-16, 'r--', label='PINN')
        axes[2, col].set_ylabel('|H + 1|')
        axes[2, col].set_xlabel('λ (affine parameter)')
        axes[2, col].set_title(f"{case['name']} - Hamiltonian")
        axes[2, col].legend(fontsize=8)
        axes[2, col].grid(True, alpha=0.3)
        
        print(f"  {case['name']:25s} | max dE/E0(PINN)={np.max(dE_pinn):.2e} | max dL/L0(PINN)={np.max(dL_pinn):.2e} | max |H+1|(PINN)={np.max(np.abs(H_pinn+1)):.2e}")
    
    plt.tight_layout()
    plt.savefig('plots/eval_conservation.png', dpi=200)
    plt.close()
    print("  Saved: plots/eval_conservation.png")


# ============================================================
# 3. EFFECTIVE POTENTIAL & PHASE SPACE
# ============================================================
def eval_phase_space(pinn):
    """Phase space portraits (r vs dr/dλ) and effective potential overlay."""
    print("\n" + "="*60)
    print("3. PHASE SPACE & EFFECTIVE POTENTIAL")
    print("="*60)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    cases = [TEST_CASES[0], TEST_CASES[2], TEST_CASES[5]]
    
    for col, case in enumerate(cases):
        r0, ur0, L = case['r0'], case['ur0'], case['L']
        uphi0 = L / (r0**2)
        state = get_initial_state(r0=r0, ur0=ur0, uphi0=uphi0)
        sol = solve_geodesic(state, [0, 300], num_points=500)
        lam = sol.t
        
        # RK45
        r_rk = sol.y[1]
        ur_rk = sol.y[4]
        
        # PINN
        pos, vel = get_pinn_velocities(pinn, r0, ur0, L, lam)
        r_pinn = pos[:, 1]
        ur_pinn = vel[:, 1]
        
        # Phase space (r vs dr/dλ)
        axes[0, col].plot(r_rk, ur_rk, 'k-', linewidth=2, label='RK45')
        axes[0, col].plot(r_pinn, ur_pinn, 'r--', linewidth=1.5, label='PINN')
        axes[0, col].set_xlabel('r / M')
        axes[0, col].set_ylabel('dr/dλ')
        axes[0, col].set_title(f"{case['name']} - Phase Space")
        axes[0, col].legend(fontsize=8)
        axes[0, col].grid(True, alpha=0.3)
        
        # Effective potential
        r_range = np.linspace(2.5, max(35, np.max(r_rk) * 1.2), 500)
        f_r = 1.0 - 2.0 / r_range
        V_eff = -1.0 + f_r * (1.0 + L**2 / r_range**2)
        
        # Effective energy from ICs
        E_eff = ur_rk[0]**2 + f_r[0] * (1.0 + L**2 / r_rk[0]**2) - 1.0
        
        axes[1, col].plot(r_range, V_eff, 'b-', linewidth=2, label='V_eff(r)')
        axes[1, col].axhline(y=E_eff, color='green', linestyle='--', alpha=0.7, label=f'E_eff={E_eff:.3f}')
        axes[1, col].plot(r_rk, ur_rk**2 + V_eff[np.searchsorted(r_range, r_rk).clip(0, len(r_range)-1)],
                         'k.', markersize=1, alpha=0.3, label='RK45 trajectory')
        axes[1, col].set_xlabel('r / M')
        axes[1, col].set_ylabel('V_eff')
        axes[1, col].set_title(f"Effective Potential (L={L})")
        axes[1, col].set_ylim(E_eff - 0.05, max(V_eff) * 1.1 if max(V_eff) > E_eff else E_eff + 0.05)
        axes[1, col].legend(fontsize=7)
        axes[1, col].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/eval_phase_space.png', dpi=200)
    plt.close()
    print("  Saved: plots/eval_phase_space.png")


# ============================================================
# 4. LONG-TERM STABILITY
# ============================================================
def eval_long_term(pinn):
    """Measure error growth rate over extended proper time."""
    print("\n" + "="*60)
    print("4. LONG-TERM STABILITY")
    print("="*60)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Use a bound orbit — most sensitive to long-term drift
    case = TEST_CASES[2]  # Unseen Bound
    r0, ur0, L = case['r0'], case['ur0'], case['L']
    uphi0 = L / (r0**2)
    state = get_initial_state(r0=r0, ur0=ur0, uphi0=uphi0)
    sol = solve_geodesic(state, [0, 300], num_points=1000)
    lam = sol.t
    
    r_true = sol.y[1]
    phi_true = sol.y[2]
    x_true = r_true * np.cos(phi_true)
    y_true = r_true * np.sin(phi_true)
    
    out = predict_trajectory(pinn, r0, ur0, L, lam)
    r_pred, phi_pred = out[:, 1], out[:, 2]
    x_pred = r_pred * np.cos(phi_pred)
    y_pred = r_pred * np.sin(phi_pred)
    
    # Euclidean error over time
    err = np.sqrt((x_true - x_pred)**2 + (y_true - y_pred)**2)
    
    axes[0].plot(lam, err, 'r-', linewidth=1.5)
    axes[0].set_xlabel('λ (proper time)')
    axes[0].set_ylabel('Euclidean Error (M)')
    axes[0].set_title(f"Error Growth: {case['name']}")
    axes[0].grid(True, alpha=0.3)
    
    # Error growth rate (log scale)
    axes[1].semilogy(lam, err + 1e-10, 'r-', linewidth=1.5)
    axes[1].set_xlabel('λ (proper time)')
    axes[1].set_ylabel('Log Error (M)')
    axes[1].set_title('Log-scale Error Growth')
    axes[1].grid(True, alpha=0.3)
    
    # Fit exponential growth rate if possible
    mask = err > 1e-6
    if np.sum(mask) > 10:
        log_err = np.log(err[mask] + 1e-10)
        lam_mask = lam[mask]
        coeffs = np.polyfit(lam_mask, log_err, 1)
        gamma = coeffs[0]
        axes[1].plot(lam_mask, np.exp(np.polyval(coeffs, lam_mask)), 'k--', 
                    label=f'gamma = {gamma:.4f}')
        axes[1].legend()
        stability = "STABLE" if gamma < 0.01 else ("MARGINAL" if gamma < 0.05 else "UNSTABLE")
        print(f"  Lyapunov-like exponent gamma = {gamma:.4f} ({stability})")
    
    plt.tight_layout()
    plt.savefig('plots/eval_long_term.png', dpi=200)
    plt.close()
    print("  Saved: plots/eval_long_term.png")


# ============================================================
# 5. COMPUTATIONAL EFFICIENCY
# ============================================================
def eval_efficiency(pinn):
    """Benchmark PINN inference vs RK45 solve time."""
    print("\n" + "="*60)
    print("5. COMPUTATIONAL EFFICIENCY")
    print("="*60)
    
    case = TEST_CASES[0]
    r0, ur0, L = case['r0'], case['ur0'], case['L']
    uphi0 = L / (r0**2)
    state = get_initial_state(r0=r0, ur0=ur0, uphi0=uphi0)
    
    lam_array = np.linspace(0, 300, 1000).astype(np.float32)
    
    # Warmup
    _ = predict_trajectory(pinn, r0, ur0, L, lam_array)
    _ = solve_geodesic(state, [0, 300], num_points=1000)
    
    # RK45 timing
    t0 = time.time()
    for _ in range(50):
        _ = solve_geodesic(state, [0, 300], num_points=1000)
    rk45_time = (time.time() - t0) / 50
    
    # PINN timing (single trajectory)
    t0 = time.time()
    for _ in range(50):
        _ = predict_trajectory(pinn, r0, ur0, L, lam_array)
    pinn_time = (time.time() - t0) / 50
    
    # PINN batch timing (100 trajectories at once)
    scalers = pinn.scalers
    batch_inputs = np.zeros((100 * 1000, 4), dtype=np.float32)
    for j in range(100):
        s = j * 1000
        batch_inputs[s:s+1000, 0] = lam_array / scalers['lam_scale']
        batch_inputs[s:s+1000, 1] = (r0 - scalers['r0_mean']) / scalers['r0_std']
        batch_inputs[s:s+1000, 2] = (ur0 - scalers['ur0_mean']) / scalers['ur0_std']
        batch_inputs[s:s+1000, 3] = (L - scalers['L_mean']) / scalers['L_std']
    
    batch_t = torch.tensor(batch_inputs).to(device)
    # Warmup
    with torch.no_grad():
        _ = pinn(batch_t)
    
    t0 = time.time()
    for _ in range(10):
        with torch.no_grad():
            _ = pinn(batch_t)
    batch_time = (time.time() - t0) / 10 / 100  # per trajectory
    
    print(f"  RK45 (single):     {rk45_time*1000:.2f} ms/trajectory")
    print(f"  PINN (single):     {pinn_time*1000:.2f} ms/trajectory")
    print(f"  PINN (batch 100):  {batch_time*1000:.2f} ms/trajectory")
    print(f"  Speedup (single):  {rk45_time/pinn_time:.1f}x")
    print(f"  Speedup (batch):   {rk45_time/batch_time:.1f}x")


# ============================================================
# MAIN
# ============================================================
def evaluate_unified_model():
    pinn = load_model()
    if pinn is None:
        return
    
    print(f"Unified PINN loaded on {device}")
    print(f"Testing on {len(TEST_CASES)} unseen initial conditions\n")
    
    eval_trajectory_accuracy(pinn)
    eval_conservation(pinn)
    eval_phase_space(pinn)
    eval_long_term(pinn)
    eval_efficiency(pinn)
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE — All plots saved to plots/")
    print("="*60)


if __name__ == "__main__":
    evaluate_unified_model()
