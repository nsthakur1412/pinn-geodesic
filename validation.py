import os
import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
from pinn_model import GeodesicPINN, compute_physics_loss
from experiments import prepare_training_data, train_pinn

plt.rcParams.update({'font.size': 12, 'figure.dpi': 150})
RK45_COLOR = '#2c7bb6'
PINN_COLOR = '#d7191c'

def run_extrapolation_test(sol, force_retrain=False):
    alphas = [0.0, 0.5, 0.9]
    lam_coll, lam_data, target_data, _ = prepare_training_data(sol, sample_ratio=0.1, lam_max=250.0)
    
    # RK45 Full data for evaluation
    lam_full = torch.linspace(sol.t[0], sol.t[-1], 1000).view(-1, 1)
    rk45_full = np.zeros((1000, 3))
    for i in range(3):
        rk45_full[:, i] = np.interp(lam_full.numpy().flatten(), sol.t, sol.y[i])
    rk45_full_tensor = torch.tensor(rk45_full, dtype=torch.float32)
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle("Extrapolation beyond training window — r(λ) trajectory")
    
    for idx, alpha in enumerate(alphas):
        print(f"Training extrapolation model alpha={alpha}...")
        torch.manual_seed(42)
        pinn, _, _ = train_pinn(lam_coll, lam_data, target_data, alpha=alpha, epochs=3000)
        pinn.eval()
        
        with torch.no_grad():
            pred_full = pinn(lam_full.to(pinn.network[0].weight.device)).cpu()
            
        in_domain = lam_full.flatten() <= 250.0
        out_domain = lam_full.flatten() > 250.0
        
        in_mae = torch.mean(torch.abs(pred_full[in_domain, 1] - rk45_full_tensor[in_domain, 1])).item()
        out_mae = torch.mean(torch.abs(pred_full[out_domain, 1] - rk45_full_tensor[out_domain, 1])).item()
        degradation = out_mae / (in_mae + 1e-12)
        
        print(f"alpha={alpha}  in-domain MAE={in_mae:.4e}  out-domain MAE={out_mae:.4e}  degradation ratio={degradation:.2f}x")
        
        ax = axes[idx]
        ax.plot(lam_full.numpy(), rk45_full[:, 1], color=RK45_COLOR, label="RK45 Ground Truth", linewidth=2)
        ax.plot(lam_full.numpy(), pred_full[:, 1].numpy(), color=PINN_COLOR, label="PINN Prediction", linestyle='--')
        
        ax.axvspan(0, 250, color='gray', alpha=0.08, lw=0)
        ax.axvline(250, color='black', linestyle='--', label='training boundary')
        
        ax.text(0.98, 0.85, f"In-domain MAE: {in_mae:.2e}\nOut-domain MAE: {out_mae:.2e}", 
                transform=ax.transAxes, ha='right', va='top', bbox=dict(facecolor='white', alpha=0.8))
        
        titles = {0.0: "α=0.0 (data only)", 0.5: "α=0.5 (balanced)", 0.9: "α=0.9 (physics-heavy)"}
        ax.set_title(titles[alpha])
        ax.set_ylabel("r(λ)")
        if idx == 0:
            ax.legend(loc='upper left')
            
    axes[-1].set_xlabel("Affine parameter λ")
    plt.tight_layout()
    plt.savefig("plots/validation1_extrapolation.png")
    plt.close()

def run_sparsity_test(sol, force_retrain=False):
    filepath = "results/sparsity_sweep.pkl"
    if os.path.exists(filepath) and not force_retrain:
        with open(filepath, "rb") as f:
            results = pickle.load(f)
    else:
        sample_ratios = [0.01, 0.02, 0.05, 0.10, 0.20, 0.40]
        alphas = [0.0, 0.5, 0.9]
        seeds = [0, 1, 2]
        
        results = []
        for sr in sample_ratios:
            for a in alphas:
                for s in seeds:
                    print(f"Sparsity sweep: SR={sr}, Alpha={a}, Seed={s}")
                    torch.manual_seed(s)
                    lam_coll, lam_data, target_data, _ = prepare_training_data(sol, sample_ratio=sr)
                    
                    pinn, _, _ = train_pinn(lam_coll, lam_data, target_data, alpha=a, epochs=3000)
                    pinn.eval()
                    
                    lam_coll_eval = lam_coll.clone().detach().requires_grad_(True)
                    pred = pinn(lam_coll_eval)
                    
                    # Compute MAE vs RK45 (interpolated)
                    rk45_coll = np.zeros((len(lam_coll), 3))
                    for i in range(3):
                        rk45_coll[:, i] = np.interp(lam_coll.cpu().detach().numpy().flatten(), sol.t, sol.y[i])
                    rk45_coll_tensor = torch.tensor(rk45_coll, dtype=torch.float32, device=pred.device)
                    
                    mae = torch.mean(torch.abs(pred - rk45_coll_tensor)).item()
                    
                    ones = torch.ones_like(lam_coll_eval)
                    dt_dlam = torch.autograd.grad(pred[:, 0:1], lam_coll_eval, grad_outputs=ones, retain_graph=True, create_graph=False)[0]
                    r_pred = pred[:, 1:2]
                    E_pred = (1.0 - 2.0 / r_pred) * dt_dlam
                    energy_drift = (torch.std(E_pred) / torch.mean(E_pred).abs()).item()
                    
                    results.append({'sample_ratio': sr, 'alpha': a, 'seed': s, 'mae': mae, 'energy_drift': energy_drift})
                    
                    # Incremental save
                    with open(filepath, "wb") as f:
                        pickle.dump(results, f)
            
    # Print table
    print(f"{'sample_ratio':<15} | {'alpha=0.0 MAE':<15} | {'alpha=0.5 MAE':<15} | {'alpha=0.9 MAE':<15}")
    unique_sr = sorted(list(set(r['sample_ratio'] for r in results)))
    for sr in unique_sr:
        maes = {0.0: [], 0.5: [], 0.9: []}
        for r in results:
            if r['sample_ratio'] == sr:
                maes[r['alpha']].append(r['mae'])
        print(f"{sr:<15} | {np.mean(maes[0.0]):<15.4e} | {np.mean(maes[0.5]):<15.4e} | {np.mean(maes[0.9]):<15.4e}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    colors = {0.0: '#e66101', 0.5: '#5e3c99', 0.9: '#1a9641'}
    labels = {0.0: "α=0.0 (no physics)", 0.5: "α=0.5", 0.9: "α=0.9 (physics-heavy)"}
    
    for a in [0.0, 0.5, 0.9]:
        mean_maes, std_maes, mean_ed, std_ed = [], [], [], []
        for sr in unique_sr:
            sr_maes = [r['mae'] for r in results if r['alpha'] == a and r['sample_ratio'] == sr]
            sr_eds = [r['energy_drift'] for r in results if r['alpha'] == a and r['sample_ratio'] == sr]
            mean_maes.append(np.mean(sr_maes))
            std_maes.append(np.std(sr_maes))
            mean_ed.append(np.mean(sr_eds))
            std_ed.append(np.std(sr_eds))
            
        mean_maes = np.array(mean_maes)
        std_maes = np.array(std_maes)
        mean_ed = np.array(mean_ed)
        std_ed = np.array(std_ed)
        
        ax1.plot(unique_sr, mean_maes, marker='o', color=colors[a], label=labels[a])
        ax1.fill_between(unique_sr, mean_maes - std_maes, mean_maes + std_maes, color=colors[a], alpha=0.2)
        
        ax2.plot(unique_sr, mean_ed, marker='o', color=colors[a], label=labels[a])
        ax2.fill_between(unique_sr, mean_ed - std_ed, mean_ed + std_ed, color=colors[a], alpha=0.2)

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xticks(unique_sr)
    ax1.set_xticklabels([f"{int(sr*100)}%" for sr in unique_sr])
    ax1.set_xlabel("Training data fraction")
    ax1.set_ylabel("Trajectory MAE (log scale)")
    ax1.set_title("Accuracy vs supervision density")
    ax1.legend()
    ax1.grid(True, which="both", ls="--")
    
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xticks(unique_sr)
    ax2.set_xticklabels([f"{int(sr*100)}%" for sr in unique_sr])
    ax2.axhline(0.01, color='black', linestyle='--', label='1% drift threshold')
    ax2.set_xlabel("Training data fraction")
    ax2.set_ylabel("Relative energy drift |σ(E)/μ(E)|")
    ax2.set_title("Physical conservation vs supervision density")
    ax2.grid(True, which="both", ls="--")
    
    plt.tight_layout()
    plt.savefig("plots/validation2_sparsity.png")
    plt.close()

def run_noise_test(sol, force_retrain=False):
    filepath = "results/noise_sweep.pkl"
    if os.path.exists(filepath) and not force_retrain:
        with open(filepath, "rb") as f:
            results = pickle.load(f)
    else:
        noise_levels = [0.0, 0.01, 0.02, 0.05, 0.10]
        alphas = [0.0, 0.5, 0.9]
        seeds = [0, 1, 2]
        
        results = []
        for nl in noise_levels:
            for a in alphas:
                for s in seeds:
                    print(f"Noise sweep: NL={nl}, Alpha={a}, Seed={s}")
                    torch.manual_seed(s)
                    lam_coll, lam_data, target_data, _ = prepare_training_data(sol, sample_ratio=0.1)
                    
                    coord_std = target_data.std(dim=0, keepdim=True)
                    noise = torch.randn_like(target_data) * nl * coord_std
                    noisy_target = target_data + noise
                    
                    pinn, _, _ = train_pinn(lam_coll, lam_data, noisy_target, alpha=a, epochs=3000)
                    pinn.eval()
                    
                    lam_coll_eval = lam_coll.clone().detach().requires_grad_(True)
                    pred = pinn(lam_coll_eval)
                    
                    rk45_coll = np.zeros((len(lam_coll), 3))
                    for i in range(3):
                        rk45_coll[:, i] = np.interp(lam_coll.cpu().detach().numpy().flatten(), sol.t, sol.y[i])
                    rk45_coll_tensor = torch.tensor(rk45_coll, dtype=torch.float32, device=pred.device)
                    
                    mae_clean = torch.mean(torch.abs(pred - rk45_coll_tensor)).item()
                    
                    ones = torch.ones_like(lam_coll_eval)
                    dt_dlam = torch.autograd.grad(pred[:, 0:1], lam_coll_eval, grad_outputs=ones, retain_graph=True, create_graph=False)[0]
                    r_pred = pred[:, 1:2]
                    E_pred = (1.0 - 2.0 / r_pred) * dt_dlam
                    energy_drift = (torch.std(E_pred) / torch.mean(E_pred).abs()).item()
                    
                    results.append({'noise_level': nl, 'alpha': a, 'seed': s, 'mae_clean': mae_clean, 'energy_drift': energy_drift})
                    
                    # Incremental save
                    with open(filepath, "wb") as f:
                        pickle.dump(results, f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    colors = {0.0: '#e66101', 0.5: '#5e3c99', 0.9: '#1a9641'}
    labels = {0.0: "α=0.0 (no physics)", 0.5: "α=0.5", 0.9: "α=0.9 (physics-heavy)"}
    unique_nl = sorted(list(set(r['noise_level'] for r in results)))
    
    for a in [0.0, 0.5, 0.9]:
        mean_maes, std_maes, mean_ed, std_ed = [], [], [], []
        for nl in unique_nl:
            nl_maes = [r['mae_clean'] for r in results if r['alpha'] == a and r['noise_level'] == nl]
            nl_eds = [r['energy_drift'] for r in results if r['alpha'] == a and r['noise_level'] == nl]
            mean_maes.append(np.mean(nl_maes))
            std_maes.append(np.std(nl_maes))
            mean_ed.append(np.mean(nl_eds))
            std_ed.append(np.std(nl_eds))
            
        mean_maes = np.array(mean_maes)
        std_maes = np.array(std_maes)
        mean_ed = np.array(mean_ed)
        std_ed = np.array(std_ed)
        
        # Use simple 0, 1, 2, 3, 4 index for x to space points evenly
        x_pos = np.arange(len(unique_nl))
        ax1.plot(x_pos, mean_maes, marker='o', color=colors[a], label=labels[a])
        ax1.fill_between(x_pos, mean_maes - std_maes, mean_maes + std_maes, color=colors[a], alpha=0.2)
        
        ax2.plot(x_pos, mean_ed, marker='o', color=colors[a], label=labels[a])
        ax2.fill_between(x_pos, mean_ed - std_ed, mean_ed + std_ed, color=colors[a], alpha=0.2)

    x_labels = [f"{int(nl*100)}%" for nl in unique_nl]
    ax1.set_xticks(np.arange(len(unique_nl)))
    ax1.set_xticklabels(x_labels)
    ax1.set_yscale('log')
    ax1.set_xlabel("Noise level (% of coordinate std)")
    ax1.set_ylabel("MAE vs clean ground truth")
    ax1.set_title("Trajectory accuracy under noisy supervision")
    ax1.legend()
    ax1.grid(True, which="both", ls="--")
    
    ax2.set_xticks(np.arange(len(unique_nl)))
    ax2.set_xticklabels(x_labels)
    ax2.set_yscale('log')
    ax2.axhline(0.01, color='black', linestyle='--', label='1% drift threshold')
    ax2.set_xlabel("Noise level (% of coordinate std)")
    ax2.set_ylabel("Relative energy drift")
    ax2.set_title("Physical conservation under noisy supervision")
    ax2.grid(True, which="both", ls="--")
    
    plt.tight_layout()
    plt.savefig("plots/validation3_noise_robustness.png")
    plt.close()

def run_ode_generalisation_test(sol, force_retrain=False):
    filepath = "results/ode_generalisation.pkl"
    if os.path.exists(filepath) and not force_retrain:
        with open(filepath, "rb") as f:
            results = pickle.load(f)
    else:
        alphas = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]
        results = []
        
        lam_train = torch.linspace(0, 500, 500).view(-1, 1).to('cuda' if torch.cuda.is_available() else 'cpu')
        lam_heldout = torch.linspace(0.5, 499.5, 500).view(-1, 1).to(lam_train.device)
        
        lam_train.requires_grad_(True)
        lam_heldout.requires_grad_(True)
        
        _, lam_data, target_data, _ = prepare_training_data(sol, sample_ratio=0.1)
        
        print(f"{'alpha':<10} | {'res_train':<15} | {'res_heldout':<15} | {'gap%':<10}")
        
        for a in alphas:
            torch.manual_seed(42)
            pinn, _, _ = train_pinn(lam_train, lam_data, target_data, alpha=a, epochs=3000)
            pinn.eval()
            
            res_train = compute_physics_loss(pinn, lam_train).item()
            res_heldout = compute_physics_loss(pinn, lam_heldout).item()
            gap = abs(res_heldout - res_train) / (res_train + 1e-12)
            
            print(f"{a:<10.1f} | {res_train:<15.4e} | {res_heldout:<15.4e} | {gap*100:<10.2f}%")
            results.append({'alpha': a, 'res_train': res_train, 'res_heldout': res_heldout, 'gap': gap})
            
            # Incremental save
            with open(filepath, "wb") as f:
                pickle.dump(results, f)

    alphas = [r['alpha'] for r in results]
    res_train = [r['res_train'] for r in results]
    res_heldout = [r['res_heldout'] for r in results]
    gaps = [r['gap'] for r in results]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(alphas, res_train, 'b-', marker='o', label="Training collocation residual")
    ax1.plot(alphas, res_heldout, 'r--', marker='s', label="Held-out collocation residual")
    ax1.fill_between(alphas, res_train, res_heldout, color='red', alpha=0.15, label="generalisation gap")
    
    ax1.set_yscale('log')
    ax1.set_xlabel("α (physics weight)")
    ax1.set_ylabel("ODE residual (log scale)")
    ax1.set_title("ODE satisfaction at unseen collocation points")
    ax1.grid(True, which="both", ls="--")
    
    ax2 = ax1.twinx()
    ax2.bar(alphas, gaps, width=0.05, color='green', alpha=0.3, label="Relative gap")
    ax2.set_ylabel("Relative gap |res_held - res_train| / res_train", color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    
    # Combine legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper center')
    
    plt.tight_layout()
    plt.savefig("plots/validation4_ode_generalisation.png")
    plt.close()

if __name__ == "__main__":
    os.makedirs("plots", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    with open("data/trajectories.pkl", "rb") as f:
        datasets = pickle.load(f)
    sol = datasets['bound']

    print("\n=== EXPERIMENT 1: Extrapolation ===")
    run_extrapolation_test(sol)

    print("\n=== EXPERIMENT 2: Data Sparsity ===")
    run_sparsity_test(sol)

    print("\n=== EXPERIMENT 3: Noise Robustness ===")
    run_noise_test(sol)

    print("\n=== EXPERIMENT 4: Held-out ODE Residual ===")
    run_ode_generalisation_test(sol)

    print("\nAll validation plots saved to plots/")
