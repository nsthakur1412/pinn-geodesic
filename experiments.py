import torch
import torch.optim as optim
import numpy as np
import pickle
import os
import time
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from pinn_model import GeodesicPINN, get_total_loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def prepare_training_data(sol, sample_ratio=0.1, lam_max=None):
    """
    sol: scipy.integrate.solve_ivp output
    lam_max: if provided, only use data up to lam_max for training (for extrapolation test)
    Returns lam_collocation, lam_data, target_data
    """
    lam = sol.t
    y = sol.y # shape (6, N)
    
    if lam_max is not None:
        valid_idx = lam <= lam_max
        lam = lam[valid_idx]
        y = y[:, valid_idx]
        
    N = len(lam)
    
    # 1. Target data subset (10% + IC)
    num_samples = max(1, int(N * sample_ratio))
    
    # Always include IC
    indices = [0]
    # Randomly sample the rest
    if num_samples > 1:
        sampled_indices = np.random.choice(np.arange(1, N), size=num_samples-1, replace=False)
        indices.extend(sampled_indices)
        
    indices = np.sort(indices)
    
    lam_data = torch.tensor(lam[indices], dtype=torch.float32).view(-1, 1).to(device)
    # Target data is [t, r, phi]
    target_data = torch.tensor(y[0:3, indices].T, dtype=torch.float32).to(device)
    
    # 2. Collocation points (dense, for physics loss)
    lam_collocation = torch.tensor(lam, dtype=torch.float32).view(-1, 1).to(device)
    lam_collocation.requires_grad = True
    
    return lam_collocation, lam_data, target_data, lam

def train_pinn(lam_collocation, lam_data, target_data, alpha, epochs=3000, lr=1e-3):
    pinn = GeodesicPINN(hidden_layers=5, neurons_per_layer=64).to(device)
    optimizer = optim.Adam(pinn.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    history = {'total': [], 'physics': [], 'data': []}
    
    start_time = time.time()
    
    pinn.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        total_loss, phys_loss, data_loss = get_total_loss(pinn, lam_collocation, lam_data, target_data, alpha)
        
        total_loss.backward()
        optimizer.step()
        scheduler.step()
        
        history['total'].append(total_loss.item())
        history['physics'].append(phys_loss.item())
        history['data'].append(data_loss.item())
        
        if epoch % 500 == 0:
            print(f"Alpha {alpha:.1f} | Epoch {epoch}/{epochs} | Total: {total_loss.item():.2e} | Phys: {phys_loss.item():.2e} | Data: {data_loss.item():.2e}")
            
    train_time = time.time() - start_time
    print(f"Alpha {alpha:.1f} Training complete in {train_time:.2f}s")
    
    return pinn, history, train_time

def run_alpha_sweep(trajectory_data, save_dir="results"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    # We use the bound orbit for the sweep
    sol = trajectory_data['bound']
    
    # Fixed sampling across all alphas
    lam_coll, lam_data, target_data, _ = prepare_training_data(sol, sample_ratio=0.1)
    
    alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    results = {}
    
    for alpha in alphas:
        pinn, history, t_time = train_pinn(lam_coll, lam_data, target_data, alpha, epochs=2500)
        
        results[alpha] = {
            'model_state': pinn.state_dict(),
            'history': history,
            'train_time': t_time,
            'final_phys': history['physics'][-1],
            'final_data': history['data'][-1]
        }
        
    with open(os.path.join(save_dir, "alpha_sweep.pkl"), "wb") as f:
        pickle.dump(results, f)
    print("Saved alpha sweep results.")

def run_large_scale_sweep(trajectory_data, save_dir="results", n_seeds=5, epochs=2500):
    """
    Research-Grade Pareto Analysis Sweep
    Runs N seeds for 21 alpha values.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    sol = trajectory_data['bound']
    
    alphas = np.linspace(0.0, 1.0, 21) # 0.0, 0.05, 0.10, ..., 1.0
    results = []
    
    # We want consistent data sampling across all runs to ensure fair comparison
    lam_coll, lam_data, target_data, lam_full = prepare_training_data(sol, sample_ratio=0.1)
    
    pos_rk45 = torch.tensor(sol.y[0:3].T, dtype=torch.float32).to(device)
    
    print(f"Starting large-scale sweep: {len(alphas)} alphas x {n_seeds} seeds = {len(alphas)*n_seeds} runs on {device}", flush=True)
    
    for alpha in alphas:
        for seed in range(n_seeds):
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            pinn, history, t_time = train_pinn(lam_coll, lam_data, target_data, alpha, epochs=epochs)
            
            final_phys = history['physics'][-1]
            final_data = history['data'][-1]
            
            # Discard only strict NaNs so we can see the flawed trade-offs
            if np.isnan(final_phys) or np.isnan(final_data):
                print(f"Run Alpha {alpha:.2f} Seed {seed} discarded (NaN)", flush=True)
                continue
            
            # Compute Trajectory MAE and new physics metrics
            pinn.eval()
            with torch.no_grad():
                pred = pinn(lam_coll)
                mae = torch.mean(torch.abs(pred - pos_rk45)).item()
                
            # Requires grad to compute velocities for E and L
            lam_coll_eval = lam_coll.clone().detach().requires_grad_(True)
            pred_eval = pinn(lam_coll_eval)
            ones = torch.ones_like(lam_coll_eval)
            
            dt_dlam = torch.autograd.grad(pred_eval[:, 0:1], lam_coll_eval, grad_outputs=ones, retain_graph=True, create_graph=False)[0]
            dr_dlam = torch.autograd.grad(pred_eval[:, 1:2], lam_coll_eval, grad_outputs=ones, retain_graph=True, create_graph=False)[0]
            dphi_dlam = torch.autograd.grad(pred_eval[:, 2:3], lam_coll_eval, grad_outputs=ones, create_graph=False)[0]
            
            ut, ur, uphi = dt_dlam, dr_dlam, dphi_dlam
            r_pred = pred_eval[:, 1:2]
            
            # E = (1 - 2/r) * ut
            # L = r^2 * uphi
            E_pred = (1.0 - 2.0 / r_pred) * ut
            L_pred = r_pred**2 * uphi
            
            energy_drift = (torch.std(E_pred) / torch.mean(E_pred).abs()).item()
            ang_mom_drift = (torch.std(L_pred) / torch.mean(L_pred).abs()).item()
            
            f_val = 1.0 - 2.0 / r_pred
            norm_residual = -f_val * ut**2 + (1.0 / f_val) * ur**2 + r_pred**2 * uphi**2 + 1.0
            norm_residual_mean = torch.mean(torch.abs(norm_residual)).item()
            
            with torch.no_grad():
                t_err = (torch.mean(torch.abs(pred[:, 0] - pos_rk45[:, 0])) / torch.std(pos_rk45[:, 0])).item()
                r_err = (torch.mean(torch.abs(pred[:, 1] - pos_rk45[:, 1])) / torch.std(pos_rk45[:, 1])).item()
                phi_err = (torch.mean(torch.abs(pred[:, 2] - pos_rk45[:, 2])) / torch.std(pos_rk45[:, 2])).item()
                
            results.append({
                'alpha': alpha,
                'seed': seed,
                'final_phys': final_phys,
                'final_data': final_data,
                'mae': mae,
                'energy_drift': energy_drift,
                'ang_mom_drift': ang_mom_drift,
                'norm_residual_mean': norm_residual_mean,
                'rel_err_t': t_err,
                'rel_err_r': r_err,
                'rel_err_phi': phi_err
            })
            
            # Save incrementally
            with open(os.path.join(save_dir, "pareto_sweep.pkl"), "wb") as f:
                pickle.dump(results, f)
                
    print(f"Large scale sweep completed. Total valid runs: {len(results)}")

def run_extrapolation_test(trajectory_data, save_dir="results"):
    """
    Train on lambda in [0, lam_max], test on full range.
    """
    sol = trajectory_data['bound']
    # Total lam is 500. Let's train on first 250 (half orbit or so).
    lam_max = 250.0
    
    lam_coll, lam_data, target_data, lam_train = prepare_training_data(sol, sample_ratio=0.1, lam_max=lam_max)
    
    # Train with balanced alpha = 0.5
    print("\nRunning Extrapolation Test (alpha = 0.5, trained on lam <= 250)")
    pinn, history, t_time = train_pinn(lam_coll, lam_data, target_data, alpha=0.5, epochs=3000)
    
    # Test on full domain (will be done in evaluation.py)
    results = {
        'model_state': pinn.state_dict(),
        'history': history,
        'lam_max': lam_max
    }
    
    with open(os.path.join(save_dir, "extrapolation_test.pkl"), "wb") as f:
        pickle.dump(results, f)
    print("Saved extrapolation test results.")

if __name__ == "__main__":
    with open("data/trajectories.pkl", "rb") as f:
        datasets = pickle.load(f)
        
    # We keep the single alpha sweep for basic testing
    run_alpha_sweep(datasets)
    run_extrapolation_test(datasets)
    
    # Run the large-scale Pareto sweep
    run_large_scale_sweep(datasets, epochs=3000)
