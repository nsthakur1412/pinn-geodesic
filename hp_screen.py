import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import time
import csv
import itertools
import random

from unified_model import UnifiedPINN, compute_unified_physics_loss
from rk45_solver import get_initial_state, solve_geodesic
from unified_eval import TEST_CASES, predict_trajectory

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================
# HYPERPARAMETER GRID
# ============================================================
ALPHAS = [0.5, 0.7]
LAMBDA_PHYS = [1.0, 10.0, 100.0]
GAMMAS = [0.5, 1.0]
FOURIER_ENABLED = [True, False]

EPOCHS = 200
BATCH_SIZE = 512
REJECTION_THRESHOLD = 1e-3  # physics/data ratio

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run_screening_config(seed, alpha, lambda_phys_val, gamma, fourier_enabled, inputs, targets):
    set_seed(seed)
    print(f"\n--- Running Config: alpha={alpha}, lambda_phys={lambda_phys_val}, gamma={gamma}, fourier={fourier_enabled}, seed={seed} ---")
    
    dataset = TensorDataset(inputs, targets)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    pinn = UnifiedPINN(hidden_layers=5, neurons_per_layer=128, gamma=gamma, fourier_enabled=fourier_enabled).to(device)
    optimizer = optim.Adam(pinn.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    
    lambda_ic = 50.0
    lambda_cons = 0.3
    
    start_time = time.time()
    
    avg_loss = 0.0
    avg_phys = 0.0
    avg_data = 0.0
    avg_ic = 0.0
    
    for epoch in range(EPOCHS):
        epoch_loss, epoch_phys, epoch_data, epoch_ic = 0.0, 0.0, 0.0, 0.0
        
        for batch_inputs, batch_targets in dataloader:
            optimizer.zero_grad()
            
            L_phys, vel_phys, out_norm, out_phys = compute_unified_physics_loss(pinn, batch_inputs)
            
            target_pos = batch_targets[:, 0:3]
            L_data = nn.MSELoss()(out_norm, target_pos)
            
            lam_scaled = batch_inputs[:, 0]
            ic_mask = (lam_scaled == 0.0)
            
            if ic_mask.sum() > 0:
                ic_pred_norm = out_norm[ic_mask]
                ic_target_pos = target_pos[ic_mask]
                L_ic_pos = nn.MSELoss()(ic_pred_norm, ic_target_pos)
                
                ic_vel_phys = vel_phys[ic_mask]
                
                ut_mean, ut_std = pinn.scalers['ut_mean'], pinn.scalers['ut_std']
                ur_mean, ur_std = pinn.scalers['ur_mean'], pinn.scalers['ur_std']
                uphi_mean, uphi_std = pinn.scalers['uphi_mean'], pinn.scalers['uphi_std']
                
                ic_ut_norm = (ic_vel_phys[:, 0:1] - ut_mean) / ut_std
                ic_ur_norm = (ic_vel_phys[:, 1:2] - ur_mean) / ur_std
                ic_uphi_norm = (ic_vel_phys[:, 2:3] - uphi_mean) / uphi_std
                
                ic_vel_pred_norm = torch.cat([ic_ut_norm, ic_ur_norm, ic_uphi_norm], dim=1)
                ic_target_vel = batch_targets[ic_mask][:, 3:6]
                L_ic_vel = nn.MSELoss()(ic_vel_pred_norm, ic_target_vel)
                
                L_ic = L_ic_pos + L_ic_vel
            else:
                L_ic = torch.tensor(0.0, device=device)
            
            # Conservation Losses
            r_p = out_phys[:, 1:2]
            ut_p = vel_phys[:, 0:1]
            uphi_p = vel_phys[:, 2:3]
            f_p = 1.0 - 2.0 / (r_p + 1e-6)
            E_p = f_p * ut_p
            L_p = (r_p**2) * uphi_p
            
            target_r_norm = batch_targets[:, 1:2]
            target_ut_norm = batch_targets[:, 3:4]
            target_uphi_norm = batch_targets[:, 5:6]
            
            r_mean, r_std = pinn.scalers['r_mean'], pinn.scalers['r_std']
            ut_mean, ut_std = pinn.scalers['ut_mean'], pinn.scalers['ut_std']
            uphi_mean, uphi_std = pinn.scalers['uphi_mean'], pinn.scalers['uphi_std']
            
            r_t = target_r_norm * r_std + r_mean
            ut_t = target_ut_norm * ut_std + ut_mean
            uphi_t = target_uphi_norm * uphi_std + uphi_mean
            
            f_t = 1.0 - 2.0 / (r_t + 1e-6)
            E_t = f_t * ut_t
            L_t = (r_t**2) * uphi_t
            
            ur_p = vel_phys[:, 1:2]
            H_p = -f_p * (ut_p**2) + (1.0 / (f_p + 1e-6)) * (ur_p**2) + (r_p**2) * (uphi_p**2) + 1.0
            
            L_E = torch.mean((E_p - E_t)**2)
            L_L = torch.mean((L_p - L_t)**2)
            L_H = torch.mean(H_p**2)
            L_cons = L_E + L_L + L_H
            
            L_total = (alpha * lambda_phys_val * L_phys + 
                       (1 - alpha) * L_data + 
                       lambda_ic * L_ic + 
                       lambda_cons * L_cons)
            
            L_total.backward()
            torch.nn.utils.clip_grad_norm_(pinn.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += L_total.item()
            epoch_phys += L_phys.item()
            epoch_data += L_data.item()
            if ic_mask.sum() > 0:
                epoch_ic += L_ic.item()
                
        scheduler.step()
        
        avg_loss = epoch_loss / len(dataloader)
        avg_phys = epoch_phys / len(dataloader)
        avg_data = epoch_data / len(dataloader)
        avg_ic = epoch_ic / len(dataloader)
        ratio = avg_phys / (avg_data + 1e-12)
        
        # Early termination checks (allow a grace period for initial settling)
        if np.isnan(avg_loss):
            print(f"  [!] Run aborted at epoch {epoch}: Loss is NaN.")
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
            
        if epoch > 50 and avg_loss > 1e5:
            print(f"  [!] Run aborted at epoch {epoch}: Exploding loss ({avg_loss:.2e}).")
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        
        # Check physics collapse at intervals
        if epoch > 50 and epoch % 50 == 0:
            if ratio < REJECTION_THRESHOLD:
                print(f"  [!] Run aborted at epoch {epoch}: Physics collapse (ratio={ratio:.2e}).")
                return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
            
    train_time = time.time() - start_time
    print(f"  Training completed in {train_time:.2f}s | Loss: {avg_loss:.4f} | Phys/Data: {ratio:.2e}")
    
    # 2. IC Accuracy
    # Calculate directly using batch processing on lambda=0
    # Just take ICs from dataset
    ic_mask = (inputs[:, 0] == 0.0)
    ic_inputs = inputs[ic_mask]
    ic_targets_phys_r = (targets[ic_mask][:, 1] * pinn.scalers['r_std']) + pinn.scalers['r_mean']
    
    with torch.no_grad():
        _, out_phys = pinn(ic_inputs)
        ic_pred_phys_r = out_phys[:, 1]
        ic_error_tensor = torch.abs(ic_pred_phys_r - ic_targets_phys_r)
        ic_error = torch.mean(ic_error_tensor).item()
        
    # 3. Short Bound Orbit Rollout
    case = TEST_CASES[2]  # Unseen Bound
    r0, ur0, L = case['r0'], case['ur0'], case['L']
    uphi0 = L / (r0**2)
    state = get_initial_state(r0=r0, ur0=ur0, uphi0=uphi0)
    # Shorten rollout horizon to ~100
    sol = solve_geodesic(state, [0, 100], num_points=500)
    lam = sol.t.astype(np.float32)
    
    r_true = sol.y[1]
    phi_true = sol.y[2]
    x_true = r_true * np.cos(phi_true)
    y_true = r_true * np.sin(phi_true)
    
    out = predict_trajectory(pinn, r0, ur0, L, lam)
    r_pred, phi_pred = out[:, 1], out[:, 2]
    x_pred = r_pred * np.cos(phi_pred)
    y_pred = r_pred * np.sin(phi_pred)
    
    bound_deviation = np.max(np.sqrt((x_true - x_pred)**2 + (y_true - y_pred)**2))
    
    # Calculate custom proxy score
    score = bound_deviation + 10 * ic_error - 0.1 * np.log10(ratio + 1e-12)
    
    print(f"  Metrics -> IC Err: {ic_error:.4f} | Bound Dev: {bound_deviation:.4f} | Score: {score:.4f}")
    
    return avg_loss, avg_phys, avg_data, ratio, ic_error, bound_deviation, score


def main():
    print("Loading dataset for screening...")
    inputs_full, targets_full = torch.load("data/unified_dataset.pt", weights_only=False)
    
    # Data reduction (50%)
    inputs = inputs_full[::2].to(device)
    targets = targets_full[::2].to(device)
    print(f"Reduced dataset size: {len(inputs)} points")
    
    configs = list(itertools.product(ALPHAS, LAMBDA_PHYS, GAMMAS, FOURIER_ENABLED))
    print(f"Total configurations to screen: {len(configs)}")
    
    csv_file = "hp_screening_results.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['alpha', 'lambda_phys', 'gamma', 'fourier_enabled', 'random_seed', 
                         'final_total_loss', 'phys_loss', 'data_loss', 'phys_data_ratio', 
                         'ic_error', 'bound_deviation', 'score'])
    
    seed = random.randint(1000, 9999)
    print(f"Using global seed: {seed}")
    
    results = []
    
    for idx, (alpha, lambda_phys_val, gamma, fourier_enabled) in enumerate(configs):
        avg_loss, avg_phys, avg_data, ratio, ic_error, bound_deviation, score = run_screening_config(
            seed, alpha, lambda_phys_val, gamma, fourier_enabled, inputs, targets
        )
        
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([alpha, lambda_phys_val, gamma, fourier_enabled, seed,
                             avg_loss, avg_phys, avg_data, ratio, ic_error, bound_deviation, score])
            
        log_str = f"Screen {idx+1}/{len(configs)} | alpha={alpha}, lam_p={lambda_phys_val}, gam={gamma}, fourier={fourier_enabled} | Score={score:.4f}"
        with open("experiments.log", "a") as f:
            f.write(log_str + "\n")
            
        if not np.isnan(score):
            results.append({
                'config': (alpha, lambda_phys_val, gamma, fourier_enabled),
                'score': score
            })
            
    # Rank configurations
    if len(results) > 0:
        results.sort(key=lambda x: x['score'])
        print("\n" + "="*50)
        print("TOP 3 CONFIGURATIONS (Lowest Score = Best)")
        print("="*50)
        for i, res in enumerate(results[:3]):
            cfg = res['config']
            print(f"{i+1}. alpha={cfg[0]}, lambda_phys={cfg[1]}, gamma={cfg[2]}, fourier={cfg[3]} | Score: {res['score']:.4f}")
    else:
        print("\nNo configurations successfully completed the screening.")

if __name__ == "__main__":
    main()
