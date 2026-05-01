import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import time
import os
import matplotlib.pyplot as plt
import pickle
from unified_model import UnifiedPINN, compute_unified_physics_loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_unified_pinn():
    print(f"Using device: {device}")
    
    # 1. Load Data
    print("Loading unified dataset...")
    inputs, targets = torch.load("data/unified_dataset.pt")
    
    # 2. DataLoader (CRITICAL: Grouping trajectories)
    # By setting shuffle=False, the batches consist of contiguous lambda sequences
    # from the same trajectory, maintaining dynamic consistency per batch.
    dataset = TensorDataset(inputs.to(device), targets.to(device))
    batch_size = 512
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # 3. Model & Optimizers
    pinn = UnifiedPINN(hidden_layers=5, neurons_per_layer=128).to(device)
    
    epochs = 4000
    optimizer = optim.Adam(pinn.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    os.makedirs("checkpoints", exist_ok=True)
    best_loss = float('inf')
    start_epoch = 0
    
    import glob
    ckpts = glob.glob("checkpoints/ckpt_*.pt")
    if ckpts:
        latest_ckpt = max(ckpts, key=os.path.getctime)
        print(f"Loading checkpoint {latest_ckpt} to resume training...")
        ckpt = torch.load(latest_ckpt, map_location=device, weights_only=False)
        pinn.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        start_epoch = ckpt['epoch'] + 1
    
    # Hyperparameters for loss balancing
    alpha = 0.7
    lambda_phys = 5.0
    lambda_ic = 50.0
    lambda_cons = 0.3
    
    print("Starting DeepONet / Parameterized PINN training...")
    start_time = time.time()
    
    history = {'epoch': [], 'total': [], 'phys': [], 'data': [], 'ic': []}
    os.makedirs("plots", exist_ok=True)
    
    for epoch in range(start_epoch, epochs):
        epoch_loss, epoch_phys, epoch_data, epoch_ic = 0.0, 0.0, 0.0, 0.0
        
        # Strengthen physics signal early on
        lambda_phys = 5.0 if epoch <= 500 else 1.0
        
        for batch_inputs, batch_targets in dataloader:
            optimizer.zero_grad()
            
            # Physics Loss, Velocity, and Outputs (ALL in ONE forward pass!)
            L_phys, vel_phys, out_norm, out_phys = compute_unified_physics_loss(pinn, batch_inputs)
            
            # Data Loss (MSE on normalized outputs vs standardized targets)
            target_pos = batch_targets[:, 0:3]
            L_data = nn.MSELoss()(out_norm, target_pos)
            
            # IC Loss (Strong enforcement)
            # Find indices where lambda == 0 (which we oversampled 5x)
            lam_scaled = batch_inputs[:, 0]
            ic_mask = (lam_scaled == 0.0)
            
            if ic_mask.sum() > 0:
                # Position IC
                ic_pred_norm = out_norm[ic_mask]
                ic_target_pos = target_pos[ic_mask]
                L_ic_pos = nn.MSELoss()(ic_pred_norm, ic_target_pos)
                
                # Velocity IC
                # Convert physical velocity predictions to normalized scale to match targets
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
            
            # P3: Conservation Losses (E, L)
            r_p = out_phys[:, 1:2]
            ut_p = vel_phys[:, 0:1]
            uphi_p = vel_phys[:, 2:3]
            
            # Predicted E and L
            f_p = 1.0 - 2.0 / (r_p + 1e-6)
            E_p = f_p * ut_p
            L_p = (r_p**2) * uphi_p
            
            # Target E and L from batch_targets (un-normalized)
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
            
            # P1: Hamiltonian Conservation
            ur_p = vel_phys[:, 1:2]
            H_p = -f_p * (ut_p**2) + (1.0 / (f_p + 1e-6)) * (ur_p**2) + (r_p**2) * (uphi_p**2) + 1.0
            
            L_E = torch.mean((E_p - E_t)**2)
            L_L = torch.mean((L_p - L_t)**2)
            L_H = torch.mean(H_p**2)
            L_cons = L_E + L_L + L_H
            
            # P2: Final loss function with alpha-balancing
            L_total = (alpha * lambda_phys * L_phys + 
                       (1 - alpha) * L_data + 
                       lambda_ic * L_ic + 
                       lambda_cons * L_cons)
            
            L_total.backward()
            torch.nn.utils.clip_grad_norm_(pinn.parameters(), 1.0) # P4: Gradient clipping
            optimizer.step()
            
            epoch_loss += L_total.item()
            epoch_phys += L_phys.item()
            epoch_data += L_data.item()
            if ic_mask.sum() > 0:
                epoch_ic += L_ic.item()
                
        scheduler.step()
        
        avg_loss = epoch_loss / len(dataloader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(pinn.state_dict(), "checkpoints/best_model.pt")

        if epoch % 10 == 0 or epoch == epochs - 1:
            avg_phys = epoch_phys / len(dataloader)
            avg_data = epoch_data / len(dataloader)
            avg_ic = epoch_ic / len(dataloader)
            ratio = avg_phys / (avg_data + 1e-6)
            log_str = f"Epoch {epoch:3d} | Total: {avg_loss:.2e} | Phys: {avg_phys:.2e} | Data: {avg_data:.2e} | IC: {avg_ic:.2e} | Phys/Data: {ratio:.2e}"
            print(log_str)
            print(f"Epoch {epoch} | IC error: {avg_ic:.4e}")
            with open("experiments.log", "a") as f:
                f.write(log_str + "\n")
            
            # Record history
            history['epoch'].append(epoch)
            history['total'].append(avg_loss)
            history['phys'].append(avg_phys)
            history['data'].append(avg_data)
            history['ic'].append(avg_ic)
            
        # Checkpoint saving
        if epoch % 100 == 0:
            torch.save({
                'epoch': epoch,
                'model_state': pinn.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'loss': avg_loss
            }, f"checkpoints/ckpt_{epoch}.pt")
            print(f"Saved checkpoint: checkpoints/ckpt_{epoch}.pt")
            
            with open("data/unified_history.pkl", "wb") as f:
                pickle.dump(history, f)
            
    train_time = time.time() - start_time
    print(f"\nTraining completed in {train_time:.2f}s")
    
    os.makedirs("results", exist_ok=True)
    torch.save(pinn.state_dict(), "results/unified_pinn.pt")
    print("Unified Parameterized PINN saved to results/unified_pinn.pt")

if __name__ == "__main__":
    train_unified_pinn()
