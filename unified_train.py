import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import time
import matplotlib.pyplot as plt
import pickle
from unified_model import UnifiedPINN, compute_all_physics_losses

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_unified_pinn():
    print(f"Using device: {device}")
    
    # 1. Load Data
    print("Loading unified dataset...")
    inputs, targets = torch.load("data/unified_dataset.pt")
    
    # 2. DataLoader
    dataset = TensorDataset(inputs.to(device), targets.to(device))
    batch_size = 512
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 3. Model
    pinn = UnifiedPINN(hidden_layers=5, neurons_per_layer=128).to(device)
    
    if os.path.exists("results/unified_pinn.pt"):
        print("Loading pre-trained model to continue training...")
        try:
            pinn.load_state_dict(torch.load("results/unified_pinn.pt", map_location=device, weights_only=True))
        except Exception as e:
            print(f"Warning: Could not load weights ({e}). Training from scratch.")
    
    epochs = 4000
    optimizer = optim.Adam(pinn.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    # ChatGPT suggestions
    lambda_phys = 1.0
    lambda_data = 0.3
    lambda_ic = 50.0
    lambda_cons = 0.3
    
    print("Starting Unified PINN training with conservation laws...")
    start_time = time.time()
    
    history = {'epoch': [], 'total': [], 'phys': [], 'conserv': [], 'data': []}
    os.makedirs("plots", exist_ok=True)
    
    for epoch in range(epochs):
        epoch_loss, epoch_phys, epoch_conserv, epoch_data = 0.0, 0.0, 0.0, 0.0
        weights = get_weights(epoch)
        
        for batch_inputs, batch_targets in dataloader:
            optimizer.zero_grad()
            
            # Data Loss (MSE on normalized outputs vs standardized targets)
            out_norm, _ = pinn(batch_inputs)
            target_pos = batch_targets[:, 0:3]
            L_data = nn.MSELoss()(out_norm, target_pos)
            
            # Physics + Conservation Loss (single autograd pass)
            L_geodesic, L_conserv, _ = compute_all_physics_losses(pinn, batch_inputs)
            
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
            _, out_phys = pinn(batch_inputs)
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
            
            # P2: Final loss function
            L_total = (lambda_phys * L_phys + lambda_data * L_data + 
                       lambda_ic * L_ic + lambda_cons * L_cons)
            
            L_total.backward()
            torch.nn.utils.clip_grad_norm_(pinn.parameters(), 1.0) # P4: Gradient clipping
            optimizer.step()
            
            epoch_loss += L_total.item()
            epoch_phys += L_geodesic.item()
            epoch_conserv += L_conserv.item()
            epoch_data += L_data.item()
                
        scheduler.step()
        
        if epoch % 10 == 0 or epoch == epochs - 1:
            avg_loss = epoch_loss / len(dataloader)
            avg_phys = epoch_phys / len(dataloader)
            avg_data = epoch_data / len(dataloader)
            avg_ic = epoch_ic / len(dataloader)
            ratio = avg_phys / (avg_data + 1e-6)
            log_str = f"Epoch {epoch:3d} | Total: {avg_loss:.2e} | Phys: {avg_phys:.2e} | Data: {avg_data:.2e} | IC: {avg_ic:.2e} | Phys/Data: {ratio:.2e}"
            print(log_str)
            with open("experiments.log", "a") as f:
                f.write(log_str + "\n")
            
            history['epoch'].append(epoch)
            history['total'].append(avg_loss)
            history['phys'].append(avg_phys)
            history['conserv'].append(avg_conserv)
            history['data'].append(avg_data)
            
            # Save Live Plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            ax1.plot(history['epoch'], history['total'], 'k-', linewidth=2, label='Total')
            ax1.plot(history['epoch'], history['phys'], 'r--', label='Geodesic')
            ax1.plot(history['epoch'], history['conserv'], 'm--', label='Conservation')
            ax1.plot(history['epoch'], history['data'], 'b--', label='Data')
            ax1.set_yscale('log')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss (log)')
            ax1.set_title('Training Loss Components')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            # Phase boundaries
            ax1.axvline(x=500, color='gray', linestyle=':', alpha=0.5)
            ax1.axvline(x=1500, color='gray', linestyle=':', alpha=0.5)
            
            # Phase labels
            if len(history['epoch']) > 1:
                ax2.bar(['Geodesic', 'Conservation', 'Data'], 
                       [avg_phys, avg_conserv, avg_data],
                       color=['#e74c3c', '#9b59b6', '#3498db'])
                ax2.set_ylabel('Loss Value')
                ax2.set_title(f'Current Loss Breakdown (Epoch {epoch})')
                ax2.set_yscale('log')
            
            plt.tight_layout()
            plt.savefig('plots/live_unified_loss.png', dpi=150)
            plt.close()
            
            with open("data/unified_history.pkl", "wb") as f:
                pickle.dump(history, f)
            
    train_time = time.time() - start_time
    print(f"\nTraining completed in {train_time:.2f}s")
    
    os.makedirs("results", exist_ok=True)
    torch.save(pinn.state_dict(), "results/unified_pinn.pt")
    print("Unified PINN saved to results/unified_pinn.pt")

if __name__ == "__main__":
    train_unified_pinn()
