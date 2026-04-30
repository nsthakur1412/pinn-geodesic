import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import time
import os
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
    
    epochs = 500
    optimizer = optim.Adam(pinn.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    alpha = 0.5
    beta = 1.0
    
    print("Starting DeepONet / Parameterized PINN training...")
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_loss, epoch_phys, epoch_data, epoch_ic = 0.0, 0.0, 0.0, 0.0
        
        for batch_inputs, batch_targets in dataloader:
            optimizer.zero_grad()
            
            # Data Loss (MSE on normalized outputs vs standardized targets)
            out_norm, _ = pinn(batch_inputs)
            target_pos = batch_targets[:, 0:3]
            L_data = nn.MSELoss()(out_norm, target_pos)
            
            # Physics Loss & Velocity
            L_phys, vel_phys = compute_unified_physics_loss(pinn, batch_inputs)
            
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
            
            # Total Balanced Loss
            L_total = alpha * L_phys + (1.0 - alpha) * L_data + beta * L_ic
            
            L_total.backward()
            optimizer.step()
            
            epoch_loss += L_total.item()
            epoch_phys += L_phys.item()
            epoch_data += L_data.item()
            if ic_mask.sum() > 0:
                epoch_ic += L_ic.item()
                
        scheduler.step()
        
        if epoch % 50 == 0 or epoch == epochs - 1:
            avg_loss = epoch_loss / len(dataloader)
            avg_phys = epoch_phys / len(dataloader)
            avg_data = epoch_data / len(dataloader)
            avg_ic = epoch_ic / len(dataloader)
            print(f"Epoch {epoch:3d} | Total: {avg_loss:.2e} | Phys: {avg_phys:.2e} | Data: {avg_data:.2e} | IC: {avg_ic:.2e}")
            
    train_time = time.time() - start_time
    print(f"\nTraining completed in {train_time:.2f}s")
    
    os.makedirs("results", exist_ok=True)
    torch.save(pinn.state_dict(), "results/unified_pinn.pt")
    print("Unified Parameterized PINN saved to results/unified_pinn.pt")

if __name__ == "__main__":
    train_unified_pinn()
