import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import csv
import time
from scientific_framework import ScientificMLP, compute_physics_metrics, run_standard_eval, log_to_csv
from extensive_study_eval import run_extreme_tests # Re-using extreme eval logic

def run_production_training():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"STARTING STAGE 4: 4,000 EPOCH PRODUCTION RUN ON {device}")
    
    # 1. Setup
    inputs, targets = torch.load("data/unified_dataset.pt", weights_only=False)
    dataset = TensorDataset(inputs.to(device), targets.to(device))
    dataloader = DataLoader(dataset, batch_size=4096, shuffle=True)
    
    # Use the Refined 6x256 Residual Architecture
    model = ScientificMLP(hidden_layers=6, neurons_per_layer=256, use_residual=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    epochs = 4000
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    # Weights from the 'Stiff' breakthrough
    lambda_phys = 20.0
    lambda_ic = 50.0
    lambda_cons = 2.0
    
    csv_path = "results/long_duration_study.csv"
    latest_ckpt = "results/checkpoint_latest.pt"
    
    start_epoch = 0
    if os.path.exists(latest_ckpt):
        checkpoint = torch.load(latest_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"RESUMING TRAINING FROM EPOCH {start_epoch}")
    else:
        if os.path.exists(csv_path): os.remove(csv_path)
        print("STARTING FRESH TRAINING")
    
    checkpoints = [500, 1000, 2000, 3000, 4000]
    
    t0_all = time.time()
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        epoch_loss, epoch_data, epoch_phys, epoch_ic = 0.0, 0.0, 0.0, 0.0
        
        for batch_inputs, batch_targets in dataloader:
            optimizer.zero_grad()
            out_norm, _ = model(batch_inputs)
            
            # Data Loss
            L_data = nn.MSELoss()(out_norm, batch_targets[:, 0:3])
            
            # IC Loss
            ic_mask = (batch_inputs[:, 0] == 0.0)
            L_ic = nn.MSELoss()(out_norm[ic_mask], batch_targets[ic_mask, 0:3]) if ic_mask.sum() > 0 else torch.tensor(0.0, device=device)
            
            # Physics Loss
            p_loss, E, L_p, H, dE, dL, _, _, _ = compute_physics_metrics(model, batch_inputs)
            
            # Corrected Conservation: dE/dlam = 0, dL/dlam = 0, and H = -1
            # This works across ALL initial conditions simultaneously
            L_cons = torch.mean(dE**2) + torch.mean(dL**2) + torch.mean((H + 1.0)**2)
            
            # Total Loss
            loss = L_data + lambda_phys * p_loss + lambda_ic * L_ic + lambda_cons * L_cons
            loss.backward()
            
            # STABILITY: Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_data += L_data.item()
            epoch_phys += p_loss.item()
            epoch_ic += L_ic.item()
            
        scheduler.step()
        
        # Logging & Checkpointing
        if epoch in checkpoints or epoch % 10 == 0:
            avg_loss = epoch_loss / len(dataloader)
            print(f"Epoch {epoch}/{epochs} | Loss: {avg_loss:.4e}")
            
            # Only do full scientific eval at major milestones
            if epoch in checkpoints:
                eval_res = run_standard_eval(model, device)
                torch.save(model.state_dict(), f"results/model_stage4_epoch_{epoch}.pt")
                print(f"Saved checkpoint and ran full eval at epoch {epoch}")
                
                log_data = {
                    'experiment_type': 'Stage4-LongTrain',
                    'epoch': epoch,
                    'total_loss': avg_loss,
                    'data_loss': epoch_data / len(dataloader),
                    'phys_loss': epoch_phys / len(dataloader),
                    'ic_loss': epoch_ic / len(dataloader),
                    'energy_drift': eval_res['Bound']['e_drift'],
                    'angular_momentum_drift': eval_res['Bound']['l_drift'],
                    'hamiltonian_violation': eval_res['Bound']['h_violation'],
                    'bound_max_dev': eval_res['Bound']['max_dev'],
                    'escape_max_dev': eval_res['Escape']['max_dev'],
                    'capture_max_dev': eval_res['Capture']['max_dev']
                }
            else:
                # Fast logging for intermediate epochs
                log_data = {
                    'experiment_type': 'Stage4-LongTrain',
                    'epoch': epoch,
                    'total_loss': avg_loss,
                    'data_loss': epoch_data / len(dataloader),
                    'phys_loss': epoch_phys / len(dataloader),
                    'ic_loss': epoch_ic / len(dataloader),
                    'energy_drift': 0, 'angular_momentum_drift': 0, 'hamiltonian_violation': 0,
                    'bound_max_dev': 0, 'escape_max_dev': 0, 'capture_max_dev': 0
                }
            log_to_csv(log_data, file_path=csv_path)
            
            # Save latest for resume capability
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, latest_ckpt)

    t1_all = time.time()
    print(f"STAGE 4 COMPLETE. Total Time: {(t1_all-t0_all)/3600:.2f} hours.")

if __name__ == "__main__":
    run_production_training()
