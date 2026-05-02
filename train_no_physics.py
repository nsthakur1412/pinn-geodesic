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

def train_no_physics():
    print(f"Using device: {device}")
    
    # 1. Load Data
    print("Loading unified dataset for NO-PHYSICS training...")
    if not os.path.exists("data/unified_dataset.pt"):
        print("Error: data/unified_dataset.pt not found!")
        return
        
    inputs, targets = torch.load("data/unified_dataset.pt")
    
    dataset = TensorDataset(inputs.to(device), targets.to(device))
    batch_size = 512
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # 3. Model & Optimizers
    pinn = UnifiedPINN(hidden_layers=5, neurons_per_layer=128).to(device)
    
    epochs = 500  # Set to 500 as requested
    optimizer = optim.Adam(pinn.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    # Hyperparameters for loss balancing - DROPPED PHYSICS
    alpha = 0.0      # Fully data driven
    lambda_phys = 0.0
    lambda_ic = 0.0
    lambda_cons = 0.0
    
    print("Starting Data-Only training (100 epochs)...")
    start_time = time.time()
    
    history = {'epoch': [], 'total': [], 'phys': [], 'data': [], 'ic': []}
    os.makedirs("results", exist_ok=True)
    
    for epoch in range(epochs):
        epoch_loss, epoch_phys, epoch_data, epoch_ic = 0.0, 0.0, 0.0, 0.0
        
        for batch_inputs, batch_targets in dataloader:
            optimizer.zero_grad()
            
            # Forward pass
            _, out_norm = pinn(batch_inputs)
            
            # Data Loss (MSE on normalized outputs vs standardized targets)
            target_pos = batch_targets[:, 0:3]
            L_data = nn.MSELoss()(out_norm, target_pos)
            
            # Total loss (only data)
            L_total = L_data
            
            L_total.backward()
            torch.nn.utils.clip_grad_norm_(pinn.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += L_total.item()
            epoch_data += L_data.item()
                
        scheduler.step()
        
        avg_loss = epoch_loss / len(dataloader)

        if epoch % 10 == 0 or epoch == epochs - 1:
            avg_data = epoch_data / len(dataloader)
            log_str = f"Epoch {epoch:3d} | Total (Data): {avg_loss:.2e}"
            print(log_str)
            
            # Record history
            history['epoch'].append(epoch)
            history['total'].append(avg_loss)
            history['data'].append(avg_data)
            
    train_time = time.time() - start_time
    print(f"\nTraining completed in {train_time:.2f}s")
    
    torch.save(pinn.state_dict(), "results/no_physics_pinn.pt")
    print("Data-only model saved to results/no_physics_pinn.pt")

    # Save history
    with open("results/no_physics_history.pkl", "wb") as f:
        pickle.dump(history, f)

if __name__ == "__main__":
    train_no_physics()
