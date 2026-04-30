import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

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
    
    # 4. Training Config
    epochs = 3000
    optimizer = optim.Adam(pinn.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    # ===== CURRICULUM LEARNING =====
    # Phase 1 (0-500):   Learn short-time, data-heavy
    # Phase 2 (500-1500): Balance physics and data
    # Phase 3 (1500-3000): Physics-dominant for extrapolation
    def get_weights(epoch):
        if epoch < 500:
            return {'phys': 1.0, 'conserv': 1.0, 'data': 10.0}
        elif epoch < 1500:
            return {'phys': 5.0, 'conserv': 5.0, 'data': 5.0}
        else:
            return {'phys': 10.0, 'conserv': 10.0, 'data': 1.0}
    
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
            
            # Total Loss with curriculum weights
            L_total = (weights['phys'] * L_geodesic + 
                       weights['conserv'] * L_conserv +
                       weights['data'] * L_data)
            
            L_total.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(pinn.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += L_total.item()
            epoch_phys += L_geodesic.item()
            epoch_conserv += L_conserv.item()
            epoch_data += L_data.item()
                
        scheduler.step()
        
        if epoch % 50 == 0 or epoch == epochs - 1:
            n_batches = len(dataloader)
            avg_loss = epoch_loss / n_batches
            avg_phys = epoch_phys / n_batches
            avg_conserv = epoch_conserv / n_batches
            avg_data = epoch_data / n_batches
            phase = "DATA" if epoch < 500 else ("BALANCE" if epoch < 1500 else "PHYSICS")
            print(f"Epoch {epoch:4d} [{phase:7s}] | Total: {avg_loss:.2e} | Geod: {avg_phys:.2e} | Conserv: {avg_conserv:.2e} | Data: {avg_data:.2e}")
            
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
