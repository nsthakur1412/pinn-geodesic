import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import csv
from scientific_framework import ScientificMLP, compute_physics_metrics, run_standard_eval, log_to_csv, TEST_SUITE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run_experiment(exp_name, lambda_phys, lambda_ic, lambda_cons):
    print(f"\n{'='*60}")
    print(f"RUNNING EXPERIMENT: {exp_name}")
    print(f"{'='*60}")
    
    # 1. Setup
    if not os.path.exists("data/unified_dataset.pt"):
        print("Error: data/unified_dataset.pt not found!")
        return
        
    inputs, targets = torch.load("data/unified_dataset.pt", weights_only=False)
    dataset = TensorDataset(inputs.to(device), targets.to(device))
    dataloader = DataLoader(dataset, batch_size=4096, shuffle=True)
    
    model = ScientificMLP(hidden_layers=6, neurons_per_layer=256, use_residual=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    epochs = 300
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    # 2. Training Loop
    for epoch in range(epochs):
        model.train()
        epoch_loss, epoch_data, epoch_phys, epoch_ic = 0.0, 0.0, 0.0, 0.0
        
        for batch_inputs, batch_targets in dataloader:
            optimizer.zero_grad()
            
            # Forward
            out_norm, out_phys = model(batch_inputs)
            
            # Data Loss (Standardized)
            target_pos = batch_targets[:, 0:3]
            L_data = nn.MSELoss()(out_norm, target_pos)
            
            # IC Loss (if enabled)
            L_ic = torch.tensor(0.0, device=device)
            if lambda_ic > 0:
                ic_mask = (batch_inputs[:, 0] == 0.0)
                if ic_mask.sum() > 0:
                    L_ic = nn.MSELoss()(out_norm[ic_mask], target_pos[ic_mask])
            
            # Physics Loss (if enabled)
            L_phys = torch.tensor(0.0, device=device)
            L_cons = torch.tensor(0.0, device=device)
            if lambda_phys > 0 or lambda_cons > 0:
                p_loss, E, L_p, H, dE, dL, _, _, _ = compute_physics_metrics(model, batch_inputs)
                if lambda_phys > 0:
                    L_phys = p_loss
                if lambda_cons > 0:
                    # Corrected Conservation: dE/dlam = 0, dL/dlam = 0, and H = -1
                    L_cons = torch.mean(dE**2) + torch.mean(dL**2) + torch.mean((H + 1.0)**2)
            
            # Total Loss
            loss = L_data + lambda_phys * L_phys + lambda_ic * L_ic + lambda_cons * L_cons
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_data += L_data.item()
            epoch_phys += L_phys.item()
            epoch_ic += L_ic.item()
            
        scheduler.step()
        
        # 3. Periodic Evaluation and Logging
        if epoch % 50 == 0 or epoch == epochs - 1:
            eval_results = run_standard_eval(model, device)
            
            log_data = {
                'experiment_type': exp_name,
                'epoch': epoch,
                'total_loss': epoch_loss / len(dataloader),
                'data_loss': epoch_data / len(dataloader),
                'phys_loss': epoch_phys / len(dataloader),
                'ic_loss': epoch_ic / len(dataloader),
                'energy_drift': eval_results['Bound']['e_drift'],
                'angular_momentum_drift': eval_results['Bound']['l_drift'],
                'hamiltonian_violation': eval_results['Bound']['h_violation'],
                'bound_max_dev': eval_results['Bound']['max_dev'],
                'escape_max_dev': eval_results['Escape']['max_dev'],
                'capture_max_dev': eval_results['Capture']['max_dev']
            }
            log_to_csv(log_data)
            print(f"Epoch {epoch:3d} | Loss: {log_data['total_loss']:.2e} | Bound Dev: {log_data['bound_max_dev']:.2f}M")
            
    # 4. Save Final Model
    os.makedirs("results", exist_ok=True)
    torch.save(model.state_dict(), f"results/model_{exp_name.lower().replace(' ', '_')}.pt")
    return eval_results

def generate_summary_table():
    """Requirement 7: Summary table generation without pandas."""
    csv_path = "results/comparative_study.csv"
    if not os.path.exists(csv_path):
        return
        
    data = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
            
    # Get last entry for each experiment
    final_results = {}
    for row in data:
        final_results[row['experiment_type']] = row
        
    headers = ['Model', 'Bound Dev', 'Escape Dev', 'Capture Dev', 'Energy Drift', 'IC Error']
    rows = []
    for exp in ["Data-Only", "Data + IC", "Full PINN"]:
        if exp in final_results:
            r = final_results[exp]
            rows.append([
                exp, 
                f"{float(r['bound_max_dev']):.2f}", 
                f"{float(r['escape_max_dev']):.2f}", 
                f"{float(r['capture_max_dev']):.2f}", 
                f"{float(r['energy_drift']):.2e}", 
                f"{float(r['ic_loss']):.2e}"
            ])
            
    # Simple ASCII table
    def print_table(headers, rows):
        col_widths = [max(len(h), max([len(str(r[i])) for r in rows] + [0])) for i, h in enumerate(headers)]
        format_str = " | ".join(["{:<" + str(w) + "}" for w in col_widths])
        print("\n" + "="*80)
        print("FINAL COMPARISON TABLE")
        print("="*80)
        print(format_str.format(*headers))
        print("-" * (sum(col_widths) + 3 * (len(headers) - 1)))
        for r in rows:
            print(format_str.format(*r))
        print("="*80)
        
        # Save to markdown
        with open("results/final_summary.md", "w") as f:
            f.write("# Final Comparative Summary\n\n")
            f.write("| " + " | ".join(headers) + " |\n")
            f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
            for r in rows:
                f.write("| " + " | ".join(r) + " |\n")

    print_table(headers, rows)

if __name__ == "__main__":
    csv_path = "results/comparative_study.csv"
    if os.path.exists(csv_path):
        os.remove(csv_path)
        
    run_experiment("Data-Only", lambda_phys=0.0, lambda_ic=0.0, lambda_cons=0.0)
    run_experiment("Data + IC", lambda_phys=0.0, lambda_ic=50.0, lambda_cons=0.0)
    run_experiment("Full PINN", lambda_phys=20.0, lambda_ic=50.0, lambda_cons=2.0)
    
    generate_summary_table()
