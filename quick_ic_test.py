import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
import pickle
import os
from unified_model import UnifiedPINN

def quick_ic_test():
    # 1. Load Model & Scalers
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on device: {device}")
    
    pinn = UnifiedPINN(hidden_layers=5, neurons_per_layer=128).to(device)
    if os.path.exists("results/unified_pinn.pt"):
        pinn.load_state_dict(torch.load("results/unified_pinn.pt", map_location=device, weights_only=True))
        pinn.eval()
        print("Model loaded successfully.\n")
    else:
        print("Error: Model file not found at results/unified_pinn.pt")
        return

    with open("data/unified_scalers.pkl", "rb") as f:
        scalers = pickle.load(f)

    # 2. Define Test Cases (r0, ur0, L)
    test_cases = [
        {"name": "Bound Orbit", "r0": 8.0, "ur0": 0.0, "L": 3.2},
        {"name": "Escape Path", "r0": 20.0, "ur0": -0.15, "L": 6.0},
        {"name": "Capture Event", "r0": 10.0, "ur0": -0.1, "L": 2.8}
    ]

    print(f"{'Case':<15} | {'Metric':<10} | {'Target':<10} | {'Predicted':<10} | {'Error':<10}")
    print("-" * 65)

    for case in test_cases:
        # Standardize Inputs
        r0_norm = (case['r0'] - scalers['r0_mean']) / scalers['r0_std']
        ur0_norm = (case['ur0'] - scalers['ur0_mean']) / scalers['ur0_std']
        L_norm = (case['L'] - scalers['L_mean']) / scalers['L_std']
        lam_norm = 0.0 # Testing Initial Condition
        
        inputs = torch.tensor([[lam_norm, r0_norm, ur0_norm, L_norm]], dtype=torch.float32).to(device)
        
        with torch.no_grad():
            out_norm, out_phys = pinn(inputs)
            
        # Target ICs: t=0, r=r0, phi=0
        pred_t, pred_r, pred_phi = out_phys[0].cpu().numpy()
        
        # Compare
        metrics = [
            ("t", 0.0, pred_t),
            ("r", case['r0'], pred_r),
            ("phi", 0.0, pred_phi)
        ]
        
        for name, target, pred in metrics:
            err = abs(target - pred)
            print(f"{case['name']:<15} | {name:<10} | {target:<10.4f} | {pred:<10.4f} | {err:<10.4e}")
        print("-" * 65)

if __name__ == "__main__":
    quick_ic_test()
