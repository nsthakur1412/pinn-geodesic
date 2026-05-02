import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
from scientific_framework import ScientificMLP, run_standard_eval

def compare_at_200():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    configs = [
        {"name": "Stage 4 (Current)", "path": "results/checkpoint_latest.pt", "type": "checkpoint"},
        {"name": "Stage 1 (Full PINN)", "path": "results/model_pinn_256_res.pt", "type": "model"},
        {"name": "Data-Only", "path": "results/model_data_256_res.pt", "type": "model"},
        {"name": "Data + IC", "path": "results/model_data_ic_256_res.pt", "type": "model"},
    ]
    
    results = []
    
    for conf in configs:
        if not os.path.exists(conf['path']):
            print(f"Skipping {conf['name']}: {conf['path']} not found.")
            continue
            
        print(f"Evaluating {conf['name']}...")
        model = ScientificMLP(hidden_layers=6, neurons_per_layer=256, use_residual=True).to(device)
        
        try:
            state = torch.load(conf['path'], map_location=device, weights_only=False)
            if conf['type'] == "checkpoint":
                model.load_state_dict(state['model_state_dict'])
            else:
                model.load_state_dict(state)
            
            # Standard eval returns: phys_loss, E_drift, L_drift, H_viol, b_dev, e_dev, c_dev
            # Wait, the version in scientific_framework returns a DICT now based on unified_train.py?
            # Let me check scientific_framework.py one more time.
            res = run_standard_eval(model, device)
            # res is a dict: {'Bound': {...}, 'Escape': {...}, 'Capture': {...}}
            
            results.append({
                "Name": conf['name'],
                "Bound Dev": res['Bound']['max_dev'],
                "Escape Dev": res['Escape']['max_dev'],
                "H-Viol": res['Bound']['h_violation']
            })
        except Exception as e:
            print(f"Failed to evaluate {conf['name']}: {e}")

    print("\nComparison at Epoch 200 (Stage 4) vs Baselines:")
    print(f"{'Model':<20} | {'Bound Dev':<10} | {'Escape Dev':<10} | {'H-Viol':<10}")
    print("-" * 60)
    for r in results:
        print(f"{r['Name']:<20} | {r['Bound Dev']:<10.4f} | {r['Escape Dev']:<10.4f} | {r['H-Viol']:<10.4e}")

if __name__ == "__main__":
    compare_at_200()
