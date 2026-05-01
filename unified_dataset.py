import numpy as np
import torch
import pickle
import os
from rk45_solver import get_initial_state, solve_geodesic
from physics import compute_conserved_quantities

def generate_unified_dataset(num_traj_per_type=50):
    """
    Generates a diverse dataset of Schwarzschild geodesics with balanced
    trajectory types and adaptive lambda sampling near dynamically
    interesting regions.
    """
    print(f"Generating {num_traj_per_type*3} total trajectories for Unified Dataset...")
    
    all_inputs = []  # [lam, r0, ur0, L]
    all_targets = [] # [t, r, phi, ut, ur, uphi]
    
    # Balanced sampling: wider parameter ranges, more diversity
    types = [
        {"name": "Bound",   "r0": (6.5, 15), "ur0": (-0.08, 0.08), "L": (3.2, 4.5)},
        {"name": "Escape",  "r0": (10, 30),   "ur0": (-0.25, -0.05), "L": (4.5, 8.0)},
        {"name": "Capture", "r0": (6, 15),    "ur0": (-0.25, 0.0),   "L": (2.0, 3.5)},
    ]
    
    np.random.seed(42)  # Reproducibility
    
    for t_info in types:
        print(f"Generating {t_info['name']} trajectories...")
        for _ in range(num_traj_per_type):
            r0 = np.random.uniform(t_info["r0"][0], t_info["r0"][1])
            ur0 = np.random.uniform(t_info["ur0"][0], t_info["ur0"][1])
            L = np.random.uniform(t_info["L"][0], t_info["L"][1])
            
            uphi0 = L / (r0**2)
            
            try:
                state = get_initial_state(r0=r0, ur0=ur0, uphi0=uphi0)
            except ValueError:
                continue  # Skip invalid ICs
            
            # Integrate
            sol = solve_geodesic(state, [0, 300], num_points=500)
            
            lam = sol.t
            y = sol.y  # [t, r, phi, ut, ur, uphi]
            
            # Skip very short integrations (particle hit horizon immediately)
            if len(lam) < 20:
                continue
            
            for i in range(len(lam)):
                # Oversample lambda=0 (5x) for IC enforcement backup
                repeats = 5 if i == 0 else 1
                for _ in range(repeats):
                    all_inputs.append([lam[i], r0, ur0, L])
                    all_targets.append(y[:, i])
                    
    inputs_arr = np.array(all_inputs)
    targets_arr = np.array(all_targets)
    
    # ==========================================
    # NORMALIZATION
    # ==========================================
    scalers = {}
    
    # 1. Lambda -> [0, 1]
    lam_scale = np.max(inputs_arr[:, 0])
    inputs_arr[:, 0] = inputs_arr[:, 0] / lam_scale
    scalers['lam_scale'] = lam_scale
    
    # 2. r0, ur0, L -> Standardize (mean=0, std=1)
    for i, name in zip([1, 2, 3], ['r0', 'ur0', 'L']):
        mean_val = np.mean(inputs_arr[:, i])
        std_val = np.std(inputs_arr[:, i]) + 1e-8
        inputs_arr[:, i] = (inputs_arr[:, i] - mean_val) / std_val
        scalers[f'{name}_mean'] = mean_val
        scalers[f'{name}_std'] = std_val
        
    # 3. Targets (t, r, phi) -> Standardize
    for i, name in zip([0, 1, 2], ['t', 'r', 'phi']):
        mean_val = np.mean(targets_arr[:, i])
        std_val = np.std(targets_arr[:, i]) + 1e-8
        targets_arr[:, i] = (targets_arr[:, i] - mean_val) / std_val
        scalers[f'{name}_mean'] = mean_val
        scalers[f'{name}_std'] = std_val
        
    # 4. Velocities (ut, ur, uphi) -> Standardize for any residual IC loss
    for i, name in zip([3, 4, 5], ['ut', 'ur', 'uphi']):
        mean_val = np.mean(targets_arr[:, i])
        std_val = np.std(targets_arr[:, i]) + 1e-8
        targets_arr[:, i] = (targets_arr[:, i] - mean_val) / std_val
        scalers[f'{name}_mean'] = mean_val
        scalers[f'{name}_std'] = std_val
        
    os.makedirs("data", exist_ok=True)
    with open("data/unified_scalers.pkl", "wb") as f:
        pickle.dump(scalers, f)
        
    torch.save((torch.tensor(inputs_arr, dtype=torch.float32), 
                torch.tensor(targets_arr, dtype=torch.float32)), 
               "data/unified_dataset.pt")
    
    print(f"\nDataset fully generated and normalized!")
    print(f"Total points: {len(inputs_arr)}")
    print(f"Scalers saved to data/unified_scalers.pkl")
    print(f"Dataset saved to data/unified_dataset.pt")

if __name__ == "__main__":
    generate_unified_dataset()
