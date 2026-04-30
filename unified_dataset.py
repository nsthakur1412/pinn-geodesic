import numpy as np
import torch
import pickle
import os
from rk45_solver import get_initial_state, solve_geodesic

def generate_unified_dataset(num_traj_per_type=15):
    print(f"Generating {num_traj_per_type*3} total trajectories for Unified Dataset...")
    
    all_inputs = []  # [lam, r0, ur0, L]
    all_targets = [] # [t, r, phi, ut, ur, uphi]
    
    # Balanced sampling constraints
    types = [
        {"name": "Bound", "r0": (7, 10), "ur0": (-0.05, 0.05), "L": (3.0, 3.8)},
        {"name": "Escape", "r0": (15, 25), "ur0": (-0.2, -0.1), "L": (5.0, 8.0)},
        {"name": "Capture", "r0": (8, 15), "ur0": (-0.2, 0.0), "L": (2.5, 3.2)},
    ]
    
    np.random.seed(42) # Reproducibility
    
    for t_info in types:
        print(f"Generating {t_info['name']} trajectories...")
        for _ in range(num_traj_per_type):
            r0 = np.random.uniform(t_info["r0"][0], t_info["r0"][1])
            ur0 = np.random.uniform(t_info["ur0"][0], t_info["ur0"][1])
            L = np.random.uniform(t_info["L"][0], t_info["L"][1])
            
            uphi0 = L / (r0**2)
            state = get_initial_state(r0=r0, ur0=ur0, uphi0=uphi0)
            
            # Integrate up to lam=300
            sol = solve_geodesic(state, [0, 300], num_points=300)
            
            lam = sol.t
            y = sol.y # [t, r, phi, ut, ur, uphi]
            
            # To ensure the PINN doesn't overfit to specific lam arrays, we could perturb lambda.
            # But standard num_points=300 is fine.
            
            for i in range(len(lam)):
                # CRITICAL: Oversample lambda=0 ~5x to enforce strong Initial Conditions
                repeats = 5 if i == 0 else 1
                for _ in range(repeats):
                    all_inputs.append([lam[i], r0, ur0, L])
                    all_targets.append(y[:, i])
                    
    inputs_arr = np.array(all_inputs)
    targets_arr = np.array(all_targets)
    
    # ==========================================
    # NORMALIZATION (CRITICAL FOR DEEPONET)
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
    # Note: Target shape is [t, r, phi, ut, ur, uphi]
    for i, name in zip([0, 1, 2], ['t', 'r', 'phi']):
        mean_val = np.mean(targets_arr[:, i])
        std_val = np.std(targets_arr[:, i]) + 1e-8
        targets_arr[:, i] = (targets_arr[:, i] - mean_val) / std_val
        scalers[f'{name}_mean'] = mean_val
        scalers[f'{name}_std'] = std_val
        
    # 4. Velocities (ut, ur, uphi) -> Standardize for IC loss scaling
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
