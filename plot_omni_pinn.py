import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
from rk45_solver import get_initial_state, solve_geodesic
from pinn_model import GeodesicPINN, get_total_loss

device = torch.device('cpu')

def prepare_cpu_data(sol, sample_ratio=0.3):
    lam = sol.t
    y = sol.y
    N = len(lam)
    
    num_samples = max(1, int(N * sample_ratio))
    indices = [0]
    if num_samples > 1:
        sampled = np.random.choice(np.arange(1, N), size=num_samples-1, replace=False)
        indices.extend(sampled)
    indices = np.sort(indices)
    
    lam_scale = float(np.max(lam))
    lam_scaled = lam / lam_scale
    
    lam_data = torch.tensor(lam_scaled[indices], dtype=torch.float32).view(-1, 1).to(device)
    target_data = torch.tensor(y[0:6, indices].T, dtype=torch.float32).to(device)
    
    lam_collocation = torch.tensor(lam_scaled, dtype=torch.float32).view(-1, 1).to(device)
    lam_collocation.requires_grad = True
    
    return lam_collocation, lam_data, target_data, lam_scale

def train_fast_cpu_pinn(sol, epochs=500):
    lam_coll, lam_data, target_data, lam_scale = prepare_cpu_data(sol)
    
    # Ultra-lightweight network so training 12 of them takes < 1 minute
    pinn = GeodesicPINN(hidden_layers=3, neurons_per_layer=24).to(device)
    optimizer = optim.Adam(pinn.parameters(), lr=3e-3)
    
    alpha = 0.02 # Mostly data-driven to perfectly mimic the shape
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss, _, _ = get_total_loss(pinn, lam_coll, lam_data, target_data, alpha, lam_scale)
        loss.backward()
        optimizer.step()
        
    pinn.eval()
    with torch.no_grad():
        pred = pinn(lam_coll).cpu().numpy()
        
    return pred

def run_omni():
    print("Generating omni-directional ray trajectories...")
    
    # We will generate 3 types of orbits, each rotated by 4 different angles (0, 90, 180, 270)
    angles = [0.0, np.pi/2, np.pi, 3*np.pi/2]
    
    scenarios = [
        {"name": "Bound", "r0": 8.0, "ur0": 0.0, "uphi0": 0.05, "lam_span": [0, 400], "color": "#2ca02c"},
        {"name": "Escape", "r0": 40.0, "ur0": -0.2, "uphi0": 0.005, "lam_span": [0, 250], "color": "#1f77b4"},
        {"name": "Capture", "r0": 40.0, "ur0": -0.2, "uphi0": 0.002, "lam_span": [0, 150], "color": "#d62728"}
    ]
    
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw={'projection': 'polar'})
    
    # Black Hole & Photon Sphere
    theta_bg = np.linspace(0, 2*np.pi, 100)
    ax.fill(theta_bg, np.full_like(theta_bg, 2.0), color='black', label="Event Horizon (r=2M)", zorder=10)
    ax.plot(theta_bg, np.full_like(theta_bg, 3.0), color='orange', linestyle='--', label="Photon Sphere (r=3M)", zorder=9)
    
    total_rays = len(angles) * len(scenarios)
    print(f"Training {total_rays} individual PINNs...")
    
    count = 1
    for angle in angles:
        for scen in scenarios:
            print(f"[{count}/{total_rays}] Simulating {scen['name']} ray at angle {np.degrees(angle):.0f}°")
            
            # RK45 Ground Truth
            state = get_initial_state(r0=scen["r0"], ur0=scen["ur0"], uphi0=scen["uphi0"], phi0=angle)
            sol = solve_geodesic(state, scen["lam_span"], num_points=400)
            
            r_rk45 = sol.y[1]
            phi_rk45 = sol.y[2]
            
            # PINN Prediction
            pred = train_fast_cpu_pinn(sol, epochs=600)
            r_pinn = pred[:, 1]
            phi_pinn = pred[:, 2]
            
            # Plot
            # Only add labels for the first angle to avoid legend clutter
            label_rk45 = f"RK45: {scen['name']}" if angle == 0.0 else ""
            label_pinn = f"PINN: {scen['name']}" if angle == 0.0 else ""
            
            ax.plot(phi_rk45, r_rk45, color=scen["color"], linestyle='-', linewidth=2.5, alpha=0.7, label=label_rk45)
            ax.plot(phi_pinn, r_pinn, color='white', linestyle='--', linewidth=1.2, label=label_pinn)
            
            count += 1
            
    ax.set_facecolor('#0d1117') # Dark space background
    fig.patch.set_facecolor('#0d1117')
    ax.grid(color='#30363d', linestyle='--', alpha=0.6)
    ax.tick_params(colors='white')
    ax.set_rlim(0, 45)
    
    legend = ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1), facecolor='#0d1117', edgecolor='#30363d', labelcolor='white')
    ax.set_title("Omni-Directional Scattering: RK45 vs PINN", pad=20, fontsize=16, color='white', weight='bold')
    
    os.makedirs("plots", exist_ok=True)
    plt.tight_layout()
    plt.savefig("plots/omni_directional_pinn.png", dpi=300, bbox_inches='tight', facecolor='#0d1117')
    print("\nSuccessfully saved omnidirectional visualization to plots/omni_directional_pinn.png")

if __name__ == "__main__":
    run_omni()
