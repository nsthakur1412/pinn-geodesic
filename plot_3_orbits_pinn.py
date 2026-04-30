import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
from rk45_solver import get_initial_state, solve_geodesic
from pinn_model import GeodesicPINN, get_total_loss

# Force CPU to completely avoid GPU interference
device = torch.device('cpu')

def prepare_cpu_data(sol, sample_ratio=0.2):
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

def train_cpu_pinn(sol, name, epochs=1000):
    print(f"Training CPU PINN for {name} orbit...")
    lam_coll, lam_data, target_data, lam_scale = prepare_cpu_data(sol)
    
    # Tiny network for CPU speed
    pinn = GeodesicPINN(hidden_layers=3, neurons_per_layer=32).to(device)
    optimizer = optim.Adam(pinn.parameters(), lr=2e-3)
    
    # We use a very low alpha (mostly data-driven) just to prove the network can fit the shapes quickly
    alpha = 0.05 
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss, _, _ = get_total_loss(pinn, lam_coll, lam_data, target_data, alpha, lam_scale)
        loss.backward()
        optimizer.step()
        
    pinn.eval()
    with torch.no_grad():
        pred = pinn(lam_coll).cpu().numpy()
        
    return pred

def run():
    print("Generating RK45 Orbits...")
    
    # 1. Bound Orbit (E < 1)
    state_bound = get_initial_state(r0=8.0, ur0=0.0, uphi0=0.05)
    sol_bound = solve_geodesic(state_bound, [0, 500], num_points=500)
    
    # 2. Escape Orbit (E > 1, High Angular Momentum)
    # Start at r=30, moving inward fast, high L
    state_escape = get_initial_state(r0=30.0, ur0=-0.3, uphi0=0.008) # L = 7.2
    sol_escape = solve_geodesic(state_escape, [0, 300], num_points=500)
    
    # 3. Capture Orbit (E > 1, Low Angular Momentum)
    # Start at r=30, moving inward fast, low L
    state_capture = get_initial_state(r0=30.0, ur0=-0.3, uphi0=0.002) # L = 1.8
    sol_capture = solve_geodesic(state_capture, [0, 150], num_points=500)
    
    orbits = {
        "Bound": sol_bound,
        "Escape": sol_escape,
        "Capture": sol_capture
    }
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': 'polar'})
    
    # Plot Black Hole and Photon Sphere
    theta = np.linspace(0, 2*np.pi, 100)
    ax.fill(theta, np.full_like(theta, 2.0), color='black', label="Event Horizon (r=2M)", zorder=10)
    ax.plot(theta, np.full_like(theta, 3.0), color='orange', linestyle='--', label="Photon Sphere (r=3M)", zorder=9)
    
    colors = {"Bound": '#2ca02c', "Escape": '#1f77b4', "Capture": '#d62728'}
    
    for name, sol in orbits.items():
        # RK45 Truth
        r_rk45 = sol.y[1]
        phi_rk45 = sol.y[2]
        ax.plot(phi_rk45, r_rk45, color=colors[name], linestyle='-', linewidth=2, label=f"RK45: {name}")
        
        # Train and Plot PINN
        pred = train_cpu_pinn(sol, name)
        r_pinn = pred[:, 1]
        phi_pinn = pred[:, 2]
        ax.plot(phi_pinn, r_pinn, color=colors[name], linestyle='--', linewidth=2, label=f"PINN: {name}")
        
    ax.set_rlim(0, 35)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.set_title("RK45 vs PINN: Bound, Escape, and Capture Orbits", pad=20, fontsize=14)
    
    if not os.path.exists("plots"):
        os.makedirs("plots")
        
    plt.tight_layout()
    plt.savefig("plots/pinn_orbit_types.png", dpi=300, bbox_inches='tight')
    print("Successfully saved to plots/pinn_orbit_types.png")

if __name__ == "__main__":
    run()
