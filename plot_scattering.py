import numpy as np
import matplotlib.pyplot as plt
from rk45_solver import get_initial_state, solve_geodesic

def plot_scattering():
    print("Generating scattering and capture trajectories...")
    
    # We start far away (r0 = 50M) with some inward velocity
    r0 = 50.0
    ur0 = -0.15 # Inward radial velocity
    
    # We will sweep the angular velocity to change the impact parameter (b ~ L/E)
    # L = r^2 * uphi
    # Critical impact parameter for E~1 is around b = 3*sqrt(3)M ~ 5.2M
    # So critical L is around 5.2.
    # At r0=50, uphi = L / 50^2 = L / 2500
    # Let's sweep L from 4.0 to 6.5 to capture a wide array of deep plunges and wide escapes
    L_values = np.linspace(4.0, 6.5, 30)
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': 'polar'})
    
    # Plot Black Hole (Event Horizon at r=2M)
    theta = np.linspace(0, 2*np.pi, 100)
    ax.fill(theta, np.full_like(theta, 2.0), color='black', label="Event Horizon (r=2M)", zorder=10)
    
    # Plot Photon Sphere (r=3M)
    ax.plot(theta, np.full_like(theta, 3.0), color='orange', linestyle='--', label="Photon Sphere (r=3M)", zorder=9)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(L_values)))
    
    for i, L in enumerate(L_values):
        uphi0 = L / (r0**2)
        state0 = get_initial_state(r0=r0, ur0=ur0, uphi0=uphi0)
        
        # Integrate for a long lambda span so they have time to escape or capture
        sol = solve_geodesic(state0, [0, 800], num_points=2000, rtol=1e-9, atol=1e-11)
        
        r = sol.y[1]
        phi = sol.y[2]
        
        if r[-1] < 2.1:
            status = "Capture"
            ls = '-'
        else:
            status = "Escape"
            ls = '-'
            
        ax.plot(phi, r, color=colors[i], linestyle=ls, linewidth=1.5, label=f"L={L:.2f} ({status})")
        
    ax.set_rlim(0, 20) # Limit view to the central region
    ax.set_rticks([2, 3, 5, 10, 15, 20])
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.set_title("Geodesic Scattering & Capture (Schwarzschild Spacetime)", pad=20, fontsize=14)
    
    plt.tight_layout()
    plt.savefig("plots/scattering_trajectories.png", dpi=300, bbox_inches='tight')
    print("Saved scattering plot to plots/scattering_trajectories.png")

if __name__ == "__main__":
    plot_scattering()
