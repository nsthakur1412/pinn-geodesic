import numpy as np
from scipy.integrate import solve_ivp
import os
import pickle
from physics import geodesic_odes, f

def get_initial_state(r0, ur0, uphi0, phi0=0.0, t0=0.0):
    """
    Computes the full initial state [t, r, phi, ut, ur, uphi]
    ensuring the normalization condition g_mu_nu u^mu u^nu = -1 is satisfied.
    """
    f_val = f(r0)
    
    # Solve for ut: -f(r) ut^2 + (1/f(r)) ur^2 + r^2 uphi^2 = -1
    # ut^2 = (1 + ur^2 / f(r) + r^2 uphi^2) / f(r)
    ut2 = (1.0 + ur0**2 / f_val + r0**2 * uphi0**2) / f_val
    if ut2 < 0:
        raise ValueError("Invalid initial conditions: ut^2 < 0")
        
    ut0 = np.sqrt(ut2)
    
    return [t0, r0, phi0, ut0, ur0, uphi0]

def solve_geodesic(initial_state, lam_span, num_points=1000, rtol=1e-8, atol=1e-10):
    """
    Solves the geodesic equations using scipy's RK45.
    Stops integration if the particle hits the event horizon (r <= 2.01).
    """
    
    # Event function to stop integration near the event horizon
    def horizon_event(lam, state):
        return state[1] - 2.01 # Stop at r = 2.01
    horizon_event.terminal = True
    
    lam_eval = np.linspace(lam_span[0], lam_span[1], num_points)
    
    sol = solve_ivp(
        fun=geodesic_odes,
        t_span=lam_span,
        y0=initial_state,
        method='RK45',
        t_eval=lam_eval,
        events=horizon_event,
        rtol=rtol,
        atol=atol
    )
    
    return sol

def robustness_check():
    """
    Performs a tolerance sweep to establish RK45 as a reliable ground truth.
    """
    print("Running RK45 robustness check...")
    
    # Bound orbit initial condition
    state0 = get_initial_state(r0=10.0, ur0=0.0, uphi0=0.04)
    lam_span = [0, 500]
    
    # Loose tolerance
    sol_loose = solve_geodesic(state0, lam_span, rtol=1e-4, atol=1e-6)
    
    # Strict tolerance
    sol_strict = solve_geodesic(state0, lam_span, rtol=1e-10, atol=1e-12)
    
    # Compare end states (if they reached the same lam)
    min_len = min(len(sol_loose.t), len(sol_strict.t))
    
    diff = np.abs(sol_loose.y[:, :min_len] - sol_strict.y[:, :min_len])
    max_diff_pos = np.max(diff[1:3, :]) # r, phi
    max_diff_vel = np.max(diff[4:6, :]) # ur, uphi
    
    print(f"Max Position Difference (loose vs strict): {max_diff_pos:.2e}")
    print(f"Max Velocity Difference (loose vs strict): {max_diff_vel:.2e}")
    print("Robustness check complete. Strict tolerance will be used for dataset generation.\n")

def generate_datasets(save_dir="data"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    datasets = {}
    
    # 1. Bound Orbit
    # L = 4 (uphi = 0.04 at r=10), E = 0.963
    # Oscillates between periapsis and apoapsis
    print("Generating Bound Orbit...")
    state_bound = get_initial_state(r0=10.0, ur0=0.0, uphi0=0.04)
    sol_bound = solve_geodesic(state_bound, [0, 500], num_points=1000, rtol=1e-10, atol=1e-12)
    datasets['bound'] = sol_bound
    
    # 2. Strong-field Orbit (near photon sphere)
    # L = 4, unstable circular orbit is at r=4. We start at r=4.01 to see the instability.
    print("Generating Strong-field Orbit...")
    state_strong = get_initial_state(r0=4.01, ur0=0.0, uphi0=4.0/(4.01**2))
    sol_strong = solve_geodesic(state_strong, [0, 200], num_points=1000, rtol=1e-10, atol=1e-12)
    datasets['strong_field'] = sol_strong
    
    # 3. Scattering Trajectory
    # Starts far, comes in, bends, leaves
    print("Generating Scattering Trajectory...")
    # L = 15 => uphi = 15/50^2 = 0.006. ur0 is negative to move inwards
    state_scatter = get_initial_state(r0=50.0, ur0=-0.23, uphi0=0.006)
    sol_scatter = solve_geodesic(state_scatter, [0, 300], num_points=1000, rtol=1e-10, atol=1e-12)
    datasets['scattering'] = sol_scatter
    
    # 4. Near-capture Trajectory
    # Starts far, L=4, enough energy to cross the barrier at r=4
    print("Generating Near-capture Trajectory...")
    state_capture = get_initial_state(r0=20.0, ur0=-0.25, uphi0=4.0/400.0)
    sol_capture = solve_geodesic(state_capture, [0, 150], num_points=1000, rtol=1e-10, atol=1e-12)
    datasets['capture'] = sol_capture
    
    # Save to disk
    with open(os.path.join(save_dir, "trajectories.pkl"), "wb") as f:
        pickle.dump(datasets, f)
    
    print(f"Saved {len(datasets)} trajectories to {save_dir}/trajectories.pkl")
    return datasets

if __name__ == "__main__":
    robustness_check()
    generate_datasets()
