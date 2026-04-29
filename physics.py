import numpy as np

# Schwarzschild metric, equatorial plane (theta = pi/2)
# Units: G = c = M = 1

def f(r):
    """Metric component f(r) = 1 - 2M/r = 1 - 2/r"""
    return 1.0 - 2.0 / r

def geodesic_odes(lam, state):
    """
    Computes the derivatives of the state variables with respect to affine parameter lambda.
    state = [t, r, phi, ut, ur, uphi]
    where ut = dt/dlam, ur = dr/dlam, uphi = dphi/dlam.
    """
    t, r, phi, ut, ur, uphi = state
    
    # Prevent numerical issues exactly at or inside the event horizon for this solver
    if r <= 2.001:
        # In a real scenario we might use Kruskal coordinates, but for our comparison
        # we focus on r > 2M and stop integration if it hits the horizon.
        return [ut, ur, uphi, 0.0, 0.0, 0.0]
        
    f_val = f(r)
    
    # First derivatives
    dt_dlam = ut
    dr_dlam = ur
    dphi_dlam = uphi
    
    # Second derivatives (from geodesic equation: d^2x^mu / dlam^2 = - Gamma^mu_{alpha beta} u^alpha u^beta)
    # Gamma^t_{tr} = 1 / (r * (r - 2))
    dut_dlam = -2.0 * (1.0 / (r * (r - 2.0))) * ut * ur
    
    # Gamma^r_{tt} = (r - 2) / r^3
    # Gamma^r_{rr} = -1 / (r * (r - 2))
    # Gamma^r_{\phi\phi} = -(r - 2)
    dur_dlam = - ((r - 2.0) / r**3) * ut**2 + (1.0 / (r * (r - 2.0))) * ur**2 + (r - 2.0) * uphi**2
    
    # Gamma^\phi_{r\phi} = 1/r
    duphi_dlam = -2.0 * (1.0 / r) * ur * uphi
    
    return [dt_dlam, dr_dlam, dphi_dlam, dut_dlam, dur_dlam, duphi_dlam]

def compute_conserved_quantities(state):
    """
    Computes Energy (E), Angular Momentum (L), and 4-velocity normalization constraint.
    state can be a 1D array or a 2D array of shape (N, 6)
    """
    state = np.atleast_2d(state)
    t = state[:, 0]
    r = state[:, 1]
    phi = state[:, 2]
    ut = state[:, 3]
    ur = state[:, 4]
    uphi = state[:, 5]
    
    f_val = f(r)
    
    # Energy E = (1 - 2M/r) ut
    E = f_val * ut
    
    # Angular Momentum L = r^2 uphi
    L = r**2 * uphi
    
    # Normalization: g_mu_nu u^mu u^nu
    # g_tt = -f(r), g_rr = 1/f(r), g_phiphi = r^2
    norm = -f_val * ut**2 + (1.0 / f_val) * ur**2 + r**2 * uphi**2
    
    return E, L, norm
