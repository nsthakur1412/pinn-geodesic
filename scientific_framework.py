import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import os
import time
from rk45_solver import get_initial_state, solve_geodesic

# ============================================================
# ARCHITECTURE (Requirement 1: IDENTICAL)
# ============================================================
class ResidualBlock(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.linear = nn.Linear(size, size)
        self.silu = nn.SiLU()
        nn.init.xavier_normal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.silu(self.linear(x)) + x

class ScientificMLP(nn.Module):
    def __init__(self, hidden_layers=6, neurons_per_layer=256, use_residual=True):
        super().__init__()
        self.use_residual = use_residual
        
        # input dim = 4 (lam, r0, ur0, L)
        input_dim = 4
        
        if use_residual:
            self.input_layer = nn.Linear(input_dim, neurons_per_layer)
            self.input_silu = nn.SiLU()
            self.res_blocks = nn.ModuleList([
                ResidualBlock(neurons_per_layer) for _ in range(hidden_layers - 1)
            ])
            self.output_layer = nn.Linear(neurons_per_layer, 3)
        else:
            layers = []
            layers.append(nn.Linear(input_dim, neurons_per_layer))
            layers.append(nn.SiLU())
            for _ in range(hidden_layers - 1):
                layers.append(nn.Linear(neurons_per_layer, neurons_per_layer))
                layers.append(nn.SiLU())
            layers.append(nn.Linear(neurons_per_layer, 3))
            self.network = nn.Sequential(*layers)
        
        # Load scalers
        with open("data/unified_scalers.pkl", "rb") as f:
            self.scalers = pickle.load(f)
            
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, inputs):
        """
        Forward pass supporting both Residual and standard MLP.
        """
        if self.use_residual:
            x = self.input_silu(self.input_layer(inputs))
            for block in self.res_blocks:
                x = block(x)
            out_norm = self.output_layer(x)
        else:
            out_norm = self.network(inputs)
        
        # Un-standardize to get physical coordinates for physics loss
        device = inputs.device
        t_mean = torch.tensor(self.scalers['t_mean'], dtype=torch.float32, device=device)
        t_std = torch.tensor(self.scalers['t_std'], dtype=torch.float32, device=device)
        r_mean = torch.tensor(self.scalers['r_mean'], dtype=torch.float32, device=device)
        r_std = torch.tensor(self.scalers['r_std'], dtype=torch.float32, device=device)
        phi_mean = torch.tensor(self.scalers['phi_mean'], dtype=torch.float32, device=device)
        phi_std = torch.tensor(self.scalers['phi_std'], dtype=torch.float32, device=device)
        
        t_phys = out_norm[:, 0:1] * t_std + t_mean
        r_phys = out_norm[:, 1:2] * r_std + r_mean
        phi_phys = out_norm[:, 2:3] * phi_std + phi_mean
        
        out_phys = torch.cat([t_phys, r_phys, phi_phys], dim=1)
        return out_norm, out_phys

# ============================================================
# PHYSICS EVALUATION (Requirement 3: ALL experiments)
# ============================================================
def compute_physics_metrics(model, inputs):
    """
    Computes geodesic residuals and conserved quantities.
    Used for both training (PINN) and evaluation (All models).
    """
    lam_scaled = inputs[:, 0:1].clone().detach().requires_grad_(True)
    other_inputs = inputs[:, 1:4].clone().detach()
    model_inputs = torch.cat([lam_scaled, other_inputs], dim=1)
    
    out_norm, out_phys = model(model_inputs)
    t, r, phi = out_phys[:, 0:1], out_phys[:, 1:2], out_phys[:, 2:3]
    
    lam_scale = model.scalers['lam_scale']
    ones = torch.ones_like(lam_scaled)
    
    # Velocities
    ut = torch.autograd.grad(t, lam_scaled, grad_outputs=ones, create_graph=True)[0] / lam_scale
    ur = torch.autograd.grad(r, lam_scaled, grad_outputs=ones, create_graph=True)[0] / lam_scale
    uphi = torch.autograd.grad(phi, lam_scaled, grad_outputs=ones, create_graph=True)[0] / lam_scale
    
    # Accelerations
    dut_dlam = torch.autograd.grad(ut, lam_scaled, grad_outputs=ones, create_graph=True)[0] / lam_scale
    dur_dlam = torch.autograd.grad(ur, lam_scaled, grad_outputs=ones, create_graph=True)[0] / lam_scale
    duphi_dlam = torch.autograd.grad(uphi, lam_scaled, grad_outputs=ones, create_graph=True)[0] / lam_scale
    
    # Geodesic Residuals
    denom = r * (r - 2.0)
    denom = torch.where(denom.abs() < 1e-4, torch.sign(denom) * 1e-4 + 1e-6, denom)
    
    res_t = dut_dlam + 2.0 * (1.0 / denom) * ut * ur
    res_r = dur_dlam + ((r - 2.0) / (r**3 + 1e-6)) * ut**2 - (1.0 / denom) * ur**2 - (r - 2.0) * uphi**2
    res_phi = duphi_dlam + 2.0 * (1.0 / (r + 1e-6)) * ur * uphi
    
    phys_loss = torch.mean(res_t**2 + res_r**2 + res_phi**2)
    
    # Conserved Quantities
    f_r = 1.0 - 2.0 / (r + 1e-6)
    E = f_r * ut
    L = (r**2) * uphi
    H = -f_r * (ut**2) + (1.0 / (f_r + 1e-6)) * (ur**2) + (r**2) * (uphi**2)
    
    # Derivatives of Conserved Quantities (for Unified PINN conservation)
    dE_dlam = torch.autograd.grad(E, lam_scaled, grad_outputs=ones, create_graph=True)[0] / lam_scale
    dL_dlam = torch.autograd.grad(L, lam_scaled, grad_outputs=ones, create_graph=True)[0] / lam_scale
    
    return phys_loss, E, L, H, dE_dlam, dL_dlam, ut, ur, uphi

# ============================================================
# STANDARDIZED TEST SUITE (Requirement 2 & 6)
# ============================================================
TEST_SUITE = [
    {"name": "Bound",   "r0": 8.0,  "ur0": 0.0,   "L": 4.0},
    {"name": "Escape",  "r0": 15.0, "ur0": -0.1,  "L": 5.5},
    {"name": "Capture", "r0": 7.0,  "ur0": -0.2,  "L": 2.5},
]

EVAL_CACHE = {}

def run_standard_eval(model, device):
    """Evaluates the model on the 3 standard test cases and returns metrics."""
    model.eval()
    results = {}
    
    for case in TEST_SUITE:
        r0, ur0, L = case['r0'], case['ur0'], case['L']
        cache_key = (r0, ur0, L)
        
        if cache_key not in EVAL_CACHE:
            # RK45 Ground Truth
            uphi0 = L / (r0**2)
            state = get_initial_state(r0=r0, ur0=ur0, uphi0=uphi0)
            sol = solve_geodesic(state, [0, 300], num_points=1000)
            
            # Sample uniformly from the continuous solution (using dense_output)
            lam_true = np.linspace(0, sol.t[-1], 1000)
            y_true = sol.sol(lam_true) # [t, r, phi, ut, ur, uphi]
            EVAL_CACHE[cache_key] = (lam_true, y_true)
        
        lam_true, y_true = EVAL_CACHE[cache_key]
        
        # PINN Input
        scalers = model.scalers
        inp = np.zeros((len(lam_true), 4), dtype=np.float32)
        inp[:, 0] = lam_true / scalers['lam_scale']
        inp[:, 1] = (r0 - scalers['r0_mean']) / scalers['r0_std']
        inp[:, 2] = (ur0 - scalers['ur0_mean']) / scalers['ur0_std']
        inp[:, 3] = (L - scalers['L_mean']) / scalers['L_std']
        
        inp_t = torch.tensor(inp).to(device)
        
        # Evaluation with autograd for physics
        phys_loss, E, L_p, H, dE, dL, ut, ur, uphi = compute_physics_metrics(model, inp_t)
        _, out_phys = model(inp_t)
        pos_p = out_phys.detach().cpu().numpy()
        
        # Deviations
        r_true = y_true[1]
        phi_true = y_true[2]
        r_p = pos_p[:, 1]
        phi_p = pos_p[:, 2]
        
        x_true = r_true * np.cos(phi_true)
        y_t = r_true * np.sin(phi_true)
        x_p = r_p * np.cos(phi_p)
        y_p = r_p * np.sin(phi_p)
        
        max_dev = np.max(np.sqrt((x_true - x_p)**2 + (y_t - y_p)**2))
        
        # Drifts
        E_p = E.detach().cpu().numpy()
        L_val = L_p.detach().cpu().numpy()
        H_p = H.detach().cpu().numpy()
        
        E0, L0 = E_p[0], L_val[0]
        e_drift = np.mean(np.abs(E_p - E0) / (np.abs(E0) + 1e-8))
        l_drift = np.mean(np.abs(L_val - L0) / (np.abs(L0) + 1e-8))
        h_violation = np.mean(np.abs(H_p + 1.0))
        
        results[case['name']] = {
            'max_dev': max_dev,
            'e_drift': e_drift,
            'l_drift': l_drift,
            'h_violation': h_violation
        }
        
    return results

# ============================================================
# LOGGING (Requirement 4)
# ============================================================
def log_to_csv(data_dict, file_path="results/comparative_study.csv"):
    import csv
    fieldnames = [
        'experiment_type', 'epoch', 'total_loss', 'data_loss', 'phys_loss', 'ic_loss',
        'energy_drift', 'angular_momentum_drift', 'hamiltonian_violation',
        'bound_max_dev', 'escape_max_dev', 'capture_max_dev'
    ]
    
    file_exists = os.path.isfile(file_path)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(data_dict)
