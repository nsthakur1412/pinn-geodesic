import torch
import torch.nn as nn
from physics import f

class GeodesicPINN(nn.Module):
    def __init__(self, hidden_layers=5, neurons_per_layer=64):
        super().__init__()
        
        layers = []
        # Input layer: lambda (1D)
        layers.append(nn.Linear(1, neurons_per_layer))
        layers.append(nn.Tanh())
        
        # Hidden layers
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(neurons_per_layer, neurons_per_layer))
            layers.append(nn.SiLU())
            
        # Output layer: t, r, phi (3D)
        layers.append(nn.Linear(neurons_per_layer, 3))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, lam):
        # lam should be shape (N, 1)
        out = lam
        for layer in self.network[:-1]:
            out = layer(out)
        out = self.network[-1](out) # [t, r, phi]
        
        t = out[:, 0:1]
        # Coordinate Bound: Enforce r > 2.1 to aggressively avoid the coordinate singularity
        r = 2.1 + torch.nn.functional.softplus(out[:, 1:2])
        phi = out[:, 2:3]
        
        return torch.cat([t, r, phi], dim=1)

def compute_physics_loss(pinn, lam, lam_scale=1.0):
    """
    Computes the physics loss (residuals of geodesic equations)
    lam: shape (N, 1), requires_grad=True
    lam_scale: normalization factor applied to lam. Chain rule requires dividing derivatives by this scale.
    """
    # 1. Forward pass
    outputs = pinn(lam)
    t = outputs[:, 0:1]
    r = outputs[:, 1:2]
    phi = outputs[:, 2:3]
    
    # 2. First derivatives (velocities)
    # create_graph=True is required to compute higher-order derivatives
    ones = torch.ones_like(lam)
    
    dt_dlam = torch.autograd.grad(t, lam, grad_outputs=ones, create_graph=True)[0] / lam_scale
    dr_dlam = torch.autograd.grad(r, lam, grad_outputs=ones, create_graph=True)[0] / lam_scale
    dphi_dlam = torch.autograd.grad(phi, lam, grad_outputs=ones, create_graph=True)[0] / lam_scale
    
    ut, ur, uphi = dt_dlam, dr_dlam, dphi_dlam
    
    # 3. Second derivatives (accelerations)
    dut_dlam = torch.autograd.grad(ut, lam, grad_outputs=ones, create_graph=True)[0] / lam_scale
    dur_dlam = torch.autograd.grad(ur, lam, grad_outputs=ones, create_graph=True)[0] / lam_scale
    duphi_dlam = torch.autograd.grad(uphi, lam, grad_outputs=ones, create_graph=True)[0] / lam_scale
    
    # 4. Geodesic Equation Residuals
    # Add epsilon to prevent division by zero near singularity
    denom = r * (r - 2.0)
    denom = torch.where(denom.abs() < 1e-4, torch.sign(denom) * 1e-4 + 1e-6, denom)
    
    res_t = dut_dlam + 2.0 * (1.0 / denom) * ut * ur
    res_r = dur_dlam + ((r - 2.0) / (r**3 + 1e-6)) * ut**2 - (1.0 / denom) * ur**2 - (r - 2.0) * uphi**2
    res_phi = duphi_dlam + 2.0 * (1.0 / (r + 1e-6)) * ur * uphi
    
    physics_loss = (torch.mean(res_t**2) + torch.mean(res_r**2) + torch.mean(res_phi**2)) / 3.0
    return physics_loss

def get_data_loss(pinn, lam_data, target_pos):
    """
    Computes MSE loss against ground truth RK45 position data.
    lam_data: sampled affine parameters (M, 1)
    target_pos: sampled positions (M, 3) for [t, r, phi]
    """
    predictions = pinn(lam_data)
    data_loss = nn.MSELoss()(predictions, target_pos)
    return data_loss

def get_total_loss(pinn, lam_collocation, lam_data, target_data, alpha, lam_scale=1.0):
    """
    Total Loss = alpha * L_physics + (1 - alpha) * L_data
    lam_collocation: dense points for physics loss
    lam_data: subset of points for data loss
    target_data: subset of target states (M, 6) containing [t, r, phi, ut, ur, uphi]
    """
    L_physics = compute_physics_loss(pinn, lam_collocation, lam_scale)
    
    target_pos = target_data[:, 0:3]
    target_vel = target_data[:, 3:6]
    
    L_data = get_data_loss(pinn, lam_data, target_pos)
    
    # IC enforcement: pin prediction at lambda=0 to ground truth IC
    ic_lam = lam_collocation[0:1]
    ic_pos_pred = pinn(ic_lam.detach().requires_grad_(False))
    ic_pos_loss = torch.mean((ic_pos_pred - target_pos[0:1]) ** 2)
    
    # Velocity IC
    ones = torch.ones_like(ic_lam)
    ic_lam_eval = ic_lam.clone().detach().requires_grad_(True)
    ic_pos_pred_eval = pinn(ic_lam_eval)
    
    dt_dlam = torch.autograd.grad(ic_pos_pred_eval[:, 0:1], ic_lam_eval, grad_outputs=ones, retain_graph=True, create_graph=True)[0] / lam_scale
    dr_dlam = torch.autograd.grad(ic_pos_pred_eval[:, 1:2], ic_lam_eval, grad_outputs=ones, retain_graph=True, create_graph=True)[0] / lam_scale
    dphi_dlam = torch.autograd.grad(ic_pos_pred_eval[:, 2:3], ic_lam_eval, grad_outputs=ones, create_graph=True)[0] / lam_scale
    
    ic_vel_pred = torch.cat([dt_dlam, dr_dlam, dphi_dlam], dim=1)
    ic_vel_loss = torch.mean((ic_vel_pred - target_vel[0:1]) ** 2)
    
    ic_loss = ic_pos_loss + ic_vel_loss
    
    total_loss = alpha * (L_physics + ic_loss) + (1.0 - alpha) * L_data
    return total_loss, L_physics, L_data
