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
            layers.append(nn.Tanh())
            
        # Output layer: t, r, phi (3D)
        layers.append(nn.Linear(neurons_per_layer, 3))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, lam):
        # lam should be shape (N, 1)
        return self.network(lam)

def compute_physics_loss(pinn, lam):
    """
    Computes the physics loss (residuals of geodesic equations)
    lam: shape (N, 1), requires_grad=True
    """
    # 1. Forward pass
    outputs = pinn(lam)
    t = outputs[:, 0:1]
    r = outputs[:, 1:2]
    phi = outputs[:, 2:3]
    
    # 2. First derivatives (velocities)
    # create_graph=True is required to compute higher-order derivatives
    ones = torch.ones_like(lam)
    
    dt_dlam = torch.autograd.grad(t, lam, grad_outputs=ones, create_graph=True)[0]
    dr_dlam = torch.autograd.grad(r, lam, grad_outputs=ones, create_graph=True)[0]
    dphi_dlam = torch.autograd.grad(phi, lam, grad_outputs=ones, create_graph=True)[0]
    
    ut, ur, uphi = dt_dlam, dr_dlam, dphi_dlam
    
    # 3. Second derivatives (accelerations)
    dut_dlam = torch.autograd.grad(ut, lam, grad_outputs=ones, create_graph=True)[0]
    dur_dlam = torch.autograd.grad(ur, lam, grad_outputs=ones, create_graph=True)[0]
    duphi_dlam = torch.autograd.grad(uphi, lam, grad_outputs=ones, create_graph=True)[0]
    
    # 4. Geodesic Equation Residuals
    # Using the equations from physics.py
    # Prevent division by zero or negative radii
    # We add a small epsilon or mask invalid regions if necessary, 
    # but the training domain should be r > 2.0.
    
    # dut_dlam = -2.0 * (1.0 / (r * (r - 2.0))) * ut * ur
    res_t = dut_dlam + 2.0 * (1.0 / (r * (r - 2.0))) * ut * ur
    
    # dur_dlam = - ((r - 2.0) / r**3) * ut**2 + (1.0 / (r * (r - 2.0))) * ur**2 + (r - 2.0) * uphi**2
    res_r = dur_dlam + ((r - 2.0) / r**3) * ut**2 - (1.0 / (r * (r - 2.0))) * ur**2 - (r - 2.0) * uphi**2
    
    # duphi_dlam = -2.0 * (1.0 / r) * ur * uphi
    res_phi = duphi_dlam + 2.0 * (1.0 / r) * ur * uphi
    
    physics_loss = torch.mean(res_t**2) + torch.mean(res_r**2) + torch.mean(res_phi**2)
    
    # 5. Normalization Constraint (optional loss term, but we track it)
    # -f(r) ut^2 + (1/f(r)) ur^2 + r^2 uphi^2 = -1
    # f_val = 1.0 - 2.0 / r
    # norm = -f_val * ut**2 + (1.0 / f_val) * ur**2 + r**2 * uphi**2
    # norm_loss = torch.mean((norm + 1.0)**2)
    # We could add norm_loss to physics_loss, but let's stick to the geodesic ODE residuals.
    
    return physics_loss

def get_data_loss(pinn, lam_data, target_data):
    """
    Computes MSE loss against ground truth RK45 data.
    lam_data: sampled affine parameters (M, 1)
    target_data: sampled positions (M, 3) for [t, r, phi]
    """
    predictions = pinn(lam_data)
    data_loss = nn.MSELoss()(predictions, target_data)
    return data_loss

def get_total_loss(pinn, lam_collocation, lam_data, target_data, alpha):
    """
    Total Loss = alpha * L_physics + (1 - alpha) * L_data
    lam_collocation: dense points for physics loss
    lam_data, target_data: subset of points for data loss
    """
    L_physics = compute_physics_loss(pinn, lam_collocation)
    L_data = get_data_loss(pinn, lam_data, target_data)
    
    total_loss = alpha * L_physics + (1.0 - alpha) * L_data
    return total_loss, L_physics, L_data
