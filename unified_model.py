import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle

class UnifiedPINN(nn.Module):
    def __init__(self, hidden_layers=5, neurons_per_layer=128):
        super().__init__()
        
        layers = []
        # Input: [lam_scaled, r0_norm, ur0_norm, L_norm] -> 4
        layers.append(nn.Linear(4, neurons_per_layer))
        layers.append(nn.SiLU())
        
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(neurons_per_layer, neurons_per_layer))
            layers.append(nn.SiLU())
            
        # Output: [t_norm, raw_r, phi_norm] -> 3
        layers.append(nn.Linear(neurons_per_layer, 3))
        
        self.network = nn.Sequential(*layers)
        
        # Load scalers so the model can map between normalized and physical space
        with open("data/unified_scalers.pkl", "rb") as f:
            self.scalers = pickle.load(f)
            
        # Initialize weights with Xavier (Glorot)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
            
    def forward(self, inputs):
        """
        inputs shape: (N, 4)
        Returns:
            out_norm: Normalized [t, r, phi] for MSE data loss
            out_phys: Physical [t, r, phi] for Geodesic physics loss
        """
        out = self.network(inputs)
        
        t_norm = out[:, 0:1]
        raw_r = out[:, 1:2]
        phi_norm = out[:, 2:3]
        
        # Stability constraint: Wall off the r=2M singularity
        r_phys = 2.1 + F.softplus(raw_r)
        
        device = inputs.device
        
        # Standardize physical r to match target data scales
        r_mean = torch.tensor(self.scalers['r_mean'], dtype=torch.float32, device=device)
        r_std = torch.tensor(self.scalers['r_std'], dtype=torch.float32, device=device)
        r_norm = (r_phys - r_mean) / r_std
        
        # Un-standardize t and phi to physical scales
        t_mean = torch.tensor(self.scalers['t_mean'], dtype=torch.float32, device=device)
        t_std = torch.tensor(self.scalers['t_std'], dtype=torch.float32, device=device)
        t_phys = t_norm * t_std + t_mean
        
        phi_mean = torch.tensor(self.scalers['phi_mean'], dtype=torch.float32, device=device)
        phi_std = torch.tensor(self.scalers['phi_std'], dtype=torch.float32, device=device)
        phi_phys = phi_norm * phi_std + phi_mean
        
        out_norm = torch.cat([t_norm, r_norm, phi_norm], dim=1)
        out_phys = torch.cat([t_phys, r_phys, phi_phys], dim=1)
        
        return out_norm, out_phys

def compute_unified_physics_loss(pinn, inputs):
    """
    Computes physics residuals by taking derivatives ONLY with respect to lambda.
    inputs: (N, 4) tensor containing [lam_scaled, r0_norm, ur0_norm, L_norm]
    """
    # Isolate lambda to compute gradients
    lam_scaled = inputs[:, 0:1].clone().detach().requires_grad_(True)
    other_inputs = inputs[:, 1:4].clone().detach() 
    
    model_inputs = torch.cat([lam_scaled, other_inputs], dim=1)
    
    _, out_phys = pinn(model_inputs)
    t = out_phys[:, 0:1]
    r = out_phys[:, 1:2]
    phi = out_phys[:, 2:3]
    
    lam_scale = pinn.scalers['lam_scale']
    ones = torch.ones_like(lam_scaled)
    
    # First derivatives (velocities)
    dt_dlam = torch.autograd.grad(t, lam_scaled, grad_outputs=ones, create_graph=True)[0] / lam_scale
    dr_dlam = torch.autograd.grad(r, lam_scaled, grad_outputs=ones, create_graph=True)[0] / lam_scale
    dphi_dlam = torch.autograd.grad(phi, lam_scaled, grad_outputs=ones, create_graph=True)[0] / lam_scale
    
    ut, ur, uphi = dt_dlam, dr_dlam, dphi_dlam
    
    # Second derivatives (accelerations)
    dut_dlam = torch.autograd.grad(ut, lam_scaled, grad_outputs=ones, create_graph=True)[0] / lam_scale
    dur_dlam = torch.autograd.grad(ur, lam_scaled, grad_outputs=ones, create_graph=True)[0] / lam_scale
    duphi_dlam = torch.autograd.grad(uphi, lam_scaled, grad_outputs=ones, create_graph=True)[0] / lam_scale
    
    # Geodesic Residuals
    denom = r * (r - 2.0)
    denom = torch.where(denom.abs() < 1e-4, torch.sign(denom) * 1e-4 + 1e-6, denom)
    
    res_t = dut_dlam + 2.0 * (1.0 / denom) * ut * ur
    res_r = dur_dlam + ((r - 2.0) / (r**3 + 1e-6)) * ut**2 - (1.0 / denom) * ur**2 - (r - 2.0) * uphi**2
    res_phi = duphi_dlam + 2.0 * (1.0 / (r + 1e-6)) * ur * uphi
    
    # Residual Balancing (dynamic self-scaling to prevent one dimension from overpowering)
    scale_t = torch.mean(res_t.detach()**2) + 1e-8
    scale_r = torch.mean(res_r.detach()**2) + 1e-8
    scale_phi = torch.mean(res_phi.detach()**2) + 1e-8
    
    physics_loss = (torch.mean(res_t**2) / scale_t + 
                    torch.mean(res_r**2) / scale_r + 
                    torch.mean(res_phi**2) / scale_phi) / 3.0
                    
    # Return both loss and the computed physical velocities for IC enforcement
    vel_phys = torch.cat([dt_dlam, dr_dlam, dphi_dlam], dim=1)
    return physics_loss, vel_phys
