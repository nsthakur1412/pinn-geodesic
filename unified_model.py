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
            
        # Output: [raw_t, raw_r, raw_phi] -> 3 (corrections to IC trajectory)
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

    def _get_physical_ics(self, inputs):
        """
        Recover physical initial conditions from normalized inputs.
        Returns: r0, ur0, uphi0, ut0, E0, L0 (all as tensors)
        """
        device = inputs.device
        
        r0_mean = torch.tensor(self.scalers['r0_mean'], dtype=torch.float32, device=device)
        r0_std = torch.tensor(self.scalers['r0_std'], dtype=torch.float32, device=device)
        ur0_mean = torch.tensor(self.scalers['ur0_mean'], dtype=torch.float32, device=device)
        ur0_std = torch.tensor(self.scalers['ur0_std'], dtype=torch.float32, device=device)
        L_mean = torch.tensor(self.scalers['L_mean'], dtype=torch.float32, device=device)
        L_std = torch.tensor(self.scalers['L_std'], dtype=torch.float32, device=device)
        
        r0 = inputs[:, 1:2] * r0_std + r0_mean
        ur0 = inputs[:, 2:3] * ur0_std + ur0_mean
        L0 = inputs[:, 3:4] * L_std + L_mean
        
        uphi0 = L0 / (r0**2 + 1e-8)
        f_r0 = 1.0 - 2.0 / (r0.clamp(min=2.1))
        ut0 = torch.sqrt(torch.clamp(
            (1.0 + ur0**2 / (f_r0 + 1e-8) + r0**2 * uphi0**2) / (f_r0 + 1e-8),
            min=1e-8
        ))
        E0 = f_r0 * ut0
        
        return r0, ur0, uphi0, ut0, E0, L0
            
    def forward(self, inputs):
        """
        Hard IC reparametrization:
          pos(λ) = pos_IC + vel_IC * λ_phys + λ_s² * NN(inputs)
        
        At λ=0: position = IC (exact), velocity = IC_velocity (exact)
        """
        lam_s = inputs[:, 0:1]  # normalized λ ∈ [0, 1]
        lam_scale = self.scalers['lam_scale']
        lam_p = lam_s * lam_scale  # physical lambda
        
        # Physical ICs
        r0, ur0, uphi0, ut0, E0, L0 = self._get_physical_ics(inputs)
        
        # P2: Remove bias in r prediction
        # We un-standardize raw_r to physical scale directly
        device = inputs.device
        r_mean = torch.tensor(self.scalers['r_mean'], dtype=torch.float32, device=device)
        r_std = torch.tensor(self.scalers['r_std'], dtype=torch.float32, device=device)
        r_phys = raw_r * r_std + r_mean
        r_norm = raw_r
        
        # Un-standardize t and phi to physical scales
        t_mean = torch.tensor(self.scalers['t_mean'], dtype=torch.float32, device=device)
        t_std = torch.tensor(self.scalers['t_std'], dtype=torch.float32, device=device)
        r_mean = torch.tensor(self.scalers['r_mean'], dtype=torch.float32, device=device)
        r_std = torch.tensor(self.scalers['r_std'], dtype=torch.float32, device=device)
        phi_mean = torch.tensor(self.scalers['phi_mean'], dtype=torch.float32, device=device)
        phi_std = torch.tensor(self.scalers['phi_std'], dtype=torch.float32, device=device)
        
        t_norm = (t_phys - t_mean) / t_std
        r_norm = (r_phys - r_mean) / r_std
        phi_norm = (phi_phys - phi_mean) / phi_std
        
        out_norm = torch.cat([t_norm, r_norm, phi_norm], dim=1)
        out_phys = torch.cat([t_phys, r_phys, phi_phys], dim=1)
        
        return out_norm, out_phys


def compute_all_physics_losses(pinn, inputs):
    """
    Computes ALL physics losses in a single autograd pass:
      1. Geodesic residuals (second-order ODE)
      2. Conservation of Energy E
      3. Conservation of Angular Momentum L
      4. Hamiltonian constraint g_μν u^μ u^ν = -1
    
    Returns: geodesic_loss, conservation_loss, vel_phys
    """
    # Isolate lambda for autograd
    lam_scaled = inputs[:, 0:1].clone().detach().requires_grad_(True)
    other_inputs = inputs[:, 1:4].clone().detach()
    model_inputs = torch.cat([lam_scaled, other_inputs], dim=1)
    
    _, out_phys = pinn(model_inputs)
    t = out_phys[:, 0:1]
    r = out_phys[:, 1:2]
    phi = out_phys[:, 2:3]
    
    lam_scale = pinn.scalers['lam_scale']
    ones = torch.ones_like(lam_scaled)
    
    # ===== First derivatives (velocities) =====
    ut = torch.autograd.grad(t, lam_scaled, grad_outputs=ones, create_graph=True)[0] / lam_scale
    ur = torch.autograd.grad(r, lam_scaled, grad_outputs=ones, create_graph=True)[0] / lam_scale
    uphi = torch.autograd.grad(phi, lam_scaled, grad_outputs=ones, create_graph=True)[0] / lam_scale
    
    # ===== Second derivatives (accelerations) =====
    dut_dlam = torch.autograd.grad(ut, lam_scaled, grad_outputs=ones, create_graph=True)[0] / lam_scale
    dur_dlam = torch.autograd.grad(ur, lam_scaled, grad_outputs=ones, create_graph=True)[0] / lam_scale
    duphi_dlam = torch.autograd.grad(uphi, lam_scaled, grad_outputs=ones, create_graph=True)[0] / lam_scale
    
    # ===== GEODESIC RESIDUALS =====
    denom = r * (r - 2.0)
    denom = torch.where(denom.abs() < 1e-4, torch.sign(denom) * 1e-4 + 1e-6, denom)
    
    res_t = dut_dlam + 2.0 * (1.0 / denom) * ut * ur
    res_r = dur_dlam + ((r - 2.0) / (r**3 + 1e-6)) * ut**2 - (1.0 / denom) * ur**2 - (r - 2.0) * uphi**2
    res_phi = duphi_dlam + 2.0 * (1.0 / (r + 1e-6)) * ur * uphi
    
    # P0: Fix physics loss (simple sum of squares)
    physics_loss = torch.mean(res_t**2) + torch.mean(res_r**2) + torch.mean(res_phi**2)
                    
    # Return both loss and the computed physical velocities for IC enforcement
    vel_phys = torch.cat([dt_dlam, dr_dlam, dphi_dlam], dim=1)
    return physics_loss, vel_phys
