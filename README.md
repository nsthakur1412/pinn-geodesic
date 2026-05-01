# Unified Geodesic PINN 🕳️

A Physics-Informed Neural Network (PINN) for learning timelike geodesics in Schwarzschild spacetime (M=1, equatorial plane). This parameterized PINN (DeepONet-style) enforces general relativity directly into the loss function — geodesic equation residuals, energy/angular momentum conservation, and the Hamiltonian constraint — enabling accurate prediction and extrapolation of bound, escape, and plunging trajectories from a single unified architecture.

## Key Features
- **Hard IC Enforcement**: Output reparametrization guarantees exact initial conditions at λ=0 by construction
- **Conservation Laws**: Energy (E), angular momentum (L), and Hamiltonian (g_μν u^μ u^ν = -1) enforced as loss terms
- **Curriculum Learning**: 3-phase training progressively shifts from data-fitting to physics-dominance
- **Gradient Clipping**: Prevents exploding gradients from second-order autograd
- **Comprehensive Evaluation**: Trajectory accuracy, conservation violation, phase space, and efficiency benchmarks

## 🚀 Project Architecture

| File | Purpose |
|------|---------|
| `physics.py` | Schwarzschild geodesic ODEs, conserved quantities |
| `rk45_solver.py` | Ground-truth RK45 numerical solver |
| `unified_dataset.py` | Generate 150 diverse parameterized trajectories |
| `unified_model.py` | DeepONet PINN architecture + physics/conservation loss |
| `unified_train.py` | Training loop with curriculum learning |
| `unified_eval.py` | Full evaluation suite (5 metrics + visualizations) |

## ⚙️ How to Use

### 0. Activate Environment
```bash
conda activate pinn_gr
```

### 1. Generate the Dataset
```bash
python unified_dataset.py
```
Generates 150 trajectories (50 bound + 50 escape + 50 capture) saved to `data/unified_dataset.pt`.

### 2. Train the Model
```bash
python unified_train.py
```
Training uses 3 curriculum phases:
- **Phase 1 (0-500)**: Data-dominant (λ_data=10, λ_phys=1)
- **Phase 2 (500-1500)**: Balanced (λ_data=5, λ_phys=5)
- **Phase 3 (1500-3000)**: Physics-dominant (λ_data=1, λ_phys=10)

Live loss plots are saved to `plots/live_unified_loss.png`. If `results/unified_pinn.pt` exists, training resumes from that checkpoint. Delete it to train from scratch.

### 3. Evaluate
```bash
python unified_eval.py
```
Runs 5 evaluation modules:
1. **Trajectory Accuracy** — MAE, RMSE, max deviation vs RK45
2. **Conservation Violation** — ΔE/E₀, ΔL/L₀, |H+1| over proper time
3. **Phase Space & Effective Potential** — (r, dr/dλ) portraits
4. **Long-term Stability** — Error growth rate (Lyapunov exponent)
5. **Computational Efficiency** — PINN vs RK45 speedup benchmarks

All plots saved to `plots/`.

## 🔬 Loss Function

$$L_{total} = \lambda_{phys} L_{geodesic} + \lambda_{conserv} L_{conservation} + \lambda_{data} L_{data}$$

Where:
- $L_{geodesic}$: Residuals of the 3 geodesic ODEs (second derivatives)
- $L_{conservation} = L_E + L_L + L_H$: Conservation of energy, angular momentum, Hamiltonian
- $L_{data}$: MSE against RK45 ground truth
