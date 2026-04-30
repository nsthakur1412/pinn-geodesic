# Unified Geodesic PINN 🕳️

A Physics-Informed Neural Network (PINN) architecture designed to learn the geodesic equations for particle motion around a Schwarzschild black hole. Unlike standard neural networks that simply fit a curve, this parameterized PINN (DeepONet architecture) enforces the underlying general relativity physics (energy conservation, angular momentum conservation, and the equations of motion) directly into the loss function, enabling it to accurately model and extrapolate bound, escape, and plunging trajectories from a single unified architecture.

## 🚀 Project Architecture

The core files required to run and experiment with the Unified PINN:

- **`physics.py`**: Contains the core calculations for the physical residuals (Geodesic equations, Hamiltonian constraint) in Schwarzschild geometry.
- **`rk45_solver.py`**: The ground-truth numerical ODE solver (Runge-Kutta 45) used to generate the dataset.
- **`unified_dataset.py`**: Script to generate a diverse set of parameterized initial conditions and solve them using `rk45_solver.py` to create the training dataset.
- **`unified_model.py`**: Defines the DeepONet PINN architecture and the function to calculate the unified physics loss tensor.
- **`unified_train.py`**: The training loop. Combines data loss, physics loss, and initial condition loss to optimize the network.
- **`unified_eval.py`**: Evaluates the trained model, comparing its predicted trajectories to the numerical ground truth and generating visualizations.

## ⚙️ How to Use

### 1. Generate the Dataset
Before training, you need to generate the ground-truth data. This script simulates various trajectories (bound, scattering, plunging) and saves them to `data/unified_dataset.pt`.

```bash
python unified_dataset.py
```

### 2. Train the Model
Once the dataset is generated, you can train the PINN. The script automatically handles loading the dataset, initializing the neural network, and applying the combined loss function ($L_{total} = \lambda_{phys} L_{phys} + \lambda_{data} L_{data} + \lambda_{ic} L_{ic}$).

```bash
python unified_train.py
```

> **Note:** If `results/unified_pinn.pt` already exists, the script will automatically load the pre-trained weights and resume training. If you want to train from scratch, delete that file first. Live training loss plots are saved to `plots/live_unified_loss.png`.

### 3. Evaluate and Visualize
To test the accuracy and generalization of your trained model, run the evaluation script. This will plot predicted vs actual trajectories and calculate Mean Absolute Error (MAE).

```bash
python unified_eval.py
```

Outputs will be saved in the `plots/` directory.

## 🔬 Tuning the Physics Loss

If you wish to experiment with how strongly the network enforces the laws of physics, modify the static weight parameters in `unified_train.py`:

```python
lambda_phys = 10.0  # Weight for Geodesic and Hamiltonian constraints
lambda_data = 1.0   # Weight for matching the ground truth data
lambda_ic = 5.0     # Weight for enforcing initial conditions
```
Increasing `lambda_phys` forces the network to adhere more strictly to relativity at the cost of being harder to optimize.
