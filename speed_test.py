import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import time
from scientific_framework import ScientificMLP

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Testing speed on {device}")

inputs, targets = torch.load("data/unified_dataset.pt", weights_only=False)
dataset = TensorDataset(inputs.to(device), targets.to(device))
dataloader = DataLoader(dataset, batch_size=4096, shuffle=True)

model = ScientificMLP(hidden_layers=6, neurons_per_layer=256).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

t0 = time.time()
model.train()
for batch_inputs, batch_targets in dataloader:
    optimizer.zero_grad()
    out_norm, _ = model(batch_inputs)
    loss = nn.MSELoss()(out_norm, batch_targets[:, 0:3])
    loss.backward()
    optimizer.step()
t1 = time.time()
print(f"Epoch 1 (Data-only) took: {t1-t0:.2f} seconds")

# Test Physics Loss
t0 = time.time()
from scientific_framework import compute_physics_metrics
batch_inputs, _ = next(iter(dataloader))
p_loss, _, _, _, _, _, _ = compute_physics_metrics(model, batch_inputs)
p_loss.backward()
t1 = time.time()
print(f"Physics Batch (4096) took: {t1-t0:.2f} seconds")
