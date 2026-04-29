import torch
import time
from experiments import prepare_training_data, train_pinn
import pickle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open("data/trajectories.pkl", "rb") as f:
    datasets = pickle.load(f)

sol = datasets['bound']
lam_coll, lam_data, target_data, lam_full = prepare_training_data(sol, sample_ratio=0.1)

print(f"Testing 100 epochs on {device}...")
start = time.time()
pinn, history, t_time = train_pinn(lam_coll, lam_data, target_data, alpha=0.5, epochs=100)
end = time.time()

print(f"Finished in {end - start:.2f} seconds.")
