import pickle
import os

if os.path.exists("results/pareto_sweep.pkl"):
    with open("results/pareto_sweep.pkl", "rb") as f:
        data = pickle.load(f)
    print(f"\n--- SUCCESS ---")
    print(f"Completed models: {len(data)} / 105")
    print(f"-----------------\n")
else:
    print("\nThe first model is still training... no models have finished yet.\n")
