import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

def load_pareto_data(results_dir="results"):
    filepath = os.path.join(results_dir, "pareto_sweep.pkl")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"{filepath} not found. Ensure experiments.py has completed the large-scale sweep.")
    
    with open(filepath, "rb") as f:
        results = pickle.load(f)
    return results

def extract_pareto_front(l_data, l_phys):
    """
    Computes the non-dominated set (Pareto Front).
    l_data and l_phys should be 1D numpy arrays of the same length.
    Returns the indices of the Pareto-optimal points.
    """
    is_pareto = np.ones(l_data.shape[0], dtype=bool)
    for i in range(l_data.shape[0]):
        # Point i is dominated if any other point j has lower/equal data AND lower/equal phys,
        # with at least one being strictly lower.
        for j in range(l_data.shape[0]):
            if i == j:
                continue
            if l_data[j] <= l_data[i] and l_phys[j] <= l_phys[i]:
                if l_data[j] < l_data[i] or l_phys[j] < l_phys[i]:
                    is_pareto[i] = False
                    break
    return np.where(is_pareto)[0]

def plot_pareto_analysis(results, save_dir="plots"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    alphas = np.array([r['alpha'] for r in results])
    final_phys = np.array([r['final_phys'] for r in results])
    final_data = np.array([r['final_data'] for r in results])
    maes = np.array([r['mae'] for r in results])
    
    # 1. Normalization (by Median to handle extreme outliers gracefully)
    norm_phys = final_phys / np.median(final_phys)
    norm_data = final_data / np.median(final_data)
    
    # Remove outlier runs (unconverged or diverged) before Pareto extraction
    p75_phys = np.percentile(final_phys, 75)
    p75_data = np.percentile(final_data, 75)
    valid_mask = (final_phys < 5 * p75_phys) & (final_data < 5 * p75_data)
    
    norm_phys_f = norm_phys[valid_mask]
    norm_data_f = norm_data[valid_mask]
    alphas_f = alphas[valid_mask]
    maes_f = maes[valid_mask]
    
    # 2. Extract Pareto Front
    pareto_indices = extract_pareto_front(norm_data_f, norm_phys_f)
    
    # Sort the Pareto points by L_data to prevent zig-zag lines
    pareto_data = norm_data_f[pareto_indices]
    pareto_phys = norm_phys_f[pareto_indices]
    pareto_alphas = alphas_f[pareto_indices]
    
    sort_idx = np.argsort(pareto_data)
    pareto_data = pareto_data[sort_idx]
    pareto_phys = pareto_phys[sort_idx]
    pareto_alphas = pareto_alphas[sort_idx]
    
    # 3. Create the Main Trade-off Plot
    fig, ax1 = plt.subplots(figsize=(10, 8))
    
    # Scatter all runs (faint, background)
    # Map marker size to Trajectory Error (MAE). 
    # Smaller MAE = larger marker, or just use MAE directly for size, but inverse is better?
    # Let's map MAE to size: size = 200 / (1 + MAE*100) or just proportional to a base size.
    # To make low error stand out:
    sizes = 50 + 200 * np.exp(-10 * maes) 
    
    scatter = ax1.scatter(norm_data, norm_phys, c=alphas, cmap='viridis', 
                          s=sizes, alpha=0.3, edgecolors='none', label='All Runs')
    
    # Plot Pareto Front (Bold, Monotonic)
    ax1.plot(pareto_data, pareto_phys, 'k-', linewidth=2.5, zorder=4, label='Pareto Front')
    
    # Highlight Pareto points
    ax1.scatter(pareto_data, pareto_phys, c=pareto_alphas, cmap='viridis', 
                s=150, edgecolors='black', linewidths=1.5, zorder=5, label='Pareto Optimal')
    
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Normalized Data Loss (L_data / median)')
    ax1.set_ylabel('Normalized Physics Loss (L_physics / median)')
    ax1.set_title('Research-Grade Loss Trade-off Analysis (Pareto Front)')
    
    # Add colorbar
    cbar = fig.colorbar(scatter, ax=ax1)
    cbar.set_label('Alpha (Physics Weight)')
    
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    ax1.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "research_pareto_front.png"), dpi=300)
    plt.close()
    
    # 4. Statistical Performance Context Plot (Alpha vs MAE)
    unique_alphas = np.unique(alphas)
    mean_maes = []
    std_maes = []
    
    for a in unique_alphas:
        idx = (alphas == a)
        a_maes = maes[idx]
        mean_maes.append(np.mean(a_maes))
        std_maes.append(np.std(a_maes))
        
    mean_maes = np.array(mean_maes)
    std_maes = np.array(std_maes)
    
    plt.figure(figsize=(8, 6))
    plt.plot(unique_alphas, mean_maes, 'b-', linewidth=2, label='Mean Trajectory MAE')
    plt.fill_between(unique_alphas, mean_maes - std_maes, mean_maes + std_maes, color='blue', alpha=0.2, label='±1 Std Dev')
    
    # Mark the alpha values that appeared in the Pareto front
    pareto_unique_alphas = np.unique(pareto_alphas)
    for pa in pareto_unique_alphas:
        plt.axvline(x=pa, color='r', linestyle='--', alpha=0.5)
        
    plt.xlabel('Alpha (Physics Weight)')
    plt.ylabel('Trajectory Mean Absolute Error (MAE)')
    plt.title('Performance Context: Alpha vs Trajectory Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Annotate Pareto lines
    plt.text(0.05, 0.95, 'Red dashed lines indicate alphas\nfound on the Pareto Front', 
             transform=plt.gca().transAxes, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
             
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "performance_vs_alpha.png"), dpi=300)
    plt.close()
    
    # 5. Conserved Quantity Stability Plot
    if 'energy_drift' in results[0]:
        energy_drifts = np.array([r['energy_drift'] for r in results])
        
        unique_alphas = np.unique(alphas)
        mean_energy_drift = []
        std_energy_drift = []
        
        for a in unique_alphas:
            idx = (alphas == a)
            a_drifts = energy_drifts[idx]
            mean_energy_drift.append(np.mean(a_drifts))
            std_energy_drift.append(np.std(a_drifts))
            
        mean_energy_drift = np.array(mean_energy_drift)
        std_energy_drift = np.array(std_energy_drift)
        
        plt.figure(figsize=(8, 5))
        for a in unique_alphas:
            idx = (alphas == a)
            plt.scatter([a]*idx.sum(), energy_drifts[idx], alpha=0.4, s=30, color='steelblue')
        
        plt.plot(unique_alphas, mean_energy_drift, 'b-', linewidth=2, label='Mean |ΔE/E|')
        plt.fill_between(unique_alphas, mean_energy_drift - std_energy_drift,
                         mean_energy_drift + std_energy_drift, alpha=0.2, color='blue')
        
        plt.xlabel('Alpha (physics weight)')
        plt.ylabel('Relative energy drift |ΔE/E|')
        plt.title('Physical consistency vs alpha trade-off')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'conserved_quantity_stability.png'), dpi=300)
        plt.close()
    
    print("Pareto analysis plots generated successfully.")

if __name__ == "__main__":
    try:
        results = load_pareto_data()
        plot_pareto_analysis(results)
    except FileNotFoundError as e:
        print(e)
