import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
from scipy.spatial.distance import jensenshannon
from scipy.special import rel_entr
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform, cdist
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

def compute_energy_distance(X, Y):
    """
    Compute the energy distance between two sets of points X and Y.
    
    Parameters:
    X, Y: numpy arrays of shape (n_points, n_dimensions)
    
    Returns:
    float: energy distance between X and Y
    """
    nx = len(X)
    ny = len(Y)
    
    XX = cdist(X, X)
    YY = cdist(Y, Y)
    XY = cdist(X, Y)[:nx*ny]
    
    # Calculate energy distance
    term1 = 2 * np.mean(XY)
    term2 = np.mean(XX)
    term3 = np.mean(YY)
    
    energy_dist = np.sqrt(2 * term1 - term2 - term3)
    energy_values = np.sqrt(np.abs(2 * XY - np.mean(XX) - np.mean(YY)))  # Compute per-pair distances
    std_dev = np.std(energy_values, ddof=1)  # Sample standard deviation

    return energy_dist, std_dev

def jensenshannon_distance(df_subset1 : pd.DataFrame, df_subset2: pd.DataFrame, model_name="model"):
    """
    Compute and visualize the Jensen-Shannon distance and KL divergence between two subsets of data.
    Parameters
    ----------
    df_subset1
    df_subset2
    model_name

    Returns
    -------

    """
    subset1_data = df_subset1.dropna().values
    subset2_data = df_subset2.dropna().values
    
    # Compute energy distance between the raw trajectories
    energy_dist = compute_energy_distance(subset1_data, subset2_data)
    print(f"Energy Distance between the two subsets: {energy_dist}")

    subset1_data = df_subset1[['latitude', 'longitude']].dropna().values
    subset2_data = df_subset2[['latitude', 'longitude']].dropna().values
    
    # Kernel Density Estimation (KDE) for both subsets
    kde_subset1 = gaussian_kde(subset1_data.T)
    kde_subset2 = gaussian_kde(subset2_data.T)
    
    # Create grid to evaluate KDEs over a common region (latitude, longitude)
    xgrid, ygrid = np.mgrid[
        min(subset1_data[:, 0].min(), subset2_data[:, 0].min()):max(subset1_data[:, 0].max(), subset2_data[:, 0].max()):100j,
        min(subset1_data[:, 1].min(), subset2_data[:, 1].min()):max(subset1_data[:, 1].max(), subset2_data[:, 1].max()):100j
    ]
    grid_coords = np.vstack([xgrid.ravel(), ygrid.ravel()])
    
    # Evaluate the KDEs on the grid
    subset1_density = kde_subset1(grid_coords).reshape(100, 100)
    subset2_density = kde_subset2(grid_coords).reshape(100, 100)
    
    # Normalize densities to ensure they sum to 1 (turn them into probabilities)
    subset1_density /= np.sum(subset1_density)
    subset2_density /= np.sum(subset2_density)
    
    # Add a small constant to avoid zeros in the densities
    epsilon = 1e-10
    subset1_density += epsilon
    subset2_density += epsilon
    
    # Compute the average distribution M
    M = 0.5 * (subset1_density + subset2_density)
    
    # Calculate Jensen-Shannon distance using the scipy.spatial.distance.jensenshannon method
    js_distance = jensenshannon(subset1_density.ravel(), subset2_density.ravel(), base=2)
    kl_divergence = np.sum(rel_entr(subset1_density, subset2_density))
    
    print(f"KL Divergence between the two subsets: {kl_divergence}")
    print(f"Jensen-Shannon Distance between the two subsets: {js_distance}")
    
    # Plotting the KDEs for comparison
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(subset1_density, origin='lower', cmap='Blues', 
                 extent=[xgrid.min(), xgrid.max(), ygrid.min(), ygrid.max()])
    ax[0].set_title("Subset 1 Density")
    ax[1].imshow(subset2_density, origin='lower', cmap='Reds', 
                 extent=[xgrid.min(), xgrid.max(), ygrid.min(), ygrid.max()])
    ax[1].set_title("Subset 2 Density")
    plt.tight_layout()
    plt.savefig(f"./figures/{model_name}_comparison.png")
    
    return js_distance, kl_divergence, energy_dist, fig


def compute_dtw_3d_batch(traj1, traj2):
    """
    Computes the DTW distance and standard deviation for batches of 3D trajectories.

    Args:
        trajs1 (np.ndarray): First batch of trajectories, shape (N, T1, 3).
        trajs2 (np.ndarray): Second batch of trajectories, shape (N, T2, 3).

    Returns:
        np.ndarray: DTW distances for each trajectory pair (N,).
        np.ndarray: Standard deviation of DTW alignment distances for each pair (N,).
        list: List of DTW paths for each trajectory pair.
    """
    trajs1 = traj1.reshape(-1, 200, 2)
    trajs2 = traj2.reshape(-1, 200, 2)
    
    assert trajs1.shape[0] == trajs2.shape[0], "Both trajectory batches must have the same number of samples (N)."

    N = trajs1.shape[0]
    dtw_distances = np.zeros(N)
    dtw_stds = np.zeros(N)
    dtw_paths = []

    for i in range(N):
        traj1, traj2 = trajs1[i], trajs2[i]

        distance, path = fastdtw(traj1, traj2, dist=euclidean)
        alignment_distances = np.array([euclidean(traj1[i], traj2[j]) for i, j in path])
        std_dev = np.std(alignment_distances)

        # Store results
        dtw_distances[i] = distance
        dtw_stds[i] = std_dev
        dtw_paths.append(path)
    
    return dtw_distances.mean(), dtw_stds.mean(), dtw_paths


