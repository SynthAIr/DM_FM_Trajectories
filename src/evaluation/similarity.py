import numpy as np
from scipy.stats import gaussian_kde
from scipy.special import rel_entr
from scipy.spatial.distance import jensenshannon
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cdist
from tqdm.autonotebook import tqdm
import pandas as pd
import numpy as np
import random
import numpy as np
from scipy.stats import gaussian_kde
from scipy.spatial.distance import jensenshannon
from scipy.special import rel_entr
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from traffic.core import Traffic


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
    
    # Compute all pairwise distances
    XX = pdist(X)
    YY = pdist(Y)
    XY = pdist(np.vstack([X, Y]))[:nx*ny]
    
    # Calculate energy distance
    term1 = 2 * np.mean(XY)
    term2 = np.mean(XX)
    term3 = np.mean(YY)
    
    energy_dist = np.sqrt(2 * term1 - term2 - term3)
    return energy_dist

def jensenshannon_distance(df_subset1 : pd.DataFrame, df_subset2: pd.DataFrame, model_name="model"):
    # Assuming df_2 is a traffic.core.Traffic object containing trajectories
    # Split the traffic object into two halves based on a criterion
    #df_subset1 = reconstructions[0]
    #df_subset2 = reconstructions[1]
    
    # Convert the first subset to a DataFrame and extract lat/lon
    subset1_data = df_subset1[['latitude', 'longitude']].dropna().values
    subset2_data = df_subset2[['latitude', 'longitude']].dropna().values
    
    # Compute energy distance between the raw trajectories
    energy_dist = compute_energy_distance(subset1_data, subset2_data)
    print(f"Energy Distance between the two subsets: {energy_dist}")
    
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

"""
def jensenshannon_distance(reconstructions, model_name="model"):

    # Assuming df_2 is a traffic.core.Traffic object containing trajectories
    # Split the traffic object into two halves based on a criterion
    df_subset1 = reconstructions[0]
    df_subset2 = reconstructions[1]

    # Convert the first subset to a DataFrame and extract lat/lon
    subset1_data = df_subset1.data[['latitude', 'longitude']].dropna().values
    subset2_data = df_subset2.data[['latitude', 'longitude']].dropna().values

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

    ax[0].imshow(subset1_density, origin='lower', cmap='Blues', extent=[xgrid.min(), xgrid.max(), ygrid.min(), ygrid.max()])
    ax[0].set_title("Subset 1 Density")

    ax[1].imshow(subset2_density, origin='lower', cmap='Reds', extent=[xgrid.min(), xgrid.max(), ygrid.min(), ygrid.max()])
    ax[1].set_title("Subset 2 Density")

    plt.tight_layout()
    plt.savefig(f"./figures/{model_name}_comparison.png")

    return js_distance, kl_divergence

"""

"""
Generation of Synthetic Aircraft Landing Trajectories Using Generative Adversarial Networks [Codebase]

File name:
    edistance.py

Description:
    Computation of energy distance as a metric for similarity of distributions.

Author:
    Sebastiaan Wijnands
    S.C.P.Wijnands@student.tudelft.nl
    August 10, 2024    
"""


def energy_distance(x, y):
    a = cdist(x, y, "euclidean").mean()
    b = cdist(x, x, "euclidean").mean()
    c = cdist(y, y, "euclidean").mean()
    e = (2 * a - b - c)

    return e

def edistance_metric(true_data, generated_data):
    true = np.array([true_data[i].ravel() for i in range(true_data.shape[0])])
    generated = np.array([generated_data[i].ravel() for i in range(generated_data.shape[0])])   
    scaler = MinMaxScaler(feature_range=(-1, 1))
    e_distances = []
    
    for _ in tqdm(range(150)):
        ids = random.sample(range(len(true)), 350)
        true_set = true[ids]
        generated_set = generated[ids]
        true_norm = scaler.fit_transform(true_set)
        generated_norm = scaler.fit_transform(generated_set)
        e_distances.append(energy_distance(true_norm, generated_norm))
    
    return np.mean(e_distances)


