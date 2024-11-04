import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import cartopy.crs as ccrs
import cartopy.feature
from typing import Any, Dict
import os
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
import matplotlib.pyplot as plt
import numpy as np
from utils import load_config
from model.AirDiffTraj import AirDiffTraj
from utils.data_utils import TrafficDataset
from traffic.core import Traffic
from traffic.algorithms.generation import Generation
from sklearn.preprocessing import MinMaxScaler
from utils.condition_utils import load_conditions



def load_and_prepare_data(configs):
    """
    Load and prepare the dataset for the model.
    """
    dataset_config = configs['data']
    dataset = TrafficDataset.from_file(
        dataset_config["data_path"],
        features=dataset_config["features"],
        shape=dataset_config["data_shape"],
        scaler=MinMaxScaler(feature_range=(-1, 1)),
        info_params={
            "features": dataset_config["info_features"],
            "index": dataset_config["info_index"],
        },
        conditional_features = load_conditions(dataset_config) ,
        down_sample_factor=dataset_config["down_sample_factor"],
    )
    traffic = Traffic.from_file(dataset_config["data_path"])

    return dataset, traffic

def get_checkpoint_path(logger_config: Dict[str, Any]):
    """
    Get the path to the checkpoint file.
    """
    run_name = logger_config["run_name"]
    artifact_location = logger_config["artifact_location"]
    # check that the artifact location exists, otherwise raise an error
    if not os.path.exists(artifact_location):
        raise FileNotFoundError(f"Artifact directory {artifact_location} not found!")

    artifact_location = os.path.join(artifact_location, run_name)
    # check that artifact location exists, otherwise raise an error
    if not os.path.exists(artifact_location):
        raise FileNotFoundError(f"Artifact location {artifact_location} not found!")

    # get the "best_model.ckpt" file
    checkpoint = os.path.join(artifact_location, "best_model.ckpt")
    # check that the checkpoint exists, otherwise raise an error
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint file {checkpoint} not found!")

    return checkpoint

def get_models(model_config, dataset_params, checkpoint_path, dataset_scaler):
    """
    Load the trained model and create the trajectory generation model.
    """
    model = AirDiffTraj.load_from_checkpoint(checkpoint_path, dataset_params = dataset_params, config = model_config)
    model.eval()  # Set the model to evaluation mode
    print("Model loaded with checkpoint!")

    from traffic.algorithms.generation import Generation

    trajectory_generation_model = Generation(
        generation=model,
        features=dataset.parameters['features'],
        scaler=dataset_scaler,
    )
    
    print("Trajectory generation model created!")
    return model, trajectory_generation_model

def get_config_data(config_path: str, data_path: str, artifact_location: str):
    configs = load_config(config_path)
    configs["data"]["data_path"] = data_path 
    configs["logger"]["artifact_location"] = artifact_location
    
    dataset, traffic = load_and_prepare_data(configs)

    condition_config = configs["data"]

    if dataset.conditional_features is None:
        conditions = load_conditions(condition_config, dataset)
    else:
        conditions = dataset.conditional_features

    return configs, dataset, traffic, conditions


def reconstruct_and_plot(dataset, model, trajectory_generation_model, n=1000, model_name = "model"):
    # Select random samples from the dataset
    rnd = np.random.randint(0, len(dataset), (n,))
    X2, con, cat, grid = dataset[rnd]
    
    # Move data to GPU if available
    grid = grid.to("cuda")
    X_ = X2.reshape(n, 4, -1).to("cuda")
    con_ = con.reshape(n, -1)
    cat_ = cat.reshape(n, -1)
    
    print("Shapes:", con.shape, cat.shape, X_.shape)
    
    # Set model guidance scale and perform reconstruction
    #model.unet.guidance_scale = 3
    x_rec, steps = model.reconstruct(X_, con_, cat_, grid)
    
    # Plotting setup
    plt.style.use("ggplot")
    fig = plt.figure(figsize=(12, 12))
    ax1 = fig.add_subplot(1, 1, 1, projection=ccrs.EuroPP())
    ax1.coastlines()
    ax1.add_feature(cartopy.feature.BORDERS, linestyle=":", alpha=1.0)
    plt.xlabel('X values')
    plt.ylabel('Y values')
    plt.title('Plot of Reconstructed Data')
    
    # Calculate and print MSE
    mse = torch.nn.functional.mse_loss(X_, x_rec)
    print("MSE:", mse)
    
    # Colors for different sets
    colors = ["red", "blue"]
    reconstructions = []
    
    for c, data in enumerate([X_, x_rec]):
        print("Data shape:", data.cpu().numpy().shape)
        
        # Move data to CPU and reshape for inverse transformation
        data_cpu = data.cpu().numpy()
        reco_x = data_cpu.transpose(0, 2, 1).reshape(data_cpu.shape[0], -1)
        
        # Inverse scaling and traffic reconstruction
        decoded = dataset.scaler.inverse_transform(reco_x)
        reconstructed_traf = trajectory_generation_model.build_traffic(
            decoded.reshape(n, -1, 4),
            coordinates=dict(latitude=48.5, longitude=8.4),
            forward=False
        )
        
        reconstructions.append(reconstructed_traf)
        
        # Plot reconstructed data on the map
        reconstructed_traf.plot(ax1, alpha=0.5, color=colors[c], linewidth=1)
    
    # Show the plot
    plt.show()
    plt.savefig(f"./figures/{model_name}_reconstructed_data.png")
    
    return reconstructions, mse

def jensenshannon_distance(reconstructions, model_name="model"):
    import numpy as np
    from scipy.stats import gaussian_kde
    from scipy.special import rel_entr
    from scipy.spatial.distance import jensenshannon
    import matplotlib.pyplot as plt

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

def density(reconstructions, model_name="model"):
    import matplotlib.pyplot as plt

    from matplotlib.offsetbox import AnchoredText
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    from cartopy.crs import EuroPP, PlateCarree
    from cartes.utils.features import countries, ocean


    with plt.style.context("traffic"):

        fig = plt.figure(figsize=(15, 10), frameon=False)
        ax = fig.subplots(1, 2, subplot_kw=dict(projection=EuroPP()))

        for ax_ in ax:
            ax_.add_feature(countries(scale="10m", linewidth=1.5))

        vmax = None  # this trick will keep the same colorbar scale for both maps

        for i, data in enumerate([reconstructions[0], reconstructions[1]]):
            # Aggregate and query the data, then convert to xarray
            data_xarray = data.agg_latlon(
                # 10 points per integer lat/lon
                resolution=dict(latitude=10, longitude=10),
                # count the number of flights
                flight_id="nunique"
            ).query(f"flight_id > 1").to_xarray()

            # Sort the DataArray by latitude and longitude
            data_xarray = data_xarray.sortby(['latitude', 'longitude'])

            # Plot the data using pcolormesh
            cax = data_xarray.flight_id.plot.pcolormesh(
                ax=ax[i],
                cmap="viridis",
                transform=PlateCarree(),
                vmax=vmax,
                add_colorbar=False,
            )

            cbaxes = inset_axes(ax[i], "4%", "60%", loc=3)
            cb = fig.colorbar(cax, cax=cbaxes)

            # Keep this value to scale the colorbar for the second day
            vmax = cb.vmax

            text = AnchoredText(
                f"{data.start_time:%B %d, %Y}",
                loc=1,
                prop={"size": 24, "fontname": "Ubuntu"},
                frameon=True,
            )
            text.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
            ax[i].add_artist(text)

        fig.set_tight_layout(True)

        plt.savefig(f"./figures/{model_name}_density.png")


def generate_samples(model, n, c_, t):
    raise NotImplementedError("Juhu")

if __name__ == "__main__":
    config_file = "./configs/config.yaml"
    data_path = "./data/resampled/combined_traffic_resampled_200.pkl"
    artifact_location= "./artifacts"
    model_name = "AirDiffTraj_18"
    checkpoint = f"./artifacts/{model_name}/best_model.ckpt"


    config = load_config(config_file)

    config, dataset, traffic, conditions = get_config_data(config_file, data_path, artifact_location)
    config['model']["traj_length"] = dataset.parameters['seq_len']
    model, trajectory_generation_model = get_models(config["model"], dataset.parameters, checkpoint, dataset.scaler)
    dataset_config = config["data"]
    batch_size = dataset_config["batch_size"]
    
    reconstructions, mse = reconstruct_and_plot(dataset, model, trajectory_generation_model, n=1000, model_name = model_name)
    jensenshannon_distance(reconstructions, model_name = model_name)
    density(reconstructions, model_name = model_name)

    
