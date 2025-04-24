import argparse
import time
from datetime import datetime
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
from model.AirDiffTraj import AirDiffTraj, AirDiffTrajDDPM, AirDiffTrajDDIM
from model.baselines import PerturbationModel
from utils.data_utils import TrafficDataset
from traffic.core import Traffic
from traffic.algorithms.generation import Generation
from sklearn.preprocessing import MinMaxScaler
from utils.condition_utils import load_conditions
from tqdm import tqdm
from utils.helper import load_and_prepare_data, get_model, init_model_config
from evaluation.diversity import data_diversity
from evaluation.similarity import jensenshannon_distance,compute_dtw_3d_batch
from evaluation.time_series import duration_and_speed, timeseries_plot
from evaluation.fidelity import discriminative_score
from lightning.pytorch.loggers import MLFlowLogger
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from model.diffusion import Diffusion
from model.AirLatDiffTraj import Phase
from model.flow_matching import FlowMatching, Wrapper
import cvxopt
from traffic.algorithms.generation import compute_latlon_from_trackgs
import pandas as pd
import seaborn as sns
from model.baselines import PerturbationModel, TimeGAN


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

def get_models(model_config, dataset_params, checkpoint_path, dataset_scaler, d=None):
    """
    Load the trained model and create the trajectory generation model.
    """
    per = False
    if model_config["type"] == "LatDiff" or model_config["type"] == "LatFM":
        temp_conf = {"type": "TCVAE"}
        config_file = f"{model_config['vae']}/config.yaml"
        checkpoint = f"{model_config['vae']}/best_model.ckpt"
        c = load_config(config_file)
        c = c['model']
        c["traj_length"] = model_config['traj_length']
        #print(c)
        vae = get_model(temp_conf).load_from_checkpoint(checkpoint, dataset_params = dataset_params, config = c)
        #print(model_config)
        #diff = Diffusion(model_config)
        if model_config["type"] == "LatDiff":
            diff = Diffusion(model_config)
        else:
            m = FlowMatching(model_config)
            diff = Wrapper(model_config, m)
        #m = FlowMatching(model_config)
        #diff = Wrapper(model_config, m)
        print("Loading", checkpoint_path)
        model = get_model(model_config).load_from_checkpoint(checkpoint_path, dataset_params = dataset_params, config = model_config, vae=vae, generative = diff)
        #model.vae = get_model(temp_conf).load_from_checkpoint(checkpoint, dataset_params = dataset_params, config = c['model'])
        #model.phase = Phase.EVAL
    elif model_config["type"] == "FM":
        fm = FlowMatching(model_config, "0", lat=True)
        model = get_model(model_config).load_from_checkpoint(checkpoint_path, dataset_params = dataset_params, config = model_config, model = fm, cuda = "0")
        #fm = FlowMatching(model_config)
        #model = get_model(model_config)(model_config, fm)
    elif model_config["type"] == "PER":
        model = get_model(model_config)(model_config)
    else:
        model = get_model(model_config).load_from_checkpoint(checkpoint_path, dataset_params = dataset_params, config = model_config)
    if not d:
        model = model.to(device)
    else:
        model = model.to(d)
    model.eval()  # Set the model to evaluation mode
    print("Model loaded with checkpoint!")

    from traffic.algorithms.generation import Generation

    trajectory_generation_model = Generation(
        generation=model,
        features=dataset_params['features'],
        scaler=dataset_scaler,
    )
    
    print("Trajectory generation model created!")
    return model, trajectory_generation_model

def get_config_data(configs, data_path: str, artifact_location: str):
    #configs = load_config(config_path)
    configs["data"]["data_path"] = data_path 
    configs["logger"]["artifact_location"] = artifact_location
    
    dataset, traffic = load_and_prepare_data(configs['data'])

    condition_config = configs["data"]

    if dataset.conditional_features is None:
        conditions = load_conditions(condition_config, dataset)
    else:
        conditions = dataset.conditional_features

    return configs, dataset, traffic, conditions

def plot_track_groundspeed(reconstructions):
    plt.style.use("ggplot")
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.EuroPP())
    ax.coastlines()
    ax.add_feature(cartopy.feature.BORDERS, linestyle=":", alpha=1.0)
    #ax1.coastlines()
    #ax1.add_feature(cartopy.feature.BORDERS, linestyle=":", alpha=1.0)
    plt.xlabel('X values')
    plt.ylabel('Y values')
    plt.title('Plot of Real (Red) and Reconstructed Data (Blue)')
    
    # Calculate and print MSE
    # Colors for different sets
    colors = ["red", "blue"]
    labels = ["real", "synthetic"]
    simple = False


    # Assuming df contains 'latitude', 'longitude', 'groundspeed', 'timedelta', and 'track'

    # Now create a new Traffic object with the updated DataFrame
    #t_updated = Traffic(reconstructions[1].data[['timedelta', 'groundspeed', 'track', 'timestamp', 'icao24']])

    # Now plot the entire updated traffic data
    #reconstructions[1].plot(ax, alpha=0.5, color='red')

    # Show the plot
    plt.title("Flight Trajectories", fontsize=16)
    for c, data in enumerate(reconstructions):
        df = data.data
        obv = 200
        n = df.shape[0] // obv
        df["groundspeed"] = df["groundspeed"] * 18/5
        df = compute_latlon_from_trackgs(df, n, obv,coordinates=dict(latitude=47.546585, longitude=8.447731), forward=False)
        t = Traffic(df)
        t.plot(ax, alpha=0.3, color=colors[c], linewidth=0.5)

    return fig

# SEBESTIAAN
def exponentially_weighted_moving_average(data, alpha=0.3):
    """
    Apply an exponentially weighted moving average (EWMA) filter to the first three features 
    of each sample in the input array.

    Parameters:
    - data (numpy.ndarray): Input array of shape (num_samples, sequence_length, features).
    - alpha (float): Smoothing factor for EWMA. Must be between 0 and 1. Default is 0.3.
    
    Returns:
    - numpy.ndarray: Output array with the EWMA applied to the first three features.
    """
    #if data.shape[2] != 4:
       # raise ValueError("The input array must have exactly 4 features.")
    
    num_samples, sequence_length, num_features = data.shape
    output = np.copy(data)
    
    for sample in range(num_samples):
        for feature in range(3):  # Only apply to the first three features
            ewma = np.zeros(sequence_length)
            ewma[0] = data[sample, 0, feature]  # Initialize the first value with the first data point
            
            for t in range(1, sequence_length):
                ewma[t] = alpha * data[sample, t, feature] + (1 - alpha) * ewma[t-1]
            
            output[sample, :, feature] = ewma

    return output

def mse_df(df1, df2, columns_to_compare = [ 'latitude', 'longitude', 'altitude', 'timedelta', 'groundspeed', 'track_cos', 'track_sin', 'vertical_rate'] ):
    
    # Compute MSE
    ## Problem because this is after rescaling :))) 
    mse = ((df1[columns_to_compare] - df2[columns_to_compare]) ** 2).mean().mean()
    return mse

def plot_traffics(traffic_list: list, 
                  title:str = "Plot of Real (Red) and Reconstructed Data (Blue)",
                  colors = ["red", "blue"],
                  labels = ["real", "synthetic"]):


    plt.style.use("ggplot")
    fig = plt.figure(figsize=(12, 12))
    ax1 = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax1.coastlines()
    ax1.add_feature(cartopy.feature.BORDERS, linestyle=":", alpha=1.0)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(title)
    for c, t in enumerate(traffic_list):
        t.plot(ax1, alpha=0.2, color=colors[c], linewidth=0.5)

    plt.legend()
    return fig

def latlon_from_trackgs(traffic):
    df = traffic.data
    df['track'] = df.apply(
        lambda row: np.degrees(np.arctan2(row['track_sin'], row['track_cos'])), axis=1
    )

    df["timestamp"] = pd.to_timedelta(df["timedelta"], unit="s")
    df = df.reset_index()
    df = compute_latlon_from_trackgs(df, len(traffic), 200, {"latitude" : 0, "longitude": 0}, forward = False)
    df = df.set_index('index')
    return Traffic(df)

def reconstruct_and_plot(dataset, model, trajectory_generation_model, n=1000, model_name = "model", rnd = None, d=None):
    # Select random samples from the dataset
    rnd = np.random.randint(0, len(dataset), (n,)) if rnd is None else rnd
    X2, con, cat, grid = dataset[rnd]
    
    # Move data to GPU if available
    if d:
        device = d
    grid = grid.to(device)
    X_ = X2.reshape(n, len(dataset.features), -1).to(device)
    con_ = con.reshape(n, -1).to(device)
    cat_ = cat.reshape(n, -1).to(device)

    perturb = isinstance(model, PerturbationModel)
    print(perturb)
    model = model.to(device)
    
    print("Shapes:", con.shape, cat.shape, X_.shape)
    
    # Set model guidance scale and perform reconstruction
    #model.unet.guidance_scale = 3
    x_rec, steps = model.reconstruct(X_, con_, cat_, grid)
    
    # Plotting setup
    local_X = X_[:,:3,:]
    local_x_rec = x_rec[:,:3,:]
    print(local_X.shape, local_x_rec.shape)
    mse = torch.nn.functional.mse_loss(local_X, local_x_rec)
    mse_std = torch.std(((local_X - local_x_rec) ** 2), unbiased=True)
    mse_dist = torch.mean(((local_X  - local_x_rec) ** 2), dim=2).cpu().numpy()
    mse_dist_std = torch.std(((local_X  - local_x_rec) ** 2), unbiased=True, dim=2).cpu().numpy()
    mse_median = torch.median(((local_X - local_x_rec) ** 2))
    print("Median", mse_median)

    print("MSE:", mse)
    title = 'Plot of Real (Red) and Reconstructed Data (Blue)'
    # Colors for different sets
    reconstructions = []
    for c, data in enumerate([X_, x_rec]):
        print("Data shape:", data.cpu().numpy().shape)
        
        # Move data to CPU and reshape for inverse transformation
        data_cpu = data.cpu().numpy()
        reco_x = data_cpu.transpose(0, 2, 1).reshape(data_cpu.shape[0], -1)
        # Inverse scaling and traffic reconstruction
        decoded = dataset.scaler.inverse_transform(reco_x)
        decoded = dataset.inverse_airport_coordinates(decoded, rnd)
        reconstructed_traf = trajectory_generation_model.build_traffic(
            decoded.reshape(n, -1, len(dataset.features)),
            coordinates=dict(latitude=0, longitude=0),
            forward=False
        )
            #reconstructed_traf = reconstructed_traf.filter("agressive").eval()
        reconstructed_traf = latlon_from_trackgs(reconstructed_traf)
        reconstructions.append(reconstructed_traf)

        if c == 1:
            df = reconstructed_traf.data.copy()
            cols = ['longitude', 'latitude', 'altitude', 'timedelta']
            numpy_array = exponentially_weighted_moving_average(df[cols].to_numpy().reshape(-1, 200, len(cols)))
            
            # Convert back to DataFrame
            df[cols] = pd.DataFrame(numpy_array.reshape(-1,len(cols)), columns=cols)
            reconstructions.append(Traffic(df))
        
    
    fig = plot_traffics(reconstructions[:2], title = title)
    cols = ['longitude', 'latitude']
    mse_lat_lon = torch.nn.functional.mse_loss(torch.tensor(reconstructions[0].data[cols].to_numpy()), torch.tensor(reconstructions[1].data[cols].to_numpy()))
    mse_lat_lon_std = torch.std(((torch.tensor(reconstructions[0].data[cols].to_numpy()) - torch.tensor(reconstructions[1].data[cols].to_numpy())) ** 2), unbiased=True)
    # Show the plot
    #fig.savefig(f"./figures/{model_name}_reconstructed_data.png")

    mse_dict = {
            "mse": mse,
            "mse_std": mse_std,
            "mse_dist" : mse_dist,
            "mse_dist_std" : mse_dist_std,
            "mse_median" : mse_median,
            "mse_lat_lon" : mse_lat_lon,
            "mse_lat_lon_std" : mse_lat_lon_std,
            }
    
    return reconstructions, mse_dict, rnd, fig


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
                f"Density",
                loc=1,
                prop={"size": 24, "fontname": "Ubuntu"},
                frameon=True,
            )
            text.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
            ax[i].add_artist(text)

        fig.set_tight_layout(True)

        #plt.savefig(f"./figures/{model_name}_density.png")

def generate_samples(dataset, model, rnd, n=10, length=200):
    # Initialize lists to store results for each sample
    all_samples = []
    all_steps = []
    perturb = isinstance(model, PerturbationModel)
    for i in tqdm(rnd):
        # Load the i-th sample from the dataset
        x, con, cat, grid = dataset[i]
        
        # Reshape con and cat as required
        con = con.reshape(1, -1).to(device)
        cat = cat.reshape(1, -1).to(device)
        
        # Adjust the shape of x
        #x = x.view(-1, 1, 28, 28)
        
        # Move grid to the device and adjust dimensions
        grid = grid.unsqueeze(dim=0).to(device)
        #print("Shapes:", con.shape, cat.shape, x.shape)
        #print("Length", length)
        #print("Features", x.shape)
        
        # Generate samples and steps using the model
        if not perturb:
            samples, steps = model.sample(n, con, cat, grid, length, features=x.shape[0])
        else:
            samples, steps = model.reconstruct(x, con, cat, grid)

        # (steps=50, n, 7, len)
        # list (steps=50) of tensors (n, 7, len)
        
        # Append results to the lists
        all_samples.append(samples)
        all_steps.append(steps)
        
    # len(rnd), len(steps), n, 7, len
    
    return all_samples, all_steps

import torch
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy
import traffic.core as tc  # Ensure the traffic library is installed

def detach_to_tensor(tensor_list):
    """
    Detaches each tensor in the list, moves to CPU, and stacks them into a single tensor.
    """
    return np.stack([tensor.cpu().detach() for tensor in tensor_list])

def plot_from_array(t: Traffic, model_name = "model"):

    """
    Plots data from a traffic.core.Traffic object on a EuroPP projection.
    """
    plt.style.use("ggplot")
    fig = plt.figure(figsize=(12, 12))

    ax1 = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax1.coastlines()
    t.plot(ax1, alpha=0.3, color="red", linewidth=0.5)
    ax1.add_feature(cartopy.feature.BORDERS, linestyle=":", alpha=1.0)
    plt.xlabel('X values')
    plt.ylabel('Y values')
    plt.title('Generated Samples')
    plt.savefig(f"./figures/{model_name}_generated_samples.png")
    return fig

def plot_traffic_comparison(traffic_list: list, n_samples: int, output_filename: str = "traffic_comparison.png", landing = True):
    """
    Plot the first n samples of each flight from two Traffic objects side by side.
    
    Parameters:
    -----------
    traffic_list : list
        List containing two Traffic objects to compare
    n_samples : int
        Number of samples to plot for each flight
    output_filename : str
        Name of the output file to save the figure
    """
    
    if len(traffic_list) != 2:
        raise ValueError("Expected exactly 2 Traffic objects in the list")
        
    # Create a figure with two subplots side by side
    plt.style.use("ggplot")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), 
                                  subplot_kw={'projection': ccrs.EuroPP()})
    axes = [ax1, ax2]
    
    # Plot each traffic object
    for idx, traffic in enumerate(traffic_list):
        # Add map features to each subplot
        axes[idx].coastlines()
        axes[idx].add_feature(cartopy.feature.BORDERS, linestyle=":", alpha=1.0)
        
        # Extract flight trajectories
        if landing:
            t = traffic.last(minutes=n_samples).eval()
        else:
            t = traffic.first(minutes=n_samples).eval()
        
        # Plot the trajectory
        t.plot(axes[idx], alpha=0.5, color="red", linewidth=0.3)
        
        # Customize the subplot
        axes[idx].set_xlabel('Longitude')
        axes[idx].set_ylabel('Latitude')
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Create the output filename
    suffix = "landing.png" if landing else "take_off.png"
    full_filename = f"{output_filename}_{suffix}"
    
    # Save the figure
    #plt.savefig(full_filename, bbox_inches='tight', dpi=300)
    return fig

# Assuming 'samples' is a list of tensors generated by model.sample
# Detach and stack samples into a single tensor
import argparse

def get_figure_from_sample_steps(steps, dataset, length = 200):
    # len(rnd), len(steps), n, 7, len
    """
    Get a figure showing the steps of the generated samples.
    """
    # Create a figure with T subplots
    len_steps = len(steps[0])
    fig, axes = plt.subplots(1, len_steps, figsize=(20, 4), sharex=True, sharey=True)
    
    # Plot each step on a separate subplot
    for i, t in enumerate(steps):
        # rnd
        for y, s in enumerate(t):
            # steps
            detached_samples = detach_to_tensor(s).reshape(-1, len(dataset.features), length)
            reco_x = detached_samples.transpose(0, 2, 1).reshape(detached_samples.shape[0], -1)
            decoded = dataset.scaler.inverse_transform(reco_x).reshape(-1, length, len(dataset.features))[:,:,:2]
            axes[y].plot(decoded[:, :, 0], decoded[:, :, 1], "o", markersize=1)
            axes[y].axis("off")
            axes[y].set_title(f"Step {1000 - (y+1) * 200}")
    
    plt.tight_layout()
    
    return fig

def get_mse_distribution(mse_dict):
    mse_values_np = mse_dict["mse_dist"].cpu().numpy() if isinstance(mse_dict["mse_dist"], torch.Tensor) else np.array(mse_dict["mse_dist"])
    std_values_np = mse_dict["mse_dist_std"].cpu().numpy() if isinstance(mse_dict["mse_dist_std"], torch.Tensor) else np.array(mse_dict["mse_dist_std"])

    # Compute global statistics per feature across all 100 trajectories
    mse_mean = np.mean(mse_values_np, axis=0)  # Shape: (n_features,)
    mse_std = np.std(mse_values_np, axis=0)    # Shape: (n_features,)

    # Define outlier thresholds per feature (mean ± 2*std)
    mse_lower = mse_mean - 2 * mse_std
    mse_upper = mse_mean + 2 * mse_std

    # Remove outliers: Replace them with NaN (they won't appear in plots)
    mse_filtered = np.where((mse_values_np >= mse_lower) & (mse_values_np <= mse_upper), mse_values_np, np.nan)

    # Dynamically adjust subplots based on n_features
    n_features = mse_values_np.shape[1]
    n_cols = min(4, n_features)  # Max 4 columns per row
    n_rows = int(np.ceil(n_features / n_cols))

    fig_mse_dict, axes_mse_dict = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes_mse_dict = np.array(axes_mse_dict).reshape(-1)  # Flatten axes for easy indexing
    
    names = ["latitude", "longitude", "altitude"]
    for i in range(n_features):
        sns.histplot(mse_filtered[:, i], bins=50, kde=True, ax=axes_mse_dict[i])
        axes_mse_dict[i].set_title(f"MSE Dist (Feature {names[i]})")
        axes_mse_dict[i].set_xlabel("MSE")
        axes_mse_dict[i].set_ylabel("Frequency")

    plt.tight_layout()
    return fig_mse_dict

def get_logger(logger, dataset_config, config):

    if logger is not None:
        return logger

    if logger is None:
        logger_config = config["logger"]
        logger_config["tags"]["dataset"] = dataset_config["dataset"]
        #config["logger"]["tags"]['weather'] = config["model"]["weather_config"]["weather_grid"]
        logger = MLFlowLogger(
            experiment_name=logger_config["experiment_name"],
            run_name=args.model_name,
            tracking_uri=logger_config["mlflow_uri"],
            tags=logger_config["tags"],
            #artifact_location=artifact_location,
        )
        logger.experiment.log_dict(logger.run_id, config, "config.yaml")

    return logger
    

def run_refactored(args, logger = None):
    seed_everything(42)
    global device 
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

    model_name = args.model_name
    artifact_location= args.artifact_location
    checkpoint = f"{artifact_location}/{model_name}/best_model.ckpt"
    config_file = f"{artifact_location}/{model_name}/config.yaml"

    if args.perturb:
        config_file = "./configs/config_perturb.yaml"
        config = load_config(config_file)
        dataset_config = load_config(args.dataset_config)
    else:
        config = load_config(config_file)
        dataset_config = config["data"]
    logger = get_logger(logger, dataset_config, config)

    dataset_config["data_path"] = args.data_path
    dataset, traffic = load_and_prepare_data(dataset_config)

    #config['model']["data"] = dataset_config
    #config['model']["traj_length"] = dataset.parameters['seq_len']
    #config['model']["continuous_len"] = dataset.con_conditions.shape[1]

    model_config = init_model_config(config, dataset_config, dataset)

    if model_config['type'] == "TCVAE" or model_config['type'] == "VAE":
        logger.log_metrics({"type": model_config['type']})

    model, trajectory_generation_model = get_models(model_config, dataset.parameters, checkpoint, dataset.scaler)

    n = 100
    n_samples = 1
    logger.log_metrics({"n": n, "n samples per" : n_samples})
    rnd = np.random.randint(0, len(dataset), (n,))
    X_original = get_traffic_from_tensor(dataset[rnd][0].detach().numpy(), dataset, trajectory_generation_model ,rnd) 


    #if model_config['type'] == "TCVAE" or model_config['type'] == "VAE" or model_config['type'] == "LatFM":
    logger.log_metrics({"type": model_config['type']})
    reconstructions, mse_dict, rnd, fig_0 = reconstruct_and_plot(dataset, model, trajectory_generation_model, n=n, model_name = model_name, d = device, rnd=rnd)
    logger.log_metrics({"Eval_MSE": mse_dict['mse'], "Eval_MSE_std": mse_dict['mse_std']})
    logger.log_metrics({"mse_lat_lon": mse_dict['mse_lat_lon'], "mse_lat_lon_std": mse_dict['mse_lat_lon_std']})
    logger.log_metrics({"Eval_MSE_median": mse_dict['mse_median']})


    fig_mse_dict = get_mse_distribution(mse_dict)
    logger.experiment.log_figure(logger.run_id, fig_mse_dict, "figures/Eval_mse_per_feature.png")
    mse_smooth = mse_df(reconstructions[0].data, reconstructions[2].data)
    print("MSE Smooth", mse_smooth)
    logger.log_metrics({"Eval_MSE_smooth": mse_smooth})
    fig_smooth = plot_traffics([reconstructions[0],reconstructions[2]])
    logger.experiment.log_figure(logger.run_id,fig_smooth, f"figures/Eval_reconstruction_smoothed.png")

    logger.experiment.log_figure(logger.run_id,fig_0, f"figures/Eval_reconstruction.png")


    #cols = [ 'latitude', 'longitude', 'altitude']
    cols = [ 'latitude', 'longitude']
    length = model_config['traj_length']

    start_time = datetime.now()
    samples, steps = generate_samples(dataset, model, rnd, n = n_samples, length = length)
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    logger.log_metrics({"sampling_time_seconds": duration})
    
    detached_samples = detach_to_tensor(samples).reshape(-1, len(dataset.features), length)
    decoded = get_traffic_from_tensor(detached_samples, dataset, trajectory_generation_model,rnd)
    #decoded = latlon_from_trackgs(decoded)

    #df = decoded.data
    #numpy_array = exponentially_weighted_moving_average(df[['longitude', 'latitude', 'altitude']].to_numpy().reshape(-1, 200, 3))
    numpy_array = decoded.data[cols].to_numpy().reshape(-1, 200, len(cols))
    # Convert back to DataFrame
    #df[cols] = pd.DataFrame(numpy_array.reshape(-1,len(cols)), columns=cols)
    #generated_traffic = Traffic(df)
    generated_traffic = decoded

    fig_pca = data_diversity(X_original.data[cols].to_numpy().reshape(-1, 200, len(cols)), numpy_array, 'PCA', 'else', model_name=model_name)
    fig_tsne = data_diversity(X_original.data[cols].to_numpy().reshape(-1, 200, len(cols)), numpy_array, 't-SNE','else', model_name = model_name)
    logger.experiment.log_figure(logger.run_id, fig_pca, f"figures/pca.png")
    logger.experiment.log_figure(logger.run_id, fig_tsne, f"figures/tsne.png")
    #reconstructed_traf = reconstructed_traf.simplify(5e2, altitude="altitude").eval()
    ##reconstructed_traf = reconstructed_traf.filter("agressive").eval()

    JSD, KL, (e_distance, e_distance_std), fig_1 = jensenshannon_distance(X_original.data[cols],generated_traffic.data[cols] , model_name = model_name)
    logger.log_metrics({"Eval_edistance_generation": e_distance, "Eval_edistance_std": e_distance_std, "Eval_JSD_generation": JSD, "Eval_KL_generation": KL})
    logger.experiment.log_figure(logger.run_id, fig_1, f"figures/Eval_comparison_generated.png")

    fig_2 = plot_from_array(generated_traffic, model_name)
    logger.experiment.log_figure(logger.run_id, fig_2, f"figures/Eval_generated_samples.png")
    try:
        generated_traffic.to_pickle(f"{artifact_location}/{model_name}/generated_samples.pkl")
    except:
        print("Error saving samples")

    #generated_traffic = latlon_from_trackgs(generated_traffic)

    mmd_gen, mmd_gen_std = compute_partial_mmd(generated_traffic, X_original)
    print("MMD GEN", mmd_gen)
    logger.log_metrics({"mmd_gen": mmd_gen, "mmd_gen_std": mmd_gen_std})

    dtw, dtw_std, _ = compute_dtw_3d_batch(X_original.data[cols].to_numpy(),generated_traffic.data[cols].to_numpy())
    logger.log_metrics({"dtw": dtw, "dtw_std": dtw_std})

    training_trajectories = X_original
    synthetic_trajectories = generated_traffic

    fig_3 = duration_and_speed(training_trajectories, synthetic_trajectories, model_name = model_name)
    logger.experiment.log_figure(logger.run_id,fig_3, f"figures/Eval_distribution_plots.png")

    features_to_plot = ['latitude', 'longitude', 'altitude', 'timedelta']
    units = {
        'latitude': '°',
        'longitude': '°',
        'altitude': 'm',
        'timedelta': 's'
    }

    fig_4 = timeseries_plot(
        training_trajectories,
        synthetic_trajectories,
        features=features_to_plot,
        units=units,
        model_name=model_name
    )
    logger.experiment.log_figure(logger.run_id,fig_4, f"figures/Eval_timeseries_plots.png")
    
    accuracy, score, conf_matrix, tpr, tnr = discriminative_score(training_trajectories.data[['latitude', 'longitude', 'altitude']].to_numpy().reshape(-1, length, 3), synthetic_trajectories.data[['latitude', 'longitude', 'altitude']].to_numpy().reshape(-1, length, 3))
    logger.log_metrics({"Discriminator_Accuracy": accuracy, "Discriminator_Score": score, "Discriminator_TPR": tpr, "Discriminator_TNR": tnr})
    print("Accuracy on test data:", accuracy)
    print("Discriminative Score:", score)
    print("Confusion Matrix:\n", conf_matrix)
    print("True Positive Rate (TPR):", tpr)
    print("True Negative Rate (TNR):", tnr)
    
    # Visualize the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['Synthetic', 'Original'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    #plt.savefig("figures/fidelity")
    logger.experiment.log_figure(logger.run_id, disp.figure_, f"figures/fidelity.png")

    logger.finalize()


def run(args, logger = None):
    seed_everything(42)
    global device 
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    model_name = args.model_name
    data_path = args.data_path
    artifact_location= args.artifact_location
    checkpoint = f"{artifact_location}/{model_name}/best_model.ckpt"
    config_file = f"{artifact_location}/{model_name}/config.yaml"
    #dataset_config_file = f"./artifacts/{model_name}/dataset_config.yaml"

    config = load_config(config_file)
    dataset_config = config["data"]
    logger = get_logger(logger, dataset_config, config)

    dataset_config["data_path"] = args.data_path
    #_, dataset, traffic, conditions = get_config_data(config_file, data_path, artifact_location)
    dataset, traffic = load_and_prepare_data(dataset_config)
    #config["data"] = dataset_config
    config['model']["data"] = dataset_config
    config['model']["traj_length"] = dataset.parameters['seq_len']
    config['model']["continuous_len"] = dataset.con_conditions.shape[1]
    if config['model']['type'] == "TCVAE" or config['model']['type'] == "VAE":
        logger.log_metrics({"type": config['model']['type']})
    model, trajectory_generation_model = get_models(config["model"], dataset.parameters, checkpoint, dataset.scaler)
    #model.eval()
    batch_size = dataset_config["batch_size"]
    n = 100
    n_samples = 1

    logger.log_metrics({"n reconstructions": n, "n samples per" : n_samples})
    
    reconstructions, mse_dict, rnd, fig_0 = reconstruct_and_plot(dataset, model, trajectory_generation_model, n=n, model_name = model_name, d = device)
    logger.log_metrics({"Eval_MSE": mse_dict['mse'], "Eval_MSE_std": mse_dict['mse_std']})
    logger.log_metrics({"Eval_MSE_median": mse_dict['mse_median']})

            # Convert PyTorch tensors to NumPy arrays
            # Convert PyTorch tensors to NumPy arrays
    # Convert PyTorch tensors to NumPy arrays if necessary
    # Convert PyTorch tensors to NumPy arrays if necessary
    

    fig_mse_dict = get_mse_distribution(mse_dict)
    logger.experiment.log_figure(logger.run_id, fig_mse_dict, "figures/Eval_mse_per_feature.png")
    mse_smooth = mse_df(reconstructions[0].data, reconstructions[2].data)
    print("MSE Smooth", mse_smooth)
    logger.log_metrics({"Eval_MSE_smooth": mse_smooth})
    fig_smooth = plot_traffics([reconstructions[0],reconstructions[2]])
    logger.experiment.log_figure(logger.run_id,fig_smooth, f"figures/Eval_reconstruction_smoothed.png")

    logger.experiment.log_figure(logger.run_id,fig_0, f"figures/Eval_reconstruction.png")
    mmd, mmd_std = compute_partial_mmd(reconstructions[0], reconstructions[1])
    print("MMD", mmd)
    logger.log_metrics({"mmd": mmd, "mmd_std": mmd_std})

    mmd,mmd_std = compute_partial_mmd(reconstructions[0], reconstructions[2])
    print("MMD smooth", mmd)
    logger.log_metrics({"mmd_smooth": mmd, "mmd_std_smooth": mmd_std})

    #if mse_smooth < mse:
        #reconstructions[1] = reconstructions[2]
        #print("Switching to smoothed version")

    #fig_track_speed = plot_track_groundspeed(reconstructions[:2])
    #logger.experiment.log_figure(logger.run_id,fig_track_speed, f"figures/Eval_reconstruction_track_speed.png")
    #logger.experiment.log_figure(logger.run_id, fig, "figures/my_plot.png")
    #print(reconstructions[1].data)
    #cols = [ 'latitude', 'longitude', 'altitude']
    cols = [ 'latitude', 'longitude']
    if n != 1000:
        JSD, KL, (e_distance, e_distance_std), fig_1 = jensenshannon_distance(reconstructions[0].data[cols], reconstructions[1].data[cols], model_name = model_name)
        logger.log_metrics({"Eval_edistance": e_distance, "Eval_JSD": JSD, "Eval_KL": KL})
        logger.experiment.log_figure(logger.run_id, fig_1, f"figures/Eval_comparison.png")
        JSD, KL, (e_distance, e_distance_std), fig_1 = jensenshannon_distance(reconstructions[0].data[cols], reconstructions[2].data[cols], model_name = model_name)
        logger.log_metrics({"Eval_edistance_smoothed": e_distance, "Eval_JSD_smoothed": JSD, "Eval_KL_smoothed": KL})
    #density(reconstructions, model_name = model_name)

    #if False
    #fig_landing = plot_traffic_comparison(reconstructions[:2], 2, f"./figures/{model_name}_", landing = True)
    #fig_takeoff = plot_traffic_comparison(reconstructions[:2], 2, f"./figures/{model_name}_", landing = False)
    #logger.experiment.log_figure(logger.run_id, fig_landing, f"figures/landing_comparison.png")
    #logger.experiment.log_figure(logger.run_id, fig_takeoff, f"figures/takeoff_comparison.png")

    length = config['model']['traj_length']
    
    start_time = datetime.now()
    samples, steps = generate_samples(dataset, model, rnd, n = n_samples, length = length)
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    logger.log_metrics({"sampling_time_seconds": duration})
    
    #fig_99 = get_figure_from_sample_steps(steps, dataset, length)
    #fig_99.savefig(f"./figures/{model_name}_generated_steps.png")
    #logger.experiment.log_figure(logger.run_id, fig_99, f"figures/generated_steps.png")

    detached_samples = detach_to_tensor(samples).reshape(-1, len(dataset.features), length)
    reconstructed_traf = get_traffic_from_tensor(detached_samples, dataset, trajectory_generation_model, rnd)

    df = reconstructed_traf.data
    #numpy_array = exponentially_weighted_moving_average(df[cols].to_numpy().reshape(-1, 200, 3))
    
    # Convert back to DataFrame
    numpy_array = df[cols].to_numpy().reshape(-1, 200, len(cols))
    df[cols] = pd.DataFrame(numpy_array.reshape(-1,len(cols)), columns=cols)
    reconstructed_traf = Traffic(df)

    fig_pca = data_diversity(reconstructions[0].data[cols].to_numpy().reshape(-1, 200, len(cols)), numpy_array, 'PCA', 'else', model_name=model_name)
    fig_tsne = data_diversity(reconstructions[0].data[cols].to_numpy().reshape(-1, 200, len(cols)), numpy_array, 't-SNE','else', model_name = model_name)
    logger.experiment.log_figure(logger.run_id, fig_pca, f"figures/pca.png")
    logger.experiment.log_figure(logger.run_id, fig_tsne, f"figures/tsne.png")
    #reconstructed_traf = reconstructed_traf.simplify(5e2, altitude="altitude").eval()
    ##reconstructed_traf = reconstructed_traf.filter("agressive").eval()

    if n != 1000:
        JSD, KL, (e_distance, e_distance_std), fig_1 = jensenshannon_distance(reconstructions[0].data[cols],reconstructed_traf.data[cols] , model_name = model_name)
        logger.log_metrics({"Eval_edistance_generation": e_distance, "Eval_JSD_generation": JSD, "Eval_KL_generation": KL})
        logger.experiment.log_figure(logger.run_id, fig_1, f"figures/Eval_comparison_generated.png")

    fig_2 = plot_from_array(reconstructed_traf, model_name)
    logger.experiment.log_figure(logger.run_id, fig_2, f"figures/Eval_generated_samples.png")
    reconstructed_traf.to_pickle(f"./artifacts/{model_name}/generated_samples.pkl")

    reconstructed_traf.data['track'] = reconstructed_traf.data.apply(
        lambda row: np.degrees(np.arctan2(row['track_sin'], row['track_cos'])), axis=1
    )
    #fig_track_speed = plot_track_groundspeed([reconstructed_traf])
    #logger.experiment.log_figure(logger.run_id,fig_track_speed, f"figures/Eval_generation_track_speed.png")
    mmd_gen, mmd_gen_std = compute_partial_mmd(reconstructed_traf, reconstructions[0])
    print("MMD GEN", mmd_gen)
    logger.log_metrics({"mmd_gen": mmd_gen, "mmd_gen_std": mmd_gen_std})

    training_trajectories = reconstructions[0]
    #synthetic_trajectories = reconstructions[1]
    synthetic_trajectories = reconstructed_traf

    fig_3 = duration_and_speed(training_trajectories, synthetic_trajectories, model_name = model_name)
    logger.experiment.log_figure(logger.run_id,fig_3, f"figures/Eval_distribution_plots.png")

    features_to_plot = ['latitude', 'longitude', 'altitude', 'timedelta']
    units = {
        'latitude': '°',
        'longitude': '°',
        'altitude': 'm',
        'timedelta': 's'
    }

    fig_4 = timeseries_plot(
        training_trajectories,
        synthetic_trajectories,
        features=features_to_plot,
        units=units,
        model_name=model_name
    )
    logger.experiment.log_figure(logger.run_id,fig_4, f"figures/Eval_timeseries_plots.png")
    
    # NOTE THIS IS PERHAPS NOT THE BEST WAY TO DO THIS BECAUSE USES RECONSRUCTED NOT GENERATED TRAFFIC

    accuracy, score, conf_matrix, tpr, tnr = discriminative_score(training_trajectories.data[['latitude', 'longitude', 'altitude']].to_numpy().reshape(-1, length, 3), synthetic_trajectories.data[['latitude', 'longitude', 'altitude']].to_numpy().reshape(-1, length, 3))
    logger.log_metrics({"Discriminator_Accuracy": accuracy, "Discriminator_Score": score, "Discriminator_TPR": tpr, "Discriminator_TNR": tnr})
    print("Accuracy on test data:", accuracy)
    print("Discriminative Score:", score)
    print("Confusion Matrix:\n", conf_matrix)
    print("True Positive Rate (TPR):", tpr)
    print("True Negative Rate (TNR):", tnr)
    
    # Visualize the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['Synthetic', 'Original'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    #plt.savefig("figures/fidelity")
    logger.experiment.log_figure(logger.run_id, disp.figure_, f"figures/fidelity.png")

    logger.finalize()

def get_traffic_from_tensor(data, dataset, trajectory_generation_model, rnd):
    print(data.shape)

    reco_x = data.transpose(0, 2, 1).reshape(data.shape[0], -1)
    n = data.shape[0]
    # Inverse scaling and traffic reconstruction
    decoded = dataset.scaler.inverse_transform(reco_x)
    decoded = dataset.inverse_airport_coordinates(decoded, rnd)
    reconstructed_traf = trajectory_generation_model.build_traffic(
        decoded.reshape(n, -1, len(dataset.features)),
        coordinates=dict(latitude=0, longitude=0),
        forward=False
    )
    reconstructed_traf = latlon_from_trackgs(reconstructed_traf)
    return reconstructed_traf

def exponential_kernel(x, y, gamma=1e-8):
    """
    Exponential (RBF) kernel function.
    gamma: Kernel width parameter. Higher values make the kernel more localized.
    """
    return np.exp(-gamma * np.linalg.norm(x - y) ** 2)

def compute_partial_mmd(X, Y, alpha=1.0, gamma=1e-8):
    """
    Computes the α-partial MMD^2 between synthetic data X and real data Y using an exponential kernel.
    
    X: Synthetic data (n_samples, n_features)
    Y: Real data (m_samples, n_features)
    alpha: Fraction of data to be matched (0 < alpha <= 1)
    std_dev: Standard deviation for kernel scaling (None will compute based on data)
    
    Returns:
    - α-partial MMD² value
    """
    X = X.data[["longitude", "latitude", "altitude"]].to_numpy().reshape(-1,3, 200)  # Convert Traffic object to numpy array
    Y = Y.data[["longitude", "latitude", "altitude"]].to_numpy().reshape(-1,3, 200) 
    print("Comparing", X.shape, Y.shape)
    n = X.shape[0]  # Number of samples in synthetic data X
    m = Y.shape[0]  # Number of samples in real data Y
    
    # Step 1: Compute kernel matrices
    K_X = np.zeros((n, n))  # Kernel matrix for X
    K_Y = np.zeros((m, m))  # Kernel matrix for Y
    K_XY = np.zeros((m, n))  # Cross kernel matrix between X and Y

    for i in range(n):
        for j in range(i, n):
            K_X[i, j] = exponential_kernel(X[i], X[j], gamma)
            K_X[j, i] = K_X[i, j]  # Symmetry
    
    for i in range(m):
        for j in range(i, m):
            K_Y[i, j] = exponential_kernel(Y[i], Y[j], gamma)
            K_Y[j, i] = K_Y[i, j]  # Symmetry

    for i in range(n):
        for j in range(m):
            K_XY[j, i] = exponential_kernel(Y[j], X[i], gamma)

    # Step 2: Define weight vector v (for real data Y)
    v = np.ones(m) / m
    
    # Step 3: Formulate the quadratic programming problem
    # The objective is to minimize w^T K_X w + v^T K_Y v - 2 v^T K_XY w
    
    # Objective quadratic term
    P = cvxopt.matrix(K_X)  # K_X for the quadratic term
    q = cvxopt.matrix(-2 * np.dot(K_XY, v))  # Linear term in the objective
    
    # Constraints
    G = cvxopt.matrix(np.vstack([-np.eye(n), np.eye(n)]))  # Constraint on w: 0 <= w_i <= 1/(alpha * n)
    h = cvxopt.matrix(np.hstack([np.zeros(n), np.ones(n) * (1 / (alpha * n))]))  # Constraints on w
    
    # Equality constraint: sum(w) = 1
    A = cvxopt.matrix(np.ones((1, n)))  # Sum of w must be 1
    b = cvxopt.matrix(1.0)
    
    # Step 4: Solve the quadratic programming problem
    sol = cvxopt.solvers.qp(P, q, G, h, A, b)
    w = np.array(sol['x']).flatten()
    
    # Step 5: Compute the α-partial MMD² value
    mmd_squared = np.dot(w.T, np.dot(K_X, w)) + np.dot(v.T, np.dot(K_Y, v)) - 2 * np.dot(v.T, np.dot(K_XY, w))
    mmd_values = np.dot(K_X, w) + np.dot(K_Y, v) - 2 * np.dot(K_XY, w)
    std_dev = np.std(np.sqrt(np.abs(mmd_values)), ddof=1)  # Sample standard deviation
    
    return np.sqrt(mmd_squared), std_dev


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the traffic model.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="AirDiffTraj_5",
        help="Name of the model (e.g., 'AirDiffTraj_5')."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        #required=True,
        default="./data/resampled/combined_traffic_resampled_200.pkl",
        help="Path to the training data file"
    )

    parser.add_argument(
        "--dataset_config",
        type=str,
        #required=True,
        default="./configs/dataset_opensky.yaml",
        help="Path to the dataset config file"
    )

    parser.add_argument(
        "--artifact_path",
        dest="artifact_location",
        type=str,
        #required=True,
        default="./artifacts",
        help="Path to save the artifacts",
    )

    parser.add_argument(
            "--cuda",
            type=int,
            default=0,
            help="GPU to use",
            )

    parser.add_argument("--eval", dest="run_train", action='store_true')
    parser.add_argument(
            "--perturb",
            dest="perturb", 
            action='store_true')

    args = parser.parse_args()
    run_refactored(args)
    
    #run(args)
