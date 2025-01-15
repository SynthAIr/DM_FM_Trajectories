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
from model.AirDiffTraj import AirDiffTraj, AirDiffTrajDDPM, AirDiffTrajDDIM
from model.baselines import PerturbationModel
from utils.data_utils import TrafficDataset
from traffic.core import Traffic
from traffic.algorithms.generation import Generation
from sklearn.preprocessing import MinMaxScaler
from utils.condition_utils import load_conditions
from tqdm import tqdm
from utils.helper import load_and_prepare_data, get_model
from evaluation.diversity import data_diversity
from evaluation.similarity import jensenshannon_distance
from evaluation.time_series import duration_and_speed, timeseries_plot
from evaluation.fidelity import discriminative_score
from lightning.pytorch.loggers import MLFlowLogger
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from model.diffusion import Diffusion
from model.AirLatDiffTraj import Phase
from model.flow_matching import FlowMatching



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
    if model_config["type"] == "LatDiff":
        temp_conf = {"type": "TCVAE"}
        config_file = f"{model_config['vae']}/config.yaml"
        checkpoint = f"{model_config['vae']}/best_model.ckpt"
        c = load_config(config_file)
        c = c['model']
        c["traj_length"] = model_config['traj_length']
        #print(c)
        vae = get_model(temp_conf).load_from_checkpoint(checkpoint, dataset_params = dataset_params, config = c)
        #print(model_config)
        diff = Diffusion(model_config)
        print("Loading", checkpoint_path)
        model = get_model(model_config).load_from_checkpoint(checkpoint_path, dataset_params = dataset_params, config = model_config, vae=vae, generative = diff)
        #model.vae = get_model(temp_conf).load_from_checkpoint(checkpoint, dataset_params = dataset_params, config = c['model'])
        #model.phase = Phase.EVAL
    elif model_config["type"] == "FM":
        fm = FlowMatching(model_config)
        model = get_model(model_config)(model_config, fm)
    else:
        model = get_model(model_config).load_from_checkpoint(checkpoint_path, dataset_params = dataset_params, config = model_config)
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

def get_config_data(config_path: str, data_path: str, artifact_location: str):
    configs = load_config(config_path)
    configs["data"]["data_path"] = data_path 
    configs["logger"]["artifact_location"] = artifact_location
    
    dataset, traffic = load_and_prepare_data(configs['data'])

    condition_config = configs["data"]

    if dataset.conditional_features is None:
        conditions = load_conditions(condition_config, dataset)
    else:
        conditions = dataset.conditional_features

    return configs, dataset, traffic, conditions


def reconstruct_and_plot(dataset, model, trajectory_generation_model, n=1000, model_name = "model", rnd = None):
    # Select random samples from the dataset
    rnd = np.random.randint(0, len(dataset), (n,)) if rnd is None else rnd
    X2, con, cat, grid = dataset[rnd]
    
    # Move data to GPU if available
    grid = grid.to("cuda")
    X_ = X2.reshape(n, len(dataset.features), -1).to("cuda")
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
    plt.title('Plot of Real (Red) and Reconstructed Data (Blue)')
    
    # Calculate and print MSE
    mse = torch.nn.functional.mse_loss(X_, x_rec)
    print("MSE:", mse)
    
    # Colors for different sets
    colors = ["red", "blue"]
    labels = ["real", "synthetic"]
    reconstructions = []
    
    for c, data in enumerate([X_, x_rec]):
        print("Data shape:", data.cpu().numpy().shape)
        
        # Move data to CPU and reshape for inverse transformation
        data_cpu = data.cpu().numpy()
        reco_x = data_cpu.transpose(0, 2, 1).reshape(data_cpu.shape[0], -1)
        
        # Inverse scaling and traffic reconstruction
        decoded = dataset.scaler.inverse_transform(reco_x)
        reconstructed_traf = trajectory_generation_model.build_traffic(
            decoded.reshape(n, -1, len(dataset.features)),
            coordinates=dict(latitude=48.5, longitude=8.4),
            forward=False
        )
        def convert_sin_cos_to_lat_lon(traffic):
            df = traffic.data
            # Calculate latitude from sine and cosine
            df["latitude"] = np.degrees(np.arctan2(df["latitude_sin"], df["latitude_cos"]))
            
            # Calculate longitude from sine and cosine
            df["longitude"] = np.degrees(np.arctan2(df["longitude_sin"], df["longitude_cos"]))
            return Traffic(df)
        
        #reconstructed_traf = convert_sin_cos_to_lat_lon(reconstructed_traf)
        reconstructions.append(reconstructed_traf)
        
        # Plot reconstructed data on the map
        reconstructed_traf.plot(ax1, alpha=0.5, color=colors[c], linewidth=1)


    plt.legend()
    
    # Show the plot
    plt.savefig(f"./figures/{model_name}_reconstructed_data.png")
    
    return reconstructions, mse, rnd, fig


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

        plt.savefig(f"./figures/{model_name}_density.png")

def generate_samples(dataset, model, rnd, n=10, length=200):
    # Initialize lists to store results for each sample
    all_samples = []
    all_steps = []
    for i in tqdm(rnd):
        # Load the i-th sample from the dataset
        x, con, cat, grid = dataset[i]
        
        # Reshape con and cat as required
        con = con.reshape(1, -1)
        cat = cat.reshape(1, -1)
        
        # Adjust the shape of x
        #x = x.view(-1, 1, 28, 28)
        
        # Move grid to the device and adjust dimensions
        grid = grid.unsqueeze(dim=0).to("cuda")
        print("Shapes:", con.shape, cat.shape, x.shape)
        print("Length", length)
        print("Features", x.shape)
        
        # Generate samples and steps using the model
        samples, steps = model.sample(n, con, cat, grid, length, features=x.shape[0])
        # (steps=50, n, 7, len)
        # list (steps=50) of tensors (n, 7, len)
        
        # Append results to the lists
        all_samples.append(samples)
        all_steps.append(steps)
        
        # Print out shapes for verification (optional)
        #print("cat shape:", cat.shape)
        #print("grid shape:", grid.shape)
        #print("samples shape:", samples.shape)
        #print("steps length:", len(steps))

    #all_samples = list of rnd with list of tensors (n, 7, len)
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

    ax1 = fig.add_subplot(1, 1, 1, projection=ccrs.EuroPP())
    ax1.coastlines()
    t.plot(ax1, alpha=0.5, color="red", linewidth=1)
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
    plt.savefig(full_filename, bbox_inches='tight', dpi=300)
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

def run(args, logger = None):
    np.random.seed(42)
    model_name = args.model_name

    data_path = args.data_path
    artifact_location= "./artifacts"
    checkpoint = f"./artifacts/{model_name}/best_model.ckpt"
    config_file = f"./artifacts/{model_name}/config.yaml"

    config = load_config(config_file)
    runid = None
    if logger is not None:
        runid = logger.run_id

    if logger is None:
        logger_config = config["logger"]
        logger = MLFlowLogger(
            experiment_name=logger_config["experiment_name"],
            run_name=args.model_name,
            tracking_uri=logger_config["mlflow_uri"],
            tags=logger_config["tags"],
            #artifact_location=artifact_location,
        )

        if runid is not None:
            logger.run_id = runid

    dataset_config = config["data"]
    dataset_config["data_path"] = args.data_path

    logger.experiment.log_dict(logger.run_id,config, config_file)
    config, dataset, traffic, conditions = get_config_data(config_file, data_path, artifact_location)
    config["data"] = dataset_config
    config['model']["data"] = dataset_config
    config['model']["traj_length"] = dataset.parameters['seq_len']
    config['model']["continuous_len"] = dataset.con_conditions.shape[1]
    model, trajectory_generation_model = get_models(config["model"], dataset.parameters, checkpoint, dataset.scaler)
    #model.eval()
    batch_size = dataset_config["batch_size"]
    n = 200
    n_samples = 2
    logger.log_metrics({"n reconstructions": n, "n samples per" : n_samples})
    
    reconstructions, mse, rnd, fig_0 = reconstruct_and_plot(dataset, model, trajectory_generation_model, n=n, model_name = model_name)
    logger.log_metrics({"Eval_MSE": mse})
    logger.experiment.log_figure(logger.run_id,fig_0, f"figures/Eval_reconstruction.png")
    #logger.experiment.log_figure(logger.run_id, fig, "figures/my_plot.png")
    #print(reconstructions[1].data)
    JSD, KL, e_distance, fig_1 = jensenshannon_distance(reconstructions[0].data, reconstructions[1].data, model_name = model_name)
    logger.log_metrics({"Eval_edistance": e_distance, "Eval_JSD": JSD, "Eval_KL": KL})
    logger.experiment.log_figure(logger.run_id, fig_1, f"figures/Eval_comparison.png")
    #density(reconstructions, model_name = model_name)
    fig_landing = plot_traffic_comparison(reconstructions, 2, f"./figures/{model_name}_", landing = True)
    fig_takeoff = plot_traffic_comparison(reconstructions, 2, f"./figures/{model_name}_", landing = False)
    logger.experiment.log_figure(logger.run_id, fig_landing, f"figures/landing_comparison.png")
    logger.experiment.log_figure(logger.run_id, fig_takeoff, f"figures/takeoff_comparison.png")
    length = config['model']['traj_length']
    
    samples, steps = generate_samples(dataset, model, rnd, n = n_samples, length = length)
    
    #fig_99 = get_figure_from_sample_steps(steps, dataset, length)
    #fig_99.savefig(f"./figures/{model_name}_generated_steps.png")
    #logger.experiment.log_figure(logger.run_id, fig_99, f"figures/generated_steps.png")


    detached_samples = detach_to_tensor(samples).reshape(-1, len(dataset.features), length)
    reco_x = detached_samples.transpose(0, 2, 1).reshape(detached_samples.shape[0], -1)
    decoded = dataset.scaler.inverse_transform(reco_x)
    

    
    X = dataset[rnd][0].reshape(-1, length, len(dataset.features))[:,:,:2]
    X_gen = decoded.reshape(-1, length, len(dataset.features))[:,:,:2]
    fig_pca = data_diversity(X, X_gen, 'PCA', 'samples', model_name=model_name)
    #fig_tsne = data_diversity(X, X_gen, 't-SNE', model_name = model_name)
    logger.experiment.log_figure(logger.run_id, fig_pca, f"figures/pca.png")
    #logger.experiment.log_figure(logger.run_id, fig_tsne, f"figures/tsne.png")

    ### SECOND

    reconstructed_traf = trajectory_generation_model.build_traffic(
    decoded,
    coordinates=dict(latitude=48.5, longitude=8.4),
    forward=False,
    )

    JSD, KL, e_distance, fig_1 = jensenshannon_distance(reconstructions[0].data,reconstructed_traf.data , model_name = model_name)
    logger.log_metrics({"Eval_edistance_generation": e_distance, "Eval_JSD_generation": JSD, "Eval_KL_generation": KL})
    logger.experiment.log_figure(logger.run_id, fig_1, f"figures/Eval_comparison_generated.png")

    fig_2 = plot_from_array(reconstructed_traf, model_name)
    logger.experiment.log_figure(logger.run_id, fig_2, f"figures/Eval_generated_samples.png")
    reconstructed_traf.to_pickle(f"./artifacts/{model_name}/generated_samples.pkl")

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
    plt.savefig("figures/fidelity")
    logger.experiment.log_figure(logger.run_id, disp.figure_, f"figures/fidelity.png")

    logger.finalize()

def get_traffic_from_tensor(data, dataset, trajectory_generation_model):
    print(data.shape)
    reco_x = data.transpose(0, 2, 1).reshape(data.shape[0], -1)
    n = data.shape[0]
    # Inverse scaling and traffic reconstruction
    decoded = dataset.scaler.inverse_transform(reco_x)
    reconstructed_traf = trajectory_generation_model.build_traffic(
        decoded.reshape(n, -1, len(dataset.features)),
        coordinates=dict(latitude=48.5, longitude=8.4),
        forward=False
    )
    return reconstructed_traf
    
        



def run_perturbation(args, logger = None):
    np.random.seed(42)
    model_name = "PerturbationModel"

    data_path = args.data_path
    #artifact_location= "./artifacts"
    #checkpoint = f"./artifacts/{model_name}/best_model.ckpt"
    config_file = f"./configs/config.yaml"
    model = PerturbationModel()

    config = load_config(config_file)
    runid = None
    if logger is not None:
        runid = logger.run_id

    if logger is None:
        logger_config = config["logger"]
        logger = MLFlowLogger(
            experiment_name=logger_config["experiment_name"],
            run_name=args.model_name,
            tracking_uri=logger_config["mlflow_uri"],
            tags=logger_config["tags"],
            #artifact_location=artifact_location,
        )

        if runid is not None:
            logger.run_id = runid


    logger.experiment.log_dict(logger.run_id,config, config_file)
    config, dataset, traffic, conditions = get_config_data(config_file, data_path, "")
    config['model']["traj_length"] = dataset.parameters['seq_len']
    config['model']["continuous_len"] = dataset.con_conditions.shape[1]
    n = 10
    n_samples = 3
    logger.log_metrics({"n reconstructions": n, "n samples per" : n_samples})
    length = config['data']['length']

    trajectory_generation_model = Generation(
        generation=model,
        features=dataset.parameters['features'],
        scaler=dataset.scaler,
    )
    
    rnd = np.random.randint(0, len(dataset), (n,))
    samples = []

    for i in tqdm(rnd):
        # Load the i-th sample from the dataset
        x, con, cat, grid = dataset[i]
        # Generate samples and steps using the model
        sample = model.sample(x, n_samples = n_samples)
        samples.append(sample)


    #samples, steps = generate_samples(dataset, model, rnd, n = n_samples, length = length)
    detached_samples = detach_to_tensor(samples).reshape(-1, len(dataset.features), length)
    decoded = get_traffic_from_tensor(detached_samples, dataset, trajectory_generation_model)
    X_traffic = get_traffic_from_tensor(dataset[rnd][0].cpu().numpy().reshape(-1, len(dataset.features), length), dataset, trajectory_generation_model)

    JSD, KL, e_distance, fig_1 = jensenshannon_distance(X_traffic.data,decoded.data , model_name = model_name)
    logger.log_metrics({"Eval_edistance_generation": e_distance, "Eval_JSD_generation": JSD, "Eval_KL_generation": KL})
    logger.experiment.log_figure(logger.run_id, fig_1, f"figures/Eval_comparison_generated.png")

    fig_2 = plot_from_array(decoded, model_name)
    logger.experiment.log_figure(logger.run_id, fig_2, f"figures/Eval_generated_samples.png")

    training_trajectories = X_traffic
    synthetic_trajectories = decoded

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
    plt.savefig("figures/fidelity")
    logger.experiment.log_figure(logger.run_id, disp.figure_, f"figures/fidelity.png")

    logger.finalize()


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
        default="./data/resampled/combined_traffic_resampled_600.pkl",
        help="Path to the training data file"
    )

    parser.add_argument(
        "--dataset_config",
        type=str,
        #required=True,
        default="./configs/dataset_opensky.yaml",
        help="Path to the dataset config file"
    )
    
    args = parser.parse_args()
    run(args)
    #run_perturbation(args)
