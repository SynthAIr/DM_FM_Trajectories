"""
Evaluation script for the thesis
Some functions are adapted from https://github.com/SynthAIr/SynTraj
"""
from datetime import datetime
import cartopy.feature
from typing import Any, Dict
import os
from lightning.pytorch import Trainer, seed_everything
from utils import load_config
from traffic.core import Traffic
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
from model.flow_matching import FlowMatching, Wrapper
import cvxopt
from traffic.algorithms.generation import compute_latlon_from_trackgs
import pandas as pd
import seaborn as sns
from model.baselines import PerturbationModel
import torch
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy
import argparse
from traffic.algorithms.generation import Generation


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
    if model_config["type"] == "LatDiff" or model_config["type"] == "LatFM":
        temp_conf = {"type": "TCVAE"}
        config_file = f"{model_config['vae']}/config.yaml"
        checkpoint = f"{model_config['vae']}/best_model.ckpt"
        c = load_config(config_file)
        c = c['model']
        c["traj_length"] = model_config['traj_length']
        vae = get_model(temp_conf).load_from_checkpoint(checkpoint, dataset_params = dataset_params, config = c)
        if model_config["type"] == "LatDiff":
            diff = Diffusion(model_config)
        else:
            m = FlowMatching(model_config)
            diff = Wrapper(model_config, m)
        print("Loading", checkpoint_path)
        model = get_model(model_config).load_from_checkpoint(checkpoint_path, dataset_params = dataset_params, config = model_config, vae=vae, generative = diff)
    elif model_config["type"] == "FM":
        fm = FlowMatching(model_config, "0", lat=True)
        model = get_model(model_config).load_from_checkpoint(checkpoint_path, dataset_params = dataset_params, config = model_config, model = fm, cuda = "0")
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



def exponentially_weighted_moving_average(data, alpha=0.3):
    """
    This function is adapted from https://github.com/SynthAIr/TimeGAN_Trajectories

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
    """
    Calculate the Mean Squared Error (MSE) between two DataFrames for specified columns.
    Parameters
    ----------
    df1
    df2
    columns_to_compare

    Returns
    -------

    """
    
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
    """
    Reconstruct and plot the data using the given model and dataset.
    Parameters
    ----------
    dataset
    model
    trajectory_generation_model
    n
    model_name
    rnd
    d

    Returns
    -------

    """
    rnd = np.random.randint(0, len(dataset), (n,)) if rnd is None else rnd
    X2, con, cat, grid = dataset[rnd]
    
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



def generate_samples(dataset, model, rnd, n=10, length=200, device = "cuda:0"):
    """
    Generate samples from the model using the given dataset and random indices.
    Parameters
    ----------
    dataset
    model
    rnd
    n
    length
    device

    Returns
    -------

    """
    all_samples = []
    all_steps = []
    perturb = isinstance(model, PerturbationModel)
    for i in tqdm(rnd):
        x, con, cat, grid = dataset[i]
        
        # Reshape con and cat as required
        con = con.reshape(1, -1).to(device)
        cat = cat.reshape(1, -1).to(device)
        
        grid = grid.unsqueeze(dim=0).to(device)

        # Generate samples and steps using the model
        if not perturb:
            samples, steps = model.sample(n, con, cat, grid, length, features=x.shape[0])
        else:
            samples, steps = model.reconstruct(x, con, cat, grid)


        all_samples.append(samples)
        all_steps.append(steps)
        

    return all_samples, all_steps


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
    #plt.savefig(f"./figures/{model_name}_generated_samples.png")
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
        
    plt.style.use("ggplot")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), 
                                  subplot_kw={'projection': ccrs.EuroPP()})
    axes = [ax1, ax2]
    
    for idx, traffic in enumerate(traffic_list):
        # Add map features to each subplot
        axes[idx].coastlines()
        axes[idx].add_feature(cartopy.feature.BORDERS, linestyle=":", alpha=1.0)
        
        if landing:
            t = traffic.last(minutes=n_samples).eval()
        else:
            t = traffic.first(minutes=n_samples).eval()
        
        t.plot(axes[idx], alpha=0.5, color="red", linewidth=0.3)
        
        axes[idx].set_xlabel('Longitude')
        axes[idx].set_ylabel('Latitude')
    
    plt.tight_layout()
    
    suffix = "landing.png" if landing else "take_off.png"
    full_filename = f"{output_filename}_{suffix}"
    
    #plt.savefig(full_filename, bbox_inches='tight', dpi=300)
    return fig

# Assuming 'samples' is a list of tensors generated by model.sample
# Detach and stack samples into a single tensor


def get_mse_distribution(mse_dict):
    """
    Method to plot the MSE distribution of the generated samples.
    Parameters
    ----------
    mse_dict

    Returns
    -------

    """
    mse_values_np = mse_dict["mse_dist"].cpu().numpy() if isinstance(mse_dict["mse_dist"], torch.Tensor) else np.array(mse_dict["mse_dist"])
    std_values_np = mse_dict["mse_dist_std"].cpu().numpy() if isinstance(mse_dict["mse_dist_std"], torch.Tensor) else np.array(mse_dict["mse_dist_std"])

    mse_mean = np.mean(mse_values_np, axis=0)  # Shape: (n_features,)
    mse_std = np.std(mse_values_np, axis=0)    # Shape: (n_features,)

    mse_lower = mse_mean - 2 * mse_std
    mse_upper = mse_mean + 2 * mse_std

    mse_filtered = np.where((mse_values_np >= mse_lower) & (mse_values_np <= mse_upper), mse_values_np, np.nan)

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
    """
    Get the logger for the evaluation.
    Parameters
    ----------
    logger
    dataset_config
    config

    Returns
    -------

    """
    if logger is not None:
        return logger

    if logger is None:
        logger_config = config["logger"]
        logger_config["tags"]["dataset"] = dataset_config["dataset"]
        logger = MLFlowLogger(
            experiment_name=logger_config["experiment_name"],
            run_name=args.model_name,
            tracking_uri=logger_config["mlflow_uri"],
            tags=logger_config["tags"],
        )
        logger.experiment.log_dict(logger.run_id, config, "config.yaml")

    return logger
    

def run_refactored(args, logger = None):
    """
    Main function to run the evaluation of the model.
    Parameters
    ----------
    args
    logger

    Returns
    -------

    """
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
    samples, steps = generate_samples(dataset, model, rnd, n = n_samples, length = length, device = device)
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


def get_traffic_from_tensor(data, dataset, trajectory_generation_model, rnd):
    """
    Get the traffic object from the tensor data.
    Parameters
    ----------
    data
    dataset
    trajectory_generation_model
    rnd

    Returns
    -------

    """
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

    v = np.ones(m) / m
    
    P = cvxopt.matrix(K_X)  # K_X for the quadratic term
    q = cvxopt.matrix(-2 * np.dot(K_XY, v))  # Linear term in the objective
    
    G = cvxopt.matrix(np.vstack([-np.eye(n), np.eye(n)]))  # Constraint on w: 0 <= w_i <= 1/(alpha * n)
    h = cvxopt.matrix(np.hstack([np.zeros(n), np.ones(n) * (1 / (alpha * n))]))  # Constraints on w
    
    A = cvxopt.matrix(np.ones((1, n)))  # Sum of w must be 1
    b = cvxopt.matrix(1.0)
    
    sol = cvxopt.solvers.qp(P, q, G, h, A, b)
    w = np.array(sol['x']).flatten()
    
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

    parser.add_argument(
            "--perturb",
            dest="perturb", 
            action='store_true')

    args = parser.parse_args()
    run_refactored(args)
    
