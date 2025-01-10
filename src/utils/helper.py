import torch
import yaml
from sklearn.datasets import make_swiss_roll
from utils.condition_utils import load_conditions
from traffic.core import Traffic
from sklearn.preprocessing import MinMaxScaler
from utils.data_utils import TrafficDataset
from model.AirDiffTraj import AirDiffTrajDDIM ,AirDiffTrajDDPM
from typing import Tuple
from model.baselines import PerturbationModel, TimeGAN
from model.AirLatDiffTraj import LatentDiffusionTraj
from model.tcvae import TCVAE

def sample_batch(size, noise=1.0):
    x, _= make_swiss_roll(size, noise=noise)
    return x[:, [0, 2]] / 10.0

def extract(input, t, x):
    shape = x.shape
    out = torch.gather(input, 0, t.to(input.device))
    reshape = [t.shape[0]] + [1] * (len(shape) - 1)
    return out.reshape(*reshape)


def load_config(config_file):
    with open(config_file, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


def save_config(config, config_file):
    with open(config_file, "w") as f:
        yaml.dump(config, f)

def load_and_prepare_data(dataset_config):
    """
    Load and prepare the dataset for the model.
    """
    dataset = TrafficDataset.from_file(
        dataset_config["data_path"],
        features=dataset_config["features"],
        shape=dataset_config["data_shape"],
        scaler=MinMaxScaler(feature_range=(-1, 1)),
        conditional_features = load_conditions(dataset_config) ,
        variables = dataset_config["weather_grid"]["variables"]
    )
    traffic = Traffic.from_file(dataset_config["data_path"])

    return dataset, traffic

def get_model(configs):
    match configs["type"]:
        case "DDPM":
            return AirDiffTrajDDPM
        case "DDIM":
            return AirDiffTrajDDIM
        case "PER":
            return PerturbationModel
        case "LatDiff":
            return LatentDiffusionTraj
        case "TimeGAN":
            return TimeGAN
        case "TCVAE":
            return TCVAE
        case "VAE":
            return TCVAE
        case _:
            return AirDiffTrajDDPM

def extract_geographic_info(
    trajectories: Traffic,
    lon_padding: float = 1,
    lat_padding: float = 1,
) -> Tuple[float, float, float, float, float, float]:

    # Determine the geographic bounds for plotting
    lon_min = trajectories.data["longitude"].min()
    lon_max = trajectories.data["longitude"].max()
    lat_min = trajectories.data["latitude"].min()
    lat_max = trajectories.data["latitude"].max()

    geographic_extent = [
        lon_min - lon_padding,
        lon_max + lon_padding,
        lat_min - lat_padding,
        lat_max + lat_padding,
    ]

    return geographic_extent
