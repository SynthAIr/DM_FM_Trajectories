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
from model.flow_matching import AirFMTraj, FlowMatching, Wrapper
from model.diffusion import Diffusion

def sample_batch(size, noise=1.0):
    x, _= make_swiss_roll(size, noise=noise)
    return x[:, [0, 2]] / 10.0

def extract(input, t, x):
    shape = x.shape
    out = torch.gather(input, 0, t.to(input.device))
    reshape = [t.shape[0]] + [1] * (len(shape) - 1)
    return out.reshape(*reshape)


def load_config_2(config_file):
    with open(config_file, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

#from ruamel.yaml import YAML
def load_config(file_path):
    """Load YAML and sort keys alphabetically."""
    """Load YAML file and sort keys alphabetically."""
    def recursively_sort_dict(d):
        if isinstance(d, dict):
            return {k: recursively_sort_dict(v) for k, v in sorted(d.items())}
        elif isinstance(d, list):
            return [recursively_sort_dict(i) for i in d]
        return d

    with open(file_path, "r") as f:
        data = yaml.safe_load(f)
    return recursively_sort_dict(data)


def save_config(config, config_file):
    #with open(config_file, "w") as f:
    #    yaml.dump(config, f)
    with open(config_file, "w") as f:
        yaml.dump(config, f, sort_keys=True, default_flow_style=False)

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
        variables = dataset_config["weather_grid"]["variables"] if dataset_config["weather_grid"]["enabled"] else [],
        metar=dataset_config["metar"],
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
        case "LatFM":
            return LatentDiffusionTraj
        case "LatDiff":
            return LatentDiffusionTraj
        case "TimeGAN":
            return TimeGAN
        case "TCVAE":
            return TCVAE
        case "VAE":
            return TCVAE
        case "FM":
            return AirFMTraj
        case _:
            raise NotImplemetedError("Invalid model name")

def init_config(config, dataset_config, args, experiment = "None"):
    config["logger"]["artifact_location"] = args.artifact_location
    config["logger"]["tags"]['dataset'] = dataset_config["dataset"]
    config["logger"]["tags"]['weather'] = str(config["model"]["weather_config"]["weather_grid"])
    config["logger"]["tags"]['experiment'] = experiment
    return config

def init_model_config(config, dataset_config, dataset):
    model_config = config["model"]
    model_config["data"] = dataset_config
    model_config["in_channels"] = len(dataset_config["features"])
    model_config["out_ch"] = len(dataset_config["features"])
    model_config["weather_config"]["variables"] = len(dataset_config["weather_grid"]["variables"])
    model_config["weather_config"]["weather_grid"] = dataset_config["weather_grid"]["enabled"]
    # print(f"*******dataset parameters: {dataset.parameters}")
    model_config["traj_length"] = dataset.parameters['seq_len']
    model_config["continuous_len"] = dataset.con_conditions.shape[1]
    return model_config

def get_model_train(dataset, model_config, dataset_config, args, pretrained_VAE = True):
    if model_config["type"] == "LatDiff" or model_config["type"] == "LatFM":
        temp_conf = {"type": "TCVAE"}
        config_file = f"{model_config['vae']}/config.yaml"
        checkpoint = f"{model_config['vae']}/best_model.ckpt"
        c = load_config(config_file)
        c = c['model']
        c["traj_length"] = dataset.parameters['seq_len']
        c['data'] = dataset_config
        if pretrained_VAE:
            print("Initing with pretrained VAE")
            vae = get_model(temp_conf).load_from_checkpoint(checkpoint, dataset_params = dataset.parameters, config = c)
        else:
            print("Initing not pretrained VAE")
            vae = get_model(temp_conf)(c)
        vae.eval()

        if model_config["type"] == "LatDiff":
            print("Initing LatDiff")
            diff = Diffusion(model_config, args.cuda)
        else:
            print("Initing LatFM")
            m = FlowMatching(model_config, args.cuda)
            diff = Wrapper(model_config, m, args.cuda)
        #model = get_model(model_config).load_from_checkpoint("artifacts/AirLatDiffTraj_5/best_model.ckpt", dataset_params = dataset.aset_params, config = model_config, vae=vae, generative = diff)
        model = get_model(model_config)(model_config, vae, diff)
    elif model_config["type"] == "FM":
        model_config["traj_length"] = dataset.parameters['seq_len']
        fm = FlowMatching(model_config, args.cuda, lat=True)
        model = get_model(model_config)(model_config, fm, args.cuda)
    else:
        model = get_model(model_config)(model_config)
    return model

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
