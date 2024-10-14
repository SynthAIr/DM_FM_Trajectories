import argparse
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
from model.Traj_UNet import Guide_UNet2
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
    #model = Guide_UNet2.load_from_checkpoint(checkpoint_path, map_location=torch.device('cuda'))
    model = Guide_UNet2.load_from_checkpoint(checkpoint_path, dataset_params = dataset_params, config = model_config)
    model.eval()  # Set the model to evaluation mode
    print("Model loaded with checkpoint!")
    
    """
    trajectory_generation_model = Generation(
        generation=trained_model,
        features=trained_model.hparams.dataset_params["features"],
        scaler=dataset_scaler,
    )
    """
    print("Trajectory generation model created!")

    return model

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

def generate_samples(model, n, c_, t):
    raise NotImplementedError("Juhu")

def run(args):
    config = load_config(args.config_file)
    args.checkpoint = f"./artifacts/AirDiffTraj/best_model.ckpt"
    #checkpoint_path = get_checkpoint_path(config["logger"])
    config, dataset, traffic, conditions = get_config_data(args.config_file, args.data_path, args.artifact_location)
    config['model']["traj_length"] = dataset.parameters['seq_len']

    model = get_models(config["model"], dataset.parameters, args.checkpoint, dataset.scaler)

    _, con, cat = dataset[0]
    print(con.shape, cat.shape)


    # Download and load the training dataset
    dataset_config = config["data"]
    batch_size = dataset_config["batch_size"]
   #train_dataset = FashionMNIST(root='./data', train=True, transform=transform)
    x, con, cat = dataset[0]
    con = con.reshape(1, -1)
    cat = cat.reshape(1, -1)
    #x = x.view(-1, 1, 28, 28)
    n = 10
    samples = model.sample(n, con, cat)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Parser for training a model with PyTorch Lightning"
    )

    args = parser.parse_args()
    args.config_file = "./configs/config.yaml"
    args.data_path = "./data/OpenSky_EHAM_LIMC.pkl"
    args.artifact_location= "./artifacts"
    args.checkpoint = "./artifacts/AirDiffTraj_3/best_model.ckpt"

    run(args)
