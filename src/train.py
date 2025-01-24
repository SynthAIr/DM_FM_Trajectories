import argparse
import os
from typing import Any, Dict, Tuple
import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from sklearn.preprocessing import MinMaxScaler
import yaml
from utils import TrafficDataset
from utils.helper import load_config, save_config, load_and_prepare_data, get_model
from utils.train_utils import get_dataloaders
from utils.condition_utils import load_conditions
from model.AirLatDiffTraj import Phase
from model.diffusion import Diffusion
from model.tcvae import TCVAE
from lightning.pytorch.accelerators import find_usable_cuda_devices
from model.flow_matching import FlowMatching, Wrapper


def train(
    train_config: Dict[str, Any],
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    logger: MLFlowLogger,
    artifact_location: str,
) -> None:
    seed_everything(train_config["seed"], workers=True)

    # Configure the trainer with specifics from the train_config.
    trainer = Trainer(
        accelerator=train_config["accelerator"],
        devices=find_usable_cuda_devices(train_config["devices"]),
        max_epochs=train_config["epochs"],
        gradient_clip_val=train_config["gradient_clip_val"],
        log_every_n_steps=train_config["log_every_n_steps"],
        strategy="ddp_find_unused_parameters_true",
        logger=logger,
        callbacks=[
            EarlyStopping(
                monitor="valid_loss",
                patience=train_config["early_stop_patience"],
            ),
            ModelCheckpoint(
                monitor="valid_loss",
                dirpath=artifact_location,
                filename="best_model",
                save_top_k=1,
                mode="min",
            ),
        ],
    )

    # Set precision for matrix multiplication
    # torch.set_float32_matmul_precision("highest") # Default used by PyTorch
    # torch.set_float32_matmul_precision("high") # Faster, but less precise
    # torch.set_float32_matmul_precision("medium") # Even faster, but also less precise
    torch.set_float32_matmul_precision(precision=train_config["precision"])
    # Start the model training and validation process.
    trainer.fit(model, train_loader, val_loader)
    # Optionally evaluate the model on test data using the best model checkpoint.
    trainer.test(model, test_loader, ckpt_path="best")

def setup_logger(args, configs):
    """Setup the logger with MLFlow configurations."""
    logger_config = configs["logger"]
    run_name, artifact_location = get_unique_run_name_and_artile_location(logger_config)

    logger = MLFlowLogger(
        experiment_name=logger_config["experiment_name"],
        run_name=run_name,
        tracking_uri=logger_config["mlflow_uri"],
        tags=logger_config["tags"],
    )
    print("Logger setup!")
    return logger, run_name, artifact_location


def run(args: argparse.Namespace):
    configs = load_config(args.config_file)
    dataset_config = load_config(args.dataset_config)
    configs["logger"]["artifact_location"] = args.artifact_location
    configs["logger"]["tags"]['dataset'] = dataset_config["dataset"]
    #configs["logger"]["tags"]['weather_grid'] = configs["model"]["weather_config"]["weather_grid"]

    # Setup logger with MLFlow with configurations read from the file.
    l_logger, run_name, artifact_location = setup_logger(args, configs)

    #dataset_config = configs["data"]
    dataset_config["data_path"] = args.data_path
    dataset, traffic = load_and_prepare_data(dataset_config)
    #conditional_features = load_conditions(dataset_config)
    print(dataset.data.shape)
    print(dataset.con_conditions.shape, dataset.cat_conditions.shape)

    train_loader, val_loader, test_loader = get_dataloaders(
        dataset,
        dataset_config["train_ratio"],
        dataset_config["val_ratio"],
        dataset_config["batch_size"],
        dataset_config["test_batch_size"],
    )
    print("Dataset loaded!")


    print("Dataset loaded!")
    model_config = configs["model"]
    model_config["data"] = dataset_config
    # print(f"*******dataset parameters: {dataset.parameters}")
    model_config["traj_length"] = dataset.parameters['seq_len']
    model_config["continuous_len"] = dataset.con_conditions.shape[1]
    print(f"*******model parameters: {model_config}")

    if model_config["type"] == "LatDiff" or model_config["type"] == "LatFM":
        temp_conf = {"type": "TCVAE"}
        config_file = f"{model_config['vae']}/config.yaml"
        checkpoint = f"{model_config['vae']}/best_model.ckpt"
        c = load_config(config_file)
        c = c['model']
        c["traj_length"] = dataset.parameters['seq_len']
        c['data'] = dataset_config
        vae = get_model(temp_conf).load_from_checkpoint(checkpoint, dataset_params = dataset.parameters, config = c)
        vae.eval()

        if model_config["type"] == "LatDiff":
            print("Initing LatDiff")
            diff = Diffusion(model_config)
        else:
            print("Initing LatFM")
            m = FlowMatching(model_config)
            diff = Wrapper(model_config, m)
        #model = get_model(model_config).load_from_checkpoint("artifacts/AirLatDiffTraj_5/best_model.ckpt", dataset_params = dataset.aset_params, config = model_config, vae=vae, generative = diff)
        model = get_model(model_config)(model_config, vae, diff)
    elif model_config["type"] == "FM":
        model_config["traj_length"] = dataset.parameters['seq_len']
        fm = FlowMatching(model_config)
        model = get_model(model_config)(model_config, fm)
    else:
        model = get_model(model_config)(model_config)

        

    print("Model built!")

    # Initiate training with the setup configurations and prepared dataset and model.
    train_config = configs["train"]
    train(train_config, model, train_loader, val_loader, test_loader, l_logger, artifact_location)
    # Save configuration used for the training in the logger's artifact location.
    configs["data"] = dataset_config
    save_config(configs, os.path.join(artifact_location, "config.yaml"))
    #l_logger.
    checkpoint_path = artifact_location + "/best_model.ckpt"
    model_size = os.path.getsize(checkpoint_path) / (1024 * 1024)  #
    l_logger.log_metrics({"Size (MB)": model_size})

    return l_logger, run_name, artifact_location


def get_unique_run_name_and_artile_location(
    logger_config: Dict[str, Any]
) -> Tuple[str, str]:
    run_name = logger_config["run_name"]
    artifact_location = logger_config["artifact_location"]
    os.makedirs(artifact_location, exist_ok=True)
    artifact_location = os.path.join(artifact_location, run_name)
    # check if run_name already exists, then add a suffix (account for already existing suffixes)
    if os.path.exists(artifact_location):
        suffix = 1
        while os.path.exists(artifact_location + f"_{suffix}"):
            suffix += 1
        run_name = run_name + f"_{suffix}"
        artifact_location = artifact_location + f"_{suffix}"

    return run_name, artifact_location



if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Train a Air traffic trajectory generation model"
    )
    parser.add_argument(
        "--config_file",
        type=str,
        #required=True,
        default="./configs/config.yaml",
        help="Path to the config file"
    )

    parser.add_argument(
        "--dataset_config",
        type=str,
        #required=True,
        default="./configs/dataset_opensky.yaml",
        help="Path to the dataset config file"
    )
    
    parser.add_argument(
        "--data_path",
        type=str,
        #required=True,
        default="./data/OpenSky_EHAM_LIMC.pkl",
        help="Path to the training data file"
    )
    parser.add_argument(
        "--artifact_path",
        dest="artifact_location",
        type=str,
        #required=True,
        default="./artifacts",
        help="Path to save the artifacts",
    )
    args = parser.parse_args()
    run(args)
    
