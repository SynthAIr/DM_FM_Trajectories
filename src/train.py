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
        devices=train_config["devices"],
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

    if False:
        trainer = Trainer(
        accelerator=train_config["accelerator"],
        devices=train_config["devices"],
        max_epochs=train_config["epochs"],
        gradient_clip_val=train_config["gradient_clip_val"],
        log_every_n_steps=train_config["log_every_n_steps"],
        strategy="ddp_find_unused_parameters_true",
        logger=logger,
        callbacks=[
            EarlyStopping(
                monitor="valid_loss_diffusion",
                patience=train_config["early_stop_patience"],
            ),
            ModelCheckpoint(
                monitor="valid_loss_diffusion",
                dirpath=artifact_location,
                filename="best_model",
                save_top_k=1,
                mode="min",
            ),
        ],
    )
        model.encoder.eval()
        model.decoder.eval()
        model.lsr.eval()
        trainer.fit(model, train_loader, val_loader)
        # Optionally evaluate the model on test data using the best model checkpoint.
        trainer.test(model, test_loader, ckpt_path="best")
        model.phase = Phase.EVAL



def run(args: argparse.Namespace):
    configs = load_config(args.config_file)
    configs["data"]["data_path"] = args.data_path
    configs["logger"]["artifact_location"] = args.artifact_location

    # Setup logger with MLFlow with configurations read from the file.
    logger_config = configs["logger"]
    run_name, artifact_location = get_unique_run_name_and_artile_location(logger_config)
    l_logger = MLFlowLogger(
        experiment_name=logger_config["experiment_name"],
        run_name=run_name,
        tracking_uri=logger_config["mlflow_uri"],
        tags=logger_config["tags"],
        #artifact_location=artifact_location,
    )
    print("Logger setup!")

    # Dataset preparation and loading.
    dataset_config = configs["data"]
    dataset, traffic = load_and_prepare_data(configs)
    #conditional_features = load_conditions(dataset_config)
    """
    dataset = TrafficDataset.from_file(
        dataset_config["data_path"],
        features=dataset_config["features"],
        shape=dataset_config["data_shape"],
        scaler=MinMaxScaler(feature_range=(-1, 1)),
        info_params={
            "features": dataset_config["info_features"],
            "index": dataset_config["info_index"],
        },
        conditional_features= conditional_features,
        down_sample_factor=dataset_config["down_sample_factor"],
        variables = dataset_config["weather_grid"]["variables"]
    )
    """
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
    # print(f"*******dataset parameters: {dataset.parameters}")
    model_config["traj_length"] = dataset.parameters['seq_len']
    model_config["continuous_len"] = dataset.con_conditions.shape[1]
    print(f"*******model parameters: {model_config}")
    if model_config["type"] == "LatDiff":
        temp_conf = {"type": "TCVAE"}
        config_file = f"{model_config['vae']}/config.yaml"
        checkpoint = f"{model_config['vae']}/best_model.ckpt"
        c = load_config(config_file)
        vae = get_model(temp_conf).load_from_checkpoint(checkpoint, dataset_params = dataset.parameters, config = c['model'])
        diff = Diffusion(model_config)
        model = get_model(model_config)(model_config, vae, diff)
        model.phase = Phase.DIFFUSION
        model.vae.eval()
    else:
        model = get_model(model_config)(model_config)

        

    print("Model built!")

    # Initiate training with the setup configurations and prepared dataset and model.
    train_config = configs["train"]
    train(train_config, model, train_loader, val_loader, test_loader, l_logger, artifact_location)
    # Save configuration used for the training in the logger's artifact location.
    save_config(configs, os.path.join(artifact_location, "config.yaml"))

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
    
