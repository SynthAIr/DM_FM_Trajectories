import argparse
from datetime import datetime
import os
from typing import Any, Dict, Tuple
import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, Callback
from lightning.pytorch.loggers import MLFlowLogger
from utils.helper import load_config, save_config, load_and_prepare_data, get_model, init_config, init_model_config, get_model_train
from utils.train_utils import get_dataloaders
from model.diffusion import Diffusion
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

    class FlexibleDeviceCheckCallback(Callback):
        def __init__(self, expected_device_ids):
            """
            Args:
                expected_device_ids (list[int]): The expected GPU device IDs, e.g., [0], [1].
            """
            self.expected_device_ids = expected_device_ids

        def on_train_start(self, trainer, pl_module):
            # Get the root device dynamically
            active_device = trainer.strategy.root_device

            # Check if the active device index is in the expected list
            if active_device.type == "cuda":
                print(f"Training is running on device: {active_device} ({torch.cuda.get_device_name(active_device.index)})")
                assert active_device.index in self.expected_device_ids, (
                    f"Trainer is running on GPU {active_device.index}, but expected one of: {self.expected_device_ids}."
                )
            else:
                #raise RuntimeError("Training is not running on a GPU. Please check your setup.")
                print("Running on CPU")


    # Configure the trainer with specifics from the train_config.
    trainer = Trainer(
        accelerator=train_config["accelerator"],
        #devices=find_usable_cuda_devices(train_config["devices"]),
        devices=[train_config["devices"]],
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
            FlexibleDeviceCheckCallback(expected_device_ids=[train_config["devices"]]),  # Add the flexible check
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

def setup_logger(args, config):
    """Setup the logger with MLFlow configurations."""
    logger_config = config["logger"]
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
    config = load_config(args.config_file)

    dataset_config = load_config(args.dataset_config)
    config = init_config(config, dataset_config, args, experiment = "cloud coverage real")
    l_logger, run_name, artifact_location = setup_logger(args, config)

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
    model_config = init_model_config(config, dataset_config, dataset)
    print(f"*******model parameters: {model_config}")
    model = get_model_train(dataset, model_config, dataset_config, args)

    print("Model built!")

    # Initiate training with the setup configurations and prepared dataset and model.
    train_config = config["train"]
    train_config["devices"] = args.cuda
    start_time = datetime.now()
    train(train_config, model, train_loader, val_loader, test_loader, l_logger, artifact_location)
    end_time = datetime.now()

    # Save configuration used for the training in the logger's artifact location.
    config["data"] = dataset_config
    save_config(config, os.path.join(artifact_location, "config.yaml"))
    #l_logger.
    checkpoint_path = artifact_location + "/best_model.ckpt"
    model_size = os.path.getsize(checkpoint_path) / (1024 * 1024)  #
    l_logger.log_metrics({"Size (MB)": model_size})
    l_logger.log_metrics({"training_time_seconds": (end_time - start_time).total_seconds()})

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

    parser.add_argument(
            "--cuda",
            type=int,
            default=0,
            help="GPU to use",
            )


    args = parser.parse_args()
    run(args)
    
