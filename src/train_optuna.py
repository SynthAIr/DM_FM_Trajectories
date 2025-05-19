import argparse
from datetime import datetime
import os
from typing import Any, Dict, Tuple
import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, Callback
from optuna.integration import PyTorchLightningPruningCallback
from lightning.pytorch.loggers import MLFlowLogger
from utils.helper import load_config, save_config, load_and_prepare_data, get_model, init_config, init_model_config, get_model_train
from utils.train_utils import get_dataloaders
import optuna


def train(
    train_config: Dict[str, Any],
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    logger: MLFlowLogger,
    artifact_location: str,
    trial: optuna.trial.Trial
) -> None:
    """
    Train the model with the given configuration and data loaders.
    Parameters
    ----------
    train_config
    model
    train_loader
    val_loader
    test_loader
    logger
    artifact_location
    trial

    Returns
    -------

    """
    seed_everything(train_config["seed"], workers=True)

    class FlexibleDeviceCheckCallback(Callback):
        def __init__(self, expected_device_ids):
            self.expected_device_ids = expected_device_ids

        def on_train_start(self, trainer, pl_module):
            active_device = trainer.strategy.root_device
            if active_device.type == "cuda":
                print(f"Training is running on device: {active_device} ({torch.cuda.get_device_name(active_device.index)})")
                assert active_device.index in self.expected_device_ids, (
                    f"Trainer is running on GPU {active_device.index}, but expected one of: {self.expected_device_ids}."
                )
            else:
                print("Running on CPU")

    trainer = Trainer(
        accelerator=train_config["accelerator"],
        devices=[train_config["devices"]],
        max_epochs=train_config["epochs"],
        gradient_clip_val=train_config["gradient_clip_val"],
        log_every_n_steps=train_config["log_every_n_steps"],
        strategy="auto",
        #strategy="ddp_ipfind_unused_parameters_true",
        logger=logger,
        callbacks=[
            PyTorchLightningPruningCallback(trial, monitor="valid_loss"),
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
            FlexibleDeviceCheckCallback(expected_device_ids=[train_config["devices"]]),
        ],
    )

    torch.set_float32_matmul_precision(precision=train_config["precision"])
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader, ckpt_path="best")


def setup_logger(args, config):
    logger_config = config["logger"]
    run_name, artifact_location = get_unique_run_name_and_artile_location(logger_config)

    logger_config["experiment_name"] = logger_config["experiment_name"] + "_optuna" 
    logger = MLFlowLogger(
        experiment_name=logger_config["experiment_name"],
        run_name=run_name,
        tracking_uri=logger_config["mlflow_uri"],
        tags=logger_config["tags"],
    )
    print("Logger setup!")
    return logger, run_name, artifact_location


def run(args: argparse.Namespace, trial: optuna.Trial = None):
    """
    Pipeline to do a hyperparameter search with Optuna for each model with different parameters.
    Parameters
    ----------
    args
    trial

    Returns
    -------

    """
    config = load_config(args.config_file)
    dataset_config = load_config(args.dataset_config)
    config = init_config(config, dataset_config, args, experiment="cloud coverage real")

    if hasattr(args, 'model_name'):
        config['logger']['run_name'] = args.model_name

    #model_str = "fm"
    #model_str = "vae"
    #model_str = "lfm"
    model_str = "ldm"

    if trial:
        if model_str == "diffusion":
            config['train']['learning_rate'] = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
            config['model']['guidance_scale'] = trial.suggest_categorical("guidance_scale", [0, 1, 3, 5, 10.0])
            config['model']['diffusion']['beta_end'] = trial.suggest_float("beta_end", 0.03, 0.08)
            config['model']['diffusion']['num_diffusion_timesteps'] = trial.suggest_categorical("num_diffusion_timesteps", [300, 400, 500])
            config['model']['diffusion']['beta_schedule'] = trial.suggest_categorical("beta_schedule", ["linear", "cosine"])
        elif model_str == "fm":
            config['train']['learning_rate'] = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
            config['model']['guidance_scale'] = trial.suggest_categorical("guidance_scale", [0, 1, 3, 5, 10])
            config['model']['num_res_blocks'] = trial.suggest_categorical("num_res_blocks", [4, 6, 8, 10])
            config['model']['attr_dim'] = trial.suggest_categorical("attr_dim", [32, 64, 128])
        elif model_str == "vae":
            config['train']['learning_rate'] = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
            #config['model']['kld_coef'] = trial.suggest_categorical("kld_coef", [0.000001, 0.0001, 0.01, 1])
            category_hdims = trial.suggest_categorical("h_dims", [0, 1, 2])
            dataset_config["batch_size"] = 1024

            if category_hdims == 0:
                config['model']['h_dims'] = [256, 128, 256]
            if category_hdims == 1:
                config['model']['h_dims'] = [256, 512, 512, 256]
            if category_hdims == 2:
                config['model']['h_dims'] = [256, 512, 256]

        elif model_str == "lfm":
            config['train']['learning_rate'] = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
            category_vae = trial.suggest_categorical("vae", [0, 1])
            config['model']['guidance_scale'] = trial.suggest_categorical("guidance_scale", [0, 1, 3, 5, 10])
            config['model']['num_res_blocks'] = trial.suggest_categorical("num_res_blocks", [4, 6, 8, 10])

            if category_vae == 0:
                config['model']['vae'] = "./artifacts/VAE_2"
            if category_vae == 1:
                config['model']['vae'] = "./artifacts/VAE_4"

        elif model_str == "ldm":
            config['train']['learning_rate'] = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
            category_vae = trial.suggest_categorical("vae", [0, 1])
            config['model']['guidance_scale'] = trial.suggest_categorical("guidance_scale", [0, 1, 3, 5, 10])
            config['model']['num_res_blocks'] = trial.suggest_categorical("num_res_blocks", [4, 6, 8, 10])

            if category_vae == 0:
                config['model']['vae'] = "./artifacts/VAE_2"
            if category_vae == 1:
                config['model']['vae'] = "./artifacts/VAE_4"
            config['model']['diffusion']['beta_end'] = trial.suggest_float("beta_end", 0.03, 0.08)
            config['model']['diffusion']['num_diffusion_timesteps'] = trial.suggest_categorical("num_diffusion_timesteps", [300, 500])
            config['model']['diffusion']['beta_schedule'] = trial.suggest_categorical("beta_schedule", ["linear", "cosine"])


        


    l_logger, run_name, artifact_location = setup_logger(args, config)

    if trial:
        if model_str == "diffusion":
            l_logger.log_metrics({"lr": config['train']['learning_rate']})
            l_logger.log_metrics({"guidance_scale": config['model']['guidance_scale']})
            l_logger.log_metrics({"beta_end": config['model']['diffusion']['beta_end']})
            l_logger.log_metrics({"num_diffusion_timesteps": config['model']['diffusion']['num_diffusion_timesteps']})
            l_logger.log_metrics({"beta_schedule": config['model']['diffusion']['beta_schedule']})
        elif model_str == "fm":
            l_logger.log_metrics({"lr": config['train']['learning_rate']})
            l_logger.log_metrics({"guidance_scale": config['model']['guidance_scale']})
            l_logger.log_metrics({"num_res_blocks": config['model']['num_res_blocks']})
            l_logger.log_metrics({"attr_dim": config['model']['attr_dim']})
        elif model_str == "vae":
            l_logger.log_metrics({"lr": config['train']['learning_rate']})
            l_logger.log_metrics({"kld_coef": config['model']['kld_coef']})
            l_logger.log_metrics({"h_dims": str(config['model']['h_dims'])})
        elif model_str == "lfm":
            l_logger.log_metrics({"lr": config['train']['learning_rate']})
            l_logger.log_metrics({"guidance_scale": config['model']['guidance_scale']})
            l_logger.log_metrics({"num_res_blocks": config['model']['num_res_blocks']})
            l_logger.log_metrics({"vae": config['model']['vae']})
        elif model_str == "ldm":
            l_logger.log_metrics({"lr": config['train']['learning_rate']})
            l_logger.log_metrics({"guidance_scale": config['model']['guidance_scale']})
            l_logger.log_metrics({"beta_end": config['model']['diffusion']['beta_end']})
            l_logger.log_metrics({"num_diffusion_timesteps": config['model']['diffusion']['num_diffusion_timesteps']})
            l_logger.log_metrics({"beta_schedule": config['model']['diffusion']['beta_schedule']})
            l_logger.log_metrics({"vae": category_vae})


    dataset_config["data_path"] = args.data_path
    dataset, traffic = load_and_prepare_data(dataset_config)

    train_loader, val_loader, test_loader = get_dataloaders(
        dataset,
        dataset_config["train_ratio"],
        dataset_config["val_ratio"],
        dataset_config["batch_size"],
        dataset_config["test_batch_size"],
    )

    model_config = init_model_config(config, dataset_config, dataset)
    model = get_model_train(dataset, model_config, dataset_config, args)

    train_config = config["train"]
    train_config["devices"] = args.cuda
    start_time = datetime.now()
    train(train_config, model, train_loader, val_loader, test_loader, l_logger, artifact_location, trial)
    end_time = datetime.now()

    config["data"] = dataset_config
    save_config(config, os.path.join(artifact_location, "config.yaml"))
    checkpoint_path = artifact_location + "/best_model.ckpt"
    model_size = os.path.getsize(checkpoint_path) / (1024 * 1024)
    l_logger.log_metrics({"Size (MB)": model_size})
    l_logger.log_metrics({"training_time_seconds": (end_time - start_time).total_seconds()})

    l_logger.experiment.log_dict(l_logger.run_id, config, "config.yaml")
    return l_logger, run_name, artifact_location


def get_unique_run_name_and_artile_location(logger_config: Dict[str, Any]) -> Tuple[str, str]:
    run_name = logger_config["run_name"]
    artifact_location = logger_config["artifact_location"]
    os.makedirs(artifact_location, exist_ok=True)
    artifact_location = os.path.join(artifact_location, run_name)
    if os.path.exists(artifact_location):
        suffix = 1
        while os.path.exists(artifact_location + f"_{suffix}"):
            suffix += 1
        run_name = run_name + f"_{suffix}"
        artifact_location = artifact_location + f"_{suffix}"
    return run_name, artifact_location


def objective(trial: optuna.Trial):
    args = argparse.Namespace(
        #config_file="./configs/config.yaml",
        #config_file="./configs/config_fm.yaml",
        #config_file="./configs/config_fm_lat.yaml",
        config_file="./configs/config_lat.yaml",
        #config_file="./configs/tcvae_config.yaml",
        dataset_config="./configs/dataset_landing_transfer.yaml",
        data_path="/mnt/data/synthair/synthair_diffusion/data/resampled/combined_traffic_resampled_landing_LSZH_200.pkl",
        artifact_location="./artifacts",
        cuda=0,
        pruning=True
    )
    logger, run_name, artifact_location = run(args, trial)
    return logger.experiment.get_run(logger.run_id).data.metrics.get("valid_loss", 1e6)


if __name__ == "__main__":
    def run_optuna_study():
        pruner = optuna.pruners.MedianPruner()
        study = optuna.create_study(direction="minimize", pruner=pruner)
        study.optimize(objective, n_trials=30)
        print("Best trial:")
        print(study.best_trial.params)

    run_optuna_study()

