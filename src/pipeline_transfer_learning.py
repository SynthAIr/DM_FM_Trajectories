import argparse
import numpy as np
import lightning.pytorch as pl
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch import seed_everything
from utils.helper import load_and_prepare_data, get_model, save_config, load_config
from evaluate import get_models
from train import setup_logger, get_dataloaders, train


from evaluation.similarity import jensenshannon_distance

def load_model(pretrained=False):
    """Placeholder for model loading function."""
    return get_model(pretrained=pretrained)  # Replace with actual model loading function

def train_and_evaluate(model, train_loader, val_loader, logger, split):
    """Train the model and evaluate its performance."""
    trainer = pl.Trainer(logger=logger, max_epochs=10)  # Adjust epochs as needed
    trainer.fit(model, train_loader, val_loader)
    results = trainer.validate(model, val_loader)
    print(f"Split {split}: {results}")
    return results

def init_config(config, dataset_config, args):
    config["logger"]["artifact_location"] = args.artifact_location
    config["logger"]["tags"]['dataset'] = dataset_config["dataset"]
    config["logger"]["tags"]['weather'] = str(config["model"]["weather_config"]["weather_grid"])
    config["logger"]["tags"]['experiment'] = "transfer learning"
    return config

def init_model_config(config, dataset_config, dataset):
    model_config = config["model"]
    model_config["data"] = dataset_config
    model_config["in_channels"] = len(dataset_config["features"])
    model_config["out_ch"] = len(dataset_config["features"])
    model_config["weather_config"]["variables"] = len(dataset_config["weather_grid"]["variables"])
    # print(f"*******dataset parameters: {dataset.parameters}")
    model_config["traj_length"] = dataset.parameters['seq_len']
    model_config["continuous_len"] = dataset.con_conditions.shape[1]
    return model_config

from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np

def reduce_dataloader(dataloader, keep_fraction=0.2):
    dataset = dataloader.dataset  # Get the dataset from the original dataloader
    num_samples = int(len(dataset) * keep_fraction)  # Calculate the number of samples to keep
    indices = np.random.choice(len(dataset), num_samples, replace=False)  # Randomly select indices
    sampler = SubsetRandomSampler(indices)
    
    return DataLoader(dataset, batch_size=dataloader.batch_size, sampler=sampler, num_workers=dataloader.num_workers)

def get_model_2(config, dataset, model_config, args):
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


def a():
    config = load_config(args.config_file)
    dataset_config = load_config(args.dataset_config)
    config = init_config(config, dataset_config, args)
    l_logger, run_name, artifact_location = setup_logger(args, config)

    dataset_config["data_path"] = args.data_path
    dataset, traffic = load_and_prepare_data(dataset_config)
    model_config = init_model_config(config, dataset_config, dataset)

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
    print(f"*******model parameters: {model_config}")


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



def run(args):
    checkpoint = f"./artifacts/{args.model_name}/best_model.ckpt"
    config_file = f"./artifacts/{args.model_name}/config.yaml"
    config = load_config(config_file)
    dataset_config = load_config(args.data_path)
    config = init_config(config, dataset_config, args)

    dataset_config["data_path"] = args.data_path
    dataset, traffic = load_and_prepare_data(dataset_config)
    model_config = init_model_config(config, dataset_config, dataset)

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
    print(f"*******model parameters: {model_config}")
    train_config = config["train"]
    train_config["devices"] = args.cuda
    train_config["epochs"] = 20
    
    for split in args.split:
        print(f"Training with {split} of the dataset...")
        train_loader_reduced = reduce_dataloader(train_loader, keep_fraction=split)
        
        config["logger"]["tags"]['split'] = "{split}"
        config["logger"]["tags"]['pretrained'] = "False"
        l_logger, run_name, artifact_location = setup_logger(args, config)
        # Train non-pretrained model
        model_non_pretrained = get_model_2(config, dataset, model_config, args)
        train(train_config, model_non_pretrained, train_loader_reduced, val_loader, test_loader, l_logger, artifact_location)
        # Train pretrained model
        config["logger"]["tags"]['pretrained'] = "True"
        l_logger, run_name, artifact_location = setup_logger(args, config)
        model_pretrained = get_models(config, dataset.parameters, checkpoint, dataset.scaler)        
        train(train_config, model_pretrained, train_loader_reduced, val_loader, test_loader, l_logger, artifact_location)

        config["data"] = dataset_config
        save_config(config, os.path.join(artifact_location, "config.yaml"))

if __name__ == "__main__":
    seed_everything(42)
    parser = argparse.ArgumentParser(description="Run the traffic model.")
    parser.add_argument("--model_name", type=str, default="AirDiffTraj_5", help="Name of the model.")
    parser.add_argument("--data_path", type=str, default="./data/resampled/combined_traffic_resampled_landing_EHAM_200.pkl", help="Path to training data.")
    parser.add_argument("--artifact_location", type=str, default="/mnt/data/synthair/synthair_diffusion/data/experiments/transfer_learning/artifacts", help="Path to training data.")
    parser.add_argument("--cuda", type=int, default=0, help="Path to training data.")
    args = parser.parse_args()
    args.split = [0.05, 0.2, 0.5, 1.0]
    run(args)

