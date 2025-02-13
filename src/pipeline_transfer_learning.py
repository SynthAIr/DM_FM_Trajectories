import argparse
import numpy as np
import lightning.pytorch as pl
import torch
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch import seed_everything
from utils.helper import load_and_prepare_data, get_model, save_config, load_config, init_config, init_model_config, get_model_train
from evaluate import get_models,reconstruct_and_plot, plot_traffics, compute_partial_mmd
from train import setup_logger, get_dataloaders, train
from model.flow_matching import FlowMatching, Wrapper
from model.diffusion import Diffusion
from evaluation.similarity import compute_energy_distance
import os
from torch.utils.data import DataLoader, SubsetRandomSampler
from traffic.algorithms.generation import Generation
import numpy as np

def train_and_evaluate(model, train_loader, val_loader, logger, split):
    """Train the model and evaluate its performance."""
    trainer = pl.Trainer(logger=logger, max_epochs=10)  # Adjust epochs as needed
    trainer.fit(model, train_loader, val_loader)
    results = trainer.validate(model, val_loader)
    print(f"Split {split}: {results}")
    return results

def reduce_dataloader(dataloader, keep_fraction=0.2):
    dataset = dataloader.dataset  # Get the dataset from the original dataloader
    num_samples = int(len(dataset) * keep_fraction)  # Calculate the number of samples to keep
    indices = np.random.choice(len(dataset), num_samples, replace=False)  # Randomly select indices
    sampler = SubsetRandomSampler(indices)
    
    return DataLoader(dataset, batch_size=dataloader.batch_size, sampler=sampler, num_workers=dataloader.num_workers)


def local_eval(model, dataset, trajectory_generation_model, n, device, l_logger, split):
    l_logger.log_metrics({"dataset_samples": int(split * len(dataset) * 0.8 * 0.8)})
    reconstructions, (mse, mse_std), rnd, fig_0 = reconstruct_and_plot(dataset, model, trajectory_generation_model, n=n, d=device)
    fig_smooth = plot_traffics([reconstructions[0],reconstructions[2]])
    l_logger.experiment.log_figure(l_logger.run_id,fig_smooth, f"figures/Eval_reconstruction_smoothed.png")
    l_logger.log_metrics({"Eval_MSE": mse, "Eval_MSE_std": mse_std})

    #subset1_data = reconstructions[0].data.dropna().values
    subset1_data = reconstructions[0].data[['latitude', 'longitude', 'altitude', 'groundspeed']].dropna().values
    #subset2_data = df_subset2[['latitude', 'longitude']].dropna().values
    subset2_data = reconstructions[2].data[['latitude', 'longitude', 'altitude', 'groundspeed']].dropna().values
    #subset2_data = .data.dropna().values

    # Compute energy distance between the raw trajectories
    energy_dist, edist_std = compute_energy_distance(subset1_data, subset2_data)
    l_logger.log_metrics({"edist": energy_dist, "edist_std": edist_std})

    mmd, mmd_std = compute_partial_mmd(reconstructions[0], reconstructions[2])
    l_logger.log_metrics({"mmd": mmd, "mmd_std": mmd_std})



def run(args):
    checkpoint = f"./artifacts/{args.model_name}/best_model.ckpt"
    config_file = f"./artifacts/{args.model_name}/config.yaml"
    config = load_config(config_file)
    dataset_config = load_config(args.dataset_path)
    config = init_config(config, dataset_config, args, experiment = "transfer learning")

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
    train_config["epochs"] = 100
    config["logger"]["experiment_name"] = "transfer learning"
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    n = 100
    for split in args.split:
        print(f"Training with {split} of the dataset...")
        train_loader_reduced = reduce_dataloader(train_loader, keep_fraction=split)
        
        config["logger"]["tags"]['split'] = f"{split}"
        config["logger"]["tags"]['pretrained'] = "False"
        l_logger, run_name, artifact_location = setup_logger(args, config)
        l_logger.log_metrics({"split": split})
        # Train non-pretrained model
        model_non_pretrained = get_model_train(dataset, model_config,dataset_config, args)
        train(train_config, model_non_pretrained, train_loader_reduced, val_loader, test_loader, l_logger, artifact_location)

        trajectory_generation_model = Generation(
            generation=model_non_pretrained,
            features=dataset.parameters['features'],
            scaler=dataset.scaler,
        )
        
        local_eval(model_non_pretrained, dataset, trajectory_generation_model, n, device, l_logger, split)

        save_config(config, os.path.join(artifact_location, "config.yaml"))
        model_non_pretrained = model_non_pretrained.to("cpu")
        # Train pretrained model
        config["logger"]["tags"]['pretrained'] = "True"
        l_logger, run_name, artifact_location = setup_logger(args, config)
        l_logger.log_metrics({"split": split})
        model_pretrained, trajectory_generation_model = get_models(config['model'], dataset.parameters, checkpoint, dataset.scaler, device)        
        train(train_config, model_pretrained, train_loader_reduced, val_loader, test_loader, l_logger, artifact_location)

        local_eval(model_pretrained, dataset, trajectory_generation_model, n, device, l_logger, split)
        config["data"] = dataset_config
        save_config(config, os.path.join(artifact_location, "config.yaml"))
        

if __name__ == "__main__":
    seed_everything(42)
    parser = argparse.ArgumentParser(description="Run the traffic model.")
    parser.add_argument("--model_name", type=str, default="AirDiffTraj_5", help="Name of the model.")
    parser.add_argument("--data_path", type=str, default="./data/resampled/combined_traffic_resampled_landing_EIDW_200.pkl", help="Path to training data.")
    parser.add_argument("--dataset_path", type=str, default="./configs/dataset_landing_transfer.yaml", help="Path to training data.")
    parser.add_argument("--artifact_location", type=str, default="/mnt/data/synthair/synthair_diffusion/data/experiments/transfer_learning/artifacts", help="Path to training data.")
    parser.add_argument("--cuda", type=int, default=0, help="Path to training data.")
    args = parser.parse_args()
    args.split = [0.05, 0.2, 0.5, 1.0]
    run(args)

