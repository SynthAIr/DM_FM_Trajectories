import argparse
import numpy as np
import lightning.pytorch as pl
import torch
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch import seed_everything
from utils.helper import load_and_prepare_data, get_model, save_config, load_config, init_config, init_model_config, get_model_train
from evaluate import get_models,reconstruct_and_plot, plot_traffics, compute_partial_mmd, get_mse_distribution
from train import setup_logger, get_dataloaders, train
from model.flow_matching import FlowMatching, Wrapper
from model.diffusion import Diffusion
from evaluation.similarity import compute_energy_distance, compute_dtw_3d_batch
import os
from torch.utils.data import DataLoader, SubsetRandomSampler
from traffic.algorithms.generation import Generation
import numpy as np
from scipy.spatial.distance import euclidean
from evaluation.diversity import data_diversity

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
    reconstructions, mse_dict, rnd, fig_0 = reconstruct_and_plot(dataset, model, trajectory_generation_model, n=n, d=device)
    fig_smooth = plot_traffics([reconstructions[0],reconstructions[1]])
    l_logger.experiment.log_figure(l_logger.run_id,fig_smooth, f"figures/Eval_reconstruction_smoothed.png")
    l_logger.log_metrics({"Eval_MSE": mse_dict["mse"], "Eval_MSE_std": mse_dict["mse_std"]})

    #subset1_data = reconstructions[0].data.dropna().values
    

    fig_mse_dict = get_mse_distribution(mse_dict)
    l_logger.experiment.log_figure(l_logger.run_id, fig_mse_dict, "figures/Eval_mse_per_feature.png")

    #cols = [ 'latitude', 'longitude', 'altitude', 'groundspeed']
    #cols = [ 'latitude', 'longitude', 'altitude']
    cols = [ 'latitude', 'longitude']
    subset1_data = reconstructions[0].data[cols].dropna().values
    #subset2_data = df_subset2[['latitude', 'longitude']].dropna().values
    subset2_data = reconstructions[1].data[cols].dropna().values
    #subset2_data = .data.dropna().values

    mmd, mmd_std = compute_partial_mmd(reconstructions[0],reconstructions[1] )
    l_logger.log_metrics({"mmd": mmd, "mmd_std": mmd_std})

    # Compute energy distance between the raw trajectories
    energy_dist, edist_std = compute_energy_distance(subset1_data, subset2_data)
    l_logger.log_metrics({"edist": energy_dist, "edist_std": edist_std})
    
    dtw, dtw_std, _ = compute_dtw_3d_batch(subset1_data, subset2_data)
    l_logger.log_metrics({"dtw": dtw, "dtw_std": dtw_std})

    tn = 2
    fig_pca = data_diversity(subset1_data.reshape(-1, 200, tn), subset2_data.reshape(-1, 200, tn), 'PCA', 'else', model_name=str(split))
    fig_tsne = data_diversity(subset1_data.reshape(-1, 200, tn), subset2_data.reshape(-1, 200, tn), 't-SNE','else', model_name = str(split))
    l_logger.experiment.log_figure(l_logger.run_id, fig_pca, f"figures/pca.png")
    l_logger.experiment.log_figure(l_logger.run_id, fig_tsne, f"figures/tsne.png")
    


def train_encoder(vae, train_config, train_loader_reduced, val_loader, test_loader, config):
    vae_logger, run_name, artifact_location = setup_logger(args, config) 
    vae.train()
    train(train_config, vae, train_loader_reduced, val_loader, test_loader, vae_logger, artifact_location)
    vae.eval()
    return run_name


def run(args):
    checkpoint = f"./artifacts/{args.model_name}/best_model.ckpt"
    config_file = f"./artifacts/{args.model_name}/config.yaml"
    config = load_config(config_file)
    dataset_config = load_config(args.dataset_config)
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
        model_non_pretrained = get_model_train(dataset, model_config,dataset_config, args, pretrained_VAE = False)
        
        def autoencoder_config():
            config_file = f"{model_config['vae']}/config.yaml"
            c = load_config(config_file)
            c['model'] = init_model_config(c, dataset_config, dataset)
            c["logger"]["tags"]['split'] = f"{split}"
            c["logger"]["tags"]['pretrained'] = "False"
            c["logger"]["experiment_name"] = "transfer learning"
            c['model']["traj_length"] = dataset.parameters['seq_len']
            c['model']['data'] = dataset_config
            return c

        if model_config["type"] == "LatDiff" or model_config["type"] == "LatFM":
            print("Training autoencoder")
            c = autoencoder_config()
            train_encoder(model_non_pretrained.vae, train_config, train_loader_reduced, val_loader, test_loader, c)

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
        if model_config["type"] == "LatDiff" or model_config["type"] == "LatFM":
            print("Training autoencoder")
            c = autoencoder_config()
            train_encoder(model_pretrained.vae, train_config, train_loader_reduced, val_loader, test_loader, c)
        train(train_config, model_pretrained, train_loader_reduced, val_loader, test_loader, l_logger, artifact_location)

        local_eval(model_pretrained, dataset, trajectory_generation_model, n, device, l_logger, split)
        config["data"] = dataset_config
        save_config(config, os.path.join(artifact_location, "config.yaml"))


def run_experiment(args):
    experiment_name = "transfer learning EIDW track"
    checkpoint = f"/mnt/data/synthair/synthair_diffusion/data/experiments_track/{args.experiment}/pretrained/{args.model_name}/best_model.ckpt"
    config_file = f"/mnt/data/synthair/synthair_diffusion/data/experiments_track/{args.experiment}/pretrained/{args.model_name}/config.yaml"
    config = load_config(config_file)
    dataset_config = load_config(args.dataset_config)
    config = init_config(config, dataset_config, args, experiment = experiment_name)
    dataset_config["data_path"] = args.data_path
    dataset, traffic = load_and_prepare_data(dataset_config)
    model_config = init_model_config(config, dataset_config, dataset)
    print(dataset.data.shape)
    print(dataset.con_conditions.shape, dataset.cat_conditions.shape)

    dataset_config["batch_size"] = 128
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
    config["logger"]["experiment_name"] = experiment_name
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    n = 100
    for split in args.split:
        print(f"Training with {split} of the dataset...")
        train_loader_reduced = reduce_dataloader(train_loader, keep_fraction=split)
        
        def autoencoder_config():
            config_file = f"{model_config['vae']}/config.yaml"
            c = load_config(config_file)
            c['model'] = init_model_config(c, dataset_config, dataset)
            c["logger"]["tags"]['split'] = f"{split}"
            c["logger"]["artifact_location"] = args.artifact_location
            c["logger"]["tags"]['pretrained'] = "False"
            c["logger"]["experiment_name"] = experiment_name
            c['model']["traj_length"] = dataset.parameters['seq_len']
            c['model']['data'] = dataset_config
            return c

        # Train pretrained model
        config["logger"]["tags"]['pretrained'] = "True"
        l_logger, run_name, artifact_location = setup_logger(args, config)
        l_logger.log_metrics({"split": split})
        l_logger.log_metrics({"channel_size": config["model"]["ch"]})
        #l_logger.log_metrics({"ch_mult": config["model"]["ch_mult"]})
        l_logger.log_metrics({"num_res_blocks": config["model"]["num_res_blocks"]})
        model_pretrained, trajectory_generation_model = get_models(config['model'], dataset.parameters, checkpoint, dataset.scaler, device)        

        if split != 0.0:
            if model_config["type"] == "LatDiff" or model_config["type"] == "LatFM":
                print("Training autoencoder")
                c = autoencoder_config()
                run_name = train_encoder(model_pretrained.vae, train_config, train_loader_reduced, val_loader, test_loader, c)
                l_logger.log_metrics({"vae": run_name})

            train(train_config, model_pretrained, train_loader_reduced, val_loader, test_loader, l_logger, artifact_location)
    
        local_eval(model_pretrained, dataset, trajectory_generation_model, n, device, l_logger, split)
        config["data"] = dataset_config
        if split != 0.0:
            save_config(config, os.path.join(artifact_location, "config.yaml"))
        model_size = os.path.getsize(checkpoint) / (1024 * 1024)  #
        l_logger.log_metrics({"Size (MB)": model_size})


def get_lowest_model_folder(base_path, model_name):
    existing_folders = []
    default_folder = None
    
    if os.path.exists(base_path):
        for folder in os.listdir(base_path):
            if folder == model_name:
                default_folder = folder
            elif folder.startswith(f"{model_name}_"):
                try:
                    num = int(folder.split("_")[-1])
                    existing_folders.append(num)
                except ValueError:
                    continue
    
    if default_folder:
        return default_folder
    
    existing_folders.sort()
    return f"{model_name}_{existing_folders[0]}" if existing_folders else model_name

def increment_model_name(model_name):
    if "_" in model_name:
        name, num = model_name.rsplit("_", 1)
        return f"{name}_{int(num) + 1}"
    return f"{model_name}_1"

def run_eval(args):
    checkpoint = f"/mnt/data/synthair/synthair_diffusion/data/experiments/{args.experiment}/pretrained/{args.model_name}/best_model.ckpt"
    config_file = f"/mnt/data/synthair/synthair_diffusion/data/experiments/{args.experiment}/pretrained/{args.model_name}/config.yaml"
    artifact_location = args.artifact_location
    config = load_config(config_file)
    dataset_config = load_config(args.dataset_config)
    config = init_config(config, dataset_config, args, experiment = "transfer learning EIDW")

    dataset_config["data_path"] = args.data_path
    dataset, traffic = load_and_prepare_data(dataset_config)
    model_config = init_model_config(config, dataset_config, dataset)

    print(dataset.data.shape)
    print(dataset.con_conditions.shape, dataset.cat_conditions.shape)
    print("Dataset loaded!")
    print(f"*******model parameters: {model_config}")
    train_config = config["train"]
    train_config["devices"] = args.cuda
    train_config["epochs"] = 100
    config["logger"]["experiment_name"] = "transfer learning EIDW"
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    n = 100
    #model_name = start_model
    split = 0.0
    #print(f"Training with {split} of the dataset...")
    config["logger"]["tags"]['eval'] = "True"
    #config["logger"]["tags"]['pretrained'] = "False"
    l_logger, run_name, artifact_location = setup_logger(args, config)
    l_logger.log_metrics({"split": split})

    #checkpoint = f"{artifact_location}/{model_name}/best_model.ckpt"
    model, trajectory_generation_model = get_models(config['model'], dataset.parameters, checkpoint, dataset.scaler, device)        

    local_eval(model, dataset, trajectory_generation_model, n, device, l_logger, split)


if __name__ == "__main__":
    seed_everything(42)
    parser = argparse.ArgumentParser(description="Run the traffic model.")
    parser.add_argument("--model_name", type=str, default="AirDiffTraj_5", help="Name of the model.")
    parser.add_argument("--data_path", type=str, default="/mnt/data/synthair/synthair_diffusion/data/resampled/combined_traffic_resampled_landing_EIDW_200.pkl", help="Path to training data.")
    parser.add_argument("--dataset_config", type=str, default="./configs/dataset_landing_transfer.yaml", help="Path to training data.")
    parser.add_argument("--artifact_location", type=str, default="/mnt/data/synthair/synthair_diffusion/data/experiments/transfer_learning_EIDW/artifacts", help="Path to training data.")
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--cuda", type=int, default=0, help="Path to training data.")
    parser.add_argument("--eval", dest="run_train", action='store_true')
    parser.add_argument("--eval_all", dest="eval_all", action='store_true')
    args = parser.parse_args()
    args.split = [0.0, 0.05, 0.2, 0.5, 1.0]
    #args.split = [0.0]
    if args.run_train:
        print("Running EVAL")
        run_eval(args)
        #run(args)
    elif args.eval_all:
        print("Running EVAL ALL")
        run_eval(args)
    else:
        print("Running Experiments")
        run_experiment(args)

