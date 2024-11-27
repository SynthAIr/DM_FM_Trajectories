import argparse
import os
import sys
import numpy as np
from lightning.pytorch.loggers import MLFlowLogger
from utils.helper import load_and_prepare_data, get_model
from evaluate import get_config_data, get_models, load_config, reconstruct_and_plot
from evaluation.similarity import jensenshannon_distance

def run(args, logger = None):
    model_name = args.model_name

    w = np.arange(0, 10, 2)

    config_file = "./configs/config.yaml"
    data_path = "./data/resampled/combined_traffic_resampled_600.pkl"
    artifact_location= "./artifacts"
    checkpoint = f"./artifacts/{model_name}/best_model.ckpt"

    config = load_config(config_file)
    if logger is None:
        logger_config = config["logger"]
        logger = MLFlowLogger(
            experiment_name=logger_config["experiment_name"],
            run_name=args.model_name,
            tracking_uri=logger_config["mlflow_uri"],
            tags=logger_config["tags"],
            artifact_location=artifact_location,
        )

    config, dataset, traffic, conditions = get_config_data(config_file, data_path, artifact_location)
    config['model']["traj_length"] = dataset.parameters['seq_len']
    config['model']["continuous_len"] = dataset.con_conditions.shape[1]
    model, trajectory_generation_model = get_models(config["model"], dataset.parameters, checkpoint, dataset.scaler)
    dataset_config = config["data"]
    batch_size = dataset_config["batch_size"]
    
    rnd = None
    for w_i in w:
        model.guidance_scale = w_i
        reconstructions, mse, rnd, fig_0 = reconstruct_and_plot(dataset, model, trajectory_generation_model, n=30, model_name = model_name, rnd = rnd)
        logger.log_metrics({f"Eval_MSE_{w_i}": mse})
        logger.log_figure(fig_0, f"Eval_{w_i}_reconstruction.png")
        #print(reconstructions[1].data)
        JSD, KL, e_distance, fig_1 = jensenshannon_distance(reconstructions, model_name = model_name)
        logger.log_figure(fig_1, f"Eval_{w_i}_comparison.png")
        logger.log_metrics({f"Eval_edistance_{w_i}": e_distance, f"Eval_JSD_{w_i}": JSD, f"Eval_KL_{w_i}": KL})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the traffic model.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="AirDiffTraj_5",
        help="Name of the model (e.g., 'AirDiffTraj_5')."
    )
    args = parser.parse_args()
    run(args)
