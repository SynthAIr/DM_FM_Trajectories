import argparse
from train import run as train_run
from evaluate import run_refactored as evaluate_run
from utils.helper import load_config

def run(args):
    """
    Run the training and evaluation pipeline.
    Parameters
    ----------
    args

    Returns
    -------

    """
    configs = load_config(args.config_file)
    logger_config = configs["logger"]
    l_logger, run_name, artifact_location = train_run(args)
    args.model_name = run_name
    evaluate_run(args, l_logger)


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

    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Name of the model (e.g., 'AirDiffTraj_5')."
        )

    parser.add_argument(
        "--dataset_config",
        type=str,
        #required=True,
        default="./configs/dataset_opensky.yaml",
        help="Path to the dataset config file"
    )

    parser.add_argument(
            "--perturb",
            dest="perturb", 
            action='store_true')

    parser.add_argument(
            "--cuda",
            type=int,
            default=0,
            help="GPU to use",
            )


    args = parser.parse_args()
    run(args)
    
