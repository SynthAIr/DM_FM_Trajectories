# Diffusion- and Flow-Matching-based models for generating synthetic landing trajectories in aviation

Code used for all models and experiments in the Master's thesis of Olav Finne Præsteng Larsen, NTNU/SINTEF 2025.

## Installation

The project contains a `pyproject.toml` file that specifies the dependencies and the build system. The project uses `poetry` for dependency management and packaging. To install the SynTraj package locally, clone the repository and install the dependencies using poetry:

```bash
git clone git@github.com:SynthAIr/DM_FM_Trajectories.git
cd DM_FM_Trajectories
poetry install
cd src
```

## Project Structure
The project is organized into the following main directories:
- `configs`: Contains configuration files for different models and datasets.
- `data`: Contains the raw and processed data files used for training and evaluation.
- `evaluation`: Metrics and scripts for evaluating the models.
- `models`: Contains the implementation of the models used for trajectory generation.
- `utils`: Utility functions and classes used throughout the project.

## Train

The `train.py` script is used to train a flight trajectory generation model. 
The models are saved in a folder with the model name. The folder contains the best model and the model configuration file. 

### Arguments

| Argument | Type | Default | Description                                               |
|----------|------|---------|-----------------------------------------------------------|
| `--config_file` | `str` | `./configs/config.yaml` | Path to the main configuration file for the model config. |
| `--dataset_config` | `str` | `./configs/dataset_landing_transfer.yaml` | Path to the dataset-specific configuration file.          |
| `--data_path` | `str` | `./data/resampled/combined_traffic_resampled_200.pkl` | Path to the training dataset file.                        |
| `--artifact_path` / `--artifact_location` | `str` | `./artifacts` | Directory where training artifacts will be saved.         |
| `--cuda` | `int` | `0` | GPU index to use for training.                            |

### Example

```bash
python train.py \
  --config_file ./configs/config.yaml \
  --dataset_config ./configs/dataset_landing_transfer.yaml \
  --data_path ./data/resampled/combined_traffic_resampled_200.pkl \
  --artifact_path ./artifacts \
  --cuda 0 
```

### Evaluate
The `evaluate.py` script is used to evaluate the flight trajectory generation model. 

### Arguments

| Argument | Type | Default | Description                                           |
|----------|------|---------|-------------------------------------------------------|
| `--model_name` | `str` | `AirDiffTraj` | Name of the model to run.                             |
| `--data_path` | `str` | `./data/resampled/combined_traffic_resampled_200.pkl` | Path to the input data file.                          |
| `--dataset_config` | `str` | `./configs/dataset_landing_transfer.yaml` | Path to the dataset configuration file.               |
| `--artifact_path` / `--artifact_location` | `str` | `./artifacts` | Directory to store artifacts such as outputs or logs. |
| `--cuda` | `int` | `0` | GPU index to use.                                     |
| `--perturb` | `flag` | `False` | Run Evaluate for perturbation model.                  |

### Example

Run the model in evaluation on GPU 0:

```bash
python evaluate.py \
  --model_name AirDiffTraj \
  --data_path ./data/resampled/combined_traffic_resampled_200.pkl \
  --dataset_config ./configs/dataset_landing_transfer.yaml \
  --artifact_path ./artifacts \
  --cuda 0 \
```
### Training and Evaluation

The `train_evaluate_pipeline.py` script combines training and evaluation for the flight trajectory generation model. 

### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--config_file` | `str` | `./configs/config.yaml` | Path to the main training configuration file. |
| `--artifact_path` / `--artifact_location` | `str` | `./artifacts` | Directory to save artifacts (models, logs, etc.). |
| `--data_path` | `str` | `./data/resampled/combined_traffic_resampled_200.pkl` | Path to the input data file.                        |
| `--dataset_config` | `str` | `./configs/dataset_landing_transfer.yaml` | Path to the dataset configuration file.             |
| `--model_name` | `str` | `None` | Optional name for the model (e.g., `AirDiffTraj`). |
| `--perturb` | `flag` | `False` | Apply perturbations to input data for robustness testing. |
| `--cuda` | `int` | `0` | GPU index to use. |

### Example

Train and evaluate a model on GPU 0:

```bash
python train_evaluate_pipeline.py \
  --config_file ./configs/config.yaml \
  --data_path ./data/resampled/combined_traffic_resampled_200.pkl \
  --artifact_path ./artifacts \
  --model_name AirDiffTraj \
  --dataset_config ./configs/dataset_landing_transfer.yaml \
  --cuda 0 
```

### Transfer Learning

## 🔁 Transfer Learning Pipeline

The `pipeline_transfer_learning.py` script is designed for transfer learning experiments using the flight trajectory generation model. It allows running training, single evaluation, or batch evaluations across a dataset split. The script supports GPU configuration.
It trains and evaluates the pretrained model on the different datasplits.
### Arguments

| Argument | Type | Default                                                            | Description |
|----------|------|--------------------------------------------------------------------|-------------|
| `--model_name` | `str` | `AirDiffTraj`                                                      | Name of the model to use. |
| `--data_path` | `str` | `./data/resampled/combined_traffic_resampled_landing_EIDW_200.pkl` | Path to the dataset used for training/evaluation. |
| `--dataset_config` | `str` | `./configs/dataset_landing_transfer.yaml`                          | Configuration file for the dataset. |
| `--artifact_location` | `str` | `./data/experiments/transfer_learning_EIDW/artifacts`              | Directory where experiment artifacts will be saved. |
| `--experiment` | `str` | *(required)*                                                       | Name or identifier of the experiment to run. |
| `--cuda` | `int` | `0`                                                                | GPU index to use. |
| `--eval` | `flag` | `False`                                                            | Run a single evaluation using the specified model and data. |
| `--eval_all` | `flag` | `False`                                                            | Run evaluations over all predefined data splits. |

> **Note:** The data is automatically split using the following proportions: `[0.0, 0.05, 0.2, 0.5, 1.0]`

### Example

Run a transfer learning experiment using GPU 0:

```bash
python pipeline_transfer_learning.py \
  --model_name AirDiffTraj_5 \
  --data_path ./data/resampled/combined_traffic_resampled_landing_EIDW_200.pkl \
  --dataset_config ./configs/dataset_landing_transfer.yaml \
  --artifact_location ./data/experiments/transfer_learning_EIDW/artifacts \
  --experiment my_experiment \
  --cuda 0


### Train Optuna
Script to optimize the hyperparameters of the model using Optuna. The script uses the `optuna` library to perform hyperparameter optimization. The script is designed to be run from the command line.

### Example
Run Optuna for hyperparameter optimization on GPU 0:

```bash
python train_optuna.py
```

### Preprocess

The `preprocess.py` script is used to prepare raw air traffic data for model training. It supports different data sources (e.g., OpenSky, Eurocontrol) and includes a mode for preprocessing landing trajectories specifically.

### Arguments

| Argument | Type | Default           | Description |
|----------|------|-------------------|-------------|
| `--ADEP` | `str` | `EHAM`            | Departure airport ICAO code. |
| `--ADES` | `str` | `LSZH`            | Arrival airport ICAO code. |
| `--data_dir` | `str` | `./data/Opensky` | Directory containing the raw input data. |
| `--data_source` | `str` | `OpenSky`         | Data source identifier. Either `"OpenSky"` or `"Eurocontrol"` by default. |
| `--landing` | `flag` | `False`           | Preprocess the data for **landing trajectories** only (overrides data source to `"landing"`). |

### Example

Preprocess a standard origin-destination dataset from OpenSky:

```bash
python preprocess.py \
  --ADES LSZH \
  --data_dir ./data/landing \
  --data_source OpenSky \
  --landing
```

### Merge Datasets
The script is used to process, interpolate, and combine all air traffic trajectory data into a single `.pkl` file for further 
use in training or evaluation. It resamples trajectories to a uniform length and supports configurations for specific airports or datasets.

### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--length` | `int` | `200` | Target length for resampling all trajectories. |
| `--data_dir` | `str` | `./data` | Directory containing input raw data files. |

> ⚠️ **Note:** The `--data_source` argument is **overwritten** by the script and is not user-configurable from the command line. Output is saved to:  
`./data/combined_traffic_resampled_landing_200.pkl`

### Example

Run the resampling and combining process with a trajectory length of 200:

```bash
python resample_combine.py \
  --length 200 \
  --data_dir ./data
```
