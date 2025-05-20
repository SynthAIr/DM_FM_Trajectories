# Repository for Master's Thesis using Diffusion- and Flow Matching-based models for generating synthetic aircraft trajectories

Code used for all models and experiments in the thesis

## Installation

The projects contains `pyproject.toml` file that specifies the dependencies and the build system. The project uses `poetry` for dependency management and packaging. To install the SynTraj package locally, clone the repository and install the dependencies using poetry:

```bash
git clone git@github.com:SynthAIr/DM_FM_Trajectories.git
cd SynTraj
poetry install
```

The project also contains a `Makefile` that provides shortcuts for common tasks. You can use the `make` command to run these tasks. To see the available tasks, run the following command:

```bash
make help
```
## Project Structure

The project is structured as follows:

```
DM_FM                               # Root directory 
├── config                          # Configuration files for the models and training
│   ├── fcvae_config.yaml           # Configuration file for the FCVAE model
│   ├── runtime_env.yaml            # Configuration file for the runtime environment
│   └── tcvae_config.yaml           # Configuration file for the TCVAE model
├── Makefile                        # Makefile for common tasks
├── pyproject.toml                  # Poetry configuration file
├── README.md                       # Project README file
├── setup.py                        # Setup file for packaging      
├── src                             # Main package directory 
│   ├── eval                        # Directory for evaluation scripts
│   │   ├── trajectory_distances    # Directory for trajectory distance metrics
│   │   └── eval_logics.py          # Evaluation logic for BlueSky simulation
│   ├── evaluate.py                 # Script for evaluating the generated trajectories
│   ├── generate.py                 # Script for generating trajectories, clusters, and latent space projection
│   ├── get_data.py                 # Script for downloading the Eurocontrol data from a MinIO server
│   ├── models                      # Directory for the generative models 
│   │   ├── fcvae.py                # Fully Connected Variational Autoencoder (FCVAE) model
│   │   ├── tcvae.py                # Temporal Convolutional Variational Autoencoder (TCVAE) model
│   ├── networks                    # Directory for the neural network modules
│   │   ├── fully_connected.py      # Fully connected neural network module
│   │   ├── recurrent.py            # Recurrent neural network module
│   │   ├── temporal_convolution.py # Temporal convolutional neural network module
│   ├── preprocess_data.py          # Script for preprocessing the Eurocontrol data
│   ├── train.py                    # Script for training the generative models
│   ├── utils                       # Directory for utility functions
│   │   ├── builders_utils.py       # Utility functions for building the models  
│   │   ├── data_utils.py           # Utility functions for loading and processing the data
│   │   ├── diffusion_utils.py      # Utility functions for diffusion models
│   │   ├── network_utils.py        # Utility functions for building the neural networks  
│   │   ├── plot_utils.py           # Utility functions for plotting the data and results  
│   │   └── train_utils.py          # Utility functions for training the models  
│   └── vae                         # Directory for the Variational Autoencoder (VAE) modules  
│       ├── abstract_vae.py         # Abstract VAE class
│       └── lsr.py                  # Latent Space Regularization (LSR) module
├── tests                           # Directory for tests
└── inlcude                         # Directory for submodules like TimeVQVAE that is supported in the syntraj application, but not part of the main syntraj repository
```


