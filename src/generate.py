
import argparse
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
import matplotlib.pyplot as plt
import numpy as np
from utils import load_config
from model.Traj_UNet import Guide_UNet2


def run(args):
    config = load_config(args.config_file)
    args.checkpoint = f"./artifacts/{config['model']['type']}/best_model.ckpt"

    model = Guide_UNet2.load_from_checkpoint(args.checkpoint, map_location=torch.device('cuda'))

    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        #transforms.Lambda(lambda x: x / 255),
        transforms.Lambda(lambda x: (x > 0.5).float())
    ])

    # Download and load the training dataset
    dataset_config = config["data"]
    batch_size = dataset_config["batch_size"]
    #train_dataset = FashionMNIST(root='./data', train=True, transform=transform)
    
    n = 10
    c = 3

    
    #x = x.view(-1, 1, 28, 28)
    c_ = torch.zeros(10).to(model.device)
    c_[c] = 1
    print(c_.shape)
    t = 20
    samples = model.sample(n, c_, t).reshape(n, 1, 28, 28)
    print(samples.shape)

    #print(samples)
    # Reshape and move the samples to the CPU
    # Define grid dimensions (e.g., 4x4 grid for 16 images)
    num_rows = 4
    num_cols = 4

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 12))

    # Flatten the axes array for easy iteration
    axes = axes.flatten()


    # Hide any unused subplots
    for i in range(n, len(axes)):
        axes[i].axis('off')

    # Adjust layout
    plt.tight_layout()
    plt.show()



    # Create a figure and a set of subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 12))

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    # Iterate over the samples and plot each image
    for i in range(n):
        ax = axes[i]
        s = samples[i]
        s *= 255
        print(s.shape)
        ax.imshow(s.cpu().detach().numpy().transpose((1,2,0)), cmap='gray')
        ax.axis('off')  # Hide the axis

    for i in range(n, len(axes)):
        axes[i].axis('off')

    # Adjust layout
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Parser for training a model with PyTorch Lightning"
    )

    args = parser.parse_args()
    args.config_file = "./configs/config.yaml"
    args.artifact_location= "./artifacts"
    args.checkpoint = "./artifacts/diffusion_1/best_model.ckpt"

    run(args)
