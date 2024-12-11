import math
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
from tqdm import tqdm
from torch.nn import functional as F

class Generative(nn.LightningModule):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x, t, con, cat, grid):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        pass

    def sample(self, n, features, length, con, cat, grid, sampling="ddpm"):
        pass

    def reconstruct(self, x, con, cat, grid):
        pass

