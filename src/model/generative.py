import math
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
from tqdm import tqdm
from torch.nn import functional as F
import lightning as L

class Generative(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x, t, con, cat, grid):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        pass

    def sample(self, n, con, cat, grid, features , length,  sampling="ddpm"):
        pass

    def reconstruct(self, x, con, cat, grid):
        pass

