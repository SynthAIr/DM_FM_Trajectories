import torch

class Generative(torch.nn.Module):
    """
    Abstract class for all the generative models used to make it easier to work and switch between them.
    """

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

