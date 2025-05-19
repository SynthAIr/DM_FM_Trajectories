from argparse import Namespace
from typing import Dict, Union
import torch
from torch.nn import functional as F
import lightning as L
from model.tcvae import VAE, VampPriorLSR
from model.generative import Generative
from typing import Tuple
from enum import Enum

class Phase(Enum):
    """
    Enum to represent the different phases of the model in terms of training and evaluation.
    """
    VAE = "vae"
    DIFFUSION = "diffusion"
    EVAL = "eval"

class LatentDiffusionTraj(L.LightningModule):
    """
    Latent Diffusion Model for Trajectory Generation.
    """
    def __init__(self, config: Union[Dict, Namespace], vae: VAE, generative: Generative) -> None:
        super().__init__()
        self.config = config
        self.lr = config["lr"]
        self.generative_model = generative
        self.vae = vae
        self.phase = Phase.DIFFUSION
        self.s = 1

    def reconstruct(self, x, con, cat, grid):
        """
        Reconstruct the input data using the VAE and generative model.
        Parameters
        ----------
        x
        con
        cat
        grid

        Returns
        -------

        """
        with torch.no_grad():
            z = self.vae.get_latent_n(x, con, cat, grid, 1)
            z_hat, _ = self.generative_model.reconstruct(z, con, cat, grid)
            print("Reconstruction loss", F.mse_loss(z_hat.float(), z))
            z_hat = z_hat.squeeze(1)
            x_hat = self.vae.decode(z_hat, con, cat, grid)
        _ = []
        return x_hat, _

    def sample(self, n, con, cat, grid, length = 200, features=8, sampling="ddpm"):
        """
        Sample from the generative model using the VAE to decode
        Parameters
        ----------
        n
        con
        cat
        grid
        length
        features
        sampling

        Returns
        -------

        """
        self.eval()
        features = 1
        length = self.generative_model.ch

        x_t, steps = self.generative_model.sample(n = n, 
                                                  con = con,
                                                  cat = cat,
                                                  grid = grid,
                                                  features=features,
                                                  length=length,
                                                  sampling = sampling)
        x_t = x_t.squeeze(1)
        with torch.no_grad():
            x_hat = self.vae.decode(x_t, con, cat, grid)
        return x_hat, steps

    def training_step(self, batch, batch_idx):
        """
        This is called at the end of the training step, before the optimizer step.
        Parameters
        ----------
        batch
        batch_idx

        Returns
        -------

        """
        match self.phase:
            case Phase.DIFFUSION:
                x, con, cat, grid = batch
                con = con.unsqueeze(0).repeat(self.s, 1, 1).view(-1, *con.shape[1:])
                cat = cat.unsqueeze(0).repeat(self.s, 1, 1).view(-1, *cat.shape[1:])
                batch[1] = con
                batch[2] = cat
                batch[3] = grid
                with torch.no_grad():
                    z = self.vae.get_latent_n(x, con, cat, grid, self.s)
                batch[0] = z
                loss = self.generative_model.training_step(batch, batch_idx)
                self.log("train_loss", loss)
                return loss
        raise ValueError(f"Invalid phase {self.phase}")

    def forward(self, x, con, cat, grid) -> Tuple[Tuple, torch.Tensor, torch.Tensor]:               # Overwrite the forward method for conditioning
        """
        Forward pass through the model. This is called when you call the model directly.
        Parameters
        ----------
        x
        con
        cat
        grid

        Returns
        -------

        """
        z = self.vae.get_latent(x, con, cat, grid)
        z = z.unsqueeze(1)
        z_hat = self.generative_model(z, con, cat, grid)
        z_hat = z_hat.squeeze(1)
        return self.vae.decode(z_hat, con, cat, grid)

    def validation_step(self, batch, batch_idx):
        """
        This is called during validation by pytorch lightning .
        Parameters
        ----------
        batch
        batch_idx

        Returns
        -------

        """
        match self.phase:
            case Phase.DIFFUSION:
                x, con, cat, grid = batch
                con = con.unsqueeze(0).repeat(self.s, 1, 1).view(-1, *con.shape[1:])
                cat = cat.unsqueeze(0).repeat(self.s, 1, 1).view(-1, *cat.shape[1:])
                batch[1] = con
                batch[2] = cat
                batch[3] = grid
                with torch.no_grad():
                    z = self.vae.get_latent_n(x, con, cat, grid, self.s)
                batch[0] = z
                loss = self.generative_model.validation_step(batch, batch_idx)
                self.log("valid_loss", loss)
                return loss
        raise ValueError(f"Invalid phase {self.phase}")

    def test_step(self, batch, batch_idx):
        """
        This is called during testing by pytorch lightning .
        Parameters
        ----------
        batch
        batch_idx

        Returns
        -------

        """
        match self.phase:
            case Phase.DIFFUSION:
                x, con, cat, grid = batch
                con = con.unsqueeze(0).repeat(self.s, 1, 1).view(-1, *con.shape[1:])
                cat = cat.unsqueeze(0).repeat(self.s, 1, 1).view(-1, *cat.shape[1:])
                batch[1] = con
                batch[2] = cat
                batch[3] = grid
                with torch.no_grad():
                    z = self.vae.get_latent_n(x, con, cat, grid, self.s)
                batch[0] = z
                loss = self.generative_model.test_step(batch, batch_idx)
                self.log("test_loss", loss)
                return loss
        raise ValueError(f"Invalid phase {self.phase}")
    
    def configure_optimizers(self):
        """
        Configure the optimizer for the model.
        Returns
        -------

        """
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """
        This is called after the optimizer step, at the end of the batch.
        Parameters
        ----------
        outputs
        batch
        batch_idx

        Returns
        -------

        """
        self.generative_model.on_train_batch_end(outputs, batch, batch_idx)
