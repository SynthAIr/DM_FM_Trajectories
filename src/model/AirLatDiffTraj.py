import torch
import numpy
import torch.nn as nn
from argparse import Namespace
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch.nn import functional as F
import lightning as L

from model.tcvae import TCN
from utils import DatasetParams, TrafficDataset
from model.tcvae import VAE, VampPriorLSR
from model.tcvae import TCDecoder, TCEncoder
from model.generative import Generative
from typing import Tuple
from model.diffusion import Diffusion
from enum import Enum

class Phase(Enum):
    VAE = "vae"
    DIFFUSION = "diffusion"
    EVAL = "eval"

class LatentDiffusionTraj(L.LightningModule):
    def __init__(self, config: Union[Dict, Namespace], vae: VAE, generative: Generative) -> None:
        super().__init__()
        self.config = config
        self.lr = config["lr"]
        self.generative_model = generative
        self.vae = vae
        self.phase = Phase.DIFFUSION
        self.s = 2

    def reconstruct(self, x, con, cat, grid):
        #return self.vae.reconstruct(x, con, cat, grid)
        with torch.no_grad():
            z = self.vae.get_latent_n(x, con, cat, grid, 1)
            #z = z.unsqueeze(1)
            z_hat, _ = self.generative_model.reconstruct(z, con, cat, grid)
            self.log("reconstruction_loss", F.mse_loss(z_hat.float(), z))
            print("Reconstruction loss", F.mse_loss(z_hat.float(), z))
            print(z[0])
            print(z_hat[0])
            #z_hat = z
            z_hat = z_hat.squeeze(1)
            x_hat = self.vae.decode(z_hat)
        _ = []
        return x_hat, _

    def sample(self, n,con, cat, grid, length = 200, features=8, sampling="ddpm"):
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
            x_hat = self.vae.decode(x_t)
            #x_hat = self.out_activ(self.decoder(x_t))
        return x_hat, steps

    def training_step(self, batch, batch_idx):
        match self.phase:
            #case Phase.VAE:
                #return self.vae.training_step(batch, batch_idx)
            case Phase.DIFFUSION:
                x, con, cat, grid = batch
                con = con.unsqueeze(0).repeat(self.s, 1, 1).view(-1, *con.shape[1:])
                cat = cat.unsqueeze(0).repeat(self.s, 1, 1).view(-1, *cat.shape[1:])
                #grid = grid.unsqueeze(0).repeat(self.s, 1, 1, 1, 1, 1).view(-1, *grid.shape[1:])
                batch[1] = con
                batch[2] = cat
                batch[3] = grid
                #x, con, cat, grid = batch
                with torch.no_grad():
                    z = self.vae.get_latent_n(x, con, cat, grid, self.s)
                #z = z.unsqueeze(1)
                #print(z.shape, batch[1].shape)
                batch[0] = z
                loss = self.generative_model.training_step(batch, batch_idx)
                self.log("train_loss", loss)
                return loss
        raise ValueError(f"Invalid phase {self.phase}")

    def forward(self, x, con, cat, grid) -> Tuple[Tuple, torch.Tensor, torch.Tensor]:               # Overwrite the forward method for conditioning
        z = self.vae.get_latent(x, con, cat, grid)
        z = z.unsqueeze(1)
        #print("z", z.shape)
        z_hat = self.generative_model(z, con, cat, grid)
        z_hat = z_hat.squeeze(1)
        return self.vae.decode(z_hat)

    def validation_step(self, batch, batch_idx):
        match self.phase:
            #case Phase.VAE:
                #return self.vae.validation_step(batch, batch_idx)
            case Phase.DIFFUSION:
                x, con, cat, grid = batch
                con = con.unsqueeze(0).repeat(self.s, 1, 1).view(-1, *con.shape[1:])
                cat = cat.unsqueeze(0).repeat(self.s, 1, 1).view(-1, *cat.shape[1:])
                #grid = grid.unsqueeze(0).repeat(self.s, 1, 1, 1, 1, 1).view(-1, *grid.shape[1:])
                batch[1] = con
                batch[2] = cat
                batch[3] = grid
                with torch.no_grad():
                    z = self.vae.get_latent_n(x, con, cat, grid, self.s)
                #z = z.unsqueeze(1)
                #print(z.shape, batch[1].shape, batch[2].shape, batch[3].shape)
                batch[0] = z
                loss = self.generative_model.validation_step(batch, batch_idx)
                self.log("valid_loss", loss)
                return loss
        raise ValueError(f"Invalid phase {self.phase}")

    def test_step(self, batch, batch_idx):
        match self.phase:
            #case Phase.VAE:
                #return self.vae.test_step(batch, batch_idx)
            case Phase.DIFFUSION:
                #x, con, cat, grid = batch
                x, con, cat, grid = batch
                con = con.unsqueeze(0).repeat(self.s, 1, 1).view(-1, *con.shape[1:])
                cat = cat.unsqueeze(0).repeat(self.s, 1, 1).view(-1, *cat.shape[1:])
                #grid = grid.unsqueeze(0).repeat(self.s, 1, 1, 1, 1, 1).view(-1, *grid.shape[1:])
                batch[1] = con
                batch[2] = cat
                batch[3] = grid
                with torch.no_grad():
                    z = self.vae.get_latent_n(x, con, cat, grid, self.s)
                #z = z.unsqueeze(1)
                batch[0] = z
                loss = self.generative_model.test_step(batch, batch_idx)
                self.log("test_loss", loss)
                return loss
        raise ValueError(f"Invalid phase {self.phase}")
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """This is called after the optimizer step, at the end of the batch."""
        #print("Called this on train batch end hooks")
        self.generative_model.on_train_batch_end(outputs, batch, batch_idx)
