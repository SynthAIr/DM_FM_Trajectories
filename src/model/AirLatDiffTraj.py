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
        self.phase = Phase.VAE

    def generative_forward(self, x, con, cat, grid) -> torch.Tensor:
        z = self.vae.get_latent(x, con, cat, grid)
        z = z.unsqueeze(1)
        #print("z", z.shape)
        z_hat = self.generative_model(z, con, cat, grid)
        z_hat = z_hat.squeeze(1)
        return z_hat

    def eval_forward(self, x, con, cat, grid) -> torch.Tensor:
        z = self.vae.get_latent(x, con, cat, grid)
        z = z.unsqueeze(1)
        #print("z", z.shape)
        z_hat = self.generative_model(z, con, cat, grid)
        z_hat = z_hat.squeeze(1)
        return self.vae.decode(z_hat)
    
    def forward(self, x, con, cat, grid) -> Tuple[Tuple, torch.Tensor, torch.Tensor]:    
        match self.phase:
            case Phase.VAE:
                return self.vae(x, con, cat, grid)
            case Phase.DIFFUSION:
                return self.generative_forward(x, con, cat, grid)
            case Phase.EVAL:
                z = self.generative_forward(x, con, cat, grid)
                x_hat = self.vae.decoder(z)
                return x_hat, []

        raise ValueError(f"Invalid phase {self.phase}")

    def reconstruct(self, x, con, cat, grid):
        #return self.vae.reconstruct(x, con, cat, grid)
        with torch.no_grad():
            z = self.vae.get_latent(x, con, cat, grid)
            z = z.unsqueeze(1)
            z_hat, _ = self.generative_model.reconstruct(z, con, cat, grid)
            z_hat = z_hat.squeeze(1)
            x_hat = self.vae.decode(z_hat)
        _ = []
        return x_hat, _

    def sample(self, n,con, cat, grid, length = 200, features=8, sampling="ddpm"):
        self.eval()
        x_t, steps = self.generative_model.sample(n, con, cat, grid, 1, 256, sampling)
        with torch.no_grad():
            x_hat = self.out_activ(self.decoder(x_t))
        return x_hat, steps

    def training_step(self, batch, batch_idx):
        match self.phase:
            case Phase.VAE:
                return self.vae.training_step(batch, batch_idx)
            case Phase.DIFFUSION:
                x, con, cat, grid = batch
                z = self.vae.get_latent(x, con, cat, grid)
                z = z.unsqueeze(1)
                batch[0] = z
                loss = self.generative_model.training_step(batch, batch_idx)
                self.log("train_loss", loss)
                return loss
        raise ValueError(f"Invalid phase {self.phase}")
    
    def validation_step(self, batch, batch_idx):
        match self.phase:
            case Phase.VAE:
                return self.vae.validation_step(batch, batch_idx)
            case Phase.DIFFUSION:
                x, con, cat, grid = batch
                z = self.vae.get_latent(x, con, cat, grid)
                z = z.unsqueeze(1)
                batch[0] = z
                loss = self.generative_model.validation_step(batch, batch_idx)
                self.log("valid_loss", loss)
                return loss
        raise ValueError(f"Invalid phase {self.phase}")

    def test_step(self, batch, batch_idx):
        match self.phase:
            case Phase.VAE:
                return self.vae.test_step(batch, batch_idx)
            case Phase.DIFFUSION:
                x, con, cat, grid = batch
                z = self.vae.get_latent(x, con, cat, grid)
                z = z.unsqueeze(1)
                batch[0] = z
                loss = self.generative_model.test_step(batch, batch_idx)
                self.log("test_loss", loss)
                return loss
        raise ValueError(f"Invalid phase {self.phase}")
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


class AirLatDiffTraj(VAE):

    _required_hparams = VAE._required_hparams + [
        "sampling_factor",
        "kernel_size",
        "dilation_base",
        "n_components",
    ]

    def __init__(
        self,
        config: Union[Dict, Namespace],
    ) -> None:
        super().__init__(config)

        self.phase = Phase.VAE
        self.config = config
        self.lr = config["lr"]  # Explore this - might want it lower when training on the full dataset

        #config["cond_embed_dim"] = get_cond_len(dataset_params['conditional_features'], seq_len = self.dataset_params["seq_len"]) if self.conditional else 0
        #config["cond_embed_dim"] =config['length']

        self.encoder = TCEncoder(
                #input_dim=self.dataset_params["input_dim"],
            input_dim=self.config["in_channels"],
            out_dim=self.hparams.encoding_dim,
            h_dims=self.hparams.h_dims[::-1],
            kernel_size=self.hparams.kernel_size,
            dilation_base=self.hparams.dilation_base,
            sampling_factor=self.hparams.sampling_factor,
            dropout=self.hparams.dropout,
            h_activ=nn.ReLU()
            )

        self.decoder = TCDecoder(
            input_dim=self.hparams.encoding_dim,
            out_dim=self.config["in_channels"],
            h_dims=self.hparams.h_dims[::-1],
            seq_len=self.config["traj_length"],
            kernel_size=self.hparams.kernel_size,
            dilation_base=self.hparams.dilation_base,
            sampling_factor=self.hparams.sampling_factor,
            dropout=self.hparams.dropout,
            h_activ=nn.ReLU(),
        )
        h_dim = self.hparams.h_dims[-1] * (int(self.config["traj_length"] / self.hparams.sampling_factor))

        self.lsr = VampPriorLSR(
            original_dim=self.config["in_channels"],
            original_seq_len=self.config["traj_length"],
            input_dim=h_dim,
            cond_length=0,
            out_dim=self.hparams.encoding_dim,
            encoder=self.encoder,
            n_components=self.hparams.n_components,
        )
        self.diffusion = Diffusion(config)
        self.out_activ = nn.Identity()

    def vae_forward(self, x, con, cat, grid) -> Tuple[Tuple, torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        q = self.lsr(h)
        z = q.rsample()
        x_hat = self.out_activ(self.decoder(z))
        return self.lsr.dist_params(q), z, x_hat
    
    def diffusion_forward(self, x, con, cat, grid) -> torch.Tensor:
        h = self.encoder(x)
        q = self.lsr(h)
        z = q.rsample()
        z = z.unsqueeze(1)
        #print("z", z.shape)
        z_hat = self.diffusion(z, con, cat, grid)
        z_hat = z_hat.squeeze(1)
        return z_hat

    def eval_forward(self, x, con, cat, grid) -> torch.Tensor:
        h = self.encoder(x)
        q = self.lsr(h)
        z = q.rsample()
        z = z.unsqueeze(1)
        #print("z", z.shape)
        z_hat = self.diffusion(z, con, cat, grid)
        z_hat = z_hat.squeeze(1)
        x_hat = self.out_activ(self.decoder(z_hat))
        return x_hat, []

    def forward(self, x, con, cat, grid) -> Tuple[Tuple, torch.Tensor, torch.Tensor]:    
        match self.phase:
            case Phase.VAE:
                return self.vae_forward(x, con, cat, grid)
            case Phase.DIFFUSION:
                return self.diffusion_forward(x, con, cat, grid)
            case Phase.EVAL:
                return self.vae_forward(x, con, cat, grid)
        return self.vae_forward(x)

    def reconstruct(self, x, con, cat, grid):
        with torch.no_grad():
            h = self.encoder(x)
            q = self.lsr(h)
            z = q.rsample()
            #z = z.unsqueeze(1)
            #z_hat, _ = self.diffusion.reconstruct(z, con, cat, grid)
            #z_hat = z_hat.squeeze(1)
            z_hat = z
            x_hat = self.out_activ(self.decoder(z_hat))
        _ = []
        return x_hat, _

    def get_distribution(self, c=None) -> torch.tensor:
        pseudo_means, pseudo_scales = self.lsr.get_distribution(c)
        return pseudo_means, pseudo_scales

    def sample(self, n,con, cat, grid, length = 256, features=1, sampling="ddpm"):
        self.eval()
        x_t, steps = self.diffusion.sample(n, con, cat, grid, length, features, sampling)
        with torch.no_grad():
            x_hat = self.out_activ(self.decoder(x_t))
        return x_hat, steps

    def vae_step(self, batch, batch_idx):
        return super().training_step(batch, batch_idx)

    def diffusion_step(self, batch, batch_idx):
        x, con, cat, grid = batch
        h = self.encoder(x)
        q = self.lsr(h)
        z = q.rsample()
        z = z.unsqueeze(1)
        #print("z", z.shape)
        return self.diffusion.step(z, con, cat, grid)


    def step(self, batch, batch_idx):
        match self.phase:
            case Phase.VAE:
                return self.vae_step(batch, batch_idx)
            case Phase.DIFFUSION:
                return self.diffusion_step(batch, batch_idx)
    
        raise ValueError(f"Invalid phase {self.phase}")

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        match self.phase:
            case Phase.VAE:
                self.log("elbo", loss, on_step=True, on_epoch=True, sync_dist=True)
            case Phase.DIFFUSION:
                self.log("train_loss_diffusion", loss, on_step=True, on_epoch=True, sync_dist=True)
        return loss


    def validation_step(self, batch, batch_idx):
        match self.phase:
            case Phase.VAE:
                loss = super().validation_step(batch, batch_idx)
                self.log("valid_loss", loss, on_step=True, on_epoch=True, sync_dist=True)
            case Phase.DIFFUSION: 
                loss = self.step(batch, batch_idx)
                self.log("valid_loss_diffusion", loss, on_step=True, on_epoch=True, sync_dist=True)
            case Phase.EVAL:
                loss = self.step(batch, batch_idx)
                self.log("valid_loss_eval", loss, on_step=True, on_epoch=True, sync_dist=True)
            case _:
                loss = self.step(batch, batch_idx)
        return loss

    def test_step(self, batch, batch_idx):
        match self.phase:
            case Phase.VAE:
                loss = super().test_step(batch, batch_idx)
                self.log("test_loss", loss, on_step=True, on_epoch=True, sync_dist=True)
            case Phase.DIFFUSION:
                loss = self.step(batch, batch_idx)
                self.log("test_loss_diffusion", loss, on_step=True, on_epoch=True, sync_dist=True)
            case Phase.EVAL:
                loss = self.step(batch, batch_idx)
                self.log("test_loss_eval", loss, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

