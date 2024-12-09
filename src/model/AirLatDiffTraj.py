import torch
import numpy
import torch.nn as nn
from argparse import Namespace
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch.nn import functional as F

from model.tcvae import TCN
from utils import DatasetParams, TrafficDataset
from model.tcvae import VAE, VampPriorLSR
from model.tcvae import TCDecoder, TCEncoder

from typing import Tuple
from model.diffusion import Diffusion, Unet
from enum import Enum

class Phase(Enum):
    VAE = "vae"
    DIFFUSION = "diffusion"
    EVAL = "eval"

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

        @property
        def example_input_array(self):
            # return torch.rand(1, self.dataset_params["input_dim"], self.dataset_params["seq_len"]), torch.rand(1, 1, self.dataset_params["seq_len"]) # Example (x,c) 
            return torch.rand(1, self.config["in_channels"], self.config["traj_length"]) # Example X

        #self.conditional = config.get("conditional", False) 
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
        z = self.lsr.sample(h)
        x_hat = self.diffusion(z, con, cat, grid)
        return x_hat

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
            params, z, x_hat = self.forward(x, con, cat, grid)
        return x_hat, []

    def get_distribution(self, c=None) -> torch.Tensor:
        pseudo_means, pseudo_scales = self.lsr.get_distribution(c)
        return pseudo_means, pseudo_scales

    def sample(self, n,con, cat, grid, length = 200, features=8, sampling="ddpm"):
        self.eval()
        x_t, steps = self.diffusion.sample(n, con, cat, grid, length, features, sampling)
        with torch.no_grad():
            x_hat = self.out_activ(self.decoder(x_t))
        return x_hat, steps

    def vae_step(self, batch, batch_idx):
        return super().training_step(batch, batch_idx)

    def step(self, batch, batch_idx):
        match self.phase:
            case Phase.VAE:
                return self.vae_step(batch, batch_idx)
            case Phase.DIFFUSION:
                return self.diffusion.step(batch, batch_idx)
    
        raise ValueError(f"Invalid phase {self.phase}")

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        match self.phase:
            case Phase.VAE:
                self.log("train_loss_vae", loss, on_step=True, on_epoch=True, sync_dist=True)
            case Phase.DIFFUSION:
                self.log("train_loss_diffusion", loss, on_step=True, on_epoch=True, sync_dist=True)
        return loss


    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        match self.phase:
            case Phase.VAE:
                self.log("valid_loss_vae", loss, on_step=True, on_epoch=True, sync_dist=True)
            case Phase.DIFFUSION:
                self.log("valid_loss_diffusion", loss, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        match self.phase:
            case Phase.VAE:
                self.log("test_loss_vae", loss, on_step=True, on_epoch=True, sync_dist=True)
            case Phase.DIFFUSION:
                self.log("test_loss_diffusion", loss, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

