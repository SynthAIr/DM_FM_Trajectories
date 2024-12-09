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
from model.diffusion import GaussianDiffusion, Unet
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
        dataset_params: DatasetParams,
        config: Union[Dict, Namespace],
    ) -> None:
        super().__init__(dataset_params, config)

        @property
        def example_input_array(self):
            # return torch.rand(1, self.dataset_params["input_dim"], self.dataset_params["seq_len"]), torch.rand(1, 1, self.dataset_params["seq_len"]) # Example (x,c) 
            return torch.rand(1, self.dataset_params["input_dim"], self.dataset_params["seq_len"]) # Example X

        self.conditional = config.get("conditional", False) 
        self.phase = Phase.VAE

        #config["cond_embed_dim"] = get_cond_len(dataset_params['conditional_features'], seq_len = self.dataset_params["seq_len"]) if self.conditional else 0
        #config["cond_embed_dim"] =config['length']

        self.encoder = TCEncoder(
            input_dim=self.dataset_params["input_dim"],
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
            out_dim=self.dataset_params["input_dim"],
            h_dims=self.hparams.h_dims[::-1],
            seq_len=self.dataset_params["seq_len"],
            kernel_size=self.hparams.kernel_size,
            dilation_base=self.hparams.dilation_base,
            sampling_factor=self.hparams.sampling_factor,
            dropout=self.hparams.dropout,
            h_activ=nn.ReLU(),
        )
        h_dim = self.hparams.h_dims[-1] * (int(self.dataset_params["seq_len"] / self.hparams.sampling_factor))

        self.lsr = VampPriorLSR(
            original_dim=self.dataset_params["input_dim"],
            original_seq_len=self.dataset_params["seq_len"],
            input_dim=h_dim,
            cond_length=0,
            out_dim=self.hparams.encoding_dim,
            encoder=self.encoder,
            n_components=self.hparams.n_components,
        )
        self.unet = Unet(
            dim = h_dim,
            dim_mults = (1, 2, 4, 8),
            num_classes = 6,
            cond_drop_prob = 0.5
        )
        self.diffusion = GaussianDiffusion(
            self.unet,
            image_size = h_dim,
            timesteps = 1000
        )

        self.out_activ = nn.Identity()

    def vae_forward(self, x) -> Tuple[Tuple, torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        q = self.lsr(h)
        z = q.rsample()
        x_hat = self.out_activ(self.decoder(z))
        return self.lsr.dist_params(q), z, x_hat

    def diffusion_forward(self, x, c=None) -> torch.Tensor:
        h = self.encoder(x)
        z = self.lsr.sample(h)
        x_hat = self.diffusion(z)
        return x_hat

    def forward(self, x, c=None) -> Tuple[Tuple, torch.Tensor, torch.Tensor]:               # Overwrite the forward method for conditioning
        match self.phase:
            case Phase.VAE:
                return self.vae_forward(x)
            case Phase.DIFFUSION:
                return self.diffusion_forward(x)
            case Phase.EVAL:
                return self.vae_forward(x)

    def get_distribution(self, c=None) -> torch.Tensor:
        pseudo_means, pseudo_scales = self.lsr.get_distribution(c)
        return pseudo_means, pseudo_scales

    def test_step(self, batch, batch_idx):
        x,c, info = batch
        _, _, x_hat = self.forward(x,c)
        loss = F.mse_loss(x_hat, x)
        self.log("hp/test_loss", loss)
        return torch.transpose(x, 1, 2), torch.transpose(x_hat, 1, 2), info

    def sample(self, n,con, cat, grid, length = 200, features=8):
        self.eval()
        con = con.to(self.device)
        cat = cat.to(self.device)
        steps = []
        with torch.no_grad():
            #Fix this
            x_t = torch.randn(n, *(features, length), device=self.device)
            for i in range(self.n_steps-1, -1, -1):
                x_t = self.sample_step(x_t,con, cat,grid, i)
                if i % 200 == 0:
                    steps.append(x_t.clone().detach())

            x_hat = self.out_activ(self.decoder(x_t))
        return x_hat, steps

    def sample_step(self, x, con, cat, grid, t):
        pass

    def step(self, batch, batch_idx):
        x, con, cat, grid = batch
        x_t, noise, t = self.forward_process(x)
        pred_noise = self.reverse_process(x_t, t, con, cat, grid)
        loss = F.mse_loss(noise.float(), pred_noise)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log("train_loss", loss, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """This is called after the optimizer step, at the end of the batch."""
        #print("Called this on train batch end hooks")
        if self.config['diffusion']['ema']:
            self.ema_helper.update(self)

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log("valid_loss", loss, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log("test_loss", loss, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)



class AirDiffTrajDDPM(AirLatDiffTraj):
    def __init__(self, dataset_params, config):
        super().__init__(dataset_params, config)

    def sample_step(self, x, con, cat, grid, t):
        # From DDPM
        # z = z * lamba
        z = torch.randn_like(x, device=x.device) if t > 1 else 0
        tt =  torch.tensor([t]).to(device=x.device)
        eps_t = self.reverse_process(x, tt, con, cat, grid)
        #print(eps_t.shape, x.shape, z.shape, self.alpha[t], self.beta[t])
        x_tminusone = 1/torch.sqrt(self.alpha[t]) * (x - (1-self.alpha[t])/(torch.sqrt(1-self.alpha_bar[t])) * eps_t) + torch.sqrt(self.beta[t]) * z
        return x_tminusone

class AirDiffTrajDDIM(AirLatDiffTraj):
    def __init__(self, dataset_params, config):
        super().__init__(dataset_params, config)

    def sample_step(self, x, con, cat, grid, t):
        l = 1
        if t <= 1:
            l = 0

        tt =  torch.tensor([t]).to(device=x.device)
        eps_t = self.reverse_process(x, tt, con, cat, grid)


        x_tminusone = 1/torch.sqrt(self.alpha[t]) * (x - torch.sqrt(1-self.alpha_bar[t]) * eps_t) + l * torch.sqrt(1 - self.alpha_bar[t-1]) * eps_t
        return x_tminusone
