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
    
    def forward(self, x, c=None) -> Tuple[Tuple, torch.Tensor, torch.Tensor]:               # Overwrite the forward method for conditioning
        h = self.encoder(x)
        q = self.lsr(h, c)
        z = q.rsample()
        x_hat = self.out_activ(self.decoder(z))
        return self.lsr.dist_params(q), z, x_hat

    def get_distribution(self, c=None) -> torch.Tensor:
        pseudo_means, pseudo_scales = self.lsr.get_distribution(c)
        return pseudo_means, pseudo_scales

    def test_step(self, batch, batch_idx):
        x,c, info = batch
        _, _, x_hat = self.forward(x,c)
        loss = F.mse_loss(x_hat, x)
        self.log("hp/test_loss", loss)
        return torch.transpose(x, 1, 2), torch.transpose(x_hat, 1, 2), info


