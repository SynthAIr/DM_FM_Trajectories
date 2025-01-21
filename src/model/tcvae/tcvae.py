"""
Implementation of the Temporal Convolutional Variational Autoencoder (TCVAE) model, based on  https://github.com/kruuZHAW/deep-traffic-generation-paper

"""
from typing import List, Optional

import torch.nn as nn
import torch.nn.functional as F
# from torch.nn.utils import weight_norm #  deprecated in favor of torch.nn.utils.parametrizations.weight_norm
from torch.nn.utils.parametrizations import weight_norm
from argparse import Namespace
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch.nn import functional as F
from utils import DatasetParams, TrafficDataset
from model.tcvae.vae import VAE, VampPriorLSR, NormalLSR
from typing import Tuple
from model.AirDiffTraj import EmbeddingBlock


class TemporalBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2,
        out_activ: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()

        self.left_padding = (kernel_size - 1) * dilation

        self.conv = weight_norm(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=1,
                dilation=dilation,
            )
        )

        self.out_activ = out_activ
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self) -> None:
        self.conv.weight.data.normal_(0, 0.01)

    def forward(self, x):
        x = F.pad(x, (self.left_padding, 0), "constant", 0)
        x = self.conv(x)
        x = self.out_activ(x) if self.out_activ is not None else x
        x = self.dropout(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2,
        h_activ: Optional[nn.Module] = None,
        is_last: bool = False,
    ) -> None:
        super().__init__()

        self.is_last = is_last

        self.tmp_block1 = TemporalBlock(
            in_channels,
            out_channels,
            kernel_size,
            dilation,
            dropout,
            h_activ,
        )

        self.tmp_block2 = TemporalBlock(
            out_channels,
            out_channels,
            kernel_size,
            dilation,
            dropout,
            h_activ if not is_last else None,  # deactivate last activation
        )

        # Optional convolution for matching in_channels and out_channels sizes
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )
        self.init_weights()

    def init_weights(self) -> None:
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        y = self.tmp_block1(x)
        y = self.tmp_block2(y)
        r = x if self.downsample is None else self.downsample(x)
        return y + r


class TCN(nn.Module):

    def __init__(
        self,
        input_dim: int,
        out_dim: int,
        h_dims: List[int],
        kernel_size: int,
        dilation_base: int,
        h_activ: Optional[nn.Module] = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        layer_dims = [input_dim] + h_dims + [out_dim]
        self.n_layers = len(layer_dims) - 1
        layers = []

        for index in range(self.n_layers):
            dilation = dilation_base**index
            in_channels = layer_dims[index]
            out_channels = layer_dims[index + 1]
            is_last = index == (self.n_layers - 1)
            layer = ResidualBlock(
                in_channels,
                out_channels,
                kernel_size,
                dilation,
                dropout,
                h_activ,
                is_last,
            )
            layers.append(layer)

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCDecoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        out_dim: int,
        h_dims: List[int],
        seq_len: int,
        kernel_size: int,
        dilation_base: int,
        sampling_factor: int,
        h_activ: Optional[nn.Module] = None,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.sampling_factor = sampling_factor

        self.decode_entry = nn.Linear(input_dim, h_dims[0] * int(seq_len / sampling_factor))

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=sampling_factor),
            TCN(
                h_dims[0],
                out_dim,
                h_dims[1:],
                kernel_size,
                dilation_base,
                h_activ,
                dropout,
            ),
        )

    def forward(self, x):
        x = self.decode_entry(x)
        b, _ = x.size()
        x = x.view(b, -1, int(self.seq_len / self.sampling_factor))
        x_hat = self.decoder(x)
        return x_hat


    
class TCEncoder(nn.Module):
    def __init__(self,
        input_dim: int,
        out_dim: int,
        h_dims: List[int],
        kernel_size: int,
        dilation_base: int,
        sampling_factor: int,
        h_activ: Optional[nn.Module] = None,
        dropout: float = 0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            TCN(
                input_dim=input_dim,
                out_dim=out_dim,
                h_dims=h_dims,
                kernel_size=kernel_size,
                dilation_base=dilation_base,
                dropout=dropout,
                h_activ=h_activ,
            ),
            nn.AvgPool1d(sampling_factor),
            nn.Flatten(),)

    def forward(self, x):
        x = self.encoder(x)
        return x

class TCVAE(VAE):

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

        self.encoder = TCEncoder(
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
            input_dim=self.hparams.encoding_dim* (2 if self.conditional else 1),
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
        
        if self.config['type'] == 'TCVAE':
            print("Initing with VampPrior")
            self.lsr = VampPriorLSR(
                original_dim=self.config["in_channels"],
                original_seq_len=self.config["traj_length"],
                input_dim=h_dim,
                cond_length=0,
                out_dim=self.hparams.encoding_dim,
                encoder=self.encoder,
                n_components=self.hparams.n_components,
            )
        else:
            print("Initing with NormalLSR")
            self.lsr = NormalLSR(
                input_dim = h_dim,
                out_dim=self.hparams.encoding_dim,
                config=self.config)

        self.weather_config = self.config["weather_config"] if self.config != None else None
        self.dataset_config = self.config["data"] if self.config != None else None
        self.continuous_len = self.config["continuous_len"] if self.config != None else 0
        #print(self.continuous_len)
        #print(self.hparams.encoding_dim)
        if self.conditional:
            self.cond = EmbeddingBlock(self.continuous_len, 0, self.hparams.encoding_dim, weather_config = self.weather_config, dataset_config = self.dataset_config)
        self.out_activ = nn.Identity()
    
    def forward(self, x, con, cat, grid) -> Tuple[Tuple, torch.Tensor, torch.Tensor]:               # Overwrite the forward method for conditioning
        h = self.encoder(x)
        q = self.lsr(h, con, cat, grid)
        z = q.rsample()
        if self.conditional:
            cond = self.cond(con, cat, grid)
            z_lat = torch.cat((z , cond), dim=1)
        else:
            z_lat = z
        x_hat = self.out_activ(self.decoder(z_lat))
        return self.lsr.dist_params(q), z, x_hat
    
    def reconstruct(self, x, con, cat, grid):
        self.eval()
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        con = con.to(self.device)
        cat = cat.to(self.device)
        grid = grid.to(self.device)
        with torch.no_grad():
            #print("recon vae")
            return self.forward(x, con, cat, grid)[2], []

    def get_distribution(self, c=None) -> torch.Tensor:
        pseudo_means, pseudo_scales = self.lsr.get_distribution(c)
        return pseudo_means, pseudo_scales

    def decode(self, z, con, cat, grid):
        if self.conditional:
            cond = self.cond(con, cat, grid)
            z = torch.cat((z , cond), dim=1)
        return self.out_activ(self.decoder(z))

