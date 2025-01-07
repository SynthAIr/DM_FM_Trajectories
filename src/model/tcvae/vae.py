"""
Implementation of Latent Space Regularization Networks (LSR), based on  https://github.com/kruuZHAW/deep-traffic-generation-paper
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import (
    Distribution,
    Independent,
    MixtureSameFamily,
    Normal,
)
from torch.distributions.categorical import Categorical
from utils.data_utils import DatasetParams


class LSR(nn.Module):

    def __init__(self, input_dim: int, out_dim: int, fix_prior: bool = False):
        super().__init__()

        self.input_dim = input_dim
        self.out_dim = out_dim
        self.fix_prior = fix_prior

    def forward(self, hidden: torch.Tensor) -> Distribution:
        raise NotImplementedError()

    def dist_params(self, p: Distribution) -> Tuple:
        raise NotImplementedError()

    def get_posterior(self, dist_params: Tuple) -> Distribution:
        raise NotImplementedError()

    def get_prior(self, batch_size: int) -> Distribution:
        raise NotImplementedError()

    def get_distribution(self, condition = None) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

class CustomMSF(MixtureSameFamily):

    def rsample(self, sample_shape=torch.Size()):
        assert (
            self.component_distribution.has_rsample
        ), "component_distribution attribute should implement rsample() method"

        weights = self.mixture_distribution._param
        comp = nn.functional.gumbel_softmax(weights, hard=True).unsqueeze(-1)
        samples = self.component_distribution.rsample(sample_shape)
        return (comp * samples).sum(dim=1)


class NormalLSR(LSR):
    def __init__(self, input_dim: int, out_dim: int):
        super().__init__(input_dim, out_dim)

        self.z_loc = nn.Linear(input_dim, out_dim)
        z_log_var_layers = []
        z_log_var_layers.append(nn.Linear(input_dim, out_dim))
        z_log_var_layers.append(nn.Hardtanh(min_val=-6.0, max_val=2.0))
        self.z_log_var = nn.Sequential(*z_log_var_layers)

        self.out_dim = out_dim
        self.dist = Normal

        self.prior_loc = nn.Parameter(torch.zeros((1, out_dim)), requires_grad=False)
        self.prior_log_var = nn.Parameter(torch.zeros((1, out_dim)), requires_grad=False)
        self.register_parameter("prior_loc", self.prior_loc)
        self.register_parameter("prior_log_var", self.prior_log_var)

    def forward(self, hidden) -> Distribution:
        loc = self.z_loc(hidden)
        log_var = self.z_log_var(hidden)
        return Independent(self.dist(loc, (log_var / 2).exp()), 1)

    def dist_params(self, p: Independent) -> List[torch.Tensor]:
        return [p.base_dist.loc, p.base_dist.scale]

    def get_posterior(self, dist_params: List[torch.Tensor]) -> Independent:
        return Independent(self.dist(dist_params[0], dist_params[1]), 1)

    def get_prior(self) -> Independent:
        return Independent(self.dist(self.prior_loc, (self.prior_log_var / 2).exp()), 1)


class VampPriorLSR(LSR):

    def __init__(
        self,
        original_dim: int,
        original_seq_len: int,
        input_dim: int,
        out_dim: int,
        cond_length: int,
        encoder: nn.Module,
        n_components: int,
    ):
        super().__init__(input_dim, out_dim)

        self.original_dim = original_dim
        self.seq_len = original_seq_len
        self.encoder = encoder
        self.n_components = n_components
        cond_length = cond_length

        self.conditions = torch.rand(size=(n_components, cond_length))

        self.dist = MixtureSameFamily
        self.comp = Normal
        self.mix = Categorical

        z_loc_layers = []
        z_loc_layers.append(nn.Linear(input_dim + cond_length, out_dim))
        self.z_loc = nn.Sequential(*z_loc_layers)

        z_log_var_layers = []
        z_log_var_layers.append(nn.Linear(input_dim + cond_length, out_dim))
        z_log_var_layers.append(nn.Hardtanh(min_val=-6.0, max_val=2.0))
        self.z_log_var = nn.Sequential(*z_log_var_layers)

        self.idle_input = torch.autograd.Variable(torch.eye(n_components, n_components), requires_grad=False)

        if torch.cuda.is_available():
            self.idle_input = self.idle_input.cuda()

        pseudo_inputs_layers = []
        pseudo_inputs_layers.append(nn.Linear(n_components, n_components))
        pseudo_inputs_layers.append(nn.ReLU())
        pseudo_inputs_layers.append(
            nn.Linear(
                n_components,
                ((original_dim) * original_seq_len),
            )
        )
        pseudo_inputs_layers.append(nn.Hardtanh(min_val=-1.0, max_val=1.0))
        self.pseudo_inputs_NN = nn.Sequential(*pseudo_inputs_layers)

        prior_log_var_layers = []
        prior_log_var_layers.append(nn.Linear(input_dim + cond_length, out_dim))
        prior_log_var_layers.append(nn.Hardtanh(min_val=-6.0, max_val=2.0))
        self.prior_log_var_NN = nn.Sequential(*z_log_var_layers)

        self.prior_weights = nn.Parameter(torch.ones((1, n_components)), requires_grad=True)

        self.register_parameter("prior_weights", self.prior_weights)

    def forward(self, hidden: torch.Tensor, con: torch.Tensor = None, cat: torch.Tensor = None, grid: torch.Tensor = None) -> Distribution:

        loc = self.z_loc(hidden)
        log_var = self.z_log_var(hidden)
        scales = (log_var / 2).exp()

        X = self.pseudo_inputs_NN(self.idle_input)

        X = X.view((X.shape[0], self.original_dim,self.seq_len))
        # X, pseudo_c = X[:, :-1, :], X[:, -1]
        pseudo_h = self.encoder(X)
        pseudo_c = self.conditions.to(pseudo_h.device)

        self.prior_means = self.z_loc(pseudo_h)
        self.prior_log_vars = self.prior_log_var_NN(pseudo_h)

        return Independent(self.comp(loc, scales), 1)

    def dist_params(self, p: MixtureSameFamily) -> Tuple:
        return [p.base_dist.loc, p.base_dist.scale]

    def get_posterior(self, dist_params: Tuple) -> Distribution:
        return Independent(self.comp(dist_params[0], dist_params[1]), 1)

    def get_prior(self) -> MixtureSameFamily:
        return self.dist(
            self.mix(logits=self.prior_weights.view(self.n_components)),
            Independent(
                self.comp(
                    self.prior_means,
                    (self.prior_log_vars / 2).exp(),
                ),
                1,
            ),
        )

    def get_distribution(self, condition = None) -> Tuple[torch.Tensor, torch.Tensor]:
        pseudo_X = self.pseudo_inputs_NN(self.idle_input)
        pseudo_X = pseudo_X.view((pseudo_X.shape[0], self.original_dim, self.seq_len)) 
        pseudo_h = self.encoder(pseudo_X)

        if condition != None:
            condition = condition.repeat(pseudo_h.shape[0], 1)      
            pseudo_h = torch.cat((pseudo_h, condition), axis=1)

        pseudo_means = self.z_loc(pseudo_h)
        pseudo_scales = (self.z_log_var(pseudo_h) / 2).exp()

        return pseudo_means, pseudo_scales

"""
Implementation of abstract classes for Variational Autoencoder (VAE) in Lightning, based on  https://github.com/kruuZHAW/deep-traffic-generation-paper
"""

from argparse import Namespace
from typing import Dict, Tuple, Union

import lightning as L
import torch
import torch.nn as nn
from cartopy.crs import EuroPP
from torch.distributions.distribution import Distribution
from torch.nn import functional as F

class Abstract(L.LightningModule):

    _required_hparams = [
        "lr",
        "lr_step_size",
        "lr_gamma",
        "dropout",
    ]

    def __init__(self, config: Union[Dict, Namespace]) -> None:
        super().__init__()
        self.config = config
        self._check_hparams(config)
        self.save_hyperparameters(config)


    def _check_hparams(self, hparams: Union[Dict, Namespace]):
        for hparam in self._required_hparams:
            if isinstance(hparams, Namespace):
                if hparam not in vars(hparams).keys():
                    raise AttributeError(f"Can't set up network, {hparam} is missing.")
            elif isinstance(hparams, dict):
                if hparam not in hparams.keys():
                    raise AttributeError(f"Can't set up network, {hparam} is missing.")
            else:
                raise TypeError(f"Invalid type for hparams: {type(hparams)}.")




class AE(Abstract):

    _required_hparams = Abstract._required_hparams + [
        "encoding_dim",
        "h_dims",
    ]

    def __init__(self, config: Union[Dict, Namespace]) -> None:
        super().__init__(config)

        self.encoder: nn.Module
        self.decoder: nn.Module
        self.out_activ: nn.Module

    def configure_optimizers(self) -> dict:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.hparams.lr_step_size,
            gamma=self.hparams.lr_gamma,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }

    def on_train_start(self) -> None:
        params = {**self.hparams, "valid_loss": 1, "test_loss": 1}

        self.logger.log_hyperparams(params)

    def forward(self, x, con, cat, grid):
        z = self.encoder(x,None)
        x_hat = self.out_activ(self.decoder(z))
        return z, x_hat

    def get_latent(self, x, con, cat, grid):
        z = self.encoder(x)
        return z

    def training_step(self, batch, batch_idx):
        x, con, cat, grid = batch
        z, x_hat = self.forward(x, con, cat, grid)
        loss = F.mse_loss(x_hat, x)
        #self.log("train_loss", loss)
        return loss, z

    def validation_step(self, batch, batch_idx):
        x, con, cat, grid = batch
        z, x_hat = self.forward(x, con, cat, grid)
        loss = F.mse_loss(x_hat, x)
        #self.log("valid_loss", loss)
        return loss, z

    def test_step(self, batch, batch_idx):
        x, con, cat, grid = batch
        z, x_hat = self.forward(x,con, cat, grid)
        loss = F.mse_loss(x_hat, x)
        #self.log("test_loss", loss)
        return loss, z

    def decode(self, z):
        return self.out_activ(self.decoder(z))

class VAE(AE):

    _required_hparams = AE._required_hparams + [
        "kld_coef",
        "llv_coef",
        "scale",
        "fix_prior",
    ]

    def __init__(
        self,
        config: Union[Dict, Namespace],
    ) -> None:
        super().__init__(config)

        self.pseudo_gamma = 0.1

        self.scale = nn.Parameter(torch.Tensor([self.hparams.scale]), requires_grad=True)

        self.lsr: LSR

    def forward(self, x, con, cat, grid) -> Tuple[Tuple, torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        q = self.lsr(h)
        z = q.rsample()
        x_hat = self.out_activ(self.decoder(z))
        return self.lsr.dist_params(q), z, x_hat

    def get_latent(self, x, con, cat, grid):
        h = self.encoder(x)
        q = self.lsr(h)
        z = q.rsample()
        return z

    def training_step(self, batch, batch_idx):
        x, con, cat, grid = batch
        dist_params, z, x_hat = self.forward(x,con, cat, grid)

        llv_loss = -self.gaussian_likelihood(x, x_hat)
        llv_coef = self.hparams.llv_coef
        q_zx = self.lsr.get_posterior(dist_params)
        p_z = self.lsr.get_prior()
        kld_loss = self.kl_divergence(z, q_zx, p_z)
        kld_coef = self.hparams.kld_coef

        elbo = kld_coef * kld_loss + llv_coef * llv_loss
        elbo = elbo.mean()

        if self.hparams.reg_pseudo:
            pseudo_X = self.lsr.pseudo_inputs_NN(self.lsr.idle_input)
            pseudo_X = pseudo_X.view((pseudo_X.shape[0], x.shape[1], x.shape[2]))
            pseudo_dist_params, pseudo_z, pseudo_x_hat = self.forward(pseudo_X, con, cat, grid)

            pseudo_llv_loss = -self.gaussian_likelihood(pseudo_X, pseudo_x_hat)
            pseudo_q_zx = self.lsr.get_posterior(pseudo_dist_params)
            pseudo_kld_loss = self.kl_divergence(pseudo_z, pseudo_q_zx, p_z)

            pseudo_elbo = kld_coef * pseudo_kld_loss + llv_coef * pseudo_llv_loss
            pseudo_elbo = (x.shape[0] / pseudo_X.shape[0]) * pseudo_elbo.mean()
            elbo = elbo + self.pseudo_gamma * pseudo_elbo
        """
        self.log_dict(
            {
                "train_loss": elbo,
                "kl_loss": kld_loss.mean(),
                "recon_loss": llv_loss.mean(),
            }
        )
        """
        return elbo, q_zx, p_z, z, kld_loss.mean(), llv_loss.mean()

    def validation_step(self, batch, batch_idx):
        x, con, cat, grid = batch
        _, z, x_hat = self.forward(x,con, cat, grid)
        loss = F.mse_loss(x_hat, x)
        #self.log("valid_loss", loss)
        return loss, z

    def test_step(self, batch, batch_idx):
        x, con, cat, grid = batch
        _,z, x_hat = self.forward(x,con, cat, grid)
        loss = F.mse_loss(x_hat, x)
        #self.log("test_loss", loss)
        return loss, z

    def gaussian_likelihood(self, x: torch.Tensor, x_hat: torch.Tensor):
        mean = x_hat
        dist = torch.distributions.Normal(mean, self.scale)
        log_pxz = dist.log_prob(x)
        dims = [i for i in range(1, len(x.size()))]
        return log_pxz.sum(dim=dims)

    def kl_divergence(self, z: torch.Tensor, p: Distribution, q: Distribution) -> torch.Tensor:
        log_p = p.log_prob(z)
        log_q = q.log_prob(z)
        return log_p - log_q

    def gen_loss(self, x: torch.Tensor, x_hat: torch.Tensor, gamma: torch.Tensor):
        HALF_LOG_TWO_PI = 0.91893

        loggamma = torch.log(gamma)
        return torch.square((x - x_hat) / gamma) / 2.0 + loggamma + HALF_LOG_TWO_PI

    def kl_loss(self, mu: torch.Tensor, std: torch.Tensor):
        logstd = torch.log(std)
        return (torch.square(mu) + torch.square(std) - 2 * logstd - 1) / 2.0

    def get_distribution(self, c=None) -> torch.Tensor:
        raise NotImplementedError("get_distribution() must be implemented in subclass.")
