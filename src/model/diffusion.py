import math
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
from model.AirDiffTraj import EmbeddingBlock, make_beta_schedule, EMAHelper, gather, get_timestep_embedding, UNET
from model.generative import Generative
from tqdm import tqdm
from torch.nn import functional as F

class Diffusion(Generative):
    def __init__(self, config):
        super().__init__()
        self.dataset_config = config["data"]
        config = config["model"]
        self.config = config
        self.ch = config["ch"] * 4
        self.attr_dim = config["attr_dim"]
        self.guidance_scale = config["guidance_scale"]
        self.ch_mult = config["ch_mult"]
        self.num_res_blocks = config["num_res_blocks"]
        self.num_resolutions = len(self.ch_mult)
        self.attn_resolutions = config["attn_resolutions"]
        self.dropout = config["dropout"]
        self.resamp_with_conv = config["resamp_with_conv"]
        #jself.in_channels = config["in_channels"]
        self.in_channels = 1
        self.resolution = self.ch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.use_linear_attn = config["use_linear_attn"]
        #self.attn_type = config["attn_type"]
        self.unet = UNET(config, resolution = self.resolution, in_channels = self.in_channels)

        self.weather_config = config["weather_config"]
        self.continuous_len = config["continuous_len"]

        self.guide_emb = EmbeddingBlock(self.continuous_len, 0, self.ch, weather_config = self.weather_config, dataset_config = self.dataset_config)
        self.place_emb = EmbeddingBlock(self.continuous_len, 0, self.ch, weather_config = self.weather_config, dataset_config = self.dataset_config)

        diff_config = config["diffusion"]
        self.n_steps = diff_config["num_diffusion_timesteps"]
        self.beta = make_beta_schedule(diff_config['beta_schedule'], self.n_steps, diff_config["beta_start"], diff_config["beta_end"], self.device)
        #self.register_buffer("beta", self.beta)

        self.alpha = 1. - self.beta
        #self.register_buffer("alpha", self.alpha)

        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        #self.alpha_bar = self.alpha_bar.to(self.device)
        #self.register_buffer("alpha_bar", self.alpha_bar)

        self.lr = config["lr"]  # Explore this - might want it lower when training on the full dataset

        if config['diffusion']['ema']:
            self.ema_helper = EMAHelper(mu=config['diffusion']['ema_rate'])
            self.ema_helper.register(self)
        else:
            self.ema_helper = None

    """
    q(x_t| x_0) = N(mean, var)
    """
    def q_xt_x0(self, x0, t, debug=False):
        self.alpha_bar = self.alpha_bar.to(x0.device)
        mean = gather(self.alpha_bar, t) ** 0.5 * x0
        var = 1 - gather(self.alpha_bar, t)
        eps = torch.randn_like(x0, device=x0.device)
        return mean + (var ** 0.5) * eps, eps  # also returns noise

    def forward_process(self, x0):
        t = torch.randint(low=0, high=self.n_steps,
                          size=(len(x0) // 2 + 1,), device=self.device)
        t = torch.cat([t, self.n_steps - t - 1], dim=0)[:len(x0)]
        # Get the noised images (xt) and the noise (our target)
        xt, noise = self.q_xt_x0(x0, t)
        return xt, noise, t

    def reverse_process(self, x_t, t, con, cat, grid):
        """
        :param cat: Cateogrical attributes
        :param con: Continuous attributes
        :param x_t: X with noise for t timesteps
        :param t: Timestep
        :return:
        """
        #print(con.shape, cat.shape, grid.shape)
        guide_emb = self.guide_emb(con, cat, grid)
        place_vector_con = torch.zeros(con.shape, device=self.device)
        place_vector_cat = torch.zeros(cat.shape, device=self.device)
        place_vector_con = place_vector_con.type_as(con)
        place_vector_cat = place_vector_cat.type_as(cat)
        place_vector_grid = torch.zeros(grid.shape, device=self.device)
        place_vector_grid = place_vector_grid.type_as(grid)
        place_emb = self.place_emb(place_vector_con, place_vector_cat, place_vector_grid)
        
        #print("reverse",x_t.shape)
        cond_noise = self.unet(x_t, t, guide_emb)
        uncond_noise = self.unet(x_t, t, place_emb)
        pred_noise = cond_noise + self.guidance_scale * (cond_noise -
                                                             uncond_noise)
        return pred_noise

    def forward(self, x, con, cat, grid):
        x_t, noise, t = self.forward_process(x)
        x_hat = self.reverse_process(x_t, t, con, cat, grid)
        return x_t, noise, x_hat

    def reconstruct(self, x, con, cat, grid):
        #print("diff reconstruct", x.shape)
        self.eval()
        con = con.to(self.device)
        cat = cat.to(self.device)
        steps = []
        with torch.no_grad():
            #Fix this
            t = torch.tensor([self.n_steps-1], device=x.device)
            x_t, noise = self.q_xt_x0(x, t)
            for i in tqdm(range(self.n_steps-1, -1, -1)):
                x_t = self.sample_step_ddpm(x_t,con, cat,grid, i)
                if i % 200 == 0:
                    steps.append(x_t.clone().detach())

        return x_t, steps

    def sample(self, n,con, cat, grid, length = 200, features=8, sampling="ddpm"):
        self.eval()
        con = con.to(self.device)
        cat = cat.to(self.device)
        steps = []
        with torch.no_grad():
            #Fix this
            x_t = torch.randn(n, *(features, length), device=self.device)
            for i in range(self.n_steps-1, -1, -1):
                if sampling == "ddim":
                    x_t = self.sample_step_ddim(x_t,con, cat,grid, i)
                else:
                    x_t = self.sample_step_ddpm(x_t,con, cat,grid, i)
                if i % 200 == 0:
                    steps.append(x_t.clone().detach())
        return x_t, steps

    def step(self, x, con, cat, grid):
        x_t, noise, t = self.forward_process(x)
        #print(x_t.shape)
        pred_noise = self.reverse_process(x_t, t, con, cat, grid)
        #print("real", noise.shape)
        #print("predicted",pred_noise.shape)
        loss = F.mse_loss(noise.float(), pred_noise)
        return loss


    def sample_step_ddpm(self, x, con, cat, grid, t):
        # From DDPM
        # z = z * lamba
        z = torch.randn_like(x, device=x.device) if t > 1 else 0
        tt =  torch.tensor([t]).to(device=x.device)
        eps_t = self.reverse_process(x, tt, con, cat, grid)
        #print(eps_t.shape, x.shape, z.shape, self.alpha[t], self.beta[t])
        x_tminusone = 1/torch.sqrt(self.alpha[t]) * (x - (1-self.alpha[t])/(torch.sqrt(1-self.alpha_bar[t])) * eps_t) + torch.sqrt(self.beta[t]) * z
        return x_tminusone

    def sample_step_ddim(self, x, con, cat, grid, t):
        l = 1
        if t <= 1:
            l = 0

        tt =  torch.tensor([t]).to(device=x.device)
        eps_t = self.reverse_process(x, tt, con, cat, grid)

        x_tminusone = 1/torch.sqrt(self.alpha[t]) * (x - torch.sqrt(1-self.alpha_bar[t]) * eps_t) + l * torch.sqrt(1 - self.alpha_bar[t-1]) * eps_t
        return x_tminusone

    def training_step(self, batch, batch_idx):
        x, con, cat, grid = batch
        loss = self.step(x, con, cat, grid)
        #self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, con, cat, grid = batch
        loss = self.step(x, con, cat, grid)
        #self.log("valid_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, con, cat, grid = batch
        loss = self.step(x, con, cat, grid)
        #self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
