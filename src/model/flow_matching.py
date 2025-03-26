import math
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
from model.AirDiffTraj import EmbeddingBlock, make_beta_schedule, EMAHelper, gather, get_timestep_embedding, UNET
from tqdm import tqdm
from torch.nn import functional as F
from flow_matching.path import CondOTProbPath, MixtureDiscreteProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler
from torch.nn.parallel import DistributedDataParallel
from torchmetrics.aggregation import MeanMetric
from model.generative import Generative
import gc
import lightning as L
from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler
from flow_matching.solver import MixtureDiscreteEulerSolver
from flow_matching.solver.ode_solver import ODESolver
from flow_matching.utils import ModelWrapper


def skewed_timestep_sample(num_samples: int, device: torch.device) -> torch.Tensor:
    P_mean = -1.2
    P_std = 1.2
    rnd_normal = torch.randn((num_samples,), device=device)
    sigma = (rnd_normal * P_std + P_mean).exp()
    time = 1 / (1 + sigma)
    time = torch.clip(time, min=0.0001, max=1.0)
    return time

class FlowMatching(Generative):

    def __init__(self, config, cuda=0, lat = False):
        super().__init__()
        self.config = config
        self.dataset_config = config["data"]
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
        self.ch = config["ch"] * 4
        self.in_channels = 8 if lat else 1
        #self.resolution = config['traj_length']
        self.resolution = config['traj_length'] if lat else self.ch

        self.unet = UNET(config, resolution = self.resolution, in_channels = self.in_channels)
        #self.optimizer_cfg = optimizer_cfg
        #self.lr_scheduler_cfg = lr_scheduler_cfg
        self.discrete = False
        # Metrics
        self.path = CondOTProbPath()
        self.skewed_timesteps = True
        #self.MASK_TOKEN = 256
        self.weather_config = config["weather_config"]
        self.continuous_len = config["continuous_len"]
        self.guidance_scale = 0.0

        self.guide_emb = EmbeddingBlock(self.continuous_len, 0, self.ch, weather_config = self.weather_config, dataset_config = self.dataset_config)
        self.place_emb = EmbeddingBlock(self.continuous_len, 0, self.ch, weather_config = self.weather_config, dataset_config = self.dataset_config)


        if config['diffusion']['ema']:
            self.ema_helper = EMAHelper(mu=config['diffusion']['ema_rate'])
            self.ema_helper.register(self)
        else:
            self.ema_helper = None

    def forward(self, x, t, con, cat, grid):
        t = torch.zeros(x.shape[0], device=x.device) + t

        if self.guidance_scale == 0.0:
            guide_emb = self.guide_emb(con, cat, grid)
            return self.unet(x, t, guide_emb)

        guide_emb = self.guide_emb(con, cat, grid)
        place_vector_con = torch.zeros(con.shape, device=self.device)
        place_vector_cat = torch.zeros(cat.shape, device=self.device)
        place_vector_con = place_vector_con.type_as(con)
        place_vector_cat = place_vector_cat.type_as(cat)
        place_vector_grid = torch.zeros(grid.shape, device=self.device)
        place_vector_grid = place_vector_grid.type_as(grid)
        place_emb = self.place_emb(place_vector_con, place_vector_cat, place_vector_grid)
        
        #print("reverse",x_t.shape)
        cond_noise = self.unet(x, t, guide_emb)
        uncond_noise = self.unet(x, t, place_emb)

        result = cond_noise + self.guidance_scale * (cond_noise - uncond_noise)
        return result

    def step(self, x, con, cat, grid):

        # Generate conditioning
        # Scaling from [-1, 1] to [0, 1]
        #samples = x * 2.0 - 1.0
        samples = x
        noise = torch.randn_like(samples, device=self.device)

        t = (
            skewed_timestep_sample(samples.size(0), device=self.device)
            if self.skewed_timesteps
            else torch.rand(samples.size(0), device=self.device)
        )

        path_sample = self.path.sample(t=t, x_0=noise, x_1=samples)
        x_t, u_t = path_sample.x_t, path_sample.dx_t

        #with torch.cuda.amp.autocast():
        loss = (self(x_t, t, con, cat, grid) - u_t).pow(2).mean()
        if torch.isnan(loss):
            print("Loss is Nan, SElf-Ut",(self(x_t, t, con, cat, grid) - u_t).mean())
            print("Loss is Nan, SELF",(self(x_t, t, con, cat, grid)).mean())

        return loss



    def on_train_batch_end(self, outputs, batch, batch_idx):
        """This is called after the optimizer step, at the end of the batch."""
        #print("Called this on train batch end hooks")
        if self.config['diffusion']['ema']:
            self.ema_helper.update(self)

class Wrapper(ModelWrapper):
    def __init__(self, config, model, cuda=0):
        super().__init__(model)
        self.device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
        #self.model = model.to(self.device)
        self.solver = ODESolver(velocity_model=self.model)
        self.ch = self.model.ch

    def step(self, batch, batch_idx):
        x, con, cat, grid = batch
        loss = self.model.step(x, con, cat, grid)
        return loss

    def training_step(self, batch, batch_idx):  
        loss = self.step(batch, batch_idx)
        #self.log("train_loss", loss)
        return loss
        #raise NotImplementedError("Training step not implemented")

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        #self.log("valid_loss", loss)
        return loss
        #raise NotImplementedError("Validation step not implemented")

    def test_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        #self.log("test_loss", loss)
        return loss
        #raise NotImplementedError("Test step not implemented")

    def sample(self, n, con, cat, grid, features , length,  sampling="ddpm"):
        self.device = con.device
        x_0 = torch.randn((n, features, length), dtype=torch.float32, device=self.device)

        if False:
            time_grid = get_time_discretization(nfes=ode_opts["nfe"])
        else:
            time_grid = torch.tensor([0.0, 1.0], device=self.device)

        ##print(self.solver.device())

        synthetic_samples = self.solver.sample(
            time_grid=time_grid,
            x_init=x_0,
            method="midpoint", ### Change this to "midpoint" for DDIM
            return_intermediates=False,
            #atol=ode_opts["atol"] if "atol" in ode_opts else 1e-5,
            #rtol=ode_opts["rtol"] if "atol" in ode_opts else 1e-5,
            #step_size=ode_opts["step_size"] if "step_size" in ode_opts else None,
            atol= 1e-5,
            rtol= 1e-5,
            step_size= 0.1,
            con= con,
            cat = cat,
            grid = grid,
            #cfg_scale=self.model.guidance_scale,
        )

        return synthetic_samples, []

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """This is called after the optimizer step, at the end of the batch."""
        self.model.on_train_batch_end(outputs, batch, batch_idx)

    def reconstruct(self, x, con, cat, grid):
        self.eval()
        con = con.to(self.device)
        cat = cat.to(self.device)
        steps = []
        with torch.no_grad():
            x_0 = torch.randn(x.shape, dtype=torch.float32, device=self.device)

            if False:
                time_grid = get_time_discretization(nfes=ode_opts["nfe"])
            else:
                time_grid = torch.tensor([0.0, 1.0], device=self.device)

            synthetic_samples = self.solver.sample(
                time_grid=time_grid,
                x_init=x_0,
                method="midpoint", ### Change this to "midpoint" for DDIM
                return_intermediates=False,
                #atol=ode_opts["atol"] if "atol" in ode_opts else 1e-5,
                #rtol=ode_opts["rtol"] if "atol" in ode_opts else 1e-5,
                #step_size=ode_opts["step_size"] if "step_size" in ode_opts else None,
                atol= 1e-5,
                rtol= 1e-5,
                step_size= 0.1,
                con= con,
                cat = cat,
                grid = grid,
                #cfg_scale=self.model.guidance_scale,
            )
            # Scaling to [0, 1] from [-1, 1]
            #synthetic_samples = torch.clamp(
                #synthetic_samples * 0.5 + 0.5, min=0.0, max=1.0
            #)
            #synthetic_samples = torch.floor(synthetic_samples * 255)
            #synthetic_samples = synthetic_samples.to(torch.float32) / 255.0
            print(synthetic_samples.shape)
        return synthetic_samples, []

class AirFMTraj(L.LightningModule):

    def __init__(self, config, model, cuda=0):
        super().__init__()
        self.model = Wrapper(config, model, cuda=cuda)
        self.lr = config["lr"]


    def step(self, batch, batch_idx):
        return self.model.step(batch, batch_idx)

    def training_step(self, batch, batch_idx):  
        loss = self.step(batch, batch_idx)
        self.log("train_loss", loss)
        return loss
        #raise NotImplementedError("Training step not implemented")

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log("valid_loss", loss)
        return loss
        #raise NotImplementedError("Validation step not implemented")

    def test_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log("test_loss", loss)
        return loss
        #raise NotImplementedError("Test step not implemented")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """This is called after the optimizer step, at the end of the batch."""
        self.model.on_train_batch_end(outputs, batch, batch_idx)

    def sample(self, n, con, cat, grid, length,features ,  sampling="ddpm"):
        return self.model.sample(n, con, cat, grid, features = features, length = length)

    def reconstruct(self, x, con, cat, grid):
        #return self.vae.reconstruct(x, con, cat, grid)
        return self.model.reconstruct(x, con, cat, grid)

