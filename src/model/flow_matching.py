"""
Code for Flow Matching model
Code inspired from paper "Flow Matching Guide and Codebase" by Lipman et al.
Source: https://github.com/facebookresearch/flow_matching
"""
import torch
from model.AirDiffTraj import EmbeddingBlock, make_beta_schedule, EMAHelper, gather, get_timestep_embedding, UNET
from flow_matching.path import CondOTProbPath, MixtureDiscreteProbPath
from model.generative import Generative
import lightning as L
from flow_matching.solver.ode_solver import ODESolver
from flow_matching.utils import ModelWrapper


def skewed_timestep_sample(num_samples: int, device: torch.device) -> torch.Tensor:
    """
    Sample a skewed timestep for the FM process.
    Parameters
    ----------
    num_samples
    device

    Returns
    -------

    """
    P_mean = -1.2
    P_std = 1.2
    rnd_normal = torch.randn((num_samples,), device=device)
    sigma = (rnd_normal * P_std + P_mean).exp()
    time = 1 / (1 + sigma)
    time = torch.clip(time, min=0.0001, max=1.0)
    return time

class FlowMatching(Generative):
    """
    Flow Matching model for trajectory generation.
    Merge of DiffTraj and Flow Matching model
    """

    def __init__(self, config, cuda=0, lat = False):
        super().__init__()
        self.config = config
        self.dataset_config = config["data"]
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
        self.ch = config["ch"] * 4
        self.in_channels = config["in_channels"] if lat else 1
        self.resolution = config['traj_length'] if lat else self.ch

        self.unet = UNET(config, resolution = self.resolution, in_channels = self.in_channels)
        self.discrete = False
        self.path = CondOTProbPath()
        self.skewed_timesteps = True
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
        """
        Forward pass of the model.
        Parameters
        ----------
        x
        t
        con
        cat
        grid

        Returns
        -------

        """
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
        
        cond_noise = self.unet(x, t, guide_emb)
        uncond_noise = self.unet(x, t, place_emb)

        result = cond_noise + self.guidance_scale * (cond_noise - uncond_noise)
        return result

    def step(self, x, con, cat, grid):
        """
        Takes a step in the training process.
        Parameters
        ----------
        x
        con
        cat
        grid

        Returns
        -------

        """

        samples = x
        noise = torch.randn_like(samples, device=self.device)

        t = (
            skewed_timestep_sample(samples.size(0), device=self.device)
            if self.skewed_timesteps
            else torch.rand(samples.size(0), device=self.device)
        )

        path_sample = self.path.sample(t=t, x_0=noise, x_1=samples)
        x_t, u_t = path_sample.x_t, path_sample.dx_t

        loss = (self(x_t, t, con, cat, grid) - u_t).pow(2).mean()
        if torch.isnan(loss):
            print("Loss is Nan, SElf-Ut",(self(x_t, t, con, cat, grid) - u_t).mean())
            print("Loss is Nan, SELF",(self(x_t, t, con, cat, grid)).mean())

        return loss



    def on_train_batch_end(self, outputs, batch, batch_idx):
        """
        Called to add EMA updates to the model.
        Parameters
        ----------
        outputs
        batch
        batch_idx

        Returns
        -------

        """
        if self.config['diffusion']['ema']:
            self.ema_helper.update(self)

class Wrapper(ModelWrapper):
    """
    Wrapper class for the Flow Matching model.
    """
    def __init__(self, config, model, cuda=0):
        super().__init__(model)
        self.device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
        self.solver = ODESolver(velocity_model=self.model)
        self.ch = self.model.ch

    def step(self, batch, batch_idx):
        """
        Perform a single step of the model. This is called during training, validation and testing.
        Parameters
        ----------
        batch
        batch_idx

        Returns
        -------

        """
        x, con, cat, grid = batch
        loss = self.model.step(x, con, cat, grid)
        return loss

    def training_step(self, batch, batch_idx):
        """
        Perform a single training step. This is called during training.
        Parameters
        ----------
        batch
        batch_idx

        Returns
        -------

        """
        loss = self.step(batch, batch_idx)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Perform a single validation step. This is called during validation.
        Parameters
        ----------
        batch
        batch_idx

        Returns
        -------

        """
        loss = self.step(batch, batch_idx)
        return loss

    def test_step(self, batch, batch_idx):
        """
        Perform a single test step. This is called during testing.
        Parameters
        ----------
        batch
        batch_idx

        Returns
        -------

        """
        loss = self.step(batch, batch_idx)
        return loss

    def sample(self, n, con, cat, grid, features , length,  sampling="ddpm"):
        """
        Samples using FM model and ODE solver.
        Takes random gaussian noise and solves ODE to create a probability path leading to a sample.
        Parameters
        ----------
        n
        con
        cat
        grid
        features
        length
        sampling

        Returns
        -------

        """
        self.device = con.device
        x_0 = torch.randn((n, features, length), dtype=torch.float32, device=self.device)

        if False:
            time_grid = get_time_discretization(nfes=ode_opts["nfe"])
        else:
            time_grid = torch.tensor([0.0, 1.0], device=self.device)

        synthetic_samples = self.solver.sample(
            time_grid=time_grid,
            x_init=x_0,
            method="midpoint",
            return_intermediates=False,
            #atol=ode_opts["atol"] if "atol" in ode_opts else 1e-5,
            #rtol=ode_opts["rtol"] if "atol" in ode_opts else 1e-5,
            #step_size=ode_opts["step_size"] if "step_size" in ode_opts else None,
            atol= 1e-5,
            rtol= 1e-5,
            step_size= 0.01,
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
        """
        Method needed for abstract class - samples from the model and does not reconstruct.
        Parameters
        ----------
        x
        con
        cat
        grid

        Returns
        -------

        """
        self.eval()
        self.device = con.device
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
            print(synthetic_samples.shape)
        return synthetic_samples, []

class AirFMTraj(L.LightningModule):
    """
    Flow Matching model for trajectory generation.
    Used to enable logging and training with PyTorch Lightning.
    """

    def __init__(self, config, model, cuda=0):
        super().__init__()
        self.model = Wrapper(config, model, cuda=cuda)
        self.lr = config["lr"]


    def step(self, batch, batch_idx):
        """
        Perform a single step of the model. This is called during training, validation and testing.
        Parameters
        ----------
        batch
        batch_idx

        Returns
        -------

        """
        return self.model.step(batch, batch_idx)

    def training_step(self, batch, batch_idx):
        """
        Perform a single training step. This is called during training.
        Parameters
        ----------
        batch
        batch_idx

        Returns
        -------

        """
        loss = self.step(batch, batch_idx)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Perform a single validation step. This is called during validation.
        Parameters
        ----------
        batch
        batch_idx

        Returns
        -------

        """
        loss = self.step(batch, batch_idx)
        self.log("valid_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        """
        Perform a single test step. This is called during testing.
        Parameters
        ----------
        batch
        batch_idx

        Returns
        -------

        """
        loss = self.step(batch, batch_idx)
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
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
        self.model.on_train_batch_end(outputs, batch, batch_idx)

    def sample(self, n, con, cat, grid, length,features ,  sampling="ddpm"):
        """
        Sample from the model.
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
        return self.model.sample(n, con, cat, grid, features = features, length = length)

    def reconstruct(self, x, con, cat, grid):
        """
        Reconstruct the model.
        Parameters
        ----------
        x
        con
        cat
        grid

        Returns
        -------

        """
        return self.model.reconstruct(x, con, cat, grid)

