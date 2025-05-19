import torch
from model.AirDiffTraj import EmbeddingBlock, make_beta_schedule, EMAHelper, gather, get_timestep_embedding, UNET
from model.generative import Generative
from tqdm import tqdm
from torch.nn import functional as F

class Diffusion(Generative):
    """
    Diffusion model for trajectory generation.
    Abstraction of the DiffTraj model from Source: https://github.com/Yasoz/DiffTraj (accessed August 2024)
    """
    def __init__(self, config, cuda=0):
        super().__init__()
        self.dataset_config = config["data"]
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
        self.in_channels = 1
        self.resolution = config["resolution"] if "resolution" in config.keys() else self.ch
        self.device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
        self.unet = UNET(config, resolution = self.resolution, in_channels = self.in_channels)
        self.weather_config = config["weather_config"]
        self.continuous_len = config["continuous_len"]

        self.guide_emb = EmbeddingBlock(self.continuous_len, 0, self.ch, weather_config = self.weather_config, dataset_config = self.dataset_config)
        self.place_emb = EmbeddingBlock(self.continuous_len, 0, self.ch, weather_config = self.weather_config, dataset_config = self.dataset_config)

        diff_config = config["diffusion"]
        self.n_steps = diff_config["num_diffusion_timesteps"]
        self.beta = make_beta_schedule(diff_config['beta_schedule'], self.n_steps, diff_config["beta_start"], diff_config["beta_end"], self.device)
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.lr = config["lr"]

        if config['diffusion']['ema']:
            self.ema_helper = EMAHelper(mu=config['diffusion']['ema_rate'])
            self.ema_helper.register(self)
        else:
            self.ema_helper = None

    """
    q(x_t| x_0) = N(mean, var)
    """
    def q_xt_x0(self, x0, t, debug=False):
        """
        q(x_t| x_0) = N(mean, var)
        Parameters
        ----------
        x0
        t
        debug

        Returns
        -------

        """
        self.alpha_bar = self.alpha_bar.to(x0.device)
        mean = gather(self.alpha_bar, t) ** 0.5 * x0
        var = 1 - gather(self.alpha_bar, t)
        eps = torch.randn_like(x0, device=x0.device)
        return mean + (var ** 0.5) * eps, eps  # also returns noise

    def forward_process(self, x0):
        """
        Forward process of the diffusion model.
        Parameters
        ----------
        x0

        Returns
        -------

        """
        t = torch.randint(low=0, high=self.n_steps,
                          size=(len(x0) // 2 + 1,), device=self.device)
        t = torch.cat([t, self.n_steps - t - 1], dim=0)[:len(x0)]
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
        guide_emb = self.guide_emb(con, cat, grid)
        place_vector_con = torch.zeros(con.shape, device=self.device)
        place_vector_cat = torch.zeros(cat.shape, device=self.device)
        place_vector_con = place_vector_con.type_as(con)
        place_vector_cat = place_vector_cat.type_as(cat)
        place_vector_grid = torch.zeros(grid.shape, device=self.device)
        place_vector_grid = place_vector_grid.type_as(grid)
        place_emb = self.place_emb(place_vector_con, place_vector_cat, place_vector_grid)
        
        cond_noise = self.unet(x_t, t, guide_emb)
        uncond_noise = self.unet(x_t, t, place_emb)
        pred_noise = cond_noise + self.guidance_scale * (cond_noise -
                                                             uncond_noise)
        return pred_noise

    def forward(self, x, con, cat, grid):
        """
        Forward pass through the model. This is called when you call the model directly.
        Parameters
        ----------
        x
        con
        cat
        grid

        Returns
        -------

        """
        x_t, noise, t = self.forward_process(x)
        x_hat = self.reverse_process(x_t, t, con, cat, grid)
        return x_t, noise, x_hat

    def reconstruct(self, x, con, cat, grid):
        """
        Reconstruct the input x using the diffusion model.
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

    def sample(self, n, con, cat, grid, length = 200, features=8, sampling="ddpm"):
        """
        Sample from the diffusion model.
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
        """
        Perform a single step of training. This is called during training.
        Parameters
        ----------
        x
        con
        cat
        grid

        Returns
        -------

        """
        x_t, noise, t = self.forward_process(x)
        pred_noise = self.reverse_process(x_t, t, con, cat, grid)
        loss = F.mse_loss(noise.float(), pred_noise)
        return loss


    def sample_step_ddpm(self, x, con, cat, grid, t):
        """
        Sample a single step of the diffusion model using DDPM.
        Parameters
        ----------
        x
        con
        cat
        grid
        t

        Returns
        -------

        """
        z = torch.randn_like(x, device=x.device) if t > 1 else 0
        tt =  torch.tensor([t]).to(device=x.device)
        eps_t = self.reverse_process(x, tt, con, cat, grid)
        x_tminusone = 1/torch.sqrt(self.alpha[t]) * (x - (1-self.alpha[t])/(torch.sqrt(1-self.alpha_bar[t])) * eps_t) + torch.sqrt(self.beta[t]) * z
        return x_tminusone

    def sample_step_ddim(self, x, con, cat, grid, t):
        """
        Sample a single step of the diffusion model using DDIM.
        Parameters
        ----------
        x
        con
        cat
        grid
        t

        Returns
        -------

        """
        l = 1
        if t <= 1:
            l = 0

        tt =  torch.tensor([t]).to(device=x.device)
        eps_t = self.reverse_process(x, tt, con, cat, grid)

        x_tminusone = 1/torch.sqrt(self.alpha[t]) * (x - torch.sqrt(1-self.alpha_bar[t]) * eps_t) + l * torch.sqrt(1 - self.alpha_bar[t-1]) * eps_t
        return x_tminusone

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
        x, con, cat, grid = batch
        loss = self.step(x, con, cat, grid)
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
        x, con, cat, grid = batch
        loss = self.step(x, con, cat, grid)
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
        x, con, cat, grid = batch
        loss = self.step(x, con, cat, grid)
        return loss

    def configure_optimizers(self):
        """
        Configure the optimizer for the model. This is used by pytorch lightning.
        Returns
        -------

        """
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """
        This is called at the end of each training batch. This is used to use EMA on the gradients.
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