import torch
import yaml
from sklearn.datasets import make_swiss_roll


def sample_batch(size, noise=1.0):
    x, _= make_swiss_roll(size, noise=noise)
    return x[:, [0, 2]] / 10.0

def extract(input, t, x):
    shape = x.shape
    out = torch.gather(input, 0, t.to(input.device))
    reshape = [t.shape[0]] + [1] * (len(shape) - 1)
    return out.reshape(*reshape)

def make_beta_schedule(schedule='linear', n_timesteps=1000, start=1e-5, end=1e-2, device = None):
    if schedule == 'linear':
        betas = torch.linspace(start, end, n_timesteps, device=device)
    elif schedule == "quad":
        betas = torch.linspace(start ** 0.5, end ** 0.5, n_timesteps, device=device) ** 2
    elif schedule == "sigmoid":
        betas = torch.linspace(-6, 6, n_timesteps, device=device)
        betas = torch.sigmoid(betas) * (end - start) + start
    elif schedule == "cosine":
        # Cosine schedule from DDPM++
        s = 0.004  # Small constant to adjust the starting point
        steps = torch.arange(n_timesteps + 1, device=device, dtype=torch.float32)
        alphas = torch.cos(((steps / n_timesteps) + s) / (1 + s) * torch.pi / 2) ** 2
        alphas = alphas / alphas[0]  # Normalize to ensure alphas[0] = 1
        betas = 1 - (alphas[1:] / alphas[:-1])  # Derive beta_t from alpha_t values
        betas = torch.clip(betas, start, end)  # Ensure betas are in the [start, end] range

    return betas

def load_config(config_file):
    with open(config_file, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


def save_config(config, config_file):
    with open(config_file, "w") as f:
        yaml.dump(config, f)
