import numpy as np
import os
import sys
import torch.nn as nn
from torch.nn import functional as F
import torch
import lightning as L
import matplotlib.pyplot as plt

## Main implementation based on "DiffTraj: Generating GPS Trajectory with Diffusion Probabilistic Model"
"""
class WideAndDeep(nn.Module):
    def __init__(self, embedding_dim=128, hidden_dim=256):
        super(WideAndDeep, self).__init__()

        # Wide part (linear model for continuous attributes)
        self.wide_fc = nn.Linear(5, embedding_dim)

        # Deep part (neural network for categorical attributes)
        self.depature_embedding = nn.Embedding(288, hidden_dim)
        self.sid_embedding = nn.Embedding(257, hidden_dim)
        self.eid_embedding = nn.Embedding(257, hidden_dim)
        self.deep_fc1 = nn.Linear(hidden_dim*3, embedding_dim)
        self.deep_fc2 = nn.Linear(embedding_dim, embedding_dim)
"""


class EmbdeddingBlock(nn.Module):
    def __init__(self, c_cat_in=10, c_num_in=1, c_out=128):
        super().__init__()
        self.fc_cat = nn.Linear(c_cat_in, c_out)
        self.fc_cat_2 = nn.Linear(c_out, c_out)
        self.fc_num = nn.Linear(c_num_in, c_out)

    def forward(self, c_cat, t, c_num=None):
        c_cat = F.relu(self.fc_cat(c_cat))
        c_cat = self.fc_cat_2(c_cat)
        if c_num is not None:
            c_num = self.fc_num(c_num)
            c = c_cat + c_num
        else:
            c = c_cat

        c = t + c
        return c


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, height, width, emb_dim=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        self.relu = nn.ReLU()
        self.norm = nn.BatchNorm2d(out_channels)
        self.wide_and_deep = EmbdeddingBlock(10, 1, emb_dim)
        self.temp_proj = nn.Linear(emb_dim, height * width)
        self.height = height
        self.width = width
        self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride,
                                   padding=0) if in_channels != out_channels else None

    def forward(self, x, t, c):
        x1 = self.norm(self.relu(self.conv1(x)))

        if c is not None:
            c_emb = self.wide_and_deep(c, t)
            # x1 = torch.cat([x1, t.view(-1, 128), c], dim=1)
            c_emb = self.temp_proj(c_emb)
            c_emb = c_emb.reshape(x1.shape[0], 1, self.width, self.height)
            x1 = x1 + c_emb

            # If input channels don't match output channels, use skip connection to match dimensions
        if self.skip_conv is not None:
            x = self.skip_conv(x)

        x2 = self.norm(self.relu(self.conv2(x1)))
        return x2 + x


# Example usage:
# in_channels = 3 (for RGB images), out_channels = 1 (for segmentation mask)
# cond_dim and time_dim are the dimensions of the condition and time embeddings.
class UNet(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.in_channels = 1
        self.out_channels = 1
        self.cond_dim = 10
        self.time_dim = 64

        self.downsample = nn.MaxPool2d(2, return_indices=True)
        self.size = 32
        self.resblock1 = ResNetBlock(self.in_channels, self.size, 3, 1, 1, 28, 28)
        self.resblock2 = ResNetBlock(self.size, self.size, 3, 1, 1, 14, 14)
        self.resblock3 = ResNetBlock(self.size, self.size, 3, 1, 1, 7, 7)

        self.upsample = nn.Upsample(scale_factor=2)
        # self.resblock3 = ResNetBlock(self.size, self.size, 3, 1, 1, 7, 7)
        self.resblock4 = ResNetBlock(self.size, self.size, 3, 1, 1, 14, 14)
        self.resblock5 = ResNetBlock(self.size, self.out_channels, 3, 1, 1, 28, 28)
        # self.resblock6 = ResNetBlock(64, self.out_channels, 3, 1, 1, 28, 28)

    def forward(self, x, t, c=None):
        x1 = self.resblock1(x, t, c)
        x2, x2_indc = self.downsample(x1)
        x2 = self.resblock2(x2, t, c)
        x3, x3_indc = self.downsample(x2)
        x3 = self.resblock3(x3, t, c)

        # x4 = self.upsample(x3, x3_indc, output_size=x2.shape)
        x3 = self.upsample(x3)
        x4 = self.resblock4(x3, t, c)
        x5 = self.upsample(x4)
        x5 = self.resblock5(x5, t, c)
        # x6 = self.resblock6(x5, t, c)
        return x5


class Diffusion(L.LightningModule):

    def __init__(self):
        super().__init__()
        self.num_steps = 1000
        self.beta = torch.linspace(0.0001, 0.05, self.num_steps).to(device='cuda')
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0).to(device='cuda')

        self.learning_rate = 1e-4
        self.channels = 1
        self.model = UNet()
        self.time_dim = 64

    def gather(self, consts: torch.Tensor, t: torch.Tensor):
        """Gather consts for $t$ and reshape to feature map shape"""
        c = consts.gather(-1, t)
        return c.reshape(-1, 1, 1)

    # Forward process
    def noise(self, x0, t, debug=False):
        # print(x0.shape)
        alpha_bar = self.gather(self.alpha_bar, t)
        mean = torch.sqrt(alpha_bar) * x0
        var = torch.sqrt(1 - alpha_bar)
        eps = torch.randn_like(x0)
        if debug:
            m = mean + var * eps
            fig, ax = plt.subplots(1, 4)
            ax[0].imshow(m[0][0].squeeze())
            ax[1].imshow(m[0][1].squeeze())
            ax[2].imshow(m[0][2].squeeze())
            ax[3].imshow(m[0][3].squeeze())
            plt.show()
        return mean + var * eps, eps  # also returns noise

    def forward_process(self, x):
        x0 = x
        # With this, the batch size need to be bigger than num steps
        t = torch.randint(low=0, high=self.num_steps, size=(len(x0) // 2 + 1,)).to(device='cuda')
        # t = torch.randint(low=0, high=self.num_steps, size=(1, )).to(device='cuda')
        t = torch.cat([t, self.num_steps - t - 1], dim=0)[:len(x0)]
        x_t, noise = self.noise(x0, t)

        return x_t, noise, t

    def get_timestep_embedding(self, timesteps, embedding_dim):
        assert len(timesteps.shape) == 1

        half_dim = embedding_dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        emb = emb.to(device=timesteps.device)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
        return emb

    def reverse_process(self, x_t, t, c=None):
        temb = self.get_timestep_embedding(t, self.time_dim).unsqueeze(1).repeat(1, x_t.shape[1], 1).reshape(-1,
                                                                                                             self.time_dim)
        if c is not None:
            c = c.repeat(x_t.shape[0], 1)

        # print(x_t.reshape(64, 64, -1).shape, temb.shape)
        x_t = x_t.reshape(-1, self.channels, 28, 28)

        pred_noise = self.model(x_t, temb, c)
        return pred_noise

    def forward(self, x, c=None):
        x_t, noise, t = self.forward_process(x)
        x_hat = self.reverse_process(x_t, t, c)
        return x_t, noise, x_hat

    def sample(self, n, c, t=None):
        t = t if t is not None else self.num_steps
        x0 = torch.randn(n, 1, 28, 28).to(device='cuda')
        # c0 = c.repeat(n, 1)
        t = (torch.ones(n, dtype=torch.long) * self.num_steps).to(device='cuda')
        # t = torch.randint(low=0, high=self.num_steps, size=(x0, )).to(device='cuda')
        x_hat = self.reverse_process(x0, t, c)
        return x_hat

    def step(self, batch, batch_idx):
        x, c = batch
        x_t, noise, t = self.forward_process(x)
        x_hat = self.reverse_process(x_t, t, c)
        x_hat = x_hat.reshape(x.shape[0], -1, 1, 28, 28)
        # x_hat = x_hat.mean(dim=1)
        loss = F.mse_loss(x_hat, noise, reduction='mean')
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log("valid_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

