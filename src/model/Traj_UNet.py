import math
import torch
import torch.nn as nn
import numpy as np
from types import SimpleNamespace
import torch.nn.functional as F
import lightning as L

from utils import EMAHelper

"""
Code from https://github.com/Yasoz/DiffTraj
"""

def get_timestep_embedding(timesteps, embedding_dim):
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


class Attention(nn.Module):
    def __init__(self, embedding_dim):
        super(Attention, self).__init__()
        self.fc = nn.Linear(embedding_dim, 1)

    def forward(self, x):
        # x shape: (batch_size, num_attributes, embedding_dim)
        weights = self.fc(x)  # shape: (batch_size, num_attributes, 1)
        # apply softmax along the attributes dimension
        weights = F.softmax(weights, dim=1)
        return weights


class WideAndDeep(nn.Module):
    def __init__(self, continuous_len, categorical_len,embedding_dim=128, hidden_dim=256):
        super(WideAndDeep, self).__init__()

        # Wide part (linear model for continuous attributes)
        self.wide_fc = nn.Linear(continuous_len, embedding_dim)

        #self.embeddings = nn.ModuleList([
        #    nn.Embedding(cardinality, embedding_dim) for cardinality, embedding_dim in zip(categorical_len, embedding_dim)
        #])

        self.adep_embedding = nn.Embedding(2, hidden_dim)
        self.ades_embedding = nn.Embedding(2, hidden_dim)

        #self.depature_embedding = nn.Embedding(288, hidden_dim)
        #self.sid_embedding = nn.Embedding(257, hidden_dim)
        #self.eid_embedding = nn.Embedding(257, hidden_dim)
        self.deep_fc1 = nn.Linear(hidden_dim*2, embedding_dim)
        self.deep_fc2 = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, continuous_attrs, categorical_attrs):
        # Continuous attributes
        #print(continuous_attrs.shape, categorical_attrs.shape)
        self.wide_fc = self.wide_fc.to(continuous_attrs.device)
        self.adep_embedding = self.adep_embedding.to(categorical_attrs.device)
        self.ades_embedding = self.ades_embedding.to(categorical_attrs.device)
        self.deep_fc1 = self.deep_fc1.to(categorical_attrs.device)
        self.deep_fc2 = self.deep_fc2.to(categorical_attrs.device)

        wide_out = self.wide_fc(continuous_attrs)

        # Deep part: Processing categorical features
        print(categorical_attrs.shape)
        print(categorical_attrs[:, 0].shape)
        print(categorical_attrs[:, 1].shape)
        print(categorical_attrs[:, 0].dtype)
        print(categorical_attrs[:, 1].dtype)
        adep_embedding = self.adep_embedding(categorical_attrs[:, 0])
        ades_embedding = self.ades_embedding(categorical_attrs[:, 1])

        categorical_embed = torch.cat(
            (adep_embedding, ades_embedding), dim=1)
        deep_out = F.relu(self.deep_fc1(categorical_embed))
        deep_out = self.deep_fc2(deep_out)
        combined_embed = wide_out + deep_out

        # Combine wide (continuous) and deep (categorical) outputs
        #combined_embed = wide_out
        return combined_embed


def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32,
                              num_channels=in_channels,
                              eps=1e-6,
                              affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv=True):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv1d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x,
                                            scale_factor=2.0,
                                            mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv=True):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv1d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (1, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels=None,
                 conv_shortcut=False,
                 dropout=0.1,
                 temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv1d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv1d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv1d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv1d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)
        h = h + self.temb_proj(nonlinearity(temb))[:, :, None]
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv1d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv1d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv1d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv1d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)
        b, c, w = q.shape
        q = q.permute(0, 2, 1)  # b,hw,c
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)
        # attend to values
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, w)

        h_ = self.proj_out(h_)

        return x + h_


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        ch, out_ch, ch_mult = config["ch"], config["out_ch"], tuple(
            config["ch_mult"])
        num_res_blocks = config["num_res_blocks"]
        attn_resolutions = config["attn_resolutions"]
        dropout = config["dropout"]
        in_channels = config["in_channels"]
        resolution = config["traj_length"]
        resamp_with_conv = ["resamp_with_conv"]
        num_timesteps = config["diffusion"]["num_diffusion_timesteps"]

        if config["type"] == 'bayesian':
            self.logvar = nn.Parameter(torch.zeros(num_timesteps))

        self.ch = ch
        self.temb_ch = self.ch * 4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # timestep embedding
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(self.ch, self.temb_ch),
            torch.nn.Linear(self.temb_ch, self.temb_ch),
        ])

        # downsampling
        self.conv_in = torch.nn.Conv1d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1, ) + ch_mult
        self.down = nn.ModuleList()
        block_in = None
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    ResnetBlock(in_channels=block_in,
                                out_channels=block_out,
                                temb_channels=self.temb_ch,
                                dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            skip_in = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                if i_block == self.num_res_blocks:
                    skip_in = ch * in_ch_mult[i_level]
                block.append(
                    ResnetBlock(in_channels=block_in + skip_in,
                                out_channels=block_out,
                                temb_channels=self.temb_ch,
                                dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv1d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x, t, extra_embed=None):
        assert x.shape[2] == self.resolution

        # timestep embedding
        temb = get_timestep_embedding(t, self.ch)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)
        if extra_embed is not None:
            temb = temb + extra_embed

        # downsampling
        hs = [self.conv_in(x)]
        # print(hs[-1].shape)
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                # print(i_level, i_block, h.shape)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        # print(hs[-1].shape)
        # print(len(hs))
        h = hs[-1]  # [10, 256, 4, 4]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)
        # print(h.shape)
        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                ht = hs.pop()
                if ht.size(-1) != h.size(-1):
                    h = torch.nn.functional.pad(h,
                                                (0, ht.size(-1) - h.size(-1)))
                h = self.up[i_level].block[i_block](torch.cat([h, ht], dim=1),
                                                    temb)
                # print(i_level, i_block, h.shape)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Guide_UNet(nn.Module):
    def __init__(self, config):
        super(Guide_UNet, self).__init__()
        self.config = config
        self.ch = config.model.ch * 4
        self.attr_dim = config.model.attr_dim
        self.guidance_scale = config.model.guidance_scale
        self.unet = Model(config)
        # self.guide_emb = Guide_Embedding(self.attr_dim, self.ch)
        # self.place_emb = Place_Embedding(self.attr_dim, self.ch)
        self.guide_emb = WideAndDeep(4,0,self.ch)
        self.place_emb = WideAndDeep(4,0, self.ch)

    def forward(self, x, t, attr):
        guide_emb = self.guide_emb(attr)
        place_vector = torch.zeros(attr.shape, device=attr.device)
        place_vector = place_vector.type_as(attr)

        place_emb = self.place_emb(place_vector)
        cond_noise = self.unet(x, t, guide_emb)
        uncond_noise = self.unet(x, t, place_emb)
        pred_noise = cond_noise + self.guidance_scale * (cond_noise -
                                                         uncond_noise)
        return pred_noise


def gather(consts: torch.Tensor, t: torch.Tensor):
    """Gather consts for $t$ and reshape to feature map shape"""
    t = t.to(consts.device)
    c = consts.gather(-1, t)
    return c.reshape(-1, 1, 1)

class Guide_UNet2(L.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.ch = config["ch"] * 4
        self.attr_dim = config["attr_dim"]
        self.guidance_scale = config["guidance_scale"]
        self.unet = Model(config)
        # self.guide_emb = Guide_Embedding(self.attr_dim, self.ch)
        # self.place_emb = Place_Embedding(self.attr_dim, self.ch)
        self.guide_emb = WideAndDeep(4, 0, self.ch)
        self.place_emb = WideAndDeep(4, 0, self.ch)

        diff_config = config["diffusion"]
        self.n_steps = diff_config["num_diffusion_timesteps"]
        self.beta = torch.linspace(diff_config["beta_start"],
                              diff_config["beta_end"], self.n_steps).cuda()
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.lr = config["lr"]  # Explore this - might want it lower when training on the full dataset

        if config['diffusion']['ema']:
            self.ema_helper = EMAHelper(mu=config['diffusion']['ema_rate'])
            self.ema_helper.register(self)
        else:
            self.ema_helper = None


    # Forward process
    """
    q(x_t| x_0) = N(mean, var)
    """
    def q_xt_x0(self, x0, t, debug=False):
        #mean = gather(self.alpha_bar, t) ** 0.5 * x0
        mean = gather(self.alpha_bar, t)
        x0 = x0.to(mean.device)
        mean = mean ** 0.5 * x0
        var = 1 - gather(self.alpha_bar, t)
        eps = torch.randn_like(x0, device=x0.device)
        return mean + (var ** 0.5) * eps, eps  # also returns noise

    def forward_process(self, x0):
        t = torch.randint(low=0, high=self.n_steps,
                          size=(len(x0) // 2 + 1,), device=x0.device)
        t = torch.cat([t, self.n_steps - t - 1], dim=0)[:len(x0)]
        # Get the noised images (xt) and the noise (our target)
        xt, noise = self.q_xt_x0(x0, t)
        return xt, noise, t

    def reverse_process(self, x_t, t, con, cat):
        """
        :param cat: Cateogrical attributes
        :param con: Continuous attributes
        :param x_t: X with noise for t timesteps
        :param t: Timestep
        :return:
        """
        t = t.to(x_t.device)
        con = con.to(x_t.device)
        cat = cat.to(x_t.device)
        guide_emb = self.guide_emb(con, cat)
        place_vector_con = torch.zeros(con.shape, device=con.device)
        place_vector_cat = torch.zeros(cat.shape, device=cat.device)
        place_vector_con = place_vector_con.type_as(con)
        place_vector_cat = place_vector_cat.type_as(cat)

        place_emb = self.place_emb(place_vector_con, place_vector_cat)

        cond_noise = self.unet(x_t, t, guide_emb)
        uncond_noise = self.unet(x_t, t, place_emb)
        pred_noise = cond_noise + self.guidance_scale * (cond_noise -
                                                             uncond_noise)
        return pred_noise

    def forward(self, x, con, cat):
        x_t, noise, t = self.forward_process(x)
        x_hat = self.reverse_process(x_t, t, con, cat)
        return x_t, noise, x_hat


    def sample(self, n, c, t=None):
        #raise NotImplementedError("Womp Womp")
        pass

    def step(self, batch, batch_idx):
        x, con, cat = batch
        x_t, noise, t = self.forward_process(x)
        pred_noise = self.reverse_process(x_t, t, con, cat)
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
            self.ema_helper.update(self.unet)

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
