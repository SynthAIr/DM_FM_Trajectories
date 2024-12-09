import math
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
from model.AirDiffTraj import EmbeddingBlock, make_beta_schedule, EMAHelper, gather, get_timestep_embedding
from tqdm import tqdm
from torch.nn import functional as F

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)  
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)


def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0,1,0,1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h


class LinAttnBlock(LinearAttention):
    """to match AttnBlock usage"""
    def __init__(self, in_channels):
        super().__init__(dim=in_channels, heads=1, dim_head=in_channels)


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
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

        # compute attention
        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w)
        q = q.permute(0,2,1)   # b,hw,c
        k = k.reshape(b,c,h*w) # b,c,hw
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,h,w)

        h_ = self.proj_out(h_)

        return x+h_


def make_attn(in_channels, attn_type="vanilla"):
    assert attn_type in ["vanilla", "linear", "none"], f'attn_type {attn_type} unknown'
    print(f"making attention of type '{attn_type}' with {in_channels} in_channels")
    if attn_type == "vanilla":
        return AttnBlock(in_channels)
    elif attn_type == "none":
        return nn.Identity(in_channels)
    else:
        return LinAttnBlock(in_channels)


class Unet(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution,use_timestep = True, use_linear_attn=False, attn_type="vanilla"):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = self.ch*4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        self.use_timestep = use_timestep
        if self.use_timestep:
            # timestep embedding
            self.temb = nn.Module()
            self.temb.dense = nn.ModuleList([
                torch.nn.Linear(self.ch,
                                self.temb_ch),
                torch.nn.Linear(self.temb_ch,
                                self.temb_ch),
            ])

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            skip_in = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                if i_block == self.num_res_blocks:
                    skip_in = ch*in_ch_mult[i_level]
                block.append(ResnetBlock(in_channels=block_in+skip_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x, t, extra_embed=None):
        #assert x.shape[2] == x.shape[3] == self.resolution
        """
        if context is not None:
            ## assume aligned context, cat along channel axis
            x = torch.cat((x, context), dim=1)


        if self.use_timestep:
            # timestep embedding
            assert t is not None
            temb = get_timestep_embedding(t, self.ch)
            temb = self.temb.dense[0](temb)
            temb = nonlinearity(temb)
            temb = self.temb.dense[1](temb)
        else:
            temb = None
        """
        temb = get_timestep_embedding(t, self.ch)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)
        if extra_embed is not None:
            temb = temb + extra_embed


        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](
                    torch.cat([h, hs.pop()], dim=1), temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h

    def get_last_layer(self):
        return self.conv_out.weight

class Diffusion(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.ch = config["ch"] * 4
        self.attr_dim = config["attr_dim"]
        self.guidance_scale = config["guidance_scale"]
        self.ch_mult = config["ch_mult"]
        self.num_res_blocks = config["num_res_blocks"]
        self.attn_resolutions = config["attn_resolutions"]
        self.dropout = config["dropout"]
        self.resamp_with_conv = config["resamp_with_conv"]
        self.in_channels = config["in_channels"]
        self.resolution = config["resolution"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.use_linear_attn = config["use_linear_attn"]
        #self.attn_type = config["attn_type"]

        self.unet = Unet(ch = self.ch, out_ch = self.attr_dim, ch_mult = self.ch_mult, num_res_blocks = self.num_res_blocks,
                         attn_resolutions = self.attn_resolutions, dropout = self.dropout, resamp_with_conv = self.resamp_with_conv, in_channels = self.in_channels,
                         resolution = self.resolution, use_timestep = True, use_linear_attn = False, attn_type = "vanilla")

        self.weather_config = config["weather_config"]
        self.continuous_len = config["continuous_len"]

        self.guide_emb = EmbeddingBlock(self.continuous_len, 0, self.ch, weather_config = self.weather_config)
        self.place_emb = EmbeddingBlock(self.continuous_len, 0, self.ch, weather_config = self.weather_config)

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
        self.eval()
        con = con.to(self.device)
        cat = cat.to(self.device)
        steps = []
        with torch.no_grad():
            #Fix this
            t = torch.tensor([self.n_steps-1], device=x.device)
            x_t, noise = self.q_xt_x0(x, t)
            for i in tqdm(range(self.n_steps-1, -1, -1)):
                x_t = self.sample_step(x_t,con, cat,grid, i)
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

    def step(self, batch, batch_idx):
        x, con, cat, grid = batch
        x_t, noise, t = self.forward_process(x)
        pred_noise = self.reverse_process(x_t, t, con, cat, grid)
        loss = F.mse_loss(noise.float(), pred_noise)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

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

