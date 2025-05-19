"""
# Adapted from Yazos's code in the DiffTraj paper:
# Source: https://github.com/Yasoz/DiffTraj (accessed August 2024)
# The code is gathered across the repository, from the files: Traj_UNet.py, utils.py, EMA.py, and module.py.
"""
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import lightning as L
from tqdm import tqdm
from utils.EMA import EMAHelper
from dataclasses import dataclass, field

def make_beta_schedule(schedule='linear', n_timesteps=1000, start=1e-5, end=1e-2, device = None):
    """
    Create a beta scheduler for the diffusion process.
    Parameters
    ----------
    schedule
    n_timesteps
    start
    end
    device

    Returns
    -------

    """
    if schedule == 'linear':
        betas = torch.linspace(start, end, n_timesteps, device=device)
    elif schedule == "quad":
        betas = torch.linspace(start ** 0.5, end ** 0.5, n_timesteps, device=device) ** 2
    elif schedule == "sigmoid":
        betas = torch.linspace(-6, 6, n_timesteps, device=device)
        betas = torch.sigmoid(betas) * (end - start) + start
    elif schedule == "cosine":
        s = 0.004
        steps = torch.arange(n_timesteps + 1, device=device, dtype=torch.float32)
        alphas = torch.cos(((steps / n_timesteps) + s) / (1 + s) * torch.pi / 2) ** 2
        alphas = alphas / alphas[0]
        betas = 1 - (alphas[1:] / alphas[:-1])
        betas = torch.clip(betas, start, end)

    return betas

def get_timestep_embedding(timesteps, embedding_dim):
    """
    Get the timestep embedding for the diffusion model.
    Parameters
    ----------
    timesteps
    embedding_dim

    Returns
    -------

    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = np.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb

def swish(x):
    """
    Swish activation function: "silu(x)=x∗σ(x),where σ(x) is the logistic sigmoid" https://docs.pytorch.org/docs/stable/generated/torch.nn.SiLU.html
    Parameters
    ----------
    x

    Returns
    -------

    """
    return F.silu(x)

def mish(x):
    """
    Mish activation function: "mish(x)=x∗tanh(softplus(x))"
    Parameters
    ----------
    x

    Returns
    -------

    """
    return F.mish(x)

def relu(x):
    """
    ReLU activation function: "ReLU(x)=max(0,x)"
    Parameters
    ----------
    x

    Returns
    -------

    """
    return F.relu(x)

def nonlinearity(x, function = "swish"):
    """
    Nonlinearity function for the model. Default is swish, but can be changed to mish or relu.
    Parameters
    ----------
    x
    function

    Returns
    -------

    """
    if function == "swish":
        return swish(x)

    if function == "mish":
        return mish(x)


    return x * torch.sigmoid(x)


class WeatherGrid(nn.Module):
    """
    WeatherGrid class for processing weather data using 3 convolutional layers.
    """
    def __init__(self, channels, lat_len, long_len, embedding_dim=128):
        super(WeatherGrid, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0)
        self.output_size = (lat_len - 3, long_len - 3)  # Approximated reduction through layers
        self.fc = nn.Linear(128 * self.output_size[0] * self.output_size[1], embedding_dim)

    def forward(self, x):
        """
        Forward pass through the WeatherGrid model.
        Parameters
        ----------
        x

        Returns
        -------

        """
        x = self.pool(relu(self.conv1(x)))  # (batch_size, 32, lat_len - 1, long_len - 1)
        x = self.pool(relu(self.conv2(x)))  # (batch_size, 64, lat_len - 2, long_len - 2)
        x = self.pool(relu(self.conv3(x)))  # (batch_size, 128, lat_len - 3, long_len - 3)
        x = x.view(x.size(0), -1)  # (batch_size, 128 * output_height * output_width)
        x = self.fc(x)
        
        return x
        
@dataclass
class WideAndDeepConfig:
    """
    Configuration for the Wide and Deep model.
    """
    continuous_len: int
    categorical_len: int
    embedding_dim: int = 128
    hidden_dim: int = 256
    deep_layers: list = field(default_factory=lambda: [10, 10, 5])
    
class WideAndDeep(nn.Module):
    """
    Wide and Deep model for processing continuous and categorical attributes.
    Based on the code from Source: https://github.com/Yasoz/DiffTraj
    """
    def __init__(self, wide_and_deep_config: WideAndDeepConfig):
        super(WideAndDeep, self).__init__()
        
        continuous_len = wide_and_deep_config.continuous_len
        categorical_len = wide_and_deep_config.categorical_len
        embedding_dim = wide_and_deep_config.embedding_dim
        hidden_dim = wide_and_deep_config.hidden_dim

        self.wide_fc = nn.Linear(continuous_len, embedding_dim)

        self.deep_embeddings = nn.ModuleList([
            nn.Embedding(emdim, hidden_dim) for emdim in wide_and_deep_config.deep_layers
        ])
        self.deep_fc1 = nn.Linear(hidden_dim*len(wide_and_deep_config.deep_layers), embedding_dim)
        self.deep_fc2 = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, continuous_attrs, categorical_attrs):
        """
        Forward pass through the Wide and Deep model.
        Parameters
        ----------
        continuous_attrs
        categorical_attrs

        Returns
        -------

        """
        wide_out = self.wide_fc(continuous_attrs)

        deep_embeddings = [
            embedding_layer(categorical_attrs[:, i]) 
            for i, embedding_layer in enumerate(self.deep_embeddings)
        ]
        
        # Concatenate all deep embeddings
        categorical_embed = torch.cat(deep_embeddings, dim=1)
        deep_out = nonlinearity(self.deep_fc1(categorical_embed))
        deep_out = self.deep_fc2(deep_out)
        
        
        combined_embed = wide_out + deep_out 
        return combined_embed

class WeatherBlock(nn.Module):
    """
    WeatherBlock class for processing weather data using multiple WeatherGrid instances.
    """
    def __init__(self, num_blocks, levels = 12, latitude = 105, longitude = 81, embedding_dim = 128) -> None:
        super().__init__()
        self.num_blocks = num_blocks
        self.levels = levels
        self.latitude = latitude
        self.longitude = longitude
        self.embedding_dim = embedding_dim

        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.blocks.append(WeatherGrid(levels, latitude, longitude, embedding_dim))

        self.fc1 = nn.Linear(embedding_dim*num_blocks, embedding_dim*num_blocks*2)
        self.fc2 = nn.Linear(embedding_dim*num_blocks*2, embedding_dim)

    def forward(self, x):
        """
        Forward pass through the WeatherBlock model.
        Parameters
        ----------
        x

        Returns
        -------

        """
        x = torch.cat([block(x[:,i]) for i, block in enumerate(self.blocks)], dim=1)
        x = nonlinearity(x)
        x = self.fc1(x)
        x = nonlinearity(x)
        x = self.fc2(x)
        return x

class MetarBlock(nn.Module):
    """
    MetarBlock class for processing METAR data.
    Same architecture as the WideAndDeep model, but with different predefined input dimensions.
    """
    def __init__(self, embedding_dim, hidden_dim) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.wide_fc = nn.Linear(13, embedding_dim)
        self.deep_emb = nn.Embedding(7, hidden_dim)
        self.deep_fc1 = nn.Linear(hidden_dim*1, embedding_dim)
        self.deep_fc2 = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):
        """
        Forward pass through the MetarBlock model.
        Parameters
        ----------
        x

        Returns
        -------

        """
        wide_out = self.wide_fc(x[:,:-1])
        cat = x[:,-1].int()
        categorical_embed = self.deep_emb(cat)
        deep_out = nonlinearity(self.deep_fc1(categorical_embed))
        deep_out = self.deep_fc2(deep_out)
        combined_embed = wide_out + deep_out
        return combined_embed



def get_categorical_category_counts(config):
    """
    Retrieves the total number of categories for each categorical variable in the configuration.

    Parameters:
        config (dict): Configuration dictionary containing conditional features.

    Returns:
        list: List of total number of categories for each categorical variable.
    """
    # Extract conditional features from the config
    conditional_features = config.get('conditional_features', {})
    
    # Initialize a list to hold category counts
    category_counts = []
    
    # Iterate through the features
    for feature_name, feature_config in conditional_features.items():
        if feature_config.get('type') == 'categorical':
            categories = feature_config.get('categories', [])
            category_counts.append(len(categories))  # Add the number of categories to the list
    
    return category_counts

class EmbeddingBlock(nn.Module):
    """
    EmbeddingBlock class for processing continuous and categorical attributes.
    Contains a WideAndDeep model and optional WeatherBlock and MetarBlock.
    Based on the code from Source: https://github.com/Yasoz/DiffTraj
    """
    def __init__(self, continuous_len, categorical_len, embedding_dim=128, hidden_dim=256, weather_config = None, dataset_config = None) -> None:
        super().__init__()
        wide_and_deep_config = WideAndDeepConfig(continuous_len, categorical_len, embedding_dim, hidden_dim, get_categorical_category_counts(dataset_config))
        self.wide_and_deep = WideAndDeep(wide_and_deep_config)
        self.weather_config = weather_config
        self.weather = weather_config and weather_config['weather_grid']
        self.metar = dataset_config['metar']
        
        if self.weather:
            variables = weather_config['variables']
            lat = weather_config['lat']
            lon = weather_config['lon']
            levels = weather_config['levels']
            w_type = weather_config['type']
            self.weather_block = WeatherBlock(num_blocks=variables, levels=levels, latitude=lat, longitude=lon, embedding_dim = embedding_dim)

        if self.metar:
            self.metar_block = MetarBlock(embedding_dim = embedding_dim, hidden_dim=hidden_dim)


    def forward(self, continuous_attrs, categorical_attrs, grid):
        """
        Forward pass through the EmbeddingBlock model.
        Parameters
        ----------
        continuous_attrs
        categorical_attrs
        grid

        Returns
        -------

        """
        x = self.wide_and_deep(continuous_attrs, categorical_attrs)

        if self.weather:
            x = x + self.weather_block(grid)

        if self.metar:
            x = x + self.metar_block(grid)

        return x



def Normalize(in_channels):
    """
    Normalization layer for the model. Default is GroupNorm, but can be changed to LayerNorm or InstanceNorm.
    Parameters
    ----------
    in_channels

    Returns
    -------

    """
    return torch.nn.GroupNorm(num_groups=16,
                              num_channels=in_channels,
                              eps=1e-6,
                              affine=True)


class Upsample(nn.Module):
    """
    Upsample class for upsampling the input tensor in the UNET.
    """
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
        """
        Forward method for UpSample
        Parameters
        ----------
        x

        Returns
        -------

        """
        x = torch.nn.functional.interpolate(x,
                                            scale_factor=2.0,
                                            mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    Downsample class for downsampling the input tensor in the UNET.
    """
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
        """
        Forward method for DownSample
        Parameters
        ----------
        x

        Returns
        -------

        """
        if self.with_conv:
            pad = (1, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x

    
class ResnetBlockLSTM(nn.Module):
    """
    ResnetBlockLSTM class for processing the input tensor using LSTM layers.
    Based on the code from Source: https://github.com/Yasoz/DiffTraj
    """
    def __init__(self,
                 in_channels,
                 out_channels=None,
                 use_lstm_shortcut=False,
                 dropout=0.1,
                 temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.use_lstm_shortcut = use_lstm_shortcut

        self.norm1 = Normalize(in_channels)
        self.lstm1 = nn.LSTM(input_size=in_channels,
                             hidden_size=self.out_channels,
                             batch_first=True)

        self.temb_proj = nn.Linear(temb_channels, self.out_channels)
        self.norm2 = Normalize(self.out_channels)
        self.dropout = nn.Dropout(dropout)

        self.lstm2 = nn.LSTM(input_size=self.out_channels,
                             hidden_size=self.out_channels,
                             batch_first=True)

        if self.in_channels != self.out_channels:
            if self.use_lstm_shortcut:
                self.shortcut = nn.LSTM(input_size=in_channels,
                                        hidden_size=self.out_channels,
                                        batch_first=True)
            else:
                self.nin_shortcut = nn.Linear(in_channels, self.out_channels)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)

        h, _ = self.lstm1(h)
        h = h + self.temb_proj(nonlinearity(temb))[:, None, :]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h, _ = self.lstm2(h)

        if self.in_channels != self.out_channels:
            if self.use_lstm_shortcut:
                x, _ = self.shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class ResnetBlock(nn.Module):
    """
    ResnetBlock class for processing the input tensor using convolutional layers.
    Based on the code from Source: https://github.com/Yasoz/DiffTraj
    """
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
    """
    AttnBlock used in UNET
    Based on the code from Source: https://github.com/Yasoz/DiffTraj
    """
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

def _get_resnet_block(in_channels, out_channels, temb_channels, dropout, CNN = True):
    if CNN:
        return ResnetBlock(in_channels, out_channels, dropout=dropout, temb_channels=temb_channels)
    else:
        return ResnetBlockLSTM(in_channels, out_channels, dropout=dropout, temb_channels=temb_channels)

class UNET(nn.Module):
    """
    UNET class for processing the input tensor using a U-Net architecture.
    Based on the code from Source: https://github.com/Yasoz/DiffTraj
    """
    def __init__(self, config, resolution = None, in_channels=None):
        super().__init__()
        self.config = config
        self.cnn = config["cnn"]
        ch, ch_mult = config["ch"], tuple(config["ch_mult"])
        #ch, out_ch, ch_mult = config["ch"], config["out_ch"], tuple(
            #config["ch_mult"])
        num_res_blocks = config["num_res_blocks"]
        attn_resolutions = config["attn_resolutions"]
        dropout = config["dropout"]
        in_channels = config["in_channels"]if in_channels is None else in_channels
        out_ch = in_channels
        resolution = config["traj_length"] if resolution is None else resolution
        resamp_with_conv = ["resamp_with_conv"]
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
                    _get_resnet_block(in_channels=block_in,
                                out_channels=block_out,
                                temb_channels=self.temb_ch,
                                dropout=dropout, CNN = self.cnn))
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
        self.mid.block_1 = _get_resnet_block(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout, CNN=self.cnn)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = _get_resnet_block(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout, CNN=self.cnn)

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
                    _get_resnet_block(in_channels=block_in + skip_in,
                                out_channels=block_out,
                                temb_channels=self.temb_ch,
                                dropout=dropout, CNN=self.cnn))
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
        if x.shape[2] != self.resolution:
            pass
            


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
                #print(h.shape, hs[-1].shape, temb.shape)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        h = hs[-1]  # [10, 256, 4, 4]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)
        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                ht = hs.pop()
                if ht.size(-1) != h.size(-1):
                    h = torch.nn.functional.pad(h,
                                                (0, ht.size(-1) - h.size(-1)))
                h = self.up[i_level].block[i_block](torch.cat([h, ht], dim=1),
                                                    temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


def gather(consts: torch.Tensor, t: torch.Tensor):
    """
    Gather the constants for the given timesteps and reshape to feature map shape.
    Based on the code from Source: https://github.com/Yasoz/DiffTraj
    Parameters
    ----------
    consts
    t

    Returns
    -------

    """
    c = consts.gather(-1, t)
    return c.reshape(-1, 1, 1)

class AirDiffTraj(L.LightningModule):
    """
    AirDiffTraj - Aircraft Trajectory Diffusion model.
    Originally from Source: https://github.com/Yasoz/DiffTraj but continued and modified by the authors of this repository.
    """

    def __init__(self, config):
        super().__init__()
        self.dataset_config = config["data"]
        self.config = config
        self.ch = config["ch"] * 4
        self.attr_dim = config["attr_dim"]
        self.guidance_scale = config["guidance_scale"]
        self.unet = UNET(config)

        self.weather_config = config["weather_config"]
        self.continuous_len = config["continuous_len"]

        self.guide_emb = EmbeddingBlock(self.continuous_len, 0, self.ch, weather_config = self.weather_config, dataset_config = self.dataset_config)
        self.place_emb = EmbeddingBlock(self.continuous_len, 0, self.ch, weather_config = self.weather_config, dataset_config = self.dataset_config)

        diff_config = config["diffusion"]
        self.n_steps = diff_config["num_diffusion_timesteps"]
        self.beta = make_beta_schedule(diff_config['beta_schedule'], self.n_steps, diff_config["beta_start"], diff_config["beta_end"], self.device)

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
        Forward process for the diffusion model.
        Parameters
        ----------
        x0

        Returns
        -------

        """
        t = torch.randint(low=0, high=self.n_steps,
                          size=(len(x0) // 2 + 1,), device=self.device)
        t = torch.cat([t, self.n_steps - t - 1], dim=0)[:len(x0)]
        # Get the noised images (xt) and the noise (our target)
        xt, noise = self.q_xt_x0(x0, t)
        return xt, noise, t

    def reverse_process(self, x_t, t, con, cat, grid):
        """
        Reverse process for the diffusion model.
        Uses classifier-free guidance
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
        """
        Forward pass through the model.
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
        Reconstruct the input tensor using the reverse process.
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
            t = torch.tensor([self.n_steps-1], device=x.device)
            x_t, noise = self.q_xt_x0(x, t)
            for i in tqdm(range(self.n_steps-1, -1, -1)):
                x_t = self.sample_step(x_t,con, cat,grid, i)
                if i % 200 == 0:
                    steps.append(x_t.clone().detach())

        return x_t, steps

    def sample(self, n,con, cat, grid, length = 200, features=8):
        """
        Sample from the model using the reverse process.
        Parameters
        ----------
        n
        con
        cat
        grid
        length
        features

        Returns
        -------

        """
        self.eval()
        con = con.to(self.device)
        cat = cat.to(self.device)
        steps = []
        with torch.no_grad():
            x_t = torch.randn(n, *(features, length), device=self.device)
            for i in range(self.n_steps-1, -1, -1):
                x_t = self.sample_step(x_t,con, cat,grid, i)
                if i % 200 == 0:
                    steps.append(x_t.clone().detach())
        return x_t, steps

    def sample_step(self, x, con, cat, grid, t):
        """
        Abstract method for sampling from the model.
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
        pass

    def step(self, batch, batch_idx):
        """
        Step function for the model. This is called during training and validation.
        Parameters
        ----------
        batch
        batch_idx

        Returns
        -------

        """
        x, con, cat, grid = batch
        x_t, noise, t = self.forward_process(x)
        pred_noise = self.reverse_process(x_t, t, con, cat, grid)
        loss = F.mse_loss(noise.float(), pred_noise)
        return loss

    def training_step(self, batch, batch_idx):
        """
        This is called during training by the pytorch lightning trainer.
        Parameters
        ----------
        batch
        batch_idx

        Returns
        -------

        """
        loss = self.step(batch, batch_idx)
        self.log("train_loss", loss, sync_dist=True)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """
        This is called at the end of the training batch, before optimizer.
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

    def validation_step(self, batch, batch_idx):
        """
        This is called during validation by the pytorch lightning trainer.
        Parameters
        ----------
        batch
        batch_idx

        Returns
        -------

        """
        loss = self.step(batch, batch_idx)
        self.log("valid_loss", loss, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        """
        This is called during testing by the pytorch lightning trainer.
        Parameters
        ----------
        batch
        batch_idx

        Returns
        -------

        """
        loss = self.step(batch, batch_idx)
        self.log("test_loss", loss, sync_dist=True)
        return loss

    def configure_optimizers(self):
        """
        Configure the optimizer for the pytorch lightning trainer.
        Returns
        -------

        """
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


class AirDiffTrajDDPM(AirDiffTraj):
    """
    Inherits from AirDiffTraj and implements the DDPM sampling step.
    """
    def __init__(self, config):
        super().__init__(config)

    def sample_step(self, x, con, cat, grid, t):
        """
        Sampling using DDPM
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

class AirDiffTrajDDIM(AirDiffTraj):
    """
    Inherits from AirDiffTraj and implements the DDIM sampling step.
    """
    def __init__(self, config):
        super().__init__(config)

    def sample_step(self, x, con, cat, grid, t):
        """
        Sampling using DDIM
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
