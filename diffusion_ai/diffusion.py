__all__ = [
    "abar",
    "inv_abar",
    "noisify",
    "timestep_embedding",
    "pre_conv",
    "upsample",
    "lin",
    "EmbResBlock",
    "saved",
    "DownBlock",
    "UpBlock",
    "EmbUNetModel",
    "ddim_step",
    "sample",
    "cond_sample",
]

import math
import torch
import fastcore.all as fc
import torch.nn.functional as F

from torch import nn
from functools import wraps
from fastcore.foundation import L
from einops import rearrange  # install einpos
from fastprogress import progress_bar  # install fastai

from .sgd import *
from .accel import *
from .resnet import *
from .dtaset import *
from .learner import *
from .augment import *
from .training import *
from .convolution import *
from .activations import *
from .initialization import *


def abar(t):
    """
    Compute cosine schedule for a given timestep.

    Args:
        t (torch.Tensor): Timestep.

    Returns:
        torch.Tensor: Cosine schedule value.
    """
    return (t * math.pi / 2).cos() ** 2


def inv_abar(x):
    """
    Compute the inverse of the cosine schedule.

    Args:
        x (torch.Tensor): Cosine schedule value.

    Returns:
        torch.Tensor: Corresponding timestep.
    """
    return x.sqrt().acos() * 2 / math.pi


def noisify(x0):
    """
    Add noise to the input data.

    Args:
        x0 (torch.Tensor): Input tensor.

    Returns:
        tuple: Noisy input tensor and timestep, noise tensor.
    """
    device = x0.device
    n = len(x0)
    t = (
        torch.rand(
            n,
        )
        .to(x0)
        .clamp(0, 0.999)
    )
    ε = torch.randn(x0.shape, device=device)
    abar_t = abar(t).reshape(-1, 1, 1, 1).to(device)
    xt = abar_t.sqrt() * x0 + (1 - abar_t).sqrt() * ε
    return (xt, t.to(device)), ε


def timestep_embedding(tsteps, emb_dim, max_period=10000):
    """
    Generate sinusoidal timestep embeddings.

    Args:
        tsteps (torch.Tensor): Timestep tensor.
        emb_dim (int): Embedding dimension.
        max_period (int): Maximum period for the sinusoidal embeddings.

    Returns:
        torch.Tensor: Timestep embeddings.
    """
    exponent = -math.log(max_period) * torch.linspace(
        0, 1, emb_dim // 2, device=tsteps.device
    )
    emb = tsteps[:, None].float() * exponent.exp()[None, :]
    emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
    return F.pad(emb, (0, 1, 0, 0)) if emb_dim % 2 == 1 else emb


def pre_conv(ni, nf, ks=3, stride=1, act=nn.SiLU, norm=None, bias=True):
    """
    Create a sequential layer with optional normalization and activation
    followed by a convolution.

    Args:
        ni (int): Number of input channels.
        nf (int): Number of output channels.
        ks (int): Kernel size. Default is 3.
        stride (int): Stride for the convolution. Default is 1.
        act (nn.Module): Activation function. Default is nn.SiLU.
        norm (nn.Module): Normalization function. Default is None.
        bias (bool): Whether to use bias in convolution. Default is True.

    Returns:
        nn.Sequential: Sequential layer.
    """
    layers = nn.Sequential()
    if norm:
        layers.append(norm(ni))
    if act:
        layers.append(act())
    layers.append(
        nn.Conv2d(ni, nf, stride=stride, kernel_size=ks, padding=ks // 2, bias=bias)
    )
    return layers


def upsample(nf):
    """
    Create an upsampling layer followed by a convolution.

    Args:
        nf (int): Number of feature channels.

    Returns:
        nn.Sequential: Upsampling layer.
    """
    return nn.Sequential(nn.Upsample(scale_factor=2.0), nn.Conv2d(nf, nf, 3, padding=1))


def lin(ni, nf, act=nn.SiLU, norm=None, bias=True):
    """
    Create a sequential layer with optional normalization and activation
    followed by a linear layer.

    Args:
        ni (int): Number of input features.
        nf (int): Number of output features.
        act (nn.Module): Activation function. Default is nn.SiLU.
        norm (nn.Module): Normalization function. Default is None.
        bias (bool): Whether to use bias in linear layer. Default is True.

    Returns:
        nn.Sequential: Sequential layer.
    """
    layers = nn.Sequential()
    if norm:
        layers.append(norm(ni))
    if act:
        layers.append(act())
    layers.append(nn.Linear(ni, nf, bias=bias))
    return layers


class SelfAttention(nn.Module):
    """
    Self-Attention layer for sequence data.

    Args:
        ni (int): Number of input features.
        attn_chans (int): Number of attention channels.
        transpose (bool): Whether to transpose the input. Default is True.
    """

    def __init__(self, ni, attn_chans, transpose=True):
        super().__init__()
        self.nheads = ni // attn_chans
        self.scale = math.sqrt(ni / self.nheads)
        self.norm = nn.LayerNorm(ni)
        self.qkv = nn.Linear(ni, ni * 3)
        self.proj = nn.Linear(ni, ni)
        self.t = transpose

    def forward(self, x):
        """
        Forward pass for self-attention.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying self-attention.
        """
        n, c, s = x.shape
        if self.t:
            x = x.transpose(1, 2)
        x = self.norm(x)
        x = self.qkv(x)
        x = rearrange(x, "n s (h d) -> (n h) s d", h=self.nheads)
        q, k, v = torch.chunk(x, 3, dim=-1)
        s = (q @ k.transpose(1, 2)) / self.scale
        x = s.softmax(dim=-1) @ v
        x = rearrange(x, "(n h) s d -> n s (h d)", h=self.nheads)
        x = self.proj(x)
        if self.t:
            x = x.transpose(1, 2)
        return x


class SelfAttention2D(SelfAttention):
    """
    Self-Attention layer for 2D data.

    Args:
        ni (int): Number of input features.
        attn_chans (int): Number of attention channels.
        transpose (bool): Whether to transpose the input. Default is True.
    """

    def forward(self, x):
        """
        Forward pass for 2D self-attention.

        Args:
            x (torch.Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            torch.Tensor: Output tensor with shape (n, c, h, w).
        """
        n, c, h, w = x.shape
        return super().forward(x.view(n, c, -1)).reshape(n, c, h, w)


class EmbResBlock(nn.Module):
    """
    Residual block with embedding for time-dependent U-Net.

    Args:
        n_emb (int): Dimension of the embedding.
        ni (int): Number of input channels.
        nf (int): Number of output channels.
        ks (int): Kernel size. Default is 3.
        act (nn.Module): Activation function. Default is nn.SiLU.
        norm (nn.Module): Normalization function. Default is nn.BatchNorm2d.
        attn_chans (int): Number of attention channels. Default is 0.
    """

    def __init__(
        self, n_emb, ni, nf=None, ks=3, act=nn.SiLU, norm=nn.BatchNorm2d, attn_chans=0
    ):
        super().__init__()
        if nf is None:
            nf = ni
        self.emb_proj = nn.Linear(n_emb, nf * 2)
        self.conv1 = pre_conv(ni, nf, ks, act=act, norm=norm)
        self.conv2 = pre_conv(nf, nf, ks, act=act, norm=norm)
        self.idconv = fc.noop if ni == nf else nn.Conv2d(ni, nf, 1)
        self.attn = False
        if attn_chans:
            self.attn = SelfAttention2D(nf, attn_chans)

    def forward(self, x, t):
        """
        Forward pass for the residual block.

        Args:
            x (torch.Tensor): Input tensor.
            t (torch.Tensor): Timestep embedding.

        Returns:
            torch.Tensor: Output tensor after applying residual block.
        """
        inp = x
        x = self.conv1(x)
        emb = self.emb_proj(F.silu(t))[:, :, None, None]
        scale, shift = torch.chunk(emb, 2, dim=1)
        x = x * (1 + scale) + shift
        x = self.conv2(x)
        x = x + self.idconv(inp)
        if self.attn:
            x = x + self.attn(x)
        return x


def saved(m, blk):
    """
    Save intermediate results during forward pass.

    Args:
        m (nn.Module): Module to wrap.
        blk (nn.Module): Block to save results in.

    Returns:
        nn.Module: Wrapped module.
    """
    m_ = m.forward

    @wraps(m.forward)
    def _f(*args, kwargs):
        res = m_(*args, kwargs)
        blk.saved.append(res)
        return res

    m.forward = _f
    return m


class DownBlock(nn.Module):
    """
    Downsampling block for U-Net.

    Args:
        n_emb (int): Dimension of the embedding.
        ni (int): Number of input channels.
        nf (int): Number of output channels.
        add_down (bool): Whether to add downsampling. Default is True.
        num_layers (int): Number of layers in the block. Default is 1.
        attn_chans (int): Number of attention channels. Default is 0.
    """

    def __init__(self, n_emb, ni, nf, add_down=True, num_layers=1, attn_chans=0):
        super().__init__()
        self.resnets = nn.ModuleList(
            [
                saved(
                    EmbResBlock(n_emb, ni if i == 0 else nf, nf, attn_chans=attn_chans),
                    self,
                )
                for i in range(num_layers)
            ]
        )
        self.down = (
            saved(nn.Conv2d(nf, nf, 3, stride=2, padding=1), self)
            if add_down
            else nn.Identity()
        )

    def forward(self, x, t):
        """
        Forward pass for the downsampling block.

        Args:
            x (torch.Tensor): Input tensor.
            t (torch.Tensor): Timestep embedding.

        Returns:
            torch.Tensor: Output tensor after downsampling.
        """
        self.saved = []
        for resnet in self.resnets:
            x = resnet(x, t)
        x = self.down(x)
        return x


class UpBlock(nn.Module):
    """
    Forward pass for the downsampling block.

    Args:
        x (torch.Tensor): Input tensor.
        t (torch.Tensor): Timestep embedding.

    Returns:
        torch.Tensor: Output tensor after downsampling.
    """

    def __init__(self, n_emb, ni, prev_nf, nf, add_up=True, num_layers=2, attn_chans=0):
        super().__init__()
        self.resnets = nn.ModuleList(
            [
                EmbResBlock(
                    n_emb,
                    (prev_nf if i == 0 else nf) + (ni if (i == num_layers - 1) else nf),
                    nf,
                    attn_chans=attn_chans,
                )
                for i in range(num_layers)
            ]
        )
        self.up = upsample(nf) if add_up else nn.Identity()

    def forward(self, x, t, ups):
        """
        Forward pass for the upsampling block.

        Args:
            x (torch.Tensor): Input tensor.
            t (torch.Tensor): Timestep embedding.
            ups (list): List of saved tensors for concatenation.

        Returns:
            torch.Tensor: Output tensor after upsampling.
        """
        for resnet in self.resnets:
            x = resnet(torch.cat([x, ups.pop()], dim=1), t)
        return self.up(x)


class EmbUNetModel(nn.Module):
    """
    U-Net model with embedding for time-dependent processing.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        nfs (tuple): Number of feature channels at each stage.
        num_layers (int): Number of layers in each block. Default is 1.
        attn_chans (int): Number of attention channels. Default is 8.
        attn_start (int): Start index for adding attention. Default is 1.
    """

    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        nfs=(224, 448, 672, 896),
        num_layers=1,
        attn_chans=8,
        attn_start=1,
    ):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, nfs[0], kernel_size=3, padding=1)
        self.n_temb = nf = nfs[0]
        n_emb = nf * 4
        self.emb_mlp = nn.Sequential(
            lin(self.n_temb, n_emb, norm=nn.BatchNorm1d), lin(n_emb, n_emb)
        )
        self.downs = nn.ModuleList()
        n = len(nfs)
        for i in range(n):
            ni = nf
            nf = nfs[i]
            self.downs.append(
                DownBlock(
                    n_emb,
                    ni,
                    nf,
                    add_down=i != n - 1,
                    num_layers=num_layers,
                    attn_chans=0 if i < attn_start else attn_chans,
                )
            )
        self.mid_block = EmbResBlock(n_emb, nfs[-1])

        rev_nfs = list(reversed(nfs))
        nf = rev_nfs[0]
        self.ups = nn.ModuleList()
        for i in range(n):
            prev_nf = nf
            nf = rev_nfs[i]
            ni = rev_nfs[min(i + 1, len(nfs) - 1)]
            self.ups.append(
                UpBlock(
                    n_emb,
                    ni,
                    prev_nf,
                    nf,
                    add_up=i != n - 1,
                    num_layers=num_layers + 1,
                    attn_chans=0 if i >= n - attn_start else attn_chans,
                )
            )
        self.conv_out = pre_conv(
            nfs[0], out_channels, act=nn.SiLU, norm=nn.BatchNorm2d, bias=False
        )

    def forward(self, inp):
        """
        Forward pass for the U-Net model.

        Args:
            inp (tuple): Input tensor and timestep tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x, t = inp
        temb = timestep_embedding(t, self.n_temb)
        emb = self.emb_mlp(temb)
        x = self.conv_in(x)
        saved = [x]
        for block in self.downs:
            x = block(x, emb)
        saved += [p for o in self.downs for p in o.saved]
        x = self.mid_block(x, emb)
        for block in self.ups:
            x = block(x, emb, saved)
        return self.conv_out(x)


def ddim_step(x_t, noise, abar_t, abar_t1, bbar_t, bbar_t1, eta, sig, clamp=True):
    """
    Perform a single DDIM step.

    Args:
        x_t (torch.Tensor): Current tensor.
        noise (torch.Tensor): Noise tensor.
        abar_t (torch.Tensor): Schedule value at t.
        abar_t1 (torch.Tensor): Schedule value at t-1.
        bbar_t (torch.Tensor): Beta bar value at t.
        bbar_t1 (torch.Tensor): Beta bar value at t-1.
        eta (float): Eta value.
        sig (float): Sigma value.
        clamp (bool): Whether to clamp the output. Default is True.

    Returns:
        tuple: Estimated x_0 and updated x_t.
    """
    sig = ((bbar_t1 / bbar_t).sqrt() * (1 - abar_t / abar_t1).sqrt()) * eta
    x_0_hat = (x_t - (1 - abar_t).sqrt() * noise) / abar_t.sqrt()
    if clamp:
        x_0_hat = x_0_hat.clamp(-1, 1)
    if bbar_t1 <= sig**2 + 0.01:
        sig = 0.0  # set to zero if very small or NaN
    x_t = abar_t1.sqrt() * x_0_hat + (bbar_t1 - sig**2).sqrt() * noise
    x_t += sig * torch.randn(x_t.shape).to(x_t)
    return x_0_hat, x_t


@torch.no_grad()
def sample(f, model, sz, steps, eta=1.0, clamp=True):
    """
    Generate samples using the model.

    Args:
        f (function): DDIM step function.
        model (nn.Module): U-Net model.
        sz (tuple): Size of the output tensor.
        steps (int): Number of sampling steps.
        eta (float): Eta value. Default is 1.0.
        clamp (bool): Whether to clamp the output. Default is True.

    Returns:
        list: List of generated samples.
    """
    model.eval()
    ts = torch.linspace(1 - 1 / steps, 0, steps)
    x_t = torch.randn(sz).cuda()
    preds = []
    for i, t in enumerate(progress_bar(ts)):
        t = t[None].cuda()
        abar_t = abar(t)
        noise = model((x_t, t))
        abar_t1 = abar(t - 1 / steps) if t >= 1 / steps else torch.tensor(1)
        x_0_hat, x_t = f(
            x_t,
            noise,
            abar_t,
            abar_t1,
            1 - abar_t,
            1 - abar_t1,
            eta,
            1 - ((i + 1) / 100),
            clamp=clamp,
        )
        preds.append(x_0_hat.float().cpu())
    return preds


@torch.no_grad()
def cond_sample(c, f, model, sz, steps, eta=1.0):
    """
    Generate conditional samples using the model.

    Args:
        c (int): Condition value.
        f (function): DDIM step function.
        model (nn.Module): U-Net model.
        sz (tuple): Size of the output tensor.
        steps (int): Number of sampling steps.
        eta (float): Eta value. Default is 1.0.

    Returns:
        list: List of generated samples.
    """
    ts = torch.linspace(1 - 1 / steps, 0, steps)
    x_t = torch.randn(sz).cuda()
    c = x_t.new_full((sz[0],), c, dtype=torch.int32)
    preds = []
    for i, t in enumerate(progress_bar(ts)):
        t = t[None].cuda()
        abar_t = abar(t)
        noise = model((x_t, t, c))
        abar_t1 = abar(t - 1 / steps) if t >= 1 / steps else torch.tensor(1)
        x_0_hat, x_t = f(
            x_t,
            noise,
            abar_t,
            abar_t1,
            1 - abar_t,
            1 - abar_t1,
            eta,
            1 - ((i + 1) / 100),
        )
        preds.append(x_0_hat.float().cpu())
    return preds


"""
### Flow of Execution ###

1. Import Required Libraries: Necessary libraries and functions are imported.

2. Define Utility Functions:
    - `abar(t)`: Computes the alpha bar schedule.
    - `inv_abar(x)`: Computes the inverse of the alpha bar schedule.
    - `noisify(x0)`: Adds noise to the input tensor `x0`.

3. Define Embedding and Convolution Functions:
    - `timestep_embedding(tsteps, emb_dim, max_period)`: Computes timestep embeddings.
    - `pre_conv(ni, nf, ks, stride, act, norm, bias)`: Creates a convolutional layer.
    - `upsample(nf)`: Creates an upsampling layer.
    - `lin(ni, nf, act, norm, bias)`: Creates a linear layer.

4. Define Self-Attention Classes:
    - `SelfAttention(nn.Module)`: Implements self-attention mechanism.
    - `SelfAttention2D(SelfAttention)`: Extends self-attention to 2D inputs.

5. Define Residual Block:
    - `EmbResBlock(nn.Module)`: Residual block with optional self-attention.

6. Define U-Net Blocks:
    - `DownBlock(nn.Module)`: Downsampling block.
    - `UpBlock(nn.Module)`: Upsampling block.

7. Define U-Net Model:
    - `EmbUNetModel(nn.Module)`: U-Net model with embedding for time-dependent processing.

8. Define DDIM Sampling Functions:
    - `ddim_step(x_t, noise, abar_t, abar_t1, bbar_t, bbar_t1, eta, sig, clamp)`: Performs a single DDIM step.
    - `sample(f, model, sz, steps, eta, clamp)`: Generates samples using the model.
    - `cond_sample(c, f, model, sz, steps, eta)`: Generates conditional samples using the model.

"""
