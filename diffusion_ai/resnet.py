import fastcore.all as fc

from torch import nn
from .convolution import *
from .activations import *
from .initialization import *
from functools import partial


act_gr = partial(GeneralRelu, leak=0.1, sub=0.4)


def _conv_block(ni, nf, stride, act=act_gr, norm=None, ks=3):
    """
    Creates a block of two convolutional layers with specified parameters.
    """
    return nn.Sequential(
        conv(ni, nf, stride=1, act=act, norm=norm, ks=ks),
        conv(nf, nf, stride=stride, act=None, norm=norm, ks=ks),
    )


class ResBlock(nn.Module):
    """
    A class that implements a residual block with two convolutional layers
    and an optional identity connection.
    """

    def __init__(self, ni, nf, stride=1, ks=3, act=act_gr, norm=None):
        super().__init__()
        self.convs = _conv_block(ni, nf, stride, act=act, ks=ks, norm=norm)
        # below is the residual connection
        self.idconv = fc.noop if ni == nf else conv(ni, nf, ks=1, stride=1, act=None)
        self.pool = fc.noop if stride == 1 else nn.AvgPool2d(2, ceil_mode=True)
        self.act = act()

    def forward(self, x):
        return self.act(self.convs(x) + self.idconv(self.pool(x)))
