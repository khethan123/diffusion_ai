import sys, gc, traceback
import fastcore.all as fc
import torch.nn.functional as F
import torch, matplotlib.pyplot as plt

from .learner import *
from copy import copy
from torch.nn import init
from .activations import *
from torch import tensor, nn
from contextlib import contextmanager
from torch.utils.data import default_collate


# code to be used only in jupyter notebooks
# def clean_ipython_hist():
#     # Code in this function mainly copied from IPython source
#     if not "get_ipython" in globals():
#         return
#     ip = get_ipython()
#     user_ns = ip.user_ns
#     ip.displayhook.flush()
#     pc = ip.displayhook.prompt_count + 1
#     for n in range(1, pc):
#         user_ns.pop("_i" + repr(n), None)
#     user_ns.update(dict(_i="", _ii="", _iii=""))
#     hm = ip.history_manager
#     hm.input_hist_parsed[:] = [""] * pc
#     hm.input_hist_raw[:] = [""] * pc
#     hm._i = hm._ii = hm._iii = hm._i00 = ""


def clean_tb():
    """
    Cleans the traceback of the last exception.
    """
    # h/t Piotr Czapla
    if hasattr(sys, "last_traceback"):
        traceback.clear_frames(sys.last_traceback)
        delattr(sys, "last_traceback")
    if hasattr(sys, "last_type"):
        delattr(sys, "last_type")
    if hasattr(sys, "last_value"):
        delattr(sys, "last_value")


def clean_mem():
    """
    Cleans the memory by collecting garbage and emptying the CUDA cache.
    """
    clean_tb()
    # clean_ipython_hist()
    gc.collect()
    torch.cuda.empty_cache()


class BatchTransformCB(Callback):
    """
    A callback class used to apply a transform to the batch before processing.
    """

    def __init__(self, tfm, on_train=True, on_val=True):
        fc.store_attr()

    def before_batch(self, learn):
        if (self.on_train and learn.training) or (self.on_val and not learn.training):
            learn.batch = self.tfm(learn.batch)


class GeneralRelu(nn.Module):
    """
    A class that implements a generalized ReLU activation function with optional
    leakiness, subtraction, and max value.
    """

    def __init__(self, leak=None, sub=None, maxv=None):
        super().__init__()
        self.leak, self.sub, self.maxv = leak, sub, maxv

    def forward(self, x):
        x = F.leaky_relu(x, self.leak) if self.leak is not None else F.relu(x)
        if self.sub is not None:
            x -= self.sub
        if self.maxv is not None:
            x.clamp_max_(self.maxv)
        return x


def plot_func(f, start=-5.0, end=5.0, steps=100):
    """
    Plots a function over a specified range.
    """
    x = torch.linspace(start, end, steps)
    plt.plot(x, f(x))
    plt.grid(True, which="both", ls="--")
    plt.axhline(y=0, color="k", linewidth=0.7)
    plt.axvline(x=0, color="k", linewidth=0.7)


def init_weights(m, leaky=0.0):
    """
    Initializes the weights of a module using Kaiming normal initialization.
    """
    if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        init.kaiming_normal_(m.weight, a=leaky)


def _lsuv_stats(hook, mod, inp, outp):
    acts = to_cpu(outp)
    hook.mean = acts.mean()
    hook.std = acts.std()


def lsuv_init(model, m, m_in, xb):
    """
    Initializes the weights and biases of a module using the
    LSUV (Layer-Sequential Unit-Variance) method.
    """
    h = Hook(m, _lsuv_stats)
    with torch.no_grad():
        while model(xb) is not None and (abs(h.std - 1) > 1e-3 or abs(h.mean) > 1e-3):
            m_in.bias -= h.mean
            m_in.weight.data /= h.std
    h.remove()


def conv(ni, nf, ks=3, stride=2, act=nn.ReLU, norm=None, bias=None):
    """
    Creates a convolutional layer with optional activation and normalization.
    """
    if bias is None:
        bias = not isinstance(norm, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))
    layers = [
        nn.Conv2d(ni, nf, stride=stride, kernel_size=ks, padding=ks // 2, bias=bias)
    ]
    if norm:
        layers.append(norm(nf))
    if act:
        layers.append(act())
    return nn.Sequential(*layers)


def get_model(act=nn.ReLU, nfs=None, norm=None):
    """
    Creates a model with a series of convolutional layers.
    """
    if nfs is None:
        nfs = [1, 8, 16, 32, 64]
    layers = [conv(nfs[i], nfs[i + 1], act=act, norm=norm) for i in range(len(nfs) - 1)]
    return nn.Sequential(
        *layers, conv(nfs[-1], 10, act=None, norm=False, bias=True), nn.Flatten()
    ).to(def_device)
