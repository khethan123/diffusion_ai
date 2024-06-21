import torch, random
import fastcore.all as fc

from .dtaset import *
from torch import nn
from .learner import *
from .activations import *
from torch.nn import init
from random import random, randint


def _flops(x, h, w):
    """
    Calculates the number of floating point operations (FLOPs) for a tensor.
    """
    if x.dim() < 3:
        return x.numel()
    if x.dim() == 4:
        return x.numel() * h * w


@fc.patch
def summary(self: Learner):
    """
    Prints a summary of the model including the number of parameters and MFLOPS for each module.
    """
    res = "|Module|Input|Output|Num params|MFLOPS|\n|--|--|--|--|--|\n"
    totp, totf = 0, 0

    def _f(hook, mod, inp, outp):
        nonlocal res, totp, totf
        nparms = sum(o.numel() for o in mod.parameters())
        totp += nparms
        *_, h, w = outp.shape
        flops = sum(_flops(o, h, w) for o in mod.parameters()) / 1e6
        totf += flops
        res += f"|{type(mod).__name__}|{tuple(inp[0].shape)}|{tuple(outp.shape)}|{nparms}|{flops:.1f}|\n"

    with Hooks(self.model, _f) as hooks:
        self.fit(1, lr=1, cbs=SingleBatchCB())
    print(f"Tot params: {totp}; MFLOPS: {totf:.1f}")
    if fc.IN_NOTEBOOK:
        from IPython.display import Markdown

        return Markdown(res)
    else:
        print(res)


@fc.patch
@fc.delegates(show_images)
def show_image_batch(self: Learner, max_n=9, cbs=None, **kwargs):
    """
    Displays a batch of images from the learner's data.
    """
    self.fit(1, cbs=[SingleBatchCB()] + fc.L(cbs))
    show_images(self.batch[0][:max_n], **kwargs)


class CapturePreds(Callback):
    """
    A callback class that captures the inputs, predictions, and
    targets during training.
    """

    def before_fit(self, learn):
        self.all_inps, self.all_preds, self.all_targs = [], [], []

    def after_batch(self, learn):
        self.all_inps.append(to_cpu(learn.batch[0]))
        self.all_preds.append(to_cpu(learn.preds))
        self.all_targs.append(to_cpu(learn.batch[1]))

    def after_fit(self, learn):
        self.all_preds, self.all_targs, self.all_inps = map(
            torch.cat, [self.all_preds, self.all_targs, self.all_inps]
        )


@fc.patch
def capture_preds(self: Learner, cbs=None, inps=False):
    """
    Captures the predictions of the learner for one epoch.
    """
    cp = CapturePreds()
    self.fit(1, train=False, cbs=[cp] + fc.L(cbs))
    res = cp.all_preds, cp.all_targs
    if inps:
        res = res + (cp.all_inps,)
    return res


def _rand_erase1(x, pct, xm, xs, mn, mx):
    """
    Erases a random portion of a tensor by replacing it with noise.
    """
    szx = int(pct * x.shape[-2])
    szy = int(pct * x.shape[-1])
    stx = int(random() * (1 - pct) * x.shape[-2])
    sty = int(random() * (1 - pct) * x.shape[-1])
    init.normal_(x[:, :, stx : stx + szx, sty : sty + szy], mean=xm, std=xs)
    x.clamp_(mn, mx)


def rand_erase(x, pct=0.2, max_num=4):
    """
    Erases random portions of a tensor multiple times.
    """
    xm, xs, mn, mx = x.mean(), x.std(), x.min(), x.max()
    num = randint(0, max_num)
    for i in range(num):
        _rand_erase1(x, pct, xm, xs, mn, mx)
    #     print(num)
    return x


class RandErase(nn.Module):
    """
    A module that erases random portions of its input.
    """

    def __init__(self, pct=0.2, max_num=4):
        super().__init__()
        self.pct, self.max_num = pct, max_num

    def forward(self, x):
        return rand_erase(x, self.pct, self.max_num)


def _rand_copy1(x, pct):
    """
    Copies a random portion of a tensor to another random location.
    """
    szx = int(pct * x.shape[-2])
    szy = int(pct * x.shape[-1])
    stx1 = int(random() * (1 - pct) * x.shape[-2])
    sty1 = int(random() * (1 - pct) * x.shape[-1])
    stx2 = int(random() * (1 - pct) * x.shape[-2])
    sty2 = int(random() * (1 - pct) * x.shape[-1])
    x[:, :, stx1 : stx1 + szx, sty1 : sty1 + szy] = x[
        :, :, stx2 : stx2 + szx, sty2 : sty2 + szy
    ]


def rand_copy(x, pct=0.2, max_num=4):
    """
    Copies random portions of a tensor to other locations multiple times.
    """
    num = randint(0, max_num)
    for i in range(num):
        _rand_copy1(x, pct)
    #     print(num)
    return x


class RandCopy(nn.Module):
    """
    A module that copies random portions of its input to other locations.
    """

    def __init__(self, pct=0.2, max_num=4):
        super().__init__()
        self.pct, self.max_num = pct, max_num

    def forward(self, x):
        return rand_copy(x, self.pct, self.max_num)
