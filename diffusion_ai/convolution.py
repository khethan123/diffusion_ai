from torch import nn
from typing import Mapping
from torch.utils.data import default_collate

import torch


def conv(ni, nf, ks=3, stride=2, act=True):
    """
    Creates a convolutional layer.

    Params:
        ni (int): Number of input channels.
        nf (int): Number of output channels.
        ks (int, optional): Size of the convolving kernel. Defaults to 3.
        stride (int, optional): Stride of the convolution. Defaults to 2.
        act (bool, optional): If True, applies a ReLU activation function after the convolution. Defaults to True.

    Returns:
        nn.Sequential: The created convolutional layer.
    """
    res = nn.Conv2d(ni, nf, stride=stride, kernel_size=ks, padding=ks // 2)
    if act:
        res = nn.Sequential(res, nn.ReLU())
    return res


def deconv(ni, nf, ks=3, act=True):
    """
    Creates a deconvolutional layer.

    Params:
        ni (int): Number of input channels.
        nf (int): Number of output channels.
        ks (int, optional): Size of the convolving kernel. Defaults to 3.
        act (bool, optional): If True, applies a ReLU activation function after the deconvolution. Defaults to True.

    Returns:
        nn.Sequential: The created deconvolutional layer.
    """
    layers = [
        nn.UpsamplingNearest2d(scale_factor=2),
        nn.Conv2d(ni, nf, stride=1, kernel_size=ks, padding=ks // 2),
    ]
    if act:
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def_device = "cuda" if torch.cuda.is_available() else "cpu"


def to_device(x, device=def_device):
    """
    Moves data to a device.

    Params:
        x (torch.Tensor or Mapping or iterable): The data to be moved.
        device (str, optional): The device to move the data to. Defaults to the default device.

    Returns:
        The data moved to the device.
    """
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, Mapping):  # if it's a dict
        return {k: v.to(device) for k, v in x.items()}
    return type(x)(to_device(o, device) for o in x)


def collate_device(b):
    """
    Collates a batch of data and moves it to a device.

    Params:
        b (iterable): The batch of data to be collated.

    Returns:
        The collated data moved to the device.
    """
    return to_device(default_collate(b))


# for an Auto-Encoder


def eval_a(model, loss_func, valid_dl, epoch=0):
    """
    Evaluates a model on a validation dataset.

    Params:
        model (nn.Module): The model to be evaluated.
        loss_func (function): The loss function.
        valid_dl (DataLoader): The validation data loader.
        epoch (int, optional): The current epoch. Defaults to 0.
    """
    model.eval()
    with torch.no_grad():
        tot_loss, count = 0.0, 0
        for xb, _ in valid_dl:
            pred = model(xb)
            n = len(xb)
            count += n
            tot_loss += loss_func(pred, xb).item() * n
    print(epoch, f"{tot_loss/count:.3f}")


def fit_a(epochs, model, loss_func, opt, train_dl, valid_dl):
    """
    Trains a model for a specified number of epochs.

    Params:
        epochs (int): The number of epochs to train for.
        model (nn.Module): The model to be trained.
        loss_func (function): The loss function.
        opt (torch.optim.Optimizer): The optimizer.
        train_dl (DataLoader): The training data loader.
        valid_dl (DataLoader): The validation data loader.
    """
    for epoch in range(epochs):
        model.train()
        for xb, _ in train_dl:
            loss = loss_func(model(xb), xb)
            loss.backward()
            opt.step()
            opt.zero_grad()
        eval_a(model, loss_func, valid_dl, epoch)
