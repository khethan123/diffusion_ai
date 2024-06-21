from __future__ import annotations

from operator import itemgetter
from itertools import zip_longest
from torch.utils.data import default_collate, DataLoader

import math
import numpy as np
import fastcore.all as fc
import matplotlib.pyplot as plt


# def inplace(f):
#     """
#     A decorator to modify a function to return its first argument after calling it.

#     Params:
#         f (function): The function to be modified.

#     Returns:
#         function: The modified function.
#     """

#     def _f(b):
#         f(b)
#         return b

#     return _f


class inplace:
    def __init__(self, f):
        self.f = f

    def __call__(self, b):
        self.f(b)
        return b


def collate_dict(ds):
    """
    Returns a function that collates a batch of data by selecting specified features.

    Params:
        ds (Dataset): The dataset to be collated.

    Returns:
        function: The collation function.
    """
    get = itemgetter(*ds.features)

    def _f(b):
        return get(default_collate(b))

    return _f


@fc.delegates(plt.Axes.imshow)
def show_image(im, ax=None, figsize=None, title=None, noframe=True, **kwargs):
    """
    Show a PIL or PyTorch image on `ax` (a specified axes).

    Params:
        im (torch.Tensor or np.ndarray): The image to be shown.
        ax (matplotlib.axes.Axes, optional): The axes on which to display the image.
        figsize (tuple, optional): The size of the figure to be created if no axes is provided.
        title (str, optional): The title of the image.
        noframe (bool, optional): Whether to remove the frame around the image.
        **kwargs: Additional keyword arguments passed to `plt.Axes.imshow`.

    Returns:
        matplotlib.axes.Axes: The axes with the displayed image.
    """
    if fc.hasattrs(im, ("cpu", "permute", "detach")):
        im = im.detach().cpu()
        if len(im.shape) == 3 and im.shape[0] < 5:
            im = im.permute(1, 2, 0)
    elif not isinstance(im, np.ndarray):
        im = np.array(im)
    if im.shape[-1] == 1:
        im = im[..., 0]
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    ax.imshow(im, **kwargs)
    if title is not None:
        ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    if noframe:
        ax.axis("off")
    return ax


@fc.delegates(plt.subplots, keep=True)
def subplots(
    nrows: int = 1,
    ncols: int = 1,
    figsize: tuple = None,
    imsize: int = 3,
    suptitle: str = None,
    **kwargs,
):  # fig and axs
    """
    Creates a figure and a set of subplots.

    Params:
        nrows (int, optional): The number of rows in the grid of subplots.
        ncols (int, optional): The number of columns in the grid of subplots.
        figsize (tuple, optional): The size of the figure to be created.
        imsize (int, optional): The size of the images to be displayed in the figure.
        suptitle (str, optional): The title of the figure.
        **kwargs: Additional keyword arguments passed to `plt.subplots`.

    Returns:
        tuple: The created figure and array of axes.
    """
    if figsize is None:
        figsize = (ncols * imsize, nrows * imsize)
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
    if suptitle is not None:
        fig.suptitle(suptitle)
    if nrows * ncols == 1:
        ax = np.array([ax])
    return fig, ax


@fc.delegates(subplots)
def get_grid(
    n: int,
    nrows: int = None,
    ncols: int = None,
    title: str = None,
    weight: str = "bold",
    size: int = 14,
    **kwargs,
):  # fig and axs
    """
    Returns a grid of axes.

    Params:
        n (int): The total number of axes in the grid.
        nrows (int, optional): The number of rows in the grid.
        ncols (int, optional): The number of columns in the grid.
        title (str, optional): The title of the figure.
        weight (str, optional): The weight of the title font.
        size (int, optional): The size of the title font.
        **kwargs: Additional keyword arguments passed to `subplots`.

    Returns:
        tuple: The created figure and array of axes.
    """
    if nrows:
        ncols = ncols or int(np.floor(n / nrows))
    elif ncols:
        nrows = nrows or int(np.ceil(n / ncols))
    else:
        nrows = int(math.sqrt(n))
        ncols = int(np.floor(n / nrows))
    fig, axs = subplots(nrows, ncols, **kwargs)
    for i in range(n, nrows * ncols):
        axs.flat[i].set_axis_off()
    if title is not None:
        fig.suptitle(title, weight=weight, size=size)
    return fig, axs


@fc.delegates(subplots)
def show_images(
    ims: list,
    nrows: int | None = None,
    ncols: int | None = None,
    titles: list | None = None,
    **kwargs,
):
    """
    Shows a list of images in a grid of subplots.

    Params:
        ims (list): The list of images to be shown.
        nrows (int, optional): The number of rows in the grid.
        ncols (int, optional): The number of columns in the grid.
        titles (list, optional): The list of titles for the images.
        **kwargs: Additional keyword arguments passed to `get_grid`.
    """
    axs = get_grid(len(ims), nrows, ncols, **kwargs)[1].flat
    for im, t, ax in zip_longest(ims, titles or [], axs):
        show_image(im, ax=ax, title=t)


def get_dls(train_ds, valid_ds, bs, **kwargs):
    """
    Returns a pair of data loaders for the training and validation datasets.

    Params:
        train_ds (Dataset): The training dataset.
        valid_ds (Dataset): The validation dataset.
        bs (int): The batch size.
        **kwargs: Additional keyword arguments passed to `DataLoader`.

    Returns:
        tuple: The training and validation data loaders.
    """
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True, **kwargs),
        DataLoader(valid_ds, batch_size=bs * 2, **kwargs),
    )


class DataLoaders:
    """
    A class for handling multiple data loaders.

    Attributes:
        train (DataLoader): The training data loader.
        valid (DataLoader): The validation data loader.
    """

    def __init__(self, *dls):
        """
        Initializes the DataLoaders with the provided data loaders.

        Params:
            *dls (DataLoader): The data loaders to be handled.
        """
        self.train, self.valid = dls[:2]

    @classmethod
    def from_dd(cls, dd, batch_size, as_tuple=True, **kwargs):
        """
        Creates a DataLoaders instance from a dictionary of datasets.

        Params:
            dd (dict): The dictionary of datasets.
            batch_size (int): The batch size for the data loaders.
            as_tuple (bool, optional): Whether to return the data loaders as a tuple.
            **kwargs: Additional keyword arguments passed to `get_dls`.

        Returns:
            DataLoaders: The created DataLoaders instance.
        """
        f = collate_dict(dd["train"])
        return cls(*get_dls(*dd.values(), bs=batch_size, collate_fn=f, **kwargs))
