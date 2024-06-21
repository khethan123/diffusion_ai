from copy import copy
from torch import optim
from .convolution import *
from functools import partial
from operator import attrgetter
from torcheval.metrics import Mean
from collections.abc import Mapping
from fastprogress import progress_bar, master_bar
from torch.optim.lr_scheduler import ExponentialLR

import fastcore.all as fc
import torch.nn.functional as F
import math, torch, matplotlib.pyplot as plt


class with_cbs:
    """
    A decorator class used to wrap a function with callbacks.
    """

    def __init__(self, nm):
        self.nm = nm

    def __call__(self, f):
        def _f(o, *args, **kwargs):
            try:
                o.callback(f"before_{self.nm}")
                f(o, *args, **kwargs)
                o.callback(f"after_{self.nm}")
            except globals()[f"Cancel{self.nm.title()}Exception"]:
                pass
            finally:
                o.callback(f"cleanup_{self.nm}")

        return _f


class Learner:
    """
    A class used to represent a Learner, which is an object that learns from data.
    """

    def __init__(
        self,
        model,
        dls=(0,),
        loss_func=F.mse_loss,
        lr=0.1,
        cbs=None,
        opt_func=optim.SGD,
    ):
        cbs = fc.L(cbs)
        fc.store_attr()

    @with_cbs("batch")
    def _one_batch(self):
        self.predict()
        self.callback("after_predict")
        self.get_loss()
        self.callback("after_loss")
        if self.training:
            self.backward()
            self.callback("after_backward")
            self.step()
            self.callback("after_step")
            self.zero_grad()

    @with_cbs("epoch")
    def _one_epoch(self):
        for self.iter, self.batch in enumerate(self.dl):
            self._one_batch()

    def one_epoch(self, training):
        self.model.train(training)
        self.dl = self.dls.train if training else self.dls.valid
        self._one_epoch()

    @with_cbs("fit")
    def _fit(self, train, valid):
        for self.epoch in self.epochs:
            if train:
                self.one_epoch(True)
            if valid:
                torch.no_grad()(self.one_epoch)(False)

    def fit(self, n_epochs=1, train=True, valid=True, cbs=None, lr=None):
        cbs = fc.L(cbs)  # convert to fastai's list

        for cb in cbs:
            self.cbs.append(cb)
        try:
            self.n_epochs = n_epochs
            self.epochs = range(n_epochs)
            if lr is None:
                lr = self.lr
            if self.opt_func:
                self.opt = self.opt_func(self.model.parameters(), lr)
            self._fit(train, valid)
        finally:
            for cb in cbs:
                self.cbs.remove(cb)

    def __getattr__(self, name):
        if name in ("predict", "get_loss", "backward", "step", "zero_grad"):
            return partial(self.callback, name)
        raise AttributeError(name)

    def callback(self, method_nm):
        run_cbs(self.cbs, method_nm, self)

    @property
    def training(self):
        return self.model.training


class CancelFitException(Exception):
    """
    An exception that is raised to cancel the fitting process.
    """

    pass


class CancelBatchException(Exception):
    """
    An exception that is raised to cancel the processing of a batch.
    """

    pass


class CancelEpochException(Exception):
    """
    An exception that is raised to cancel the processing of an epoch.
    """

    pass


class Callback:
    """
    A base class for callbacks that provides a default order for sorting.
    """

    order = 0


def run_cbs(cbs, method_nm, learn=None):
    """
    A function to run callbacks in order.
    """
    for cb in sorted(cbs, key=attrgetter("order")):
        method = getattr(cb, method_nm, None)
        if method is not None:
            method(learn)


class SingleBatchCB(Callback):
    """
    A callback that cancels the fitting process after a single batch.
    """

    order = 1

    def after_batch(self, learn):
        raise CancelFitException()


def to_cpu(x):
    """
    A function to move data to the CPU and convert it to the appropriate type.
    """
    if isinstance(x, Mapping):
        return {k: to_cpu(v) for k, v in x.items()}
    if isinstance(x, list):
        return [to_cpu(o) for o in x]
    if isinstance(x, tuple):
        return tuple(to_cpu(list(x)))
    res = x.detach().cpu()
    return res.float() if res.dtype == torch.float16 else res


class MetricsCB(Callback):
    """
    A callback class used to compute and log metrics during training.
    """

    def __init__(self, *ms, **metrics):
        for o in ms:
            metrics[type(o).__name__] = o
        self.metrics = metrics
        self.all_metrics = copy(metrics)
        self.all_metrics["loss"] = self.loss = Mean()

    def _log(self, d):
        print(d)

    def before_fit(self, learn):
        learn.metrics = self

    def before_epoch(self, learn):
        [o.reset() for o in self.all_metrics.values()]

    def after_epoch(self, learn):
        log = {k: f"{v.compute():.3f}" for k, v in self.all_metrics.items()}
        log["epoch"] = learn.epoch
        log["train"] = "train" if learn.model.training else "eval"
        self._log(log)

    def after_batch(self, learn):
        x, y, *_ = to_cpu(learn.batch)
        for m in self.metrics.values():
            m.update(to_cpu(learn.preds), y)
        self.loss.update(to_cpu(learn.loss), weight=len(x))


class DeviceCB(Callback):
    """
    A callback class used to move data to a specified device before training.
    """

    def __init__(self, device=def_device):
        fc.store_attr()

    def before_fit(self, learn):
        if hasattr(learn.model, "to"):
            learn.model.to(self.device)

    def before_batch(self, learn):
        learn.batch = to_device(learn.batch, device=self.device)


class TrainCB(Callback):
    """
    A callback class used to perform training operations such as prediction, loss computation, backpropagation, and optimization step.
    """

    def __init__(self, n_inp=1):
        self.n_inp = n_inp

    def predict(self, learn):
        learn.preds = learn.model(*learn.batch[: self.n_inp])

    def get_loss(self, learn):
        learn.loss = learn.loss_func(learn.preds, *learn.batch[self.n_inp :])

    def backward(self, learn):
        learn.loss.backward()

    def step(self, learn):
        learn.opt.step()

    def zero_grad(self, learn):
        learn.opt.zero_grad()


class ProgressCB(Callback):
    """
    A callback class used to display a progress bar and log training progress during training.
    """

    order = MetricsCB.order + 1

    def __init__(self, plot=False):
        self.plot = plot

    def before_fit(self, learn):
        learn.epochs = self.mbar = master_bar(learn.epochs)
        self.first = True
        if hasattr(learn, "metrics"):
            learn.metrics._log = self._log
        self.losses = []
        self.val_losses = []

    def _log(self, d):
        if self.first:
            self.mbar.write(list(d), table=True)
            self.first = False
        self.mbar.write(list(d.values()), table=True)

    def before_epoch(self, learn):
        learn.dl = progress_bar(learn.dl, leave=False, parent=self.mbar)

    def after_batch(self, learn):
        learn.dl.comment = f"{learn.loss:.3f}"
        if self.plot and hasattr(learn, "metrics") and learn.training:
            self.losses.append(learn.loss.item())
            if self.val_losses:
                self.mbar.update_graph(
                    [
                        [fc.L.range(self.losses), self.losses],
                        [
                            fc.L.range(learn.epoch).map(
                                lambda x: (x + 1) * len(learn.dls.train)
                            ),
                            self.val_losses,
                        ],
                    ]
                )

    def after_epoch(self, learn):
        if not learn.training:
            if self.plot and hasattr(learn, "metrics"):
                self.val_losses.append(learn.metrics.all_metrics["loss"].compute())
                self.mbar.update_graph(
                    [
                        [fc.L.range(self.losses), self.losses],
                        [
                            fc.L.range(learn.epoch + 1).map(
                                lambda x: (x + 1) * len(learn.dls.train)
                            ),
                            self.val_losses,
                        ],
                    ]
                )


class TrainLearner(Learner):
    """
    A subclass of Learner that overrides some methods for training.
    """

    def predict(self):
        self.preds = self.model(self.batch[0])

    def get_loss(self):
        self.loss = self.loss_func(self.preds, self.batch[1])

    def backward(self):
        self.loss.backward()

    def step(self):
        self.opt.step()

    def zero_grad(self):
        self.opt.zero_grad()


class MomentumLearner(TrainLearner):
    """
    A subclass of TrainLearner that implements momentum in the zero_grad method.
    """

    def __init__(
        self, model, dls, loss_func, lr=None, cbs=None, opt_func=optim.SGD, mom=0.85
    ):
        self.mom = mom
        super().__init__(model, dls, loss_func, lr, cbs, opt_func)

    def zero_grad(self):
        with torch.no_grad():
            for p in self.model.parameters():
                p.grad *= self.mom


class LRFinderCB(Callback):
    """
    A callback class used to find the optimal learning rate during training.
    """

    def __init__(self, gamma=1.3, max_mult=3):
        fc.store_attr()

    def before_fit(self, learn):
        self.sched = ExponentialLR(learn.opt, self.gamma)
        self.lrs, self.losses = [], []
        self.min = math.inf

    def after_batch(self, learn):
        if not learn.training:
            raise CancelEpochException()
        self.lrs.append(learn.opt.param_groups[0]["lr"])
        loss = to_cpu(learn.loss)
        self.losses.append(loss)
        if loss < self.min:
            self.min = loss
        if math.isnan(loss) or (loss > self.min * self.max_mult):
            raise CancelFitException()
        self.sched.step()

    def cleanup_fit(self, learn):
        plt.plot(self.lrs, self.losses)
        plt.xscale("log")


@fc.patch
def lr_find(self: Learner, gamma=1.3, max_mult=3, start_lr=1e-5, max_epochs=10):
    """
    A method to find the optimal learning rate by fitting the model for a few epochs
    and plotting the loss against the learning rate.
    """
    self.fit(max_epochs, lr=start_lr, cbs=LRFinderCB(gamma=gamma, max_mult=max_mult))


# Note: DEPENDENCY INJECTION
"""
It's a technique whereby one object supplies the dependencies of another object. 
Here, the `Learner` instance (`learn`) is being passed as an argument to the methods 
of the `Callback` classes. This allows the callback methods to have access to the 
state and methods of the `Learner` instance, making it easier to modify its behavior 
during training. This is a common design pattern in object-oriented programming and 
is particularly useful in scenarios like this where you want to allow for flexible 
and easily extendable code.
"""
