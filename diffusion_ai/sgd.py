from .learner import *
from matplotlib import pyplot as plt


class BaseSchedCB(Callback):
    """
    A base callback class that schedules the learning rate of the optimizer.
    """

    def __init__(self, sched):
        self.sched = sched

    def before_fit(self, learn):
        self.schedo = self.sched(learn.opt)

    def _step(self, learn):
        if learn.training:
            self.schedo.step()


class BatchSchedCB(BaseSchedCB):
    """
    A callback class that schedules the learning rate after each batch.
    """

    def after_batch(self, learn):
        self._step(learn)


class HasLearnCB(Callback):
    """
    A callback class that keeps a reference to the learner during training.
    """

    def before_fit(self, learn):
        self.learn = learn

    def after_fit(self, learn):
        self.learn = None


class RecorderCB(Callback):
    """
    A callback class that records specified metrics during training and provides a method for plotting them.
    """

    def __init__(self, **d):
        self.d = d

    def before_fit(self, learn):
        self.recs = {k: [] for k in self.d}
        self.pg = learn.opt.param_groups[0]

    def after_batch(self, learn):
        if not learn.training:
            return
        for k, v in self.d.items():
            self.recs[k].append(v(self))

    def plot(self):
        for k, v in self.recs.items():
            plt.plot(v, label=k)
            plt.legend()
            plt.show()


class EpochSchedCB(BaseSchedCB):
    """
    A callback class that schedules the learning rate after each epoch.
    """

    def after_epoch(self, learn):
        self._step(learn)
