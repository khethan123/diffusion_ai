"""
In Python, `__all__` is a list of strings that defines what names should be imported when `from <module> import *` is used.
It is used to control the importing. It specifies the public objects of that module, as interpreted by `import *`. 
It overrides the default of hiding everything that begins with an underscore.

 `__all__ = ['MixedPrecision', 'AccelerateCB']` means that when this module is imported 
 using `from <module> import *`, only the `MixedPrecision` and `AccelerateCB` classes will be imported. 
 Any other objects in the module that are not included in the `__all__` list will not be imported.

This can be useful for controlling which parts of your code are accessible when the module is imported, 
and can also help to avoid naming conflicts with other modules. It's a way to specify a public interface for a module or package.

"""

import torch

from .sgd import *
from .resnet import *
from .dtaset import *  # from dtaset import * ; # both are same
from .augment import *
from .learner import *
from .convolution import *
from .activations import *
from .initialization import *
from accelerate import Accelerator  # pip install accelerate


# when `from accel import *` is used only the below classes are imported
__all__ = ["MixedPrecision", "AccelerateCB"]


class MixedPrecision(TrainCB):
    """
    A callback for mixed precision training.
    """

    order = DeviceCB.order + 10

    def before_fit(self, learn):
        """
        Initialize the gradient scaler for automatic
        mixed precision training before fitting the model.
        """
        self.scaler = torch.cuda.amp.GradScaler()

    def before_batch(self, learn):
        """
        Enter autocast mode before processing each batch in order to use
        whichever hardware is present, either GPU or CPU.
        """
        self.autocast = torch.autocast("cuda", dtype=torch.float16)
        self.autocast.__enter__()

    def after_loss(self, learn):
        """
        Exit autocast mode after calculating the loss.
        """
        self.autocast.__exit__(None, None, None)

    def backward(self, learn):
        """
        Perform backward pass with scaled gradients.
        """
        self.scaler.scale(learn.loss).backward()

    def step(self, learn):
        """
        Update the parameters and the gradient scaler.
        """
        self.scaler.step(learn.opt)
        self.scaler.update()


class AccelerateCB(TrainCB):
    """
    A callback for accelerated training.
    Accelerate provides an easy API to make your scripts run with mixed precision
    and on any kind of distributed setting (multi-GPUs, TPUs etc.) while still
    letting you write your own training loop.
    """

    order = DeviceCB.order + 10

    def __init__(self, n_inp=1, mixed_precision="fp16"):
        """
        Initialize the accelerator.
        """
        super().__init__(n_inp=n_inp)
        self.acc = Accelerator(mixed_precision=mixed_precision)

    def before_fit(self, learn):
        """
        Prepare the model, optimizer, and data loaders for accelerated training.
        """
        learn.model, learn.opt, learn.dls.train, learn.dls.valid = self.acc.prepare(
            learn.model, learn.opt, learn.dls.train, learn.dls.valid
        )

    def backward(self, learn):
        """
        Perform backward pass with acceleration.
        """
        self.acc.backward(learn.loss)
