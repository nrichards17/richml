import math
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR, _LRScheduler
from torch.nn.functional import _pointwise_loss


class CosineRestartsLR(_LRScheduler):
    def __init__(self, optimizer, max_epochs=200, num_restarts=0, eta_min=None, eta_max=0.01,
                 amplitude_ratio=0.8, period_multiplier=2.0, min_frac=None, last_epoch=-1):

        self.max_epochs = max_epochs
        self.num_restarts = num_restarts

        if (min_frac is not None) and (eta_min is not None):
            raise Exception("Cannot define eta_min and min_frac simultaneously")
        elif min_frac is not None:
            self.eta_min = min_frac * eta_max
        elif eta_min is not None:
            self.eta_min = eta_min
        else:
            self.eta_min = 0.0  # default

        # self.eta_min = eta_min
        self.eta_max = eta_max
        self.amplitude_ratio = amplitude_ratio
        self.period_multiplier = period_multiplier

        self.period = int(self.max_epochs / np.sum(self.period_multiplier ** np.arange(self.num_restarts + 1)))
        self.period_list = self.period * (self.period_multiplier ** np.arange(num_restarts + 1))
        self.period_list = np.insert(self.period_list, 0, 0)
        self.milestone_list = np.cumsum(self.period_list)

        # self.NUM_EPOCHS = max_epochs
        super(CosineRestartsLR, self).__init__(optimizer, last_epoch)

    def calculate_lr(self, epoch):
        lr = self.eta_min
        for i in range(1, len(self.milestone_list)):
            left = self.milestone_list[i - 1]
            right = self.milestone_list[i]
            power = i - 1
            ex_period = self.period_list[i]
            if (epoch >= left) and (epoch < right):
                lr = self.eta_min + (self.amplitude_ratio ** power) * (self.eta_max - self.eta_min) * \
                     (1 + math.cos(math.pi * (epoch - left) / ex_period)) / 2
        return lr

    def get_lr(self):
        return [self.calculate_lr(self.last_epoch) for _ in self.base_lrs]


class OneCycleLR(_LRScheduler):
    def __init__(self, optimizer, max_epochs=200, eta_min=None, eta_max=0.01, end_ratio=0.2,
                 min_frac=None, last_epoch=-1):

        if (min_frac is not None) and (eta_min is not None):
            raise Exception("Cannot define eta_min and min_frac simultaneously")
        elif min_frac is not None:
            self.eta_min = min_frac * eta_max
        elif eta_min is not None:
            self.eta_min = eta_min
        else:
            self.eta_min = 0.0  # default

        self.eta_max = eta_max
        self.epsilon = 0.0

        self.max_epochs = max_epochs
        self.x3 = max_epochs
        self.x2 = int((1 - end_ratio) * max_epochs)
        self.x1 = int(self.x2 / 2)

        # initialize after setting params, b/c get_lr is called
        super(OneCycleLR, self).__init__(optimizer, last_epoch)

    def calculate_lr(self, epoch):
        if epoch < self.x1:
            lr = ((self.eta_max - self.eta_min) / self.x1) * epoch + self.eta_min
        elif epoch < self.x2:
            lr = ((self.eta_min - self.eta_max) / (self.x2 - self.x1)) * (epoch - self.x1) + self.eta_max
        elif epoch < self.x3:
            lr = ((self.epsilon - self.eta_min) / (self.x3 - self.x2)) * (epoch - self.x2) + self.eta_min
        else:
            lr = self.epsilon
        return lr

    def get_lr(self):
        """
        also accepts epoch input
        """
        return [self.calculate_lr(self.last_epoch) for _ in self.base_lrs]