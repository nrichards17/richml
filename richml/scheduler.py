import math
import numpy as np
from torch.optim.lr_scheduler import _LRScheduler


class CosineRestartsLR(_LRScheduler):
    """
    Implementing SGDR: Stochastic Gradient Descent with Warm Restarts
    by Loshchilov & Hutter
    https://arxiv.org/pdf/1608.03983.pdf

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        max_epochs (int): Maximum number of iterations.
        num_restarts (int): Number of times LR is restarted. Default: 0
        eta_min (float): Minimum learning rate. Default: 0.0
        eta_max (float): Maximum learning rate - overwrites optimizer LR. Default: 0.001
        amplitude_ratio (float): Ratio by which the maximum LR is multiplied every restart. Default: 0.8
        period_multiplier (float): Scale factor of the period of each successive restart phase. Default: 2.0
        last_epoch (int): The index of last epoch. Default: -1
    """

    def __init__(self, optimizer, max_epochs, num_restarts=0, eta_min=0.0, eta_max=0.001,
                 amplitude_ratio=0.8, period_multiplier=2.0, last_epoch=-1):

        self.max_epochs = max_epochs
        self.num_restarts = num_restarts
        self.eta_min = eta_min
        self.eta_max = eta_max
        self.amplitude_ratio = amplitude_ratio
        self.period_multiplier = period_multiplier

        # back-calculate base period from max epochs, num restarts, and period multiplier
        base_period = self.max_epochs / np.sum(self.period_multiplier ** np.arange(self.num_restarts + 1))
        period_list = base_period * (self.period_multiplier ** np.arange(num_restarts + 1))
        period_list = np.insert(period_list, 0, 0)

        # points at which LR is restarted to high value, start of cosine half-wave
        # includes the max epochs at the end
        self.restart_points = np.cumsum(period_list)

        # initialize after setting params, b/c get_lr is called
        super(CosineRestartsLR, self).__init__(optimizer, last_epoch)

    def _calculate_lr(self, epoch):
        for i in range(self.num_restarts + 1):
            # determine which restart/phase from current epoch
            left_point = self.restart_points[i]
            right_point = self.restart_points[i + 1]
            power = i
            period = right_point - left_point
            if (epoch >= left_point) and (epoch < right_point):
                lr = self.eta_min + (self.amplitude_ratio ** power) * (self.eta_max - self.eta_min) * \
                     (1 + math.cos(math.pi * (epoch - left_point) / period)) / 2
                break
        else:
            lr = self.eta_min
        return lr

    def get_lr(self):
        return [self._calculate_lr(self.last_epoch) for _ in self.base_lrs]


class OneCycleLR(_LRScheduler):
    """
    Implementing 1-Cycle policy

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        max_epochs (int): Maximum number of iterations.
        eta_min (float): Minimum learning rate. Default: 1e-5
        eta_max (float): Maximum learning rate - overwrites optimizer LR. Default: 0.001
        epsilon (float): Final LR value in end phase. Default: 0.0
        end_ratio (float): Ratio of
    """

    def __init__(self, optimizer, max_epochs, eta_min=1e-5, eta_max=0.001,
                 epsilon=0.0, end_ratio=0.1, last_epoch=-1):
        self.eta_min = eta_min
        self.eta_max = eta_max
        self.epsilon = epsilon
        if end_ratio < 0 or end_ratio > 1:
            raise ValueError('End ratio must be: 0 < x < 1')
        self.end_ratio = end_ratio

        self.max_epochs = max_epochs
        self.x3 = max_epochs
        self.x2 = (1.0 - self.end_ratio) * self.max_epochs
        self.x1 = self.x2 / 2.0

        # initialize after setting params, b/c get_lr is called
        super(OneCycleLR, self).__init__(optimizer, last_epoch)

    def _calculate_lr(self, epoch):
        """
        Calculates LR with linear segments based on three epoch points: x1, x2, x3
        """
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
        return [self._calculate_lr(self.last_epoch) for _ in self.base_lrs]