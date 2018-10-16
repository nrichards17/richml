"""
Plot learning rate over training epochs
Scheduler: CycleRestartsLR
"""

from richml.scheduler import CycleRestartsLR
from example_utils import FakeNet, plot_scheduler

import torch

if __name__ == '__main__':
    network = FakeNet()
    optimizer = torch.optim.Adam(network.parameters())

    scheduler = CycleRestartsLR(
        optimizer,
        max_epochs=1000,
        num_restarts=10,
        eta_min=0.001,
        eta_max=0.01,
        amplitude_ratio=0.9,
        period_multiplier=1.0,
        center_shift=0.8,
    )

    plot_scheduler(scheduler)
