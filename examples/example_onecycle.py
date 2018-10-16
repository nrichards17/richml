"""
Plot learning rate over training epochs
Scheduler: OneCycleLR
"""

from richml.scheduler import OneCycleLR
from example_utils import FakeNet, plot_scheduler

import torch

if __name__ == '__main__':
    network = FakeNet()
    optimizer = torch.optim.Adam(network.parameters())

    scheduler = OneCycleLR(
        optimizer,
        max_epochs=1000,
        eta_min=0.001,
        eta_max=0.01,
        epsilon=0.0,
        end_ratio=0.2
    )

    plot_scheduler(scheduler)
