import torch
import numpy as np
import matplotlib.pyplot as plt

from richml.scheduler import CosineRestartsLR, OneCycleLR, CycleRestartsLR


class FakeNet(torch.nn.Module):
    """A fake network in order to test schedulers."""

    def __init__(self):
        super().__init__()
        self.hidden = torch.nn.Linear(10, 1)

    def forward(self, x):
        return x


def plot_scheduler(scheduler):
    lr_history = []
    for epoch in range(scheduler.max_epochs):
        scheduler.step(epoch)
        lr = scheduler.get_lr()[0]
        lr_history.append(lr)

    lr_history = np.array(lr_history)
    epochs = np.arange(scheduler.max_epochs)

    plt.plot(epochs, lr_history)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Example: {}'.format(scheduler.__class__.__name__))

    plt.show()


if __name__ == '__main__':
    import argparse

    choices = ['cosine_restarts', 'one_cycle', 'cycle_restarts']
    parser = argparse.ArgumentParser(description='Plot richml scheduler examples.')
    parser.add_argument('--scheduler', metavar='SCHEDULER',
                        choices=choices,
                        help='choose of the schedulers: { %(choices)s }',
                        default='cosine_restarts')
    args = parser.parse_args()

    network = FakeNet()
    optimizer = torch.optim.Adam(network.parameters())

    if args.scheduler == 'cosine_restarts':
        scheduler = CosineRestartsLR(
            optimizer,
            max_epochs=1000,
            num_restarts=10,
            eta_min=0.001,
            eta_max=0.01,
            amplitude_ratio=0.9,
            period_multiplier=1.0,
        )
    elif args.scheduler == 'one_cycle':
        scheduler = OneCycleLR(
            optimizer,
            max_epochs=1000,
            eta_min=0.001,
            eta_max=0.01,
            epsilon=0.0,
            end_fraction=0.2
        )
    elif args.scheduler == 'cycle_restarts':
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
    else:
        pass

    plot_scheduler(scheduler)
