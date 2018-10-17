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


def plot_scheduler(scheduler, show=True):
    lr_history = []
    for epoch in range(scheduler.max_epochs):
        scheduler.step(epoch)
        lr = scheduler.get_lr()[0]
        lr_history.append(lr)

    fig, ax = plt.subplots(1, 1)
    lr_history = np.array(lr_history)
    epochs = np.arange(scheduler.max_epochs)
    ax.plot(epochs, lr_history)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Example: {}'.format(scheduler.__class__.__name__))

    # plot scheduler arguments/params on figure if available
    if hasattr(scheduler, 'params'):
        params = scheduler.params
        y_loc, x_loc, y_step = 0.95, 0.6, 0.06
        for name, value in params.items():
            plt.text(x_loc, y_loc, f'{name}: {value}', transform=ax.transAxes, fontsize=12)
            y_loc -= y_step

    if show:
        plt.show()

    return fig, ax


if __name__ == '__main__':
    import argparse

    choices = ['cosine_restarts', 'one_cycle', 'cycle_restarts']
    parser = argparse.ArgumentParser(description='Plot richml scheduler examples.')
    parser.add_argument('--scheduler', metavar='SCHEDULER',
                        choices=choices,
                        help='choose one of the schedulers: { %(choices)s }',
                        default='cosine_restarts')
    args = parser.parse_args()

    # create network and optimizer required by PyTorch schedulers
    network = FakeNet()
    optimizer = torch.optim.Adam(network.parameters())

    if args.scheduler == choices[0]:
        scheduler = CosineRestartsLR(
            optimizer,
            max_epochs=1000,
            num_restarts=2,
            eta_min=0.001,
            eta_max=0.01,
            amplitude_ratio=0.8,
            period_multiplier=1.5,
        )
    elif args.scheduler == choices[1]:
        scheduler = OneCycleLR(
            optimizer,
            max_epochs=1000,
            eta_min=0.001,
            eta_max=0.01,
            epsilon=0.0,
            end_fraction=0.2
        )
    elif args.scheduler == choices[2]:
        scheduler = CycleRestartsLR(
            optimizer,
            max_epochs=1000,
            num_restarts=5,
            eta_min=0.001,
            eta_max=0.01,
            amplitude_ratio=0.8,
            period_multiplier=1.0,
            center_shift=0.5,
        )
    else:
        pass

    _, _ = plot_scheduler(scheduler)
