import torch
import numpy as np
import matplotlib.pyplot as plt


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
    plt.show()
