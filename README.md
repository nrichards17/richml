richml
======

<div class="row">
<img src="https://github.com/nrichards17/richml/raw/master/examples/fig_cosine_restarts.png" height="200" width="300">
<img src="https://github.com/nrichards17/richml/raw/master/examples/fig_one_cycle.png" height="200" width="300">
<img src="https://github.com/nrichards17/richml/raw/master/examples/fig_cycle_restarts.png" height="200" width="300">
</div>

-------------------

Currently, this package includes:
- custom PyTorch learning rate schedulers
    * Cosine annealing w/ restarts
    * 1-Cycle
    * 1-Cycle w/ restarts

Dependencies
------------

Found in `setup.py`
```
INSTALL_REQUIRES = [
    'numpy>=1.15.0',
    'torch>=0.4.1',
]
```

Some examples also require  `matplotlib`

Installation
------------

    pip install git+https://github.com/nrichards17/richml#egg=richml

Examples
--------

### Visualize PyTorch LR Schedulers

See [examples/example_plot_scheduler.py](https://github.com/nrichards17/richml/blob/master/examples/example_plot_scheduler.py) 

Example
```
python example_plot_scheduler.py --scheduler cosine_restarts
```

```
usage: example_plot_scheduler.py [-h] [--scheduler SCHEDULER]

Plot richml scheduler examples.

optional arguments:
  -h, --help            show this help message and exit
  --scheduler SCHEDULER
                        choose one of the schedulers: { cosine_restarts,
                        one_cycle, cycle_restarts }
```
