import torch

from concern.config import Configurable, State
from mmcv.runner import build_optimizer


class OptimizerScheduler(Configurable):
    optimizer = State()
    optimizer_args = State(default={})
    learning_rate = State(autoload=False)

    def __init__(self, cmd={}, **kwargs):
        self.load_all(**kwargs)
        self.load('learning_rate', cmd=cmd, **kwargs)
        if 'lr' in cmd:
            self.optimizer_args['lr'] = cmd['lr']

    def create_optimizer(self, parameters):
        optimizer = getattr(torch.optim, self.optimizer)(
                parameters, **self.optimizer_args)
        if hasattr(self.learning_rate, 'prepare'):
            self.learning_rate.prepare(optimizer)
        return optimizer


class CustomOptimizerScheduler(Configurable):
    # NOTE: for this class, optimizer arg is ignored.
    optimizer = State()
    optimizer_args = State(default={})
    learning_rate = State(autoload=False)

    def __init__(self, cmd={}, **kwargs):
        self.load_all(**kwargs)
        self.load('learning_rate', cmd=cmd, **kwargs)
        if 'lr' in cmd:
            self.optimizer_args['lr'] = cmd['lr']

    def create_optimizer(self, model):
        optimizer = build_optimizer(model, self.optimizer_args)
        if hasattr(self.learning_rate, 'prepare'):
            self.learning_rate.prepare(optimizer)
        return optimizer