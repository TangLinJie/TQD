# Copyright (c) Open-MMLab. All rights reserved.
import numbers
from math import cos, pi
from concern.config import Configurable, State

#from mmcv.runner.hooks.hook import HOOKS, Hook


class LrUpdaterHook(Configurable):
    """LR Scheduler in MMCV.
    Args:
        by_epoch (bool): LR changes epoch by epoch
        warmup (string): Type of warmup used. It can be None(use no warmup),
            'constant', 'linear' or 'exp'
        warmup_iters (int): The number of iterations or epochs that warmup
            lasts
        warmup_ratio (float): LR used at the beginning of warmup equals to
            warmup_ratio * initial_lr
        warmup_by_epoch (bool): When warmup_by_epoch == True, warmup_iters
            means the number of epochs that warmup lasts, otherwise means the
            number of iteration that warmup lasts
    """
    by_epoch = State(default=True)
    warmup = State(default=None)
    warmup_iters = State(default=0)
    warmup_ratio = State(default=0.1)
    warmup_by_epoch = State(default=False)

    def __init__(self):
        # validate the "warmup" argument
        if self.warmup is not None:
            if self.warmup not in ['constant', 'linear', 'exp']:
                raise ValueError(
                    f'"{self.warmup}" is not a supported type for warming up, valid'
                    ' types are "constant" and "linear"')
        if self.warmup is not None:
            assert self.warmup_iters > 0, \
                '"warmup_iters" must be a positive integer'
            assert 0 < self.warmup_ratio <= 1.0, \
                '"warmup_ratio" must be in range (0,1]'

        # self.by_epoch = by_epoch
        # self.warmup = warmup
        # self.warmup_iters = warmup_iters
        # self.warmup_ratio = warmup_ratio
        # self.warmup_by_epoch = warmup_by_epoch

        if self.warmup_by_epoch:
            self.warmup_epochs = self.warmup_iters
            self.warmup_iters = None
        else:
            self.warmup_epochs = None

        self.base_lr = []  # initial lr for all param groups
        self.regular_lr = []  # expected lr if no warming up is performed

    def _set_lr(self, optimizer, lr_groups):
        for param_group, lr in zip(optimizer.param_groups,
                                    lr_groups):
            param_group['lr'] = lr

    def get_lr(self, base_lr, cur_iter, max_iters):
        raise NotImplementedError

    def get_regular_lr(self, cur_iter, max_iters):
        return [self.get_lr(_base_lr, cur_iter, max_iters) for _base_lr in self.base_lr]

    def get_warmup_lr(self, cur_iters):

        def _get_warmup_lr(cur_iters, regular_lr):
            if self.warmup == 'constant':
                warmup_lr = [_lr * self.warmup_ratio for _lr in regular_lr]
            elif self.warmup == 'linear':
                k = (1 - cur_iters / self.warmup_iters) * (1 -
                                                           self.warmup_ratio)
                warmup_lr = [_lr * (1 - k) for _lr in regular_lr]
            elif self.warmup == 'exp':
                k = self.warmup_ratio**(1 - cur_iters / self.warmup_iters)
                warmup_lr = [_lr * k for _lr in regular_lr]
            return warmup_lr

        if isinstance(self.regular_lr, dict):
            lr_groups = {}
            for key, regular_lr in self.regular_lr.items():
                lr_groups[key] = _get_warmup_lr(cur_iters, regular_lr)
            return lr_groups
        else:
            return _get_warmup_lr(cur_iters, self.regular_lr)

    def before_run(self, optimizer):
        # NOTE: when resuming from a checkpoint, if 'initial_lr' is not saved,
        # it will be set according to the optimizer params
        for group in optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])
        self.base_lr = [
            group['initial_lr'] for group in optimizer.param_groups
        ]

    # def before_train_epoch(self, runner):
    #     if self.warmup_iters is None:
    #         epoch_len = len(runner.data_loader)
    #         self.warmup_iters = self.warmup_epochs * epoch_len

    #     if not self.by_epoch:
    #         return

    #     self.regular_lr = self.get_regular_lr(runner)
    #     self._set_lr(runner, self.regular_lr)

    def before_train_iter(self, optimizer, cur_iter, max_iters):
        if not self.by_epoch:
            self.regular_lr = self.get_regular_lr(cur_iter, max_iters)
            if self.warmup is None or cur_iter >= self.warmup_iters:
                self._set_lr(optimizer, self.regular_lr)
                return self.regular_lr
            else:
                warmup_lr = self.get_warmup_lr(cur_iter)
                self._set_lr(optimizer, warmup_lr)
                return warmup_lr
        elif self.by_epoch:
            if self.warmup is None or cur_iter > self.warmup_iters:
                return
            elif cur_iter == self.warmup_iters:
                self._set_lr(optimizer, self.regular_lr)
                return self.regular_lr
            else:
                warmup_lr = self.get_warmup_lr(cur_iter)
                self._set_lr(optimizer, warmup_lr)
                return warmup_lr


#@HOOKS.register_module()
class PolyLrUpdaterHook(LrUpdaterHook):

    power = State(default=1.)
    min_lr = State(default=0.)
    def __init__(self, **kwargs):
        self.load_all(**kwargs)
        super(PolyLrUpdaterHook, self).__init__()

    def get_lr(self, base_lr, cur_iter, max_iters):
        progress = cur_iter
        max_progress = max_iters
        coeff = (1 - progress / max_progress)**self.power
        # fix lr < 0 case
        cur_lr = (base_lr - self.min_lr) * coeff + self.min_lr
        if cur_lr < self.min_lr:
            cur_lr = self.min_lr
        return cur_lr

    def prepare(self, optimizer):
        self.before_run(optimizer)

    def set_lr(self, optimizer, cur_iter, max_iters):
        return self.before_train_iter(optimizer, cur_iter, max_iters)
        

class CosineAnnealingLrUpdaterHook(LrUpdaterHook):

    min_lr = State(default=0.)
    def __init__(self, **kwargs):
        self.load_all(**kwargs)
        super(CosineAnnealingLrUpdaterHook, self).__init__()

    def get_lr(self, base_lr, cur_iter, max_iters):
        return self.annealing_cos(base_lr, self.min_lr, cur_iter / max_iters)

    def prepare(self, optimizer):
        self.before_run(optimizer)

    def annealing_cos(self, start, end, factor, weight=1):
        """Calculate annealing cos learning rate.
        Cosine anneal from `weight * start + (1 - weight) * end` to `end` as
        percentage goes from 0.0 to 1.0.
        Args:
            start (float): The starting learning rate of the cosine annealing.
            end (float): The ending learing rate of the cosine annealing.
            factor (float): The coefficient of `pi` when calculating the current
                percentage. Range from 0.0 to 1.0.
            weight (float, optional): The combination factor of `start` and `end`
                when calculating the actual starting learning rate. Default to 1.
        """
        cos_out = cos(pi * factor) + 1
        return end + 0.5 * weight * (start - end) * cos_out

    def set_lr(self, optimizer, cur_iter, max_iters):
        return self.before_train_iter(optimizer, cur_iter, max_iters)