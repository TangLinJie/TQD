from concern.config import Configurable, State
import os
import torch


class Checkpoint(Configurable):
    start_epoch = State(default=0)
    start_iter = State(default=0)
    resume = State()
    resume_optimizer = State(default=True)
    resume_base_lr = State(default=True)

    def __init__(self, **kwargs):
        self.load_all(**kwargs)

        cmd = kwargs['cmd']
        if 'start_epoch' in cmd:
            self.start_epoch = cmd['start_epoch']
        if 'start_iter' in cmd:
            self.start_iter = cmd['start_iter']
        if 'resume' in cmd:
            self.resume = cmd['resume']

    def restore_model(self, model, device, logger):
        if self.resume is None:
            return

        if not os.path.exists(self.resume):
            self.logger.warning("Checkpoint not found: " +
                                self.resume)
            return

        logger.info("Resuming from " + self.resume)
        state_dict = torch.load(self.resume, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        logger.info("Resumed from " + self.resume)

    def restore_model_v2(self, model, device, logger):
        if self.resume is None:
            return

        if not os.path.exists(self.resume):
            self.logger.warning("Checkpoint not found: " +
                                self.resume)
            return

        logger.info("Resuming model from " + self.resume)
        state_dict = torch.load(self.resume, map_location=device)['model']
        model.load_state_dict(state_dict, strict=True)
        #model.load_state_dict(state_dict, strict=False)
        logger.info("Resumed model from " + self.resume)

    def restore_optimizer(self, optimizer, device, logger):
        if self.resume is None:
            return

        if not self.resume_optimizer:
            return

        if not os.path.exists(self.resume):
            self.logger.warning("Checkpoint not found: " +
                                self.resume)
            return

        state_dict = torch.load(self.resume, map_location=device)
        if 'optimizer' not in state_dict:
            logger.warning("optimizer not in state dict, it is not be loaded.")
        else:
            logger.info("Resuming optimizer from " + self.resume)
            state_dict = state_dict['optimizer']
            optimizer.load_state_dict(state_dict, strict=True)
            logger.info("Resumed optimizer from " + self.resume)
    
    def restore_learning_rate(self, learning_rate, device, logger):
        if self.resume is None:
            return

        if not self.resume_base_lr:
            return

        if not os.path.exists(self.resume):
            self.logger.warning("Checkpoint not found: " +
                                self.resume)
            return

        state_dict = torch.load(self.resume, map_location=device)
        if 'base_lr' not in state_dict:
            logger.warning('base lr not in state dict, it is not be loaded.')
        else:
            logger.info("Resuming base lr from " + self.resume)
            state_dict = state_dict['base_lr']
            learning_rate.base_lr = state_dict
            logger.info("Resumed base lr from " + self.resume)

    def restore_counter(self):
        return self.start_epoch, self.start_iter
