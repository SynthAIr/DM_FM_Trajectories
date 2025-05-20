import torch.nn as nn

"""
Code is adapted from https://github.com/Yasoz/DiffTraj
"""


class EMAHelper(object):
    """
    Exponential Moving Average helper class for the UNET models.
    """
    def __init__(self, mu=0.999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        """
        Register the parameters of the module to the shadow dictionary.
        Parameters
        ----------
        module

        Returns
        -------

        """
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        """
        Updates the gradients based on the shadow dictionary.
        Parameters
        ----------
        module

        Returns
        -------

        """
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = self.shadow[name].data.to(param.data.device)
                self.shadow[name].data = (
                    1. -
                    self.mu) * param.data + self.mu * self.shadow[name].data

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict
