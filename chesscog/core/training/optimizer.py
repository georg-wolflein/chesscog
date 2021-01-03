import torch
import typing
from recap import CfgNode as CN


def build_optimizer_from_config(optimizer_cfg: CN, params: typing.Iterable) -> torch.optim.Optimizer:
    """Build an optimizer for neural network training from a configuration.

    Args:
        optimizer_cfg (CN): the optimizer part of the configuration object
        params (typing.Iterable): the parameters to optimize

    Raises:
        NotImplementedError: if the desired optimizer is not implemented

    Returns:
        torch.optim.Optimizer: the built optimizer
    """
    optimizers = {
        "Adam": lambda: torch.optim.Adam(params, lr=optimizer_cfg.LEARNING_RATE)
    }
    if optimizer_cfg.NAME not in optimizers:
        raise NotImplementedError
    return optimizers[optimizer_cfg.NAME]()
