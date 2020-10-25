import torch

from chesscog.utils.config import CfgNode as CN


def build_optimizer_from_config(optimizer_cfg: CN, params) -> torch.optim.Optimizer:
    optimizers = {
        "Adam": lambda: torch.optim.Adam(params, lr=optimizer_cfg.LEARNING_RATE)
    }
    if optimizer_cfg.NAME not in optimizers:
        raise NotImplementedError
    return optimizers[optimizer_cfg.NAME]()
