"""Common tasks related to models.
"""

from torch import nn
from recap import CfgNode as CN

from chesscog.core.registry import Registry

#: The global models registry
MODELS_REGISTRY = Registry()


def build_model(cfg: CN) -> nn.Module:
    """Build a CNN from a configuration.

    Args:
        cfg (CN): the configuration

    Returns:
        nn.Module: the built CNN model
    """
    model = cfg.TRAINING.MODEL
    return MODELS_REGISTRY[model.REGISTRY][model.NAME]()
