from torch import nn
from recap import CfgNode as CN

from chesscog.utils.registry import Registry

MODELS_REGISTRY = Registry()


def build_model(cfg: CN) -> nn.Module:
    model = cfg.TRAINING.MODEL
    return MODELS_REGISTRY[model.REGISTRY][model.NAME]()
