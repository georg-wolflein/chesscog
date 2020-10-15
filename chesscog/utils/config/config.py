from __future__ import annotations
import os
from typing import Dict, Any
from pathlib import Path
from yacs.config import CfgNode as _CfgNode
import typing

from chesscog.utils.io import URI

BASE_KEY = "_BASE_"


class CfgNode(_CfgNode):

    @classmethod
    def load_yaml_with_base(cls, filename: os.PathLike) -> CfgNode:
        uri = URI(filename)
        with uri.open("r") as f:
            cfg = cls.load_cfg(f)
        if BASE_KEY in cfg:
            base_cfg = cls.load_yaml_with_base(cfg[BASE_KEY])
            del cfg[BASE_KEY]
            base_cfg.merge_from_other_cfg(cfg)
            return base_cfg
        return cfg

    def merge_with_dict(self, overrides: typing.Dict[str, typing.Any]):
        for key, value in overrides.items():
            child = self
            *path_segments, prop = key.split(".")
            for path_segment in path_segments:
                if path_segment not in child:
                    child[path_segment] = CfgNode()
                child = child[path_segment]
            child[prop] = value
