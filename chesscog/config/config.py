import os
from typing import Dict, Any
from pathlib import Path
from fvcore.common.config import CfgNode as _CfgNode, PathManager
import typing


BASE_KEY = "_BASE_"


class CfgNode(_CfgNode):

    @classmethod
    def load_yaml_with_base(cls, filename: str) -> _CfgNode:
        with PathManager.open(filename, "r") as f:
            cfg = cls.load_cfg(f)
        if BASE_KEY in cfg:
            base_cfg_file = cfg[BASE_KEY]
            if base_cfg_file.startswith("~"):
                base_cfg_file = os.path.expanduser(base_cfg_file)
            if not (base_cfg_file.startswith("/") or "://" in base_cfg_file):
                base_cfg_file = os.path.join(os.path.dirname(filename),
                                             base_cfg_file)
            base_cfg = cls.load_yaml_with_base(base_cfg_file)
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
