"""Script to create config files for the grid search.

.. code-block:: console

    $ python -m chesscog.corner_detection.create_configs --help
    usage: create_configs.py [-h]
    
    Create YAML config files for grid search.
    
    optional arguments:
      -h, --help  show this help message and exit
"""

import numpy as np
import typing
import functools
from recap import URI, CfgNode as CN
import argparse

from chesscog.core import listify

parameters = {
    "EDGE_DETECTION.LOW_THRESHOLD": np.arange(80, 151, 10),
    "EDGE_DETECTION.HIGH_THRESHOLD": np.arange(100, 501, 50),
    # "EDGE_DETECTION.APERTURE": [3, 5],
    "LINE_DETECTION.THRESHOLD": np.arange(100, 251, 50),
    "RANSAC.OFFSET_TOLERANCE": np.arange(.05, .201, .05),
    "MAX_OUTLIER_INTERSECTION_POINT_RATIO_PER_LINE": np.arange(.5, .8, .2),
    # "LINE_REFINEMENT.LINE_THRESHOLD": [2, 3, 4]
}

parameters = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
              for (k, v) in parameters.items()}


@listify
def _add_parameter(key: str, values: typing.Iterable[typing.Any], cfgs: typing.List[CN]) -> list:
    for value in values:
        for cfg in cfgs:
            cfg = cfg.clone()
            cfg_node = cfg
            *key_items, final_key = key.split(".")
            for k in key_items:
                cfg_node = cfg_node[k]
            cfg_node[final_key] = value
            yield cfg


def _is_valid_cfg(cfg: CN) -> bool:
    return cfg.EDGE_DETECTION.LOW_THRESHOLD <= cfg.EDGE_DETECTION.HIGH_THRESHOLD


if __name__ == "__main__":
    argparse.ArgumentParser(
        description="Create YAML config files for grid search.").parse_args()
    cfg_folder = URI("config://corner_detection")
    cfg = CN.load_yaml_with_base(cfg_folder / "_base.yaml")
    cfgs = [cfg]
    for k, v in parameters.items():
        cfgs = _add_parameter(k, v, cfgs)
    cfgs = filter(_is_valid_cfg, cfgs)
    for i, cfg in enumerate(cfgs, 1):
        with (cfg_folder / f"generated_{i}.yaml").open("w") as f:
            cfg.dump(stream=f)
