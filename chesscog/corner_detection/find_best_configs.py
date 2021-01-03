"""Script to execute the grid search over the generated YAML config files in order to determine which configuration achieves the best performance of the chessboard localization algorithm.

.. code-block:: console

    $ python -m chesscog.corner_detection.find_best_configs --help
    usage: find_best_configs.py [-h] [--n N] [--in INPUT]
                                [--out OUT]
    
    Get the best n configs of the results obtained via grid search
    
    optional arguments:
      -h, --help  show this help message and exit
      --n N       the number of configs to retain
      --in INPUT  the CSV file containing the results of the grid
                  search
      --out OUT   the output folder for the YAML files     output folder
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from recap import URI, CfgNode as CN


def _find_best_configs(n: int, results_file: Path, output_folder: Path):
    df = pd.read_csv(results_file)
    df = df.sort_values("mistakes")
    df = df.drop_duplicates([x for x in df.columns if x != "mistakes"])
    df = df.reset_index(drop=True)
    df = df.head(n)

    output_folder.mkdir(exist_ok=True, parents=True)

    for i, row in df.iterrows():
        configs = {
            k[len("config."):]: v if not hasattr(v, "item") else v.item()
            for k, v in row.items()
            if k.startswith("config.")
        }
        cfg = CN.load_yaml_with_base("config://corner_detection/_base.yaml")
        cfg.merge_with_dict(configs)
        with (output_folder / f"{i:04d}.yaml").open("w") as f:
            cfg.dump(stream=f)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Get the best n configs of the results obtained via grid search")
    parser.add_argument("--n",
                        help="the number of configs to retain",
                        type=int, default=100)
    parser.add_argument("--in", dest="input",
                        help="the CSV file containing the results of the grid search",
                        type=str, default="results://corner_detection/evaluate.csv")
    parser.add_argument("--out",
                        help="the output folder for the YAML files",
                        type=str, default="config://corner_detection/refined")
    args = parser.parse_args()
    _find_best_configs(args.n, URI(args.input), URI(args.out))
