"""Script to train (i.e. fine-tune) the classifiers on the new dataset.

.. code-block:: console

    $ python -m chesscog.transfer_learning.train --help      
    usage: train.py [-h]
    
    Fine-tune the classifiers on the new dataset.
    
    optional arguments:
      -h, --help  show this help message and exit
"""

from recap import URI, CfgNode as CN
import typing
import torch
import logging
import argparse

from chesscog.core.training.train import train_model
from chesscog.core import device, DEVICE

logger = logging.getLogger(__name__)


def _train_model(model_type: str) -> typing.Tuple[torch.nn.Module, CN]:
    model_file = next((URI("models://") / model_type).glob("*.pt"))
    yaml_file = URI("config://transfer_learning") / \
        model_type / f"{model_file.stem}.yaml"
    cfg = CN.load_yaml_with_base(yaml_file)
    run_dir = URI("runs://transfer_learning") / model_type
    model = torch.load(model_file, map_location=DEVICE)
    model = device(model)
    is_inception = "inception" in model_file.stem.lower()
    train_model(cfg, run_dir, model, is_inception,
                model_file.stem, eval_on_train=True)


if __name__ == "__main__":
    argparse.ArgumentParser(
        description="Fine-tune the classifiers on the new dataset.").parse_args()
    for model_type in ("occupancy_classifier", "piece_classifier"):
        logger.info(f"Starting training for {model_type}")
        _train_model(model_type)
        logger.info(f"Finished training for {model_type}")
