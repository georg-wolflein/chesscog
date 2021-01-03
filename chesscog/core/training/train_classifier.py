import argparse
import logging
from recap import URI, CfgNode as CN

from chesscog.core.training import train

logger = logging.getLogger(__name__)


def train_classifier(name: str):
    """Set up CLI interface for training a classifier.

    Args:
        name (str): the name of the classifier (`"occupancy_classifier"` or `"piece_classifier"`)
    """
    configs_dir = URI("config://") / name

    def _train(config: str):
        cfg = CN.load_yaml_with_base(configs_dir / f"{config}.yaml")
        run_dir = URI("runs://") / name / config

        # Train the model and save it
        train(cfg, run_dir)

    # Read available configs
    configs = [x.stem for x in configs_dir.glob("*.yaml")
               if not x.stem.startswith("_")]

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Train the network.")
    parser.add_argument("--config", help="the configuration to train (default: all)",
                        type=str, choices=configs, default=None)
    args = parser.parse_args()

    # Train
    if args.config is None:
        logger.info("Training all configurations one by one")
        for config in configs:
            _train(config)
    else:
        logger.info(f"Training the {args.config} configuration")
        _train(args.config)
