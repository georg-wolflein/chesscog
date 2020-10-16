import logging

from chesscog.utils.io import URI
from chesscog.utils.config import CfgNode as CN
from .models import MODELS

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    config_dir = URI("config://occupancy_classifier")

    logger.info(f"Removing YAML files from {config_dir}.")
    for f in config_dir.glob("*.yaml"):
        if f.name != "_base.yaml":
            f.unlink()

    for name, model in MODELS.items():
        for center_crop in {True, False}:
            config_file = config_dir / \
                (name + ("_centercrop" if center_crop else "") + ".yaml")
            logging.info(f"Writing configuration file {config_file}")

            size = model.input_size
            C = CN()
            C._BASE_ = "config://occupancy_classifier/_base.yaml"
            C.DATASET = CN()
            C.DATASET.TRANSFORMS = CN()
            C.DATASET.TRANSFORMS.CENTER_CROP = (50, 50) \
                if center_crop else (100, 100)
            C.DATASET.TRANSFORMS.RESIZE = (size, size)
            C.TRAINING = CN()
            C.TRAINING.MODEL = name

            with config_file.open("w") as f:
                f.write(C.dump())
