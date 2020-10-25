import logging

from chesscog.utils.io import URI
from chesscog.utils.config import CfgNode as CN
from chesscog.utils.models import MODELS_REGISTRY

logger = logging.getLogger(__name__)


def create_configs(classifier: str, include_centercrop: bool = False):
    config_dir = URI("config://") / classifier

    logger.info(f"Removing YAML files from {config_dir}.")
    for f in config_dir.glob("*.yaml"):
        if f.name != "_base.yaml":
            f.unlink()

    for name, model in MODELS_REGISTRY[classifier.upper()].items():
        for center_crop in ({True, False} if include_centercrop else {False}):
            config_file = config_dir / \
                (name + ("_centercrop" if center_crop else "") + ".yaml")
            logging.info(f"Writing configuration file {config_file}")

            size = model.input_size
            C = CN()
            C._BASE_ = "config://" + classifier + "/_base.yaml"
            C.DATASET = CN()
            C.DATASET.TRANSFORMS = CN()
            C.DATASET.TRANSFORMS.CENTER_CROP = (50, 50) \
                if center_crop else None
            C.DATASET.TRANSFORMS.RESIZE = size
            C.TRAINING = CN()
            C.TRAINING.MODEL = CN()
            C.TRAINING.MODEL.REGISTRY = classifier.upper()
            C.TRAINING.MODEL.NAME = name

            if model.pretrained:
                def create_phase(epochs: int, lr: float, params: str):
                    return {
                        "EPOCHS": epochs,
                        "PARAMS": params,
                        "OPTIMIZER": {
                            "NAME": "Adam",
                            "LEARNING_RATE": lr
                        }
                    }
                C.TRAINING.PHASES = [create_phase(epochs=1, lr=.001, params="head"),
                                     create_phase(epochs=2, lr=.0001, params="all")]

            with config_file.open("w") as f:
                C.dump(stream=f)
