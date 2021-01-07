import logging
from recap import URI, CfgNode as CN

from chesscog.core.models import MODELS_REGISTRY

logger = logging.getLogger(__name__)


def create_configs(classifier: str, include_centercrop: bool = False):
    """Create the YAML configuration files for all registered models for a classifier.

    Args:
        classifier (str): the classifier (either `"occupancy_classifier"` or `"piece_classifier"`)
        include_centercrop (bool, optional): whether to create two configs per model, one including center crop and one not. Defaults to False.
    """
    config_dir = URI("config://") / classifier

    logger.info(f"Removing YAML files from {config_dir}.")
    for f in config_dir.glob("*.yaml"):
        if not f.name.startswith("_"):
            f.unlink()

    for name, model in MODELS_REGISTRY[classifier.upper()].items():
        for center_crop in ({True, False} if include_centercrop else {False}):
            config_file = config_dir / \
                (name + ("_centercrop" if center_crop else "") + ".yaml")
            logging.info(f"Writing configuration file {config_file}")

            size = model.input_size
            C = CN()
            override_base = f"config://{classifier}/_base_override_{name}.yaml"
            if URI(override_base).exists():
                C._BASE_ = override_base
            else:
                suffix = "_pretrained" if model.pretrained else ""
                C._BASE_ = f"config://{classifier}/_base{suffix}.yaml"
            C.DATASET = CN()
            C.DATASET.TRANSFORMS = CN()
            C.DATASET.TRANSFORMS.CENTER_CROP = (50, 50) \
                if center_crop else None
            C.DATASET.TRANSFORMS.RESIZE = size
            C.TRAINING = CN()
            C.TRAINING.MODEL = CN()
            C.TRAINING.MODEL.REGISTRY = classifier.upper()
            C.TRAINING.MODEL.NAME = name

            with config_file.open("w") as f:
                C.dump(stream=f)
