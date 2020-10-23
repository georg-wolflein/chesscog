import argparse
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import typing
import logging
from PIL import Image

from chesscog.occupancy_classifier import models as _models
from chesscog.occupancy_classifier.dataset import build_dataset, build_data_loader, Datasets, unnormalize
from chesscog.utils.config import CfgNode as CN
from chesscog.utils.io import URI
from chesscog.utils.training import StatsAggregator
from chesscog.utils import device, DEVICE

logger = logging.getLogger(__name__)


def _csv(model: torch.nn.Module, agg: StatsAggregator, model_name: str, mode: Datasets) -> str:
    params = sum([np.prod(p.size()) for p in model.parameters()])
    return ",".join(map(str, [model_name,
                              mode.value,
                              params,
                              agg.accuracy(),
                              *map(agg.precision, agg.classes),
                              *map(agg.recall, agg.classes),
                              *map(agg.f1_score, agg.classes),
                              *agg.confusion_matrix.flatten()
                              ]))


def _csv_heading(classes: typing.List[str]) -> str:
    def class_headings(metric: str) -> typing.List[str]:
        return [f"{metric}/{c}" for c in classes]
    return ",".join(["model",
                     "dataset",
                     "parameters",
                     "accuracy",
                     *class_headings("precision"),
                     *class_headings("recall"),
                     *class_headings("f1_score"),
                     *(f"confusion_matrix/{i}/{j}"
                       for i in range(len(classes))
                       for j in range(len(classes)))])


def evaluate(model_path: Path, datasets: typing.List[Datasets], output_folder: Path, find_mistakes: bool = False, include_heading: bool = False) -> str:
    model_name = model_path.stem
    config_file = model_path.parent / f"{model_name}.yaml"
    if not config_file.exists():
        raise ValueError("config file missing")
    cfg = CN.load_yaml_with_base(config_file)
    model = torch.load(model_path, map_location=DEVICE)
    model = device(model)
    model.eval()
    datasets = {mode: build_dataset(cfg, mode)
                for mode in datasets}
    classes = datasets[Datasets.TRAIN].classes

    def _eval():
        if include_heading:
            yield _csv_heading(classes)
        for mode, dataset in datasets.items():
            # Load dataset
            loader = build_data_loader(cfg, dataset, mode)
            # Compute statistics over whole dataset
            agg = StatsAggregator(["empty", "occupied"])
            for images, labels in device(loader):
                predictions = model(images)
                agg.add_batch(predictions, labels, **(dict(inputs=images)
                                                      if find_mistakes else dict()))

            yield _csv(model, agg, model_name, mode)
            if find_mistakes:
                imgs = torch.tensor(agg.mistakes)
                img = torchvision.utils.make_grid(
                    imgs).numpy().transpose((1, 2, 0))
                img = Image.fromarray(img)
                mistakes_file = output_folder / \
                    f"{model_name}_{mode.value}_mistakes.png"
                logger.info(f"Writing mistakes to {mistakes_file}")
                img.save(mistakes_file)
    return "\n".join(_eval())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained models")
    parser.add_argument("--model", help="the model to evaluate (if unspecified, all models in 'runs://occupancy_classifier' will be evaluated)",
                        type=str, default=None)
    parser.add_argument("--dataset", help="the dataset to evaluate (if unspecified, train and val will be evaluated)",
                        type=str, default=None, choices=[x.value for x in Datasets])
    parser.add_argument("--out", help="output folder", type=str,
                        default="results://occupancy_classifier")
    parser.add_argument("--find-mistakes", help="whether to output all misclassification images",
                        dest="find_mistakes", action="store_true")
    parser.set_defaults(find_mistakes=False)
    args = parser.parse_args()

    # Evaluate
    output_folder = URI(args.out)
    output_folder.mkdir(parents=True, exist_ok=True)
    output_csv = output_folder / "evaluate.csv"
    with output_csv.open("w") as f:
        models = list(URI("runs://occupancy_classifier").glob("*/*.pt")) \
            if args.model is None else [URI(args.model)]
        datasets = [Datasets.TRAIN, Datasets.VAL] \
            if args.dataset is None else [d for d in Datasets if d.value == args.dataset]
        for i, model in enumerate(models):
            logger.info(f"Processing model {i+1}/{len(models)}")
            f.write(evaluate(model, datasets, output_folder,
                             find_mistakes=args.find_mistakes,
                             include_heading=i == 0) + "\n")
