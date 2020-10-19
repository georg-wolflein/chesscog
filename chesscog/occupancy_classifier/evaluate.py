import argparse
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import typing
import logging

from chesscog.occupancy_classifier import models
from chesscog.occupancy_classifier.dataset import build_dataset, build_data_loader, Datasets
from chesscog.utils.config import CfgNode as CN
from chesscog.utils.io import URI
from chesscog.utils.training import StatsAggregator
from chesscog.utils import device

logger = logging.getLogger(__name__)


def _csv(agg: StatsAggregator, run: str, dataset: Datasets) -> str:
    return ",".join(map(str, [run,
                              dataset.value,
                              agg.accuracy(),
                              *map(agg.precision, agg.classes),
                              *map(agg.recall, agg.classes),
                              *map(agg.f1_score, agg.classes),
                              *agg.confusion_matrix.flatten()
                              ]))


def _csv_heading(classes: typing.List[str]) -> str:
    def class_headings(metric: str) -> typing.List[str]:
        return [f"{metric}/{c}" for c in classes]
    return ",".join(["run",
                     "dataset",
                     "accuracy",
                     *class_headings("precision"),
                     *class_headings("recall"),
                     *class_headings("f1_score"),
                     *(f"confusion_matrix/{i}/{j}"
                       for i in range(len(classes))
                       for j in range(len(classes)))])


def evaluate(run: str, include_heading: bool = False) -> str:
    run_dir = URI("runs://occupancy_classifier") / run
    cfg = CN.load_yaml_with_base(run_dir / "config.yaml")
    model_path = next(run_dir.glob("model*.pt"))
    model = torch.load(model_path)
    model = device(model)
    model.eval()
    datasets = {mode: build_dataset(cfg, mode)
                for mode in (Datasets.TRAIN, Datasets.VAL)}
    classes = datasets[Datasets.TRAIN].classes

    def _eval():
        if include_heading:
            yield _csv_heading(classes)
        for mode, dataset in datasets.items():
            # Load dataset
            loader = build_data_loader(cfg, dataset, mode)
            # Compute statistics over whole dataset
            agg = StatsAggregator(classes)
            for images, labels in device(loader):
                predictions = model(images)
                agg.add_batch(predictions, labels)
            yield _csv(agg, run, dataset)
    return "\n".join(_eval())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained models")
    runs = [x.parent.name for x in URI(
        "runs://occupancy_classifier").glob("*/model*.pt")]
    parser.add_argument("--run", help="the run to evaluate",
                        type=str, choices=runs, default=None)
    parser.add_argument("--out", help="output CSV file", type=str,
                        default="runs://occupancy_classifier/evaluation.csv")
    args = parser.parse_args()

    # Evaluate
    with URI(args.out).open("w") as f:
        if args.run is not None:
            runs = [args.run]
        for i, run in enumerate(runs):
            logger.info(f"Processing run {i+1}/{len(runs)}")
            f.write(evaluate(run, include_heading=i == 0) + "\n")
