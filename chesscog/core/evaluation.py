"""Common functions for evaluation CNNs.
"""

import argparse
import torch
import torchvision
import numpy as np
from pathlib import Path
import typing
import logging
from PIL import Image
from recap import URI, CfgNode as CN

from chesscog.core.dataset import build_dataset, build_data_loader, Datasets, unnormalize
from chesscog.core.statistics import StatsAggregator
from chesscog.core import device, DEVICE

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
    """Evaluate a model, returning the results as CSV.

    Args:
        model_path (Path): path to the model folder containing the YAML file and the saved weights
        datasets (typing.List[Datasets]): the datasets to evaluate on
        output_folder (Path): output folder for the mistake images (if applicable)
        find_mistakes (bool, optional): whether to output all mistakes as images to the output folder. Defaults to False.
        include_heading (bool, optional): whether to include a heading in the CSV output. Defaults to False.

    Raises:
        ValueError: if the YAML config file is missing

    Returns:
        str: the CSV string
    """
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
    classes = next(iter(datasets.values())).classes

    csv = []
    if include_heading:
        csv.append(_csv_heading(classes))
    for mode, dataset in datasets.items():
        # Load dataset
        loader = build_data_loader(cfg, dataset, mode)
        # Compute statistics over whole dataset
        agg = StatsAggregator(classes)
        for images, labels in device(loader):
            predictions = model(images)
            agg.add_batch(predictions, labels, **(dict(inputs=images)
                                                  if find_mistakes else dict()))

        csv.append(_csv(model, agg, model_name, mode))
        if find_mistakes:
            groundtruth, mistakes = zip(*sorted(agg.mistakes,
                                                key=lambda x: x[0]))
            imgs = torch.tensor(mistakes).permute((0, 2, 3, 1))
            imgs = unnormalize(imgs).permute((0, 3, 1, 2))
            img = torchvision.utils.make_grid(imgs, pad_value=1, nrow=4)
            img = img.numpy().transpose((1, 2, 0)) * 255
            img = Image.fromarray(img.astype(np.uint8))
            mistakes_file = output_folder / \
                f"{model_name}_{mode.value}_mistakes.png"
            logger.info(f"Writing mistakes to {mistakes_file}")
            img.save(mistakes_file)
            groundtruth_file = output_folder / \
                f"{model_name}_{mode.value}_groundtruth.csv"
            with groundtruth_file.open("w") as f:
                f.write(",".join(map(str, groundtruth)))
    return "\n".join(csv)


def perform_evaluation(classifier: str):
    """Function to set up the CLI for the evaluation script.

    Args:
        classifier (str): the classifier
    """
    parser = argparse.ArgumentParser(description="Evaluate trained models.")
    parser.add_argument("--model", help=f"the model to evaluate (if unspecified, all models in 'runs://{classifier}' will be evaluated)",
                        type=str, default=None)
    parser.add_argument("--dataset", help="the dataset to evaluate (if unspecified, train and val will be evaluated)",
                        type=str, default=None, choices=[x.value for x in Datasets])
    parser.add_argument("--out", help="output folder", type=str,
                        default=f"results://{classifier}")
    parser.add_argument("--find-mistakes", help="whether to output all misclassification images",
                        dest="find_mistakes", action="store_true")
    parser.set_defaults(find_mistakes=False)
    args = parser.parse_args()

    # Evaluate
    output_folder = URI(args.out)
    output_folder.mkdir(parents=True, exist_ok=True)
    output_csv = output_folder / "evaluate.csv"
    with output_csv.open("w") as f:
        models = list(URI(f"runs://{classifier}").glob("*/*.pt")) \
            if args.model is None else [URI(args.model)]
        datasets = [Datasets.TRAIN, Datasets.VAL] \
            if args.dataset is None else [d for d in Datasets if d.value == args.dataset]
        for i, model in enumerate(models):
            logger.info(f"Processing model {i+1}/{len(models)}")
            f.write(evaluate(model, datasets, output_folder,
                             find_mistakes=args.find_mistakes,
                             include_heading=i == 0) + "\n")
