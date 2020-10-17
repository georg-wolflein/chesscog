import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision import transforms as T
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
import typing
import copy
import argparse
import functools
import shutil

from chesscog.utils import device
from chesscog.utils.config import CfgNode as CN
from chesscog.utils.training import build_optimizer_from_config, StatsAggregator
from chesscog.utils.io import URI
from .dataset import build_datasets, build_data_loader, Datasets
from .models import MODELS

logger = logging.getLogger(__name__)


def train(cfg: CN, run_dir: Path) -> nn.Module:
    logger.info(f"Starting training in {run_dir}")

    # Create folder
    if run_dir.exists():
        logger.warning(
            f"The folder {run_dir} already exists and will be overwritten by this run")
        shutil.rmtree(run_dir, ignore_errors=True)
    run_dir.mkdir(exist_ok=True)

    # Store config
    with (run_dir / "config.yaml").open("w") as f:
        cfg.dump(stream=f)

    datasets, classes = build_datasets(cfg)
    dataset = datasets[Datasets.ALL]

    model = MODELS[cfg.TRAINING.MODEL]()
    device(model)

    best_weights, best_accuracy, best_step = copy.deepcopy(
        model.state_dict()), 0., 0

    criterion = nn.CrossEntropyLoss()

    writer = {mode: SummaryWriter(run_dir / mode.value)
              for mode in {Datasets.TRAIN, Datasets.VAL}}
    aggregator = {mode: StatsAggregator(classes)
                  for mode in {Datasets.TRAIN, Datasets.VAL}}
    loader = {mode: build_data_loader(cfg, datasets, mode)
              for mode in {Datasets.TRAIN, Datasets.VAL}}

    def log(step: int, loss: float, mode: Datasets):
        if mode == Datasets.TRAIN:
            logger.info(f"Step {step:5d}: loss {loss:.3f}")

        w, agg = (x[mode] for x in (writer, aggregator))

        w.add_scalar("Loss", loss, step)
        w.add_scalar("Accuracy", agg.accuracy(), step)
        for c in classes:
            w.add_scalar(f"Precision/{c}", agg.precision(c), step)
            w.add_scalar(f"Recall/{c}", agg.recall(c), step)
            w.add_scalar(f"F1 score/{c}", agg.f1_score(c), step)

    def perform_iteration(data: typing.Tuple[torch.Tensor, torch.Tensor], mode: Datasets):
        inputs, labels = map(device, data)
        with torch.set_grad_enabled(mode == Datasets.TRAIN):
            # Reset gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs, labels)

            if mode == Datasets.TRAIN:
                loss.backward()

        with torch.no_grad():
            aggregator[mode].add_batch(outputs, labels)

        # Perform optimisation
        if mode == Datasets.TRAIN:
            optimizer.step()

        # Return
        return loss.item()

    step = 0
    log_every_n = 100

    # Loop over training phases
    for phase in cfg.TRAINING.PHASES:

        for p in model.parameters():
            p.requires_grad = False
        parameters = list(model.parameters()) if phase.PARAMS == "all" \
            else model.params[phase.PARAMS]
        for p in parameters:
            p.requires_grad = True
        optimizer = build_optimizer_from_config(phase.OPTIMIZER,
                                                parameters)

        # Loop over epochs (passes over the whole dataset)
        for epoch in range(phase.EPOCHS):
            aggregator[Datasets.TRAIN].reset()

            # Iterate the training set
            losses = []
            for i, data in enumerate(loader[Datasets.TRAIN]):

                # Perform training iteration
                losses.append(perform_iteration(data, mode=Datasets.TRAIN))

                if step % log_every_n == 0:
                    loss = np.mean(list(losses))
                    log(step, loss, Datasets.TRAIN)
                    aggregator[Datasets.TRAIN].reset()
                    losses = []

                    # Validate entire validation dataset
                    aggregator[Datasets.VAL].reset()

                    # Iterate entire val dataset
                    perform_val_iteration = functools.partial(perform_iteration,
                                                              mode=Datasets.VAL)
                    val_losses = map(perform_val_iteration,
                                     loader[Datasets.VAL])

                    # Gather losses and log
                    val_loss = np.mean(list(val_losses))
                    log(step, val_loss, Datasets.VAL)

                # Save weights if we get a better performance
                accuracy = aggregator[Datasets.VAL].accuracy()
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_weights = copy.deepcopy(model.state_dict())
                    best_step = step

                # Get ready for next step
                step += 1

    # Clean up
    for w in writer.values():
        w.flush()
        w.close()

    logger.info("Finished training")

    logger.info(
        f"Restoring best weight state (step {best_step} with validation accuracy of {best_accuracy})")
    model.load_state_dict(best_weights)
    return model


if __name__ == "__main__":
    configs_dir = URI("config://") / "occupancy_classifier"

    def _train(config: str):
        cfg = CN.load_yaml_with_base(configs_dir / f"{config}.yaml")
        run_dir = URI("runs://") / "occupancy_classifier" / config

        # Train the model and save it
        model = train(cfg, run_dir)
        torch.save(model, run_dir / "model.pt")

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
