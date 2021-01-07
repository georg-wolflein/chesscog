"""Main implementation of model training.
"""

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import torchvision
import numpy as np
from pathlib import Path
import logging
import typing
import copy
import functools
import shutil
from recap import CfgNode as CN

from chesscog.core import device
from chesscog.core.training import build_optimizer_from_config
from chesscog.core.statistics import StatsAggregator
from chesscog.core.dataset import build_dataset, build_data_loader, Datasets
from chesscog.core.models import build_model

logger = logging.getLogger(__name__)


def train(cfg: CN, run_dir: Path) -> nn.Module:
    """Traing a model.

    Args:
        cfg (CN): the configuration object describing the model, dataset, etc.
        run_dir (Path): where to write tensorboard files, the active YAML file, and the chosen weights

    Returns:
        nn.Module: the trained model
    """
    model = build_model(cfg)
    is_inception = "inception" in cfg.TRAINING.MODEL.NAME.lower()
    train_model(cfg, run_dir, model, is_inception)


def train_model(cfg: CN, run_dir: Path, model: torch.nn.Module, is_inception: bool = False, model_name: str = None, eval_on_train: bool = False) -> nn.Module:
    """Train a model that has already been loaded.

    Args:
        cfg (CN): the configuration object describing the model, dataset, etc.
        run_dir (Path): where to write tensorboard files, the active YAML file, and the chosen weights
        model (torch.nn.Module): the loaded model
        is_inception (bool, optional): whether the model is InceptionV3. Defaults to False.
        model_name (str, optional): the name of the model (by default the last component of the run directory). Defaults to None.
        eval_on_train (bool, optional): whether to evaluate on the training set. Defaults to False.

    Returns:
        nn.Module: the trained model
    """
    logger.info(f"Starting training in {run_dir}")
    if not model_name:
        model_name = run_dir.name

    # Create folder
    if run_dir.exists():
        logger.warning(
            f"The folder {run_dir} already exists and will be overwritten by this run")
        shutil.rmtree(run_dir, ignore_errors=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Store config
    with (run_dir / f"{model_name}.yaml").open("w") as f:
        cfg.dump(stream=f)

    # Move model to device
    device(model)

    best_weights, best_accuracy, best_step = copy.deepcopy(
        model.state_dict()), 0., 0

    criterion = nn.CrossEntropyLoss()

    modes = {Datasets.TRAIN, Datasets.VAL}
    if eval_on_train:
        dataset = build_dataset(cfg, Datasets.TRAIN)
        datasets = {mode: dataset
                    for mode in modes}
    else:
        datasets = {mode: build_dataset(cfg, mode)
                    for mode in modes}
    classes = datasets[Datasets.TRAIN].classes
    loader = {mode: build_data_loader(cfg, datasets[mode], mode)
              for mode in modes}
    writer = {mode: SummaryWriter(run_dir / mode.value)
              for mode in modes}
    aggregator = {mode: StatsAggregator(classes)
                  for mode in modes}

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

            # Forward pass and compute loss
            if is_inception and mode == Datasets.TRAIN:
                # Special case for inception models
                outputs, auxiliary_outputs = model(inputs)
                loss1 = criterion(outputs, labels)
                loss2 = criterion(auxiliary_outputs, labels)
                loss = loss1 + 0.4*loss2
            else:
                outputs = model(inputs)
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

    # Ensure we're in training mode
    model.train()

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
                    model.eval()
                    aggregator[Datasets.VAL].reset()

                    # Iterate entire val dataset
                    perform_val_iteration = functools.partial(perform_iteration,
                                                              mode=Datasets.VAL)
                    val_losses = map(perform_val_iteration,
                                     loader[Datasets.VAL])

                    # Gather losses and log
                    val_loss = np.mean(list(val_losses))
                    log(step, val_loss, Datasets.VAL)
                    model.train()

                # Save weights if we get a better performance
                accuracy = aggregator[Datasets.VAL].accuracy()
                if accuracy >= best_accuracy:
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
    torch.save(model, run_dir / f"{model_name}.pt")
    with (run_dir / f"{model_name}.txt").open("w") as f:
        f.write(f"exported at step: {best_step}")
    return model
