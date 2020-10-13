import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision import transforms as T
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
import argparse

from chesscog import RUNS_DIR, CONFIG_DIR
from chesscog.config import CfgNode as CN
from chesscog.util.training import build_optimizer_from_config, AccuracyAggregator
from .dataset import build_dataset

logger = logging.getLogger(__name__)


class CNN100(nn.Module):
    def __init__(self):
        super().__init__()
        # Input size: 100x100
        self.conv1 = nn.Conv2d(3, 16, 5)  # 96
        self.pool1 = nn.MaxPool2d(2, 2)  # 48
        self.conv2 = nn.Conv2d(16, 32, 5)  # 44
        self.pool2 = nn.MaxPool2d(2, 2)  # 22
        self.conv3 = nn.Conv2d(32, 64, 3)  # 20
        self.pool3 = nn.MaxPool2d(2, 2)  # 10
        self.fc1 = nn.Linear(64 * 10 * 10, 1000)
        self.fc2 = nn.Linear(1000, 256)
        self.fc3 = nn.Linear(256, 2)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 10 * 10)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CNN50(nn.Module):
    def __init__(self):
        super().__init__()
        # Input size: 50x50
        self.conv1 = nn.Conv2d(3, 16, 3)  # 48
        self.pool1 = nn.MaxPool2d(2, 2)  # 24
        self.conv2 = nn.Conv2d(16, 32, 3)  # 22
        self.pool2 = nn.MaxPool2d(2, 2)  # 11
        self.fc1 = nn.Linear(64 * 11 * 11, 1000)
        self.fc2 = nn.Linear(1000, 2)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 11 * 11)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


NETWORKS = {
    "CNN50": CNN50,
    "CNN100": CNN100
}


def train(cfg: CN, run_dir: Path) -> nn.Module:
    logger.info(f"Starting training in {run_dir}")
    dataset, train_loader, val_loader, test_loader = build_dataset(cfg)

    model = NETWORKS[cfg.TRAINING.NETWORK]()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer_from_config(cfg.TRAINING.OPTIMIZER,
                                            model.parameters())

    step = 0

    train_writer = SummaryWriter(run_dir / "train")
    val_writer = SummaryWriter(run_dir / "val")
    accuracies = AccuracyAggregator(len(dataset.classes))
    val_accuracies = AccuracyAggregator(len(dataset.classes))
    log_every_n = 100

    for epoch in range(cfg.TRAINING.EPOCHS):

        for i, data in enumerate(train_loader):
            accuracies.reset()

            # Get the current minibatch
            inputs, labels = (x.to(device) for x in data)

            # Reset gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Backward pass
            loss = criterion(outputs, labels)
            loss.backward()

            # Log metrics
            with torch.no_grad():
                accuracies.add_batch(outputs, labels)
                logger.info(
                    f"Step {step:5d}, loss {loss.item():.3f}, acc {accuracies.accuracy:.3f}")

                train_writer.add_scalar("Loss", loss.item(), step)
                train_writer.add_scalar("Accuracy", accuracies.accuracy, step)
                for c, idx in dataset.class_to_idx.items():
                    train_writer.add_scalar(
                        f"Accuracy/{c}", accuracies[idx], step)

                if step % log_every_n == 0:
                    # Compute validation metrics
                    val_accuracies.reset()
                    val_losses = list()
                    for val_i, data in enumerate(val_loader, 1):
                        inputs, labels = (x.to(device) for x in data)
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        val_losses.append(loss.item())
                        val_accuracies.add_batch(outputs, labels)
                    val_writer.add_scalar("Loss", np.mean(val_losses), step)
                    val_writer.add_scalar(
                        "Accuracy", val_accuracies.accuracy, step)
                    for c, idx in dataset.class_to_idx.items():
                        val_writer.add_scalar(
                            f"Accuracy/{c}", val_accuracies[idx], step)
            # Perform optimisation
            optimizer.step()
            step += 1

    for writer in (train_writer, val_writer):
        writer.flush()
        writer.close()

    logger.info("Finished training")
    return model


if __name__ == "__main__":
    def _train(config: str):
        run_dir = RUNS_DIR / "occupancy_classifier" / config / \
            datetime.now().strftime("_%Y-%m-%d_%H-%M-%S")
        cfg = CN.load_yaml_with_base(
            str(CONFIG_DIR / "occupancy_classifier" / f"{config}.yaml"))
        model = train(cfg, run_dir)
        torch.save(model, run_dir / "model.pt")

    # Read available configs
    configs = [x.stem for x in (CONFIG_DIR / "occupancy_classifier").glob("*.yaml")
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
