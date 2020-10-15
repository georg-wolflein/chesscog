import torch
import numpy as np

from chesscog.utils.config import CfgNode as CN


class AccuracyAggregator():
    """Simple class for aggregating accuracy statistics between batches.
    """

    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.correct = np.zeros(self.num_classes, dtype=np.uint32)
        self.total = np.zeros(self.num_classes, dtype=np.uint32)

    def add_batch(self, one_hot_outputs: torch.Tensor, labels: torch.Tensor):
        outputs = one_hot_outputs.cpu().argmax(axis=-1)
        labels = labels.cpu()
        correct_mask = outputs == labels
        for c in range(self.num_classes):
            class_mask = labels == c
            self.correct[c] += (correct_mask & class_mask).sum()
            self.total[c] += class_mask.sum()

    def __getitem__(self, idx: int) -> float:
        """Obtain the accuracy score for a specific class.

        Args:
            idx (int): the class

        Returns:
            float: the accuracy score
        """
        total = self.total[idx]
        return self.correct[idx] / total if total != 0 else 0

    @property
    def accuracy(self) -> float:
        """Obtain the overall accuracy score.

        Returns:
            float: the accuracy score
        """
        correct = self.correct.sum()
        total = self.total.sum()
        return correct / total if total != 0 else 0


def build_optimizer_from_config(optimizer_cfg: CN, params) -> torch.optim.Optimizer:
    optimizers = {
        "Adam": lambda: torch.optim.Adam(params, lr=optimizer_cfg.LEARNING_RATE)
    }
    if optimizer_cfg.NAME not in optimizers:
        raise NotImplementedError
    return optimizers[optimizer_cfg.NAME]()
