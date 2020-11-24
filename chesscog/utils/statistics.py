import torch
import typing
import numpy as np

from recap import CfgNode as CN


def _fraction(a: float, b: float) -> float:
    return a/b if b != 0 else 0


class StatsAggregator():
    """Simple class for aggregating accuracy statistics between batches.
    """

    def __init__(self, classes: list):
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.reset()
        self.mistakes = []

    def reset(self):
        self.confusion_matrix = np.zeros((len(self.classes), len(self.classes)),
                                         dtype=np.uint32)
        self.mistakes = []

    def add_batch(self, one_hot_outputs: torch.Tensor, labels: torch.Tensor, inputs: torch.Tensor = None):
        """Add a batch to compute statistics over.

        Args:
            one_hot_outputs (torch.Tensor): the one hot outputs of the model
            labels (torch.Tensor): the groundtruth labels
            inputs (torch.Tensor, optional): the inputs (if supplied, will be used to keep track of mistakes)
        """
        outputs = one_hot_outputs.cpu().argmax(axis=-1).numpy()
        labels = labels.cpu().numpy()
        for predicted_class, _ in enumerate(self.classes):
            predicted_mask = outputs == predicted_class
            for actual_class, _ in enumerate(self.classes):
                actual_mask = labels == actual_class
                self.confusion_matrix[predicted_class,
                                      actual_class] += (actual_mask & predicted_mask).sum()
        if inputs is not None:
            mistakes_mask = outputs != labels
            mistakes = inputs[mistakes_mask].cpu().numpy()
            groundtruth = map(self.classes.__getitem__, labels[mistakes_mask])
            self.mistakes.extend(zip(groundtruth, mistakes))

    def accuracy(self) -> float:
        correct = np.trace(self.confusion_matrix)  # sum along diagonal
        total = np.sum(self.confusion_matrix)
        return _fraction(correct, total)

    def precision(self, cls: str) -> float:
        idx = self.class_to_idx[cls]
        tp = self.confusion_matrix[idx, idx]
        tp_plus_fp = self.confusion_matrix[idx].sum()
        return _fraction(tp, tp_plus_fp)

    def recall(self, cls: str) -> float:
        idx = self.class_to_idx[cls]
        tp = self.confusion_matrix[idx, idx]
        p = self.confusion_matrix[:, idx].sum()
        return _fraction(tp, p)

    def f1_score(self, cls: str) -> float:
        precision = self.precision(cls)
        recall = self.recall(cls)
        return _fraction(2 * precision * recall, precision + recall)
