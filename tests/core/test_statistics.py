import numpy as np
import torch
import pytest

from chesscog.core.statistics import StatsAggregator


@pytest.fixture
def aggregator() -> StatsAggregator:
    agg = StatsAggregator(["a", "b"])
    a_output = np.array([.9, .1, .8, .2, .9, .9, .9, .2])
    b_output = 1 - a_output
    outputs = torch.tensor(np.stack([a_output, b_output], axis=-1))
    labels = torch.tensor([0, 0, 0, 1, 0, 0, 1, 0])
    # predicted:          [0, 1, 0, 1, 0, 0, 0, 1]
    agg.add_batch(outputs, labels)
    return agg


def test_accuracy(aggregator: StatsAggregator):
    assert np.isclose(aggregator.accuracy(), 5/8)


def test_precision(aggregator: StatsAggregator):
    assert np.isclose(aggregator.precision("a"), 4/5)
    assert np.isclose(aggregator.precision("b"), 1/3)


def test_recall(aggregator: StatsAggregator):
    assert np.isclose(aggregator.recall("a"), 4/6)
    assert np.isclose(aggregator.recall("b"), 1/2)


def test_f1_score(aggregator: StatsAggregator):
    a_precision = 4/5
    a_recall = 4/6
    assert np.isclose(aggregator.f1_score("a"),
                      2 * a_precision * a_recall, a_precision + a_recall)
    b_precision = 1/3
    b_recall = 1/2
    assert np.isclose(aggregator.f1_score("b"),
                      2 * b_precision * b_recall, b_precision + b_recall)


def test_empty_batch():
    aggregator = StatsAggregator(["a", "b"])
    assert aggregator.accuracy() == 0
    assert aggregator.precision("a") == 0
    assert aggregator.precision("b") == 0
    assert aggregator.recall("a") == 0
    assert aggregator.recall("b") == 0
    assert aggregator.f1_score("a") == 0
    assert aggregator.f1_score("b") == 0


def test_reset(aggregator):
    aggregator.reset()
    assert aggregator.accuracy() == 0
    assert aggregator.precision("a") == 0
    assert aggregator.precision("b") == 0
    assert aggregator.recall("a") == 0
    assert aggregator.recall("b") == 0
    assert aggregator.f1_score("a") == 0
    assert aggregator.f1_score("b") == 0
