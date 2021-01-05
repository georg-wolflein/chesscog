import numpy as np
import torch

from chesscog.core.statistics import StatsAggregator


def test_stats_aggregator():
    agg = StatsAggregator(["a", "b"])
    a_output = np.array([.9, .1, .8, .2, .9, .9, .9, .2])
    b_output = 1 - a_output
    outputs = torch.tensor(np.stack([a_output, b_output], axis=-1))
    labels = torch.tensor([0, 0, 0, 1, 0, 0, 1, 0])
    agg.add_batch(outputs, labels)
    assert np.isclose(agg.accuracy(), 5/8)
