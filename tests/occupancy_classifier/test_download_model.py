from chesscog.occupancy_classifier.download_model import ensure_model
from chesscog.utils.io import URI


def test_ensure_model():
    ensure_model(show_size=False)
    #assert URI("models://occupancy_classifier.pt").exists()
