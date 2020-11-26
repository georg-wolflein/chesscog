import pytest
import typing
from recap import URI

from chesscog.occupancy_classifier.download_model import ensure_model as ensure_occupancy_classifier
from chesscog.piece_classifier.download_model import ensure_model as ensure_piece_classifier


@pytest.mark.parametrize("ensure_model,name", [
    (ensure_occupancy_classifier, "occupancy_classifier"),
    (ensure_piece_classifier, "piece_classifier")
])
def test_ensure_model(ensure_model: typing.Callable, name: str):
    ensure_model(show_size=False)
    assert len(list(URI(f"models://{name}").glob("*.pt"))) > 0
