import pytest
from PIL import Image

from chesscog.core.dataset.transforms import Shear, Translate, Scale


@pytest.mark.parametrize("transform", [Shear(1.),
                                       Translate(.5, .5),
                                       Scale(.5, .5)])
def test_transform_retains_size(transform):
    img = Image.new("RGB", (50, 100))
    img = transform(img)
    assert img.size == (50, 100)
