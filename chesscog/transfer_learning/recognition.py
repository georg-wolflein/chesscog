"""Script to perform a single inference using the fine-tuned system on the new dataset.

.. code-block:: console

    $ python -m chesscog.transfer_learning.recognition --help
    usage: recognition.py [-h] [--white] [--black] file
    
    Run the chess recognition pipeline on an input image
    
    positional arguments:
      file        path to the input image
    
    optional arguments:
      -h, --help  show this help message and exit
      --white     indicate that the image is from the white player's
                  perspective (default)
      --black     indicate that the image is from the black player's
                  perspective
"""

from recap import URI
import functools

from chesscog.recognition.recognition import main

if __name__ == "__main__":
    from chesscog.transfer_learning.download_models import ensure_models

    main(URI("models://transfer_learning"),
         setup=functools.partial(ensure_models, show_size=True))
