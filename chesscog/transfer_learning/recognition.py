from recap import URI
import functools

from chesscog.recognition.recognition import main

if __name__ == "__main__":
    from chesscog.transfer_learning.download_models import ensure_models

    main(URI("models://transfer_learning"),
         setup=functools.partial(ensure_models, show_size=True))
