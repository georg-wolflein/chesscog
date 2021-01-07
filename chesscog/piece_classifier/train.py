"""Script to train the candidate piece classifiers.

.. code-block:: console

    $ python -m chesscog.piece_classifier.train --help   
    usage: train.py [-h]
                    [--config {AlexNet,ResNet,CNN100_3Conv_3Pool_2FC,InceptionV3,VGG,CNN100_3Conv_3Pool_3FC}]
    
    Train the network.
    
    optional arguments:
      -h, --help            show this help message and exit
      --config {AlexNet,ResNet,CNN100_3Conv_3Pool_2FC,InceptionV3,VGG,CNN100_3Conv_3Pool_3FC}
                            the configuration to train (default: all)
"""

from chesscog.core.training import train_classifier

if __name__ == "__main__":
    train_classifier("piece_classifier")
