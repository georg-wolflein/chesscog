"""Script to train the candidate occupancy classifiers.

.. code-block:: console

    $ python -m chesscog.occupancy_classifier.train --help   
    usage: train.py [-h]
                    [--config {AlexNet,ResNet,VGG_centercrop,AlexNet_centercrop,CNN50_3Conv_1Pool_2FC,CNN100_3Conv_3Pool_2FC,CNN50_2Conv_2Pool_2FC,VGG,CNN100_3Conv_3Pool_3FC_centercrop,CNN50_3Conv_1Pool_3FC_centercrop,CNN50_2Conv_2Pool_2FC_centercrop,CNN100_3Conv_3Pool_2FC_centercrop,CNN50_3Conv_1Pool_2FC_centercrop,CNN50_2Conv_2Pool_3FC_centercrop,ResNet_centercrop,CNN50_3Conv_1Pool_3FC,CNN50_2Conv_2Pool_3FC,CNN100_3Conv_3Pool_3FC}]
    
    Train the network.
    
    optional arguments:
      -h, --help            show this help message and exit
      --config {AlexNet,ResNet,VGG_centercrop,AlexNet_centercrop,CNN50_3Conv_1Pool_2FC,CNN100_3Conv_3Pool_2FC,CNN50_2Conv_2Pool_2FC,VGG,CNN100_3Conv_3Pool_3FC_centercrop,CNN50_3Conv_1Pool_3FC_centercrop,CNN50_2Conv_2Pool_2FC_centercrop,CNN100_3Conv_3Pool_2FC_centercrop,CNN50_3Conv_1Pool_2FC_centercrop,CNN50_2Conv_2Pool_3FC_centercrop,ResNet_centercrop,CNN50_3Conv_1Pool_3FC,CNN50_2Conv_2Pool_3FC,CNN100_3Conv_3Pool_3FC}
                            the configuration to train (default: all)
"""

from chesscog.core.training import train_classifier

if __name__ == "__main__":
    train_classifier("occupancy_classifier")
