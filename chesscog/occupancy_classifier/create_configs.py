"""Script to generate the YAML configuration files for each of the candidate occupancy classification model architectures.
The candidates are defined in the :mod:`~chesscog.occupancy_classifier.models` module.

.. code-block:: console

    $ python -m chesscog.occupancy_classifier.create_configs --help
    usage: create_configs.py [-h]
    
    Generate the YAML configuration for the occupancy
    classifiers.
    
    optional arguments:
      -h, --help  show this help message and exit
"""

from chesscog.core.training import create_configs
import argparse

if __name__ == "__main__":
    argparse.ArgumentParser(
        description="Generate the YAML configuration for the occupancy classifiers.").parse_args()
    create_configs("occupancy_classifier", include_centercrop=True)
