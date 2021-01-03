"""Script to evaluate one or more occupancy classifiers.

.. code-block:: console

    $ python -m chesscog.occupancy_classifier.evaluate --help      
    usage: evaluate.py [-h] [--model MODEL]
                       [--dataset {train,val,test}] [--out OUT]
                       [--find-mistakes]
    
    Evaluate trained models.
    
    optional arguments:
      -h, --help            show this help message and exit
      --model MODEL         the model to evaluate (if unspecified,
                            all models in
                            'runs://occupancy_classifier' will be
                            evaluated)
      --dataset {train,val,test}
                            the dataset to evaluate (if unspecified,
                            train and val will be evaluated)
      --out OUT             output folder
      --find-mistakes       whether to output all misclassification
                            images
"""

from chesscog.core.evaluation import perform_evaluation

if __name__ == "__main__":
    perform_evaluation("occupancy_classifier")
