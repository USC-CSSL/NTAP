""" The Neural Text Analysis Pipeline

NTAP is a high-level interface for text analysis, with a focus on
interpreting the outputs of models for inferential purposes. The main
features of NTAP include:

* Specify models, preprocessing, and feature extraction via
  human-readable formulas (similarly to formulas in R)
* Prioritization of interpretation over prediction through the
  ``analyze`` module
* Integration with a variety of machine learning and Natural Language
  Processing packages, such as pandas, transformers, optuna, and tomotopy
* Easy learning curve for new Python users, in order to increase
  accessibility

We note that NTAP is in an **alpha** stage of development, and that bugs
are surely to emerge as usage of the package increases.

Highest-priority features that are to be added:

* Integration of more interpretability/explainability research for easy
  access in the ``analyze`` module
* Integration of plotting interface (probably matplotlib) for analysis
  and prediction
* Methods for measuring and mitigating unfair and biased models

"""

name = "ntap"

import ntap.parse 
import ntap.supervised 
import ntap.bagofwords 
import ntap.dic
import ntap.embed

