.. :changelog:

Release History
===============


0.3.1.alpha (2021-04-11)
~~~~~~~~~~~~~~~~~~~~~~~~

**Improvements**

- Begins to add ``math`` module to ``mltools`` package.
    - Module contains a number of methods for mathematical operations relevant to machine learning.
        - Adds ``sigmoid()`` function for calculating the sigmoid of an input value.
        - Adds ``mean_squared_error()`` function for calculating the error between hypothesis values and the actual value.


0.3.0.alpha (2021-04-10)
~~~~~~~~~~~~~~~~~~~~~~~~

**Improvements**

- Begins adding Neural Network model to package in new ``mltools.neuralnetwork.py`` module.
    - Makes ``NeuralNetwork`` class importable through `mltools` package.
        - i.e. ``from mltools import NeuralNetwork``
- Adds improved csv parsing method ``parse_csv_2()`` in ``mltools.fileio.py`` module.
    - Makes ``parse_csv_2()`` function importable through `mltools` package.
        - i.e. ``from mltools import parse_csv_2``
