ml-tools
========

.. image:: https://img.shields.io/pypi/l/mwtab.svg
   :target: https://choosealicense.com/licenses/bsd-3-clause-clear/
   :alt: License information

.. image:: https://img.shields.io/github/stars/cdpowell/ml-tools.svg?style=social&label=Star
   :target: https://github.com/cdpowell/ml-tools
   :alt: GitHub project


The ``ml-tools`` package is a Python library containing common tools for machine learning. The package currently
contains modules for implementing Decision Trees and Regression.


Decision Tree
~~~~~~~~~~~~~

The ``ml-tools`` package provides the `mltools.tree.Node()` object used for constructing decision trees. The object
functions off a modified ID3 algorithm to recursively build the tree based off information gain calculated with
Shannon's entropy.

.. image:: https://raw.githubusercontent.com/cdpowell/ml-tools/master/docs/decision_tree/_static/Figure_1.png
   :width: 50%
   :align: center


Regression
~~~~~~~~~~


Installation
~~~~~~~~~~~~

The ``mwtab`` package should run under Python 3.4+. Use pip_ to install the package locally. Starting with Python 3.4, pip_ is included by default.


Install on Linux, Mac OS X
--------------------------

.. code:: bash

   python3 -m pip install -e ./mw-tools


Install on Windows
------------------

.. code:: bash

   py -3 -m pip install -e ./mw-tools


.. _pip: https://pip.pypa.io