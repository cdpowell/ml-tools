#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Routines for machine learning

This package includes the following modules:

``fileio``
This module provides functions for parsing data files.

``regression``

``tree``
This module provides functions and classes for implementing decision tree models.

"""

from .fileio import parse_csv, parse_csv_2
from .tree import normalize, equidistant_discretization, equidensity_discretization, Node
from .regression import Regression
from .neuralnetwork import NeuralNetwork
from .math import mean_squared_error, sigmoid


__version__ = "0.3.1.alpha"
