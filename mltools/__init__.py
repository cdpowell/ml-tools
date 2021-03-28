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

from .fileio import parse_csv
from .tree import normalize, equidistant_discretization, equidensity_discretization, Node


__version__ = "0.2.0"