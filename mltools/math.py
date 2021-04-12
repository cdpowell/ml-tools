#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
from numpy import exp


def mean_squared_error(hypothesis, actual, derivative=False, num_values=None):
    """Method for calculating the mean squared error of input value(s).

    """
    num_values = num_values if num_values else len(hypothesis)
    return 1 / num_values * sum((hypothesis - actual) ** 2)


def sigmoid(input_value, derivative=False):
    """Method for calculating the sigmoid of an input value.

    g(in) = 1 / (1 + e^(-in))

    Derivative:
        g'(in) = g(in)*(1-g(in))

    :param input_value: Value(s) for mean squared error to be calculated for.
    :type input_value: :py:class:`float` or :py:class:`int` or
                       :py:class:`~numpy.ndarray`
    :param bool derivative: Denotes that the derivative of the sigmoid function should be calculated.
    :return: Returns the calculated sigmoid value(s).
    :rtype: :py:class:`float` or :py:class:`int` or
            :py:class:`~numpy.ndarray`
    """
    if derivative:
        sig = sigmoid(input_value)
        return sig * (1 - sig)
    return 1 / (1 + exp(input_value))
