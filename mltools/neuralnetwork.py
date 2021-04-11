#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
from numpy import array
from random import uniform
import mltools


class NeuralNetwork(object):
    """Feed-Forward Neural Network model class object.
    """

    def __init__(self, X, Y, alpha=0.001, weight_range=(-1, 1), bias_range=(0, 0.5)):
        """Initializer for Neural Network model class.

        """
        self.X = X
        self.Y = Y
        self.alpha = alpha

        # initialize weight and bias matrices
        # for now, defaults to a neural network based on nmist dataset; with 784 input nodes, 128 first-layer,
        # hidden-layer nodes, 64 second-layer, hidden-layer nodes, and 10 output nodes.
        self.h_layer_1_weights = array([[uniform(*weight_range) for x in range(784)] for y in range(128)])
        self.h_layer_2_weights = array([[uniform(*weight_range) for x in range(128)] for y in range(64)])
        self.output_layer_weights = array([[uniform(*weight_range) for x in range(64)] for y in range(10)])

        self.h_layer_1_biases = array([[uniform(*bias_range)] for x in range(self.h_layer_1_weights.shape[0])])
        self.h_layer_2_biases = array([[uniform(*bias_range)] for x in range(self.h_layer_2_weights.shape[0])])
        self.output_layer_biases = array([[uniform(*bias_range)] for x in range(self.output_layer_weights.shape[0])])

    def _calculate(self, input_value):
        """Method for predicting class label from trained model.

        :param input_value:
        :type input_value: :py:class:`~numpy.ndarray`
        """
        outputs = dict()
        outputs["layer_1"] = self.h_layer_1_weights.dot(input_value.T) + self.h_layer_1_biases
        outputs["sig_layer_1"] = mltools.sigmoid(outputs["layer_1"])
        outputs["layer_2"] = self.h_layer_2_weights.dot(outputs["sig_layer_1"]) + self.h_layer_2_biases
        outputs["sig_layer_2"] = mltools.sigmoid(outputs["layer_2"])
        outputs["layer_3"] = self.h_layer_2_weights.dot(outputs["sig_layer_2"]) + self.h_layer_2_biases
        outputs["sig_layer_3"] = mltools.sigmoid(outputs["layer_3"])

        return outputs

    def predict(self, input_value):
        """Method for predicting class label from trained model.

        :param input_value:
        :type input_value: :py:class:`~numpy.ndarray`
        """
        return self._calculate(input_value)["sig_layer_3"]

    def train(self, iterations=100, verbose=False):
        """Method to train Neural Network on a given dataset.

        Updates weight values according to:

            w.j = w.j + alpha * error * g'(in) * x.j

            where:
                in = w.dot(x) + b
                g'(in) = g(in)*(1-g(in))

        Updates biases parameters according to:

            b = b * alpha * error * g'(in)
        """
        # perform iterative training
        for i in range(iterations):

            # perform calculations for each data point
            for data_point in self.X:

            # calculate node outputs for each layer
            layer_outputs = self._calculate(self.X)

            error = array([mltools.mean_squared_error(hypotheses, self.Y[0][count]) for count, hypotheses in enumerate(layer_outputs[2])])

            print(error.shape)
            print(error)

            print(layer_outputs[2].shape)
            exit()

            # update biases
            self.output_layer_biases = self.output_layer_biases * self.alpha * error * layer_outputs[2]

            # update weights
