#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
from numpy import array, dot
from random import uniform
import mltools


class NeuralNetwork(object):
    """Feed-Forward Neural Network model class object.
    """

    def __init__(self, X, Y, alpha=0.001, weight_range=(-1, 1), bias_range=(-1, 1)):
        """Initializer for Neural Network model class.

        """
        self.X = X
        self.Y = Y
        self.alpha = alpha

        # initialize weight and bias matrices
        # for now, defaults to a neural network based on nmist dataset; with 784 input nodes, 128 first-layer,
        # hidden-layer nodes, 64 second-layer, hidden-layer nodes, and 10 output nodes.
        self.weights = dict()
        self.weights["layer_1"] = array([[uniform(*weight_range) for x in range(784)] for y in range(128)])
        self.weights["layer_2"] = array([[uniform(*weight_range) for x in range(128)] for y in range(64)])
        self.weights["layer_3"] = array([[uniform(*weight_range) for x in range(64)] for y in range(10)])
        # self.weights["layer_4"] = array([[uniform(*weight_range) for x in range(10)] for y in range(1)])

        self.bias = dict()
        self.bias["layer_1"] = array([[uniform(*bias_range)] for x in range(self.weights["layer_1"].shape[0])])
        self.bias["layer_2"] = array([[uniform(*bias_range)] for x in range(self.weights["layer_2"].shape[0])])
        self.bias["layer_3"] = array([[uniform(*bias_range)] for x in range(self.weights["layer_3"].shape[0])])
        # self.bias["layer_4"] = array([[uniform(*bias_range)] for x in range(self.weights["layer_3"].shape[0])])

    def _calculate(self, input_value):
        """Method for predicting class label from trained model.

        in = W dot x
        a = g(in)

        :param input_value:
        :type input_value: :py:class:`~numpy.ndarray`
        """
        outputs = dict()
        outputs["layer_1"] = dot(self.weights["layer_1"], input_value.T) + self.bias["layer_1"]
        outputs["sig_layer_1"] = mltools.sigmoid(outputs["layer_1"])
        outputs["layer_2"] = self.weights["layer_2"].dot(outputs["sig_layer_1"]) + self.bias["layer_2"]
        outputs["sig_layer_2"] = mltools.sigmoid(outputs["layer_2"])
        outputs["layer_3"] = self.weights["layer_3"].dot(outputs["sig_layer_2"]) + self.bias["layer_3"]
        outputs["sig_layer_3"] = mltools.sigmoid(outputs["layer_3"])
        outputs["layer_4"] = self.weights["layer_4"].dot(outputs["sig_layer_3"]) + self.bias["layer_4"]
        outputs["sig_layer_4"] = mltools.sigmoid(outputs["layer_4"])

        return outputs

    def predict(self, input_value):
        """Method for predicting class label from trained model.

        :param input_value:
        :type input_value: :py:class:`~numpy.ndarray`
        """
        return self._calculate(input_value)["sig_layer_4"]

    def train(self, iterations=25, verbose=False):
        """Method to train Neural Network on a given dataset.

        Updates weight values according to:

            w.j = w.j + alpha * error * g'(in) * x.j

            where:
                in = w.dot(x) + b
                g'(in) = g(in)*(1-g(in))
                error = actual - hypothesis

        Updates biases parameters according to:

            b = b * alpha * error * g'(in)
        """
        # perform iterative training
        for i in range(iterations):

            if verbose:
                print("Epoch {}".format(i + 1))

            # perform calculations for each data point
            for count, data_point in enumerate(self.X):

                # calculate node outputs for each layer
                layer_outputs = self._calculate(array([data_point]))

                # perform bias and weight updates
                updates = dict()
                delta = (self.Y[0][count] - layer_outputs["sig_layer_4"]) * mltools.sigmoid(layer_outputs["layer_4"], True)
                updates["l4b"] = self.bias["layer_4"] + self.alpha * delta
                updates["l4w"] = self.weights["layer_4"] + self.alpha * layer_outputs["sig_layer_4"] * delta

                delta = self.weights["layer_4"].dot(delta) * mltools.sigmoid(layer_outputs["layer_3"], True)
                updates["l3b"] = self.bias["layer_3"] + self.alpha * delta
                updates["l3w"] = self.weights["layer_3"] + self.alpha * layer_outputs["sig_layer_3"] * delta

                delta = self.weights["layer_3"].T.dot(delta) * mltools.sigmoid(layer_outputs["layer_2"], True)
                updates["l2b"] = self.bias["layer_2"] + self.alpha * delta
                updates["l2w"] = self.weights["layer_2"] + self.alpha * layer_outputs["sig_layer_2"] * delta

                delta = self.weights["layer_2"].T.dot(delta) * mltools.sigmoid(layer_outputs["layer_1"], True)
                updates["l1b"] = self.bias["layer_1"] + self.alpha * delta
                updates["l1w"] = self.weights["layer_1"] + self.alpha * layer_outputs["sig_layer_1"] * delta

                self.bias["layer_4"] = updates["l4b"]
                self.weights["layer_4"] = updates["l4w"]
                self.bias["layer_3"] = updates["l3b"]
                self.weights["layer_3"] = updates["l3w"]
                self.bias["layer_2"] = updates["l2b"]
                self.weights["layer_2"] = updates["l2w"]
                self.bias["layer_1"] = updates["l1b"]
                self.weights["layer_1"] = updates["l1w"]
