#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
from numpy import array, exp
from random import uniform


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
        # for now, defaults to a neural network with 784 input nodes (based on nmist dataset), 128 first-layer,
        # hidden-layer nodes, 64 second-layer, hidden-layer nodes, and 10 output nodes.
        self.h_layer_1_weights = array([[uniform(*weight_range) for x in range(784)] for y in range(128)])
        self.h_layer_2_weights = array([[uniform(*weight_range) for x in range(128)] for y in range(64)])
        self.output_layer_weights = array([[uniform(*weight_range) for x in range(64)] for y in range(10)])

        self.h_layer_1_biases = array([[uniform(*bias_range)] for x in range(self.h_layer_1_weights.shape[0])])
        self.h_layer_2_biases = array([[uniform(*bias_range)] for x in range(self.h_layer_2_weights.shape[0])])
        self.output_layer_biases = array([[uniform(*bias_range)] for x in range(self.output_layer_weights.shape[0])])

        # print(self.h_layer_1_weights[0].dot(self.X[0]))
        # print(self.h_layer_1_weights[1].dot(self.X[1]))
        # blah = self.h_layer_1_weights.dot(self.X.T) + self.h_layer_1_biases
        # print(len(blah[0]))

    @staticmethod
    def sigmoid(input_value):
        """Activation function for Neural Network model based on sigmoid equation.

        g(in) = 1 / (1 + e^(-in))
        """
        return 1 / (1 + exp(input_value))

    def predict(self, input_value):
        """Method for predicting class label from trained model.

        :param input_value:
        :type input_value: :py:class:`~numpy.ndarray`
        """
        layer_1_output = self.sigmoid(self.h_layer_1_weights.dot(input_value.T) + self.h_layer_1_biases)
        layer_2_output = self.sigmoid(self.h_layer_2_weights.dot(layer_1_output) + self.h_layer_2_biases)
        output = self.sigmoid(self.output_layer_weights.dot(layer_2_output) + self.output_layer_biases)

        return output

    def train(self, iterations=1000, verbose=False):
        """Method to train Neural Network on a given dataset.

        Updates weight values according to:

        w.j = w.j + alpha * error * g'(in) * x.j

        where:
            g'(in) = g(in)*(1-g(in))

        """
        for i in range(iterations):
            pass
