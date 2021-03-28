#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
mltools.tree
~~~~~~~~~~~~

This module provides functions and classes for implementing decision tree models.
"""
from copy import deepcopy
from math import log2


def normalize(data_list, feature_index):
    """Method for normalizing the feature data values in a list of data entries given the index of the feature value in
    the data entry

    :param data_list: List of data entries where each data entry is a list of feature.
    :type data_list: list
    :param feature_index: Index of feature value in each data entry.
    :type feature_index: int
    :return: None
    """
    feature_value_list = [float(value[feature_index]) for value in data_list]
    maximum = max(feature_value_list)
    minimum = min(feature_value_list)

    for data in data_list:
        data[feature_index] = (float(data[feature_index]) - minimum) / (maximum - minimum)

    return minimum, maximum


def equidistant_discretization(data_list, feature_index, number_bins=2):
    """Method for discretizing raw data based on equidistant bins.

    :param list data_list: List of data entries where each data entry is a list of feature.
    :param int feature_index: Index of feature value in each data entry.
    :param int number_bins: Number of bins to divide data into.
    :return: None
    """
    for data in data_list:
        for x in range(number_bins):
            if x * (1/number_bins) <= data[feature_index] < (x + 1) * (1/number_bins):
                data[feature_index] = x
                break
            if x == number_bins - 1:
                data[feature_index] = x
                break


def equidensity_discretization(data_list, feature_index, number_bins=2):
    """Method for discretizing raw data based on equal density bins.

    :param list data_list: List of data entries where each data entry is a list of feature.
    :param int feature_index: Index of feature value in each data entry.
    :param int number_bins: Number of bins to divide data into.
    :return: None
    """

    bin_size = len(data_list) / number_bins
    count = 0
    bin_number = 0
    for data in sorted(data_list, key=lambda x: x[feature_index]):
        if count < bin_size:
            data[feature_index] = bin_number
        else:
            count = 0
            bin_number += 1
            data[feature_index] = bin_number

        count += 1


class Node(object):
    """Decision tree node class object.

    Node object represent a single node on a decision mltools. The node object has parameters for list of children nodes if
    node has children else the node acts as a leaf and contains a class label.
    """

    def __init__(self, data_list, feature_index_set, splitting_feature_value="", background_frequencies={}, maximum_value=2):
        """Initialization method for class node.

        :param list data_list: List of data values where each data value is a list of feature values.
        :param set feature_index_set: Set of unique feature indexes.
        """
        self.data_list = deepcopy(data_list)
        self.feature_index_set = feature_index_set
        self.splitting_feature_value = splitting_feature_value
        self.children_splitting_feature_index = ""
        self.background_frequencies = background_frequencies
        self.maximum_value = maximum_value

        # CHANGE THIS BASED ON ENTROPY METHOD USED
        self.entropy = self.calc_entropy()
        # self.calc_relative_entropy(self.background_frequencies, self.maximum_value)

        self.children_nodes = []
        self.class_label = None

    def __len__(self):
        """Overrides len() method for class node.
        """
        return len(self.data_list)

    def calc_entropy(self, class_label_index=-1):
        """Method for calculating entropy (shannon entropy) for the node.

            D(x) = -Sum[P(x)log2(P(x))]

        :param int class_label_index: Index of class label value in data point (list).
        :return Entropy of class labels in self.
        """
        feature_value_list = [value[class_label_index] for value in self.data_list]
        feature_value_set = set(feature_value_list)

        entropy = 0
        for feature_value in feature_value_set:
            feature_count = feature_value_list.count(feature_value)
            if feature_count == 0:
                pass
            else:
                entropy += -1 * feature_count / len(feature_value_list) * log2(feature_count / len(feature_value_list))

        return entropy

    def calc_relative_entropy(self, background_frequencies, maximum_value, class_label_index=-1):
        """Method for calculating relative entropy (Kullback-Leibler divergence) for node.

        :param int class_label_index: Index of class label value in data point (list).
        :param dict background_frequencies: Dictionary of background frequencies
        :return Relative entropy of class labels in self.
        """
        feature_value_list = [value[class_label_index] for value in self.data_list]
        feature_value_set = set(feature_value_list)

        entropy = 0
        for feature_value in feature_value_set:
            feature_count = feature_value_list.count(feature_value)
            if feature_count == 0:
                pass
            else:
                entropy += feature_count / len(feature_value_list) * log2(
                    (feature_count / len(feature_value_list)) / (background_frequencies[feature_value])
                )

        return 1 - entropy / maximum_value

    def build_tree(self, depth=0, class_label_index=-1):
        """Method for building decision mltools.

        Method uses modified ID3 algorithm to recursively build a decision mltools.
        """

        # node only contains one class label
        if len({data[class_label_index] for data in self.data_list}) == 1:
            self.class_label = self.data_list[0][class_label_index]

        # no more features to split by
        elif not self.feature_index_set:
            class_labels = [data_point[class_label_index] for data_point in self.data_list]
            self.class_label = max(set(class_labels), key=class_labels.count)

        # continue building mltools
        elif depth != 3:

            # determine all possible splits
            possible_splits = dict()
            for feature_index in self.feature_index_set:
                feature_values = sorted({data_entry[feature_index] for data_entry in self.data_list})
                bins = dict()  # dictionary with unique feature values as keys and data points as values
                copy_data_list = deepcopy(self.data_list)
                for feature_value in feature_values:
                    count = 0
                    while count < len(copy_data_list):
                        if copy_data_list[count][feature_index] == feature_value:
                            bins.setdefault(feature_value, list()).append(copy_data_list.pop(count))
                        else:
                            count += 1

                # add possible split to list
                possible_splits[feature_index] = [Node(
                    bins[k],
                    self.feature_index_set-{feature_index},
                    k,
                    background_frequencies=self.background_frequencies,
                    maximum_value=self.maximum_value
                ) for k in bins]

            # determine optimal feature to split on
            # selects the list of children (...)[1]) with the minimum average entropy (key=...)
            self.children_splitting_feature_index, self.children_nodes = min(
                possible_splits.items(),
                key=lambda x: sum([len(n)*n.entropy/len(self.data_list) for n in x[1]])
            )

            # recursively build mltools
            for child_node in self.children_nodes:
                child_node.build_tree(depth+1, class_label_index)

        # reached maxed depth, assume class label is most common class label present in node
        else:
            class_labels = [data_point[class_label_index] for data_point in self.data_list]
            self.class_label = max(set(class_labels), key=class_labels.count)

    def predict(self, data_value, depth=0):
        """Method to predict class label of a given data point

        :param data_value:
        :return:
        """
        # node is a leaf node and value fits in leaf node, return label
        if self.class_label:
            return self.class_label

        elif self.children_nodes:
            flag = False
            for child in self.children_nodes:
                if data_value[self.children_splitting_feature_index] == child.splitting_feature_value:
                    flag = True
                    return child.predict(data_value, depth+1)

            if not flag:
                return None


