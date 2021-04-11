#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mltools.fileio
~~~~~~~~~~~~~~~

This module provides functions for parsing data files.
"""

from csv import reader
from numpy import array, genfromtxt


def parse_csv(data_path, headers=False):
    """Method for parsing data from CSV files.

    Returns tuple containing nested list of data values and list of header/feature labels if header=True.

    :param str data_path: Path to CSV data file.
    :param bool headers: Separate row 1 as column headers.
    :return: Tuple containing nested list of data values and list of header/feature labels if header=True.
    :rtype: :py:class:`tuple`
    """
    with open(data_path, "r") as fh:
        data_file = reader(fh, delimiter=",")

        feature_list = None
        data_list = list()
        for count, data_entry in enumerate(data_file):
            if count == 0 and headers:
                feature_list = data_entry
            else:
                data_list.append(data_entry)

    return data_list, feature_list


def parse_csv_2(file_path, label_index=None, headers=False):
    """Improved method for parsing data from CSV files.

    :param str file_path: Path to CSV data file.
    :param label_index: Index of class labels in data array.
    :type label_index: :py:obj:`None` or :py:class:`int`
    :param bool headers: Denotes the presences of value labels (headers) in data array.
    :return Two arrays; x values and y values.
    :rtype: :py:class:`tuple`
    """
    raw_data = genfromtxt(file_path, delimiter=",", skip_header=1 if headers else 0)

    if type(label_index) == int:
        if label_index == 0:
            return raw_data[:, 1:], array([raw_data[:, 0]])
        elif label_index == -1:
            return raw_data[:, :len(raw_data[0]) - 1], array([raw_data[:, -1]])
        else:
            # lazy handling of y values not being at the ends of the array
            raise ValueError("")

    else:
        return raw_data, None
