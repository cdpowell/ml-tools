#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
mltools.fileio
~~~~~~~~~~~~~~~

This module provides functions for parsing data files.
"""

from csv import reader


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
