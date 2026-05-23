#!/usr/bin/env python3
"""
Task 2: Load data from a file
"""
import pandas as pd


def from_file(filename, delimiter):
    """
    Loads data from a file as a pd.DataFrame

    Args:
        filename (str): the path to the file to load
        delimiter (str): the column separator

    Returns:
        pd.DataFrame: the loaded dataframe
    """
    return pd.read_csv(filename, sep=delimiter)
