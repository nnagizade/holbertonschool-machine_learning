#!/usr/bin/env python3
"""
Defines a function that creates a pd.DataFrame from a np.ndarray
"""
import pandas as pd


def from_numpy(array):
    """
    Creates a pd.DataFrame from a np.ndarray
    Each column is labeled alphabetically (A, B, C...)
    """
    # Get the number of columns in the array
    num_cols = array.shape[1]

    # Generate labels: A, B, C... (ASCII 65 is 'A')
    # Using list comprehension to create labels up to 26 columns
    col_names = [chr(65 + i) for i in range(num_cols)]

    # Create and return the DataFrame
    return pd.DataFrame(array, columns=col_names)
