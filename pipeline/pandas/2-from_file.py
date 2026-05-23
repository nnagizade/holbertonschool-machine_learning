#!/usr/bin/env python3
"""
Module containing the function from_numpy
"""
import pandas as pd


def from_numpy(array):
    """
    Creates a pd.DataFrame from a np.ndarray with alphabetical column labels
    """
    # Get the number of columns in the ndarray
    cols = array.shape[1]

    # Generate a list of uppercase letters for the column names
    # chr(65) is 'A', chr(66) is 'B', etc.
    column_names = [chr(i) for i in range(65, 65 + cols)]

    # Create the DataFrame using the array and the generated names
    df = pd.DataFrame(array, columns=column_names)

    return df
