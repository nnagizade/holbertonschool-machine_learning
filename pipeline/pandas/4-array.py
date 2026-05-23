#!/usr/bin/env python3
"""
Task 4: Convert specific columns and rows to a numpy.ndarray
"""
import pandas as pd


def array(df):
    """
    Selects the last 10 rows of the High and Close columns
    and converts them into a numpy.ndarray.

    Args:
        df (pd.DataFrame): The dataframe to process

    Returns:
        numpy.ndarray: The selected data as an array
    """
    # Select only High and Close, then take the last 10 rows
    # Finally, convert to a numpy array
    return df[['High', 'Close']].tail(10).to_numpy()
