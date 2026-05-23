#!/usr/bin/env python3
"""
Task 4: Convert specific columns and rows to a numpy.ndarray
"""


def array(df):
    """
    Selects the last 10 rows of the High and Close columns
    and converts them into a numpy.ndarray.
    """
    return df[['High', 'Close']].tail(10).to_numpy()
