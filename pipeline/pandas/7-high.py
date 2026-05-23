#!/usr/bin/env python3
"""
Task 7: Sort by High price
"""


def high(df):
    """
    Sorts the dataframe by the High price in descending order.

    Args:
        df: pd.DataFrame to be sorted

    Returns:
        The sorted pd.DataFrame
    """
    return df.sort_values(by='High', ascending=False)
