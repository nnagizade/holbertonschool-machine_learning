#!/usr/bin/env python3
"""
Task 10: Set index
"""


def index(df):
    """
    Sets the Timestamp column as the index of the dataframe.

    Args:
        df: pd.DataFrame to be modified

    Returns:
        The modified pd.DataFrame
    """
    # set_index moves the specified column to the index (row labels)
    return df.set_index('Timestamp')
