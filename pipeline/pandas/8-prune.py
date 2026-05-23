#!/usr/bin/env python3
"""
Task 8: Prune NaN values
"""


def prune(df):
    """
    Removes any entries where Close has NaN values.

    Args:
        df: pd.DataFrame to be pruned

    Returns:
        The modified pd.DataFrame
    """
    # dropna removes rows with missing values
    # subset=['Close'] ensures we only look for NaNs in that specific column
    return df.dropna(subset=['Close'])
