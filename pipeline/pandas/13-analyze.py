#!/usr/bin/env python3
"""
Task 13: Analyze the DataFrame
"""


def analyze(df):
    """
    Computes descriptive statistics for all columns except Timestamp.

    Args:
        df: pd.DataFrame to be analyzed

    Returns:
        A new pd.DataFrame containing the statistics
    """
    # 1. Drop the Timestamp column (axis=1 means column)
    # 2. Call describe() to get count, mean, std, min, 25%, 50%, 75%, max
    return df.drop(columns=['Timestamp']).describe()
