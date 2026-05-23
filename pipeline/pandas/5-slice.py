#!/usr/bin/env python3
"""
Task 5: Slice the DataFrame
"""


def slice(df):
    """
    Extracts specific columns and selects every 60th row.

    Args:
        df: pd.DataFrame to be sliced

    Returns:
        The sliced pd.DataFrame
    """
    # Select specific columns and take every 60th row
    # Syntax: df.iloc[start:stop:step]
    return df[['High', 'Low', 'Close', 'Volume_(BTC)']].iloc[::60]
