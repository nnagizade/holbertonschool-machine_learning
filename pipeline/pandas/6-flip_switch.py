#!/usr/bin/env python3
"""
Task 6: Flip and Switch
"""


def flip_switch(df):
    """
    Sorts the data in reverse chronological order and transposes it.

    Args:
        df: pd.DataFrame to be transformed

    Returns:
        The transformed pd.DataFrame
    """
    # Sort in reverse chronological order (flipping the rows)
    # Then transpose the entire dataframe (switching axes)
    return df.sort_index(ascending=False).transpose()
