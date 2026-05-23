#!/usr/bin/env python3
"""
Task 12: Hierarchy
"""
import pandas as pd
index = __import__('10-index').index


def hierarchy(df1, df2):
    """
    Concatenates two dataframes with a MultiIndex where Timestamp
    is the first level and source (bitstamp/coinbase) is the second.

    Args:
        df1: first pd.DataFrame (coinbase)
        df2: second pd.DataFrame (bitstamp)

    Returns:
        The concatenated and rearranged pd.DataFrame
    """
    # 1. Index both dataframes on their Timestamp columns using Task 10's func
    df1 = index(df1)
    df2 = index(df2)

    # 2. Filter both to the specific timestamp range: 1417411980 to 1417417980
    # Use .loc for inclusive slicing on the index
    df1_filtered = df1.loc[1417411980:1417417980]
    df2_filtered = df2.loc[1417411980:1417417980]

    # 3. Concatenate with keys to create the initial MultiIndex
    # Level 0 will be bitstamp/coinbase, Level 1 will be Timestamp
    df = pd.concat([df2_filtered, df1_filtered], keys=['bitstamp', 'coinbase'])

    # 4. Rearrange MultiIndex so Timestamp is Level 0
    # swaplevel(0, 1) flips the levels
    df = df.swaplevel(0, 1)

    # 5. Ensure chronological order by sorting the index
    df = df.sort_index()

    return df
