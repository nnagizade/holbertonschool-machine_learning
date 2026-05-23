#!/usr/bin/env python3
"""
Task 11: Concatenate DataFrames
"""
import pandas as pd
index = __import__('10-index').index


def concat(df1, df2):
    """
    Concatenates two dataframes with specific indexing and labeling rules.

    Args:
        df1: first pd.DataFrame (coinbase)
        df2: second pd.DataFrame (bitstamp)

    Returns:
        The concatenated pd.DataFrame
    """
    # 1. Index both dataframes on their Timestamp columns
    df1 = index(df1)
    df2 = index(df2)

    # 2. Filter df2 to include timestamps up to and including 1417411920
    df2_filtered = df2.loc[:1417411920]

    # 3. Concatenate: df2_filtered on top of df1
    # 4. Add keys 'bitstamp' for df2 and 'coinbase' for df1
    result = pd.concat([df2_filtered, df1], keys=['bitstamp', 'coinbase'])

    return result
