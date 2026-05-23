#!/usr/bin/env python3
"""
Task 9: Fill missing values
"""


def fill(df):
    """
    Fills missing values in the dataframe according to specific rules.

    Args:
        df: pd.DataFrame to be modified

    Returns:
        The modified pd.DataFrame
    """
    # 1. Remove the Weighted_Price column
    df = df.drop(columns=['Weighted_Price'])

    # 2. Fill missing values in Close with the previous row's value
    # ffill is shorthand for 'forward fill'
    df['Close'] = df['Close'].ffill()

    # 3. Fill missing High, Low, Open with the Close value in the same row
    # We use fillna and pass the Close series as the value provider
    df['High'] = df['High'].fillna(df['Close'])
    df['Low'] = df['Low'].fillna(df['Close'])
    df['Open'] = df['Open'].fillna(df['Close'])

    # 4. Set missing values in Volume columns to 0
    df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)
    df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(0)

    return df
