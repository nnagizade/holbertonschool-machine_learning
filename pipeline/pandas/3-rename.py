#!/usr/bin/env python3
"""
Task 3: Rename and convert timestamp
"""
import pandas as pd


def rename(df):
    """
    Renames the column Timestamp to Datetime, converts the values
    to datetime format, and displays only Datetime and Close.

    Args:
        df (pd.DataFrame): The dataframe to be modified

    Returns:
        pd.DataFrame: The modified dataframe
    """
    # Rename Timestamp column to Datetime
    df = df.rename(columns={'Timestamp': 'Datetime'})

    # Convert Datetime column values to datetime objects
    df['Datetime'] = pd.to_datetime(df['Datetime'], unit='s')

    # Display (select) only the Datetime and Close columns
    df = df[['Datetime', 'Close']]

    return df
