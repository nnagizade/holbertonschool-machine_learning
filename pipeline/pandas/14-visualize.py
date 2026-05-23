#!/usr/bin/env python3
"""
Task 14: Visualize the DataFrame
"""
from_file = __import__('2-from_file').from_file
import matplotlib.pyplot as plt
import pandas as pd


# Load the data
df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# 1. Remove the Weighted_Price column
df = df.drop(columns=['Weighted_Price'])

# 2. Rename Timestamp to Date and convert to datetime objects
df = df.rename(columns={'Timestamp': 'Date'})
df['Date'] = pd.to_datetime(df['Date'], unit='s')

# 3. Index the dataframe on Date
df = df.set_index('Date')

# 4. Fill missing values
# Close should be set to the previous row value (forward fill)
df['Close'] = df['Close'].ffill()

# High, Low, Open set to the same row's Close value
df['High'] = df['High'].fillna(df['Close'])
df['Low'] = df['Low'].fillna(df['Close'])
df['Open'] = df['Open'].fillna(df['Close'])

# Volume columns set to 0
df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)
df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(0)

# 5. Filter and Resample
# Plot from 2017 and beyond at daily intervals
df = df.loc['2017-01-01':]

# Group values of the same day (Resampling)
# High: max, Low: min, Open: mean, Close: mean, Volumes: sum
df = df.resample('D').agg({
    'High': 'max',
    'Low': 'min',
    'Open': 'mean',
    'Close': 'mean',
    'Volume_(BTC)': 'sum',
    'Volume_(Currency)': 'sum'
})

# Plotting (This part is usually provided in the main script,
# but included here to fulfill the visual aspect)
df.plot()
plt.show()
