#!/usr/bin/env python3
"""
preprocess_data.py

Cleans and transforms the raw minute-level BTC dataset
(mczielinski/bitcoin-historical-data on Kaggle) into a form that
forecast_btc.py can train an RNN on.

RAW FORMAT (one row per 60-second window, from 2012 onward)
------------------------------------------------------------
    Timestamp, Open, High, Low, Close, Volume

Note: this is the current (single-file) version of the dataset. Earlier
versions of this Holberton project were written against a two-file
Coinbase + Bitstamp release with extra columns (Volume_(BTC),
Volume_(Currency), Weighted_Price); this script targets the schema
actually shipped today.

DESIGN DECISIONS (answers to the questions the project prompt asks)
---------------------------------------------------------------------
1. "Are all of the data points useful?"
   No. The dataset starts in January 2012, when Bitcoin traded for a few
   dollars and barely anyone traded it at all -- huge stretches of
   minutes have Volume == 0 (no trade happened; the row just repeats the
   last price). Those zero-volume minutes carry no real information
   about market dynamics, so after resampling, hours with zero total
   volume are dropped rather than treated as valid observations. A
   configurable `--start-date` also lets us discard the whole
   thin-liquidity 2012-2016 era outright.

2. "Are all of the data features useful?"
   Not equally. Open/High/Low/Close are all highly correlated measures
   of "the price" within the hour. We keep Close (the value we
   ultimately predict) and derive one additional price-shape feature,
   the hourly range as a fraction of price ((High - Low) / Close), which
   captures intra-hour volatility that Close alone would miss. Volume is
   kept as-is (summed per hour) since it's a genuinely independent
   signal of trading activity. Open/High/Low are otherwise dropped once
   the range feature is computed.

3. "Should you rescale the data?"
   Yes. Close ranges from single dollars (2012) to tens of thousands of
   dollars (recent years), and Volume lives on yet another scale
   entirely -- without scaling, the loss would be dominated by whichever
   feature has the largest raw magnitude. Volume is heavily right-skewed
   (occasional huge-volume hours), so it's log1p-transformed before
   scaling to tame outliers. Every feature is then min-max scaled to
   [0, 1] using statistics computed ONLY on the training split, so
   nothing about the future leaks into training.

4. "Is the current time window relevant?"
   Two things change here:
     a. Minute-level granularity would require a 1,440-step RNN to cover
        24 hours -- unwieldy and slow to train, and noisier than
        necessary. Data is resampled to hourly candles (Open=first,
        High=max, Low=min, Close=last, Volume=sum), matching the task's
        own framing that an hour is "approximately how long the average
        transaction takes."
     b. Very old data reflects a market regime (liquidity, participants,
        exchange maturity) that no longer resembles today's BTC market,
        so `--start-date` (default 2019-01-01) discards it.

5. "How should you save this preprocessed data?"
   As a single compact `.npz` (NumPy zipped archive) containing the
   scaled train/valid/test feature matrices, the raw (unscaled) close
   prices for each split (for later error-metric reporting in original
   USD), and the scaler's min/max stats. This avoids ever re-parsing the
   multi-gigabyte raw CSV again.
"""
import argparse
import numpy as np
import pandas as pd


FEATURES = ["Close", "Range_Pct", "Volume"]
CLOSE_IDX = FEATURES.index("Close")


def load_raw(path):
    """Load the raw minute-level CSV, indexed by UTC datetime."""
    df = pd.read_csv(path)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="s", utc=True)
    df = df.set_index("Timestamp").sort_index()
    return df


def resample_hourly(df):
    """Collapse 60-second rows into 1-hour OHLCV candles."""
    agg = df.resample("1h").agg(
        {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        }
    )
    return agg


def clean(df):
    """Drop hours with no real trading activity, forward-fill small gaps."""
    df = df.dropna(how="all")
    df = df.ffill()
    df = df.dropna()
    df = df[df["Volume"] > 0]  # zero-volume hours carry no real signal
    return df


def add_features(df):
    """Derive Range_Pct and drop the now-redundant Open/High/Low."""
    df = df.copy()
    df["Range_Pct"] = (df["High"] - df["Low"]) / df["Close"]
    df["Volume"] = np.log1p(df["Volume"])  # tame the heavy right skew
    return df[["Close", "Range_Pct", "Volume"]]


def chronological_split(df, train_frac=0.7, valid_frac=0.15):
    n = len(df)
    n_train = int(n * train_frac)
    n_valid = int(n * valid_frac)
    train = df.iloc[:n_train]
    valid = df.iloc[n_train:n_train + n_valid]
    test = df.iloc[n_train + n_valid:]
    return train, valid, test


def fit_minmax(train_arr):
    mins = train_arr.min(axis=0)
    maxs = train_arr.max(axis=0)
    ranges = np.where(maxs - mins == 0, 1.0, maxs - mins)
    return mins, ranges


def apply_minmax(arr, mins, ranges):
    return (arr - mins) / ranges


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="btcusd_1-min_data.csv",
                         help="Path to the raw minute-level BTC CSV")
    parser.add_argument("--start-date", default="2019-01-01",
                         help="Discard data before this date (YYYY-MM-DD)")
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--valid-frac",
