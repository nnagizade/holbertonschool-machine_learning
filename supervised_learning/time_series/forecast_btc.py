#!/usr/bin/env python3
"""
forecast_btc.py

Creates, trains, and validates a Keras RNN that forecasts the BTC close
price one hour ahead, given the previous 24 hours of (Close, Range_Pct,
Volume) data.

Expects preprocessed_data.npz produced by preprocess_data.py.

Architecture
------------
A single LSTM layer is a natural fit here: the model only needs to learn
short-range temporal dependencies over a 24-step window, and LSTM's gating
mechanism handles that without the vanishing-gradient issues a plain
SimpleRNN would run into even at this modest sequence length.

    Input(24, num_features)
      -> LSTM(64)
      -> Dropout(0.2)
      -> Dense(32, relu)
      -> Dense(1)                (predicted, scaled next-hour Close)

Loss: mean squared error (MSE), as required by the task.
Data pipeline: tf.keras.utils.timeseries_dataset_from_array, which is a
thin, well-tested wrapper that builds and returns a tf.data.Dataset of
(window, target) pairs -- satisfying the "use a tf.data.Dataset to feed
data to your model" requirement without hand-rolling window slicing.
"""
import argparse
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


WINDOW_SIZE = 24   # past 24 hours
BATCH_SIZE = 64


def make_dataset(arr, close_idx, window_size=WINDOW_SIZE, batch_size=BATCH_SIZE,
                  shuffle=False):
    """
    Build a tf.data.Dataset of (24-hour window, next-hour close) pairs
    from a (T, num_features) array of already-scaled data.
    """
    features = arr[:-1]                       # every step can start a window
    targets = arr[window_size:, close_idx]    # close price 1 step after each window

    dataset = tf.keras.utils.timeseries_dataset_from_array(
        data=features,
        targets=targets,
        sequence_length=window_size,
        sequence_stride=1,
        shuffle=shuffle,
        batch_size=batch_size,
    )
    return dataset


def build_model(window_size, num_features):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(window_size, num_features)),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss="mse", metrics=["mae"])
    return model


def unscale_close(scaled_close, mins, ranges, close_idx):
    return scaled_close * ranges[close_idx] + mins[close_idx]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="preprocessed_data.npz")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--model-out", default="btc_forecast_model.keras")
    parser.add_argument("--plot-out", default="forecast_vs_actual.png")
    parser.add_argument("--history-plot-out", default="training_history.png")
    args = parser.parse_args()

    print("Loading preprocessed data...")
    data = np.load(args.data, allow_pickle=True)
    train_arr, valid_arr, test_arr = data["train"], data["valid"], data["test"]
    mins, ranges = data["scaler_min"], data["scaler_range"]
    close_idx = int(data["close_idx"])
    num_features = train_arr.shape[1]

    train_ds = make_dataset(train_arr, close_idx, shuffle=True)
    valid_ds = make_dataset(valid_arr, close_idx)
    test_ds = make_dataset(test_arr, close_idx)

    model = build_model(WINDOW_SIZE, num_features)
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
    ]

    print("Training...")
    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    # Chart 1: training vs validation loss over epochs
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Training loss (MSE)")
    plt.plot(history.history["val_loss"], label="Validation loss (MSE)")
    plt.title("Model training history")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (scaled MSE)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.history_plot_out)
    print(f"Training history plot saved to {args.history_plot_out}")

    print("Evaluating on the test set...")
    test_loss, test_mae = model.evaluate(test_ds)
    print(f"Test MSE (scaled): {test_loss:.6f}  Test MAE (scaled): {test_mae:.6f}")

    # Predict over the test set and report error in real USD terms too.
    y_true_scaled, y_pred_scaled = [], []
    for x_batch, y_batch in test_ds:
        preds = model.predict(x_batch, verbose=0).flatten()
        y_true_scaled.extend(y_batch.numpy())
        y_pred_scaled.extend(preds)

    y_true_usd = unscale_close(np.array(y_true_scaled), mins, ranges, close_idx)
    y_pred_usd = unscale_close(np.array(y_pred_scaled), mins, ranges, close_idx)
    rmse_usd = np.sqrt(np.mean((y_true_usd - y_pred_usd) ** 2))
    mae_usd = np.mean(np.abs(y_true_usd - y_pred_usd))
    print(f"Test RMSE: ${rmse_usd:,.2f}   Test MAE: ${mae_usd:,.2f}")

    model.save(args.model_out)
    print(f"Model saved to {args.model_out}")

    # Chart 2: predicted vs. actual close price on the test set
    plt.figure(figsize=(12, 5))
    plt.plot(y_true_usd, label="Actual close")
    plt.plot(y_pred_usd, label="Predicted close")
    plt.title("BTC next-hour close: actual vs. predicted (test set)")
    plt.xlabel("Hour (test set index)")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.plot_out)
    print(f"Plot saved to {args.plot_out}")


if __name__ == "__main__":
    main()
