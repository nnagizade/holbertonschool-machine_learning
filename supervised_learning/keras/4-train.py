#!/usr/bin/env python3
"""Trains a model using mini-batch gradient descent."""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                 verbose=True, shuffle=False):
    """Trains a model using mini-batch gradient descent.

    Args:
        network: the model to train
        data (numpy.ndarray): shape (m, nx) containing the input data
        labels (numpy.ndarray): one-hot shape (m, classes) with the
            labels of data
        batch_size: the size of the batch used for mini-batch
            gradient descent
        epochs: the number of passes through data for mini-batch
            gradient descent
        verbose (bool): determines if output should be printed during
            training
        shuffle (bool): determines whether to shuffle the batches
            every epoch. Normally, it is a good idea to shuffle, but
            for reproducibility, we have chosen to set the default to
            False

    Returns:
        the History object generated after training the model
    """
    history = network.fit(data, labels,
                           batch_size=batch_size,
                           epochs=epochs,
                           verbose=verbose,
                           shuffle=shuffle)

    return history
