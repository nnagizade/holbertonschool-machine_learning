#!/usr/bin/env python3
"""Trains a model using mini-batch gradient descent."""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                 validation_data=None, early_stopping=False,
                 patience=0, learning_rate_decay=False, alpha=0.1,
                 decay_rate=1, save_best=False, filepath=None,
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
        validation_data: the data to validate the model with, if not
            None
        early_stopping (bool): indicates whether early stopping
            should be used. Early stopping should only be performed
            if validation_data exists. Early stopping should be
            based on validation loss
        patience: the patience used for early stopping
        learning_rate_decay (bool): indicates whether learning rate
            decay should be used. Learning rate decay should only be
            performed if validation_data exists. The decay should be
            performed using inverse time decay. The learning rate
            should decay in a stepwise fashion after each epoch. Each
            time the learning rate updates, Keras should print a
            message
        alpha: the initial learning rate
        decay_rate: the decay rate
        save_best (bool): indicates whether to save the model after
            each epoch if it is the best. A model is considered the
            best if its validation loss is the lowest that the model
            has obtained
        filepath: the file path where the model should be saved
        verbose (bool): determines if output should be printed during
            training
        shuffle (bool): determines whether to shuffle the batches
            every epoch. Normally, it is a good idea to shuffle, but
            for reproducibility, we have chosen to set the default to
            False

    Returns:
        the History object generated after training the model
    """
    callbacks = []

    if learning_rate_decay and validation_data:
        def scheduler(epoch):
            """Updates the learning rate using inverse time decay."""
            return alpha / (1 + decay_rate * epoch)

        lr_decay = K.callbacks.LearningRateScheduler(scheduler,
                                                       verbose=1)
        callbacks.append(lr_decay)

    if early_stopping and validation_data:
        early_stop = K.callbacks.EarlyStopping(monitor='val_loss',
                                                patience=patience)
        callbacks.append(early_stop)

    if save_best and validation_data:
        checkpoint = K.callbacks.ModelCheckpoint(filepath,
                                                  monitor='val_loss',
                                                  save_best_only=True)
        callbacks.append(checkpoint)

    history = network.fit(data, labels,
                           batch_size=batch_size,
                           epochs=epochs,
                           validation_data=validation_data,
                           callbacks=callbacks,
                           verbose=verbose,
                           shuffle=shuffle)

    return history
