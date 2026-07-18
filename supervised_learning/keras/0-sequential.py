#!/usr/bin/env python3
"""Builds a neural network with the Keras library."""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """Builds a neural network with the Keras library.

    Args:
        nx (int): number of input features to the network
        layers (list): number of nodes in each layer of the network
        activations (list): activation functions used for each layer
        lambtha (float): L2 regularization parameter
        keep_prob (float): probability that a node will be kept for dropout

    Returns:
        the keras model
    """
    model = K.Sequential()
    regularizer = K.regularizers.l2(lambtha)

    for i in range(len(layers)):
        if i == 0:
            model.add(K.layers.Dense(
                layers[i],
                activation=activations[i],
                kernel_regularizer=regularizer,
                input_shape=(nx,)
            ))
        else:
            model.add(K.layers.Dense(
                layers[i],
                activation=activations[i],
                kernel_regularizer=regularizer
            ))

        if i != len(layers) - 1:
            model.add(K.layers.Dropout(1 - keep_prob))

    return model
