#!/usr/bin/env python3
"""
Module to create a batch normalization layer in TensorFlow
"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer for a neural network in TensorFlow

    Args:
        prev: activated output of the previous layer
        n: number of nodes in the layer to be created
        activation: activation function to be used on the output of the layer

    Returns:
        A tensor of the activated output for the layer
    """
    # Kernel initializer as specified
    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')

    # Base dense layer (unactivated output Z)
    dense_layer = tf.keras.layers.Dense(
        units=n,
        kernel_initializer=init
    )
    Z = dense_layer(prev)

    # Batch normalization layer with gamma=1, beta=0, epsilon=1e-7 initializers
    batch_norm = tf.keras.layers.BatchNormalization(
        gamma_initializer='ones',
        beta_initializer='zeros',
        epsilon=1e-7
    )
    Z_norm = batch_norm(Z)

    # Apply activation function if provided, else return unactivated output
    if activation is not None:
        return activation(Z_norm)

    return Z_norm
