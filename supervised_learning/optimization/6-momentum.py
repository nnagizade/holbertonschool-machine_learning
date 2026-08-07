#!/usr/bin/env python3
"""
Module to set up gradient descent with momentum optimization in TensorFlow
"""
import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """
    Sets up the gradient descent with momentum optimization in TensorFlow

    Args:
        alpha: learning rate
        beta1: momentum weight

    Returns:
        The optimizer
    """
    return tf.keras.optimizers.SGD(learning_rate=alpha, momentum=beta1)
