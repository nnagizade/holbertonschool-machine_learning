#!/usr/bin/env python3
"""
Module to create a learning rate decay operation in TensorFlow
using inverse time decay
"""
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, decay_step):
    """
    Creates a learning rate decay operation in TensorFlow using inverse time decay

    Args:
        alpha: original learning rate
        decay_rate: weight used to determine the rate at which alpha will decay
        decay_step: number of passes before alpha is decayed further

    Returns:
        The learning rate decay operation
    """
    return tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=alpha,
        decay_steps=decay_step,
        decay_rate=decay_rate,
        staircase=True
    )
