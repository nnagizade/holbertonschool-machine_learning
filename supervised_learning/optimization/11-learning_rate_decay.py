#!/usr/bin/env python3
"""
Module to calculate learning rate decay using inverse time decay in numpy
"""


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Updates the learning rate using inverse time decay in numpy in a stepwise fashion

    Args:
        alpha: original learning rate
        decay_rate: weight used to determine the rate at which alpha decays
        global_step: number of passes of gradient descent that have elapsed
        decay_step: number of passes before alpha is decayed further

    Returns:
        The updated value for alpha
    """
    # Calculate stepwise count using integer division
    step = global_step // decay_step

    # Inverse time decay formula: alpha / (1 + decay_rate * step)
    alpha_updated = alpha / (1 + decay_rate * step)

    return alpha_updated
