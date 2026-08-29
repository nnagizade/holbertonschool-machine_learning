#!/usr/bin/env python3
"""Initializes the Q-table"""
import numpy as np


def q_init(env):
    """Initializes the Q-table

    Args:
        env: the FrozenLakeEnv instance

    Returns:
        the Q-table as a numpy.ndarray of zeros
    """
    state_space = env.observation_space.n
    action_space = env.action_space.n

    return np.zeros((state_space, action_space))
