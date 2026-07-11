#!/usr/bin/env python3
"""Defines a neural network with one hidden layer performing binary
classification"""
import numpy as np


class NeuralNetwork:
    """Class that defines a neural network with one hidden layer
    performing binary classification"""

    def __init__(self, nx, nodes):
        """Class constructor

        Args:
            nx (int): number of input features
            nodes (int): number of nodes found in the hidden layer
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(nodes) is not int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.W1 = np.random.randn(nodes, nx)
        self.b1 = np.zeros((nodes, 1))
        self.A1 = 0

        self.W2 = np.random.randn(1, nodes)
        self.b2 = 0
        self.A2 = 0
