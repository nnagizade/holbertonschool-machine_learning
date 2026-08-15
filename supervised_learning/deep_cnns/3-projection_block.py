#!/usr/bin/env python3
"""Module that builds a projection block as described in
Deep Residual Learning for Image Recognition (2015)"""
from tensorflow import keras as K


def projection_block(A_prev, filters, s=2):
    """
    Builds a projection block as described in Deep Residual
    Learning for Image Recognition (2015)

    Args:
        A_prev: output from the previous layer
        filters: tuple or list containing F11, F3, F12, respectively:
            F11 is the number of filters in the first 1x1 convolution
            F3 is the number of filters in the 3x3 convolution
            F12 is the number of filters in the second 1x1
                convolution as well as the 1x1 convolution in the
                shortcut connection
        s: the stride of the first convolution in both the main
           path and the shortcut connection

    Returns:
        The activated output of the projection block
    """
    F11, F3, F12 = filters
    init = K.initializers.he_normal(seed=0)

    conv1 = K.layers.Conv2D(filters=F11, kernel_size=(1, 1),
                             strides=(s, s), padding='same',
                             kernel_initializer=init)(A_prev)
    bn1 = K.layers.BatchNormalization(axis=3)(conv1)
    act1 = K.layers.Activation('relu')(bn1)

    conv2 = K.layers.Conv2D(filters=F3, kernel_size=(3, 3),
                             padding='same',
                             kernel_initializer=init)(act1)
    bn2 = K.layers.BatchNormalization(axis=3)(conv2)
    act2 = K.layers.Activation('relu')(bn2)

    conv3 = K.layers.Conv2D(filters=F12, kernel_size=(1, 1),
                             padding='same',
                             kernel_initializer=init)(act2)
    bn3 = K.layers.BatchNormalization(axis=3)(conv3)

    shortcut = K.layers.Conv2D(filters=F12, kernel_size=(1, 1),
                                strides=(s, s), padding='same',
                                kernel_initializer=init)(A_prev)
    bn_shortcut = K.layers.BatchNormalization(axis=3)(shortcut)

    add = K.layers.Add()([bn3, bn_shortcut])
    output = K.layers.Activation('relu')(add)

    return output
