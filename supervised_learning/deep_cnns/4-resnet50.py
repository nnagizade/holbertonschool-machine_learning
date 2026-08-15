#!/usr/bin/env python3
"""Module that builds the ResNet-50 architecture as described in
Deep Residual Learning for Image Recognition (2015)"""
from tensorflow import keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """
    Builds the ResNet-50 architecture as described in Deep
    Residual Learning for Image Recognition (2015)

    Returns:
        The keras model
    """
    init = K.initializers.he_normal(seed=0)
    X = K.Input(shape=(224, 224, 3))

    conv1 = K.layers.Conv2D(filters=64, kernel_size=(7, 7),
                             strides=(2, 2), padding='same',
                             kernel_initializer=init)(X)
    bn1 = K.layers.BatchNormalization(axis=3)(conv1)
    act1 = K.layers.Activation('relu')(bn1)
    pool1 = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                                   padding='same')(act1)

    proj2 = projection_block(pool1, [64, 64, 256], s=1)
    id2a = identity_block(proj2, [64, 64, 256])
    id2b = identity_block(id2a, [64, 64, 256])

    proj3 = projection_block(id2b, [128, 128, 512], s=2)
    id3a = identity_block(proj3, [128, 128, 512])
    id3b = identity_block(id3a, [128, 128, 512])
    id3c = identity_block(id3b, [128, 128, 512])

    proj4 = projection_block(id3c, [256, 256, 1024], s=2)
    id4a = identity_block(proj4, [256, 256, 1024])
    id4b = identity_block(id4a, [256, 256, 1024])
    id4c = identity_block(id4b, [256, 256, 1024])
    id4d = identity_block(id4c, [256, 256, 1024])
    id4e = identity_block(id4d, [256, 256, 1024])

    proj5 = projection_block(id4e, [512, 512, 2048], s=2)
    id5a = identity_block(proj5, [512, 512, 2048])
    id5b = identity_block(id5a, [512, 512, 2048])

    avg_pool = K.layers.AveragePooling2D(pool_size=(7, 7),
                                          padding='same')(id5b)

    output = K.layers.Dense(units=1000, activation='softmax',
                             kernel_initializer=init)(avg_pool)

    model = K.models.Model(inputs=X, outputs=output)

    return model
