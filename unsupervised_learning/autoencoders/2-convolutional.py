#!/usr/bin/env python3
"""Convolutional Autoencoder Module."""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """Creates a convolutional autoencoder model.

    Args:
        input_dims (tuple): Dimensions of the model input.
        filters (list): Number of filters for each conv layer in encoder.
        latent_dims (tuple): Dimensions of latent space representation.

    Returns:
        tuple: (encoder, decoder, auto)
    """
    # Build Encoder
    inputs_enc = keras.Input(shape=input_dims)
    x = inputs_enc
    for f in filters:
        x = keras.layers.Conv2D(
            f, (3, 3), padding='same', activation='relu'
        )(x)
        x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    encoder = keras.Model(inputs=inputs_enc, outputs=x)

    # Build Decoder
    inputs_dec = keras.Input(shape=latent_dims)
    x = inputs_dec
    rev_filters = filters[::-1]

    for i, f in enumerate(rev_filters):
        if i == len(rev_filters) - 1:
            x = keras.layers.Conv2D(
                f, (3, 3), padding='valid', activation='relu'
            )(x)
        else:
            x = keras.layers.Conv2D(
                f, (3, 3), padding='same', activation='relu'
            )(x)
        x = keras.layers.UpSampling2D((2, 2))(x)

    x = keras.layers.Conv2D(
        filters=input_dims[-1],
        kernel_size=(3, 3),
        padding='same',
        activation='sigmoid'
    )(x)
    decoder = keras.Model(inputs=inputs_dec, outputs=x)

    # Build Autoencoder
    auto_inputs = keras.Input(shape=input_dims)
    auto_outputs = decoder(encoder(auto_inputs))
    auto = keras.Model(inputs=auto_inputs, outputs=auto_outputs)

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
