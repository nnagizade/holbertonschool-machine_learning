#!/usr/bin/env python3
"""Convolutional Autoencoder Model Definition"""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    Creates a convolutional autoencoder.

    Parameters:
        input_dims (tuple): Dimensions of the model input.
        filters (list): Number of filters for each conv layer in the encoder.
        latent_dims (tuple): Dimensions of the latent space representation.

    Returns:
        encoder, decoder, auto: The compiled autoencoder model and sub-models.
    """
    # ------------------ Encoder ------------------
    inputs_enc = keras.Input(shape=input_dims)
    x = inputs_enc
    for f in filters:
        x = keras.layers.Conv2D(f, (3, 3), padding='same', activation='relu')(x)
        x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    encoder = keras.Model(inputs=inputs_enc, outputs=x, name='encoder')

    # ------------------ Decoder ------------------
    inputs_dec = keras.Input(shape=latent_dims)
    x = inputs_dec
    reversed_filters = filters[::-1]

    for i, f in enumerate(reversed_filters):
        if i == len(reversed_filters) - 1:
            # Second-to-last convolution overall uses valid padding
            x = keras.layers.Conv2D(f, (3, 3), padding='valid', activation='relu')(x)
        else:
            x = keras.layers.Conv2D(f, (3, 3), padding='same', activation='relu')(x)
        x = keras.layers.UpSampling2D((2, 2))(x)

    # Last convolution overall (output layer)
    x = keras.layers.Conv2D(
        filters=input_dims[-1],
        kernel_size=(3, 3),
        padding='same',
        activation='sigmoid'
    )(x)
    decoder = keras.Model(inputs=inputs_dec, outputs=x, name='decoder')

    # ---------------- Autoencoder ----------------
    auto_inputs = keras.Input(shape=input_dims)
    auto_outputs = decoder(encoder(auto_inputs))
    auto = keras.Model(inputs=auto_inputs, outputs=auto_outputs, name='autoencoder')

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
