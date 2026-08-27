#!/usr/bin/env python3
"""Sparse Autoencoder Module."""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """Creates a sparse autoencoder model.

    Args:
        input_dims (int): Dimension of the input data.
        hidden_layers (list): Number of nodes for each hidden layer in
            the encoder.
        latent_dims (int): Dimension of the latent space representation.
        lambtha (float): Regularization parameter used for L1 regularization
            on the encoded output.

    Returns:
        tuple: (encoder, decoder, auto)
            - encoder: The encoder model
            - decoder: The decoder model
            - auto: The compiled sparse autoencoder model
    """
    # Build Encoder
    encoder_inputs = keras.Input(shape=(input_dims,))
    x = encoder_inputs
    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)

    # Latent space layer with L1 activity regularization
    regularizer = keras.regularizers.l1(lambtha)
    latent_outputs = keras.layers.Dense(
        latent_dims,
        activation='relu',
        activity_regularizer=regularizer
    )(x)
    encoder = keras.Model(inputs=encoder_inputs, outputs=latent_outputs)

    # Build Decoder
    decoder_inputs = keras.Input(shape=(latent_dims,))
    x = decoder_inputs
    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)
    decoder_outputs = keras.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = keras.Model(inputs=decoder_inputs, outputs=decoder_outputs)

    # Build Autoencoder
    auto_outputs = decoder(encoder(encoder_inputs))
    auto = keras.Model(inputs=encoder_inputs, outputs=auto_outputs)

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
