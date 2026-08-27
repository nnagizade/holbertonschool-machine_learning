#!/usr/bin/env python3
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """Builds a vanilla autoencoder network."""
    # 1. Encoder
    inputs = keras.Input(shape=(input_dims,))
    x = inputs
    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)
    latent_layer = keras.layers.Dense(latent_dims, activation='relu')(x)
    encoder = keras.Model(inputs, latent_layer)

    # 2. Decoder
    latent_inputs = keras.Input(shape=(latent_dims,))
    x = latent_inputs
    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)
    outputs = keras.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = keras.Model(latent_inputs, outputs)

    # 3. Autoencoder
    autoencoder_outputs = decoder(encoder(inputs))
    autoencoder = keras.Model(inputs, autoencoder_outputs)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, autoencoder
