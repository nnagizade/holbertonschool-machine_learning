#!/usr/bin/env python3
"""
Module for creating a Variational Autoencoder (VAE)
"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a variational autoencoder.

    Parameters:
    - input_dims (int): dimensions of the model input
    - hidden_layers (list): number of nodes for each hidden layer in the encoder
    - latent_dims (int): dimensions of the latent space representation

    Returns:
    - encoder: encoder model returning [latent_rep, mean, log_variance]
    - decoder: decoder model
    - auto: full compiled autoencoder model
    """
    # ------------------ ENCODER ------------------
    inputs = keras.Input(shape=(input_dims,))
    x = inputs
    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)

    mean = keras.layers.Dense(latent_dims, activation=None)(x)
    log_variance = keras.layers.Dense(latent_dims, activation=None)(x)

    def sampling(args):
        """Reparameterization trick for latent sampling"""
        m, log_v = args
        batch = keras.backend.shape(m)[0]
        dim = keras.backend.shape(m)[1]
        epsilon = keras.backend.random_normal(shape=(batch, dim))
        return m + keras.backend.exp(0.5 * log_v) * epsilon

    z = keras.layers.Lambda(sampling)([mean, log_variance])
    encoder = keras.Model(inputs, [z, mean, log_variance], name='encoder')

    # ------------------ DECODER ------------------
    latent_inputs = keras.Input(shape=(latent_dims,))
    x = latent_inputs
    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)
    outputs = keras.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = keras.Model(latent_inputs, outputs, name='decoder')

    # ------------------ AUTOENCODER ------------------
    auto_outputs = decoder(z)
    auto = keras.Model(inputs, auto_outputs, name='autoencoder')

    # Custom VAE Loss (Reconstruction Loss + KL Divergence)
    def vae_loss(inputs, outputs):
        recon_loss = keras.losses.binary_crossentropy(inputs, outputs)
        recon_loss *= input_dims
        kl_loss = 1 + log_variance - keras.backend.square(mean) - keras.backend.exp(log_variance)
        kl_loss = keras.backend.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        return keras.backend.mean(recon_loss + kl_loss)

    auto.compile(optimizer='adam', loss=vae_loss)

    return encoder, decoder, auto
