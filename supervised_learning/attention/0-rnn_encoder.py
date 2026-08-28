#!/usr/bin/env python3
"""Defines the RNNEncoder class for machine translation"""
import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """Encodes for machine translation"""

    def __init__(self, vocab, embedding, units, batch):
        """Class constructor.

        Args:
            vocab: integer, size of the input vocabulary
            embedding: integer, dimensionality of the embedding
                       vector
            units: integer, number of hidden units in the RNN cell
            batch: integer, batch size

        Sets the public instance attributes batch, units, embedding,
        gru
        """
        super(RNNEncoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab, output_dim=embedding)
        self.gru = tf.keras.layers.GRU(
            units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform')

    def initialize_hidden_state(self):
        """Initializes the hidden states for the RNN cell to a
        tensor of zeros.

        Returns:
            a tensor of shape (batch, units) containing the
            initialized hidden states
        """
        return tf.zeros((self.batch, self.units))

    def call(self, x, initial):
        """Performs the encoder's forward pass.

        Args:
            x: tensor of shape (batch, input_seq_len), input to the
               encoder layer as word indices within the vocabulary
            initial: tensor of shape (batch, units), initial hidden
                     state

        Returns:
            outputs, hidden
            outputs: tensor of shape (batch, input_seq_len, units),
                     outputs of the encoder
            hidden: tensor of shape (batch, units), last hidden
                    state of the encoder
        """
        x = self.embedding(x)
        outputs, hidden = self.gru(x, initial_state=initial)

        return outputs, hidden 
