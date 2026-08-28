#!/usr/bin/env python3
"""Defines the RNNDecoder class for machine translation"""
import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """Decodes for machine translation"""

    def __init__(self, vocab, embedding, units, batch):
        """Class constructor.

        Args:
            vocab: integer, size of the output vocabulary
            embedding: integer, dimensionality of the embedding
                       vector
            units: integer, number of hidden units in the RNN cell
            batch: integer, batch size

        Sets the public instance attributes embedding, gru, F
        """
        super(RNNDecoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab, output_dim=embedding)
        self.gru = tf.keras.layers.GRU(
            units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform')
        self.F = tf.keras.layers.Dense(vocab)
        self.attention = SelfAttention(units)

    def call(self, x, s_prev, hidden_states):
        """Performs the decoder's forward pass.

        Args:
            x: tensor of shape (batch, 1), previous word in the
               target sequence as an index of the target vocabulary
            s_prev: tensor of shape (batch, units), previous decoder
                    hidden state
            hidden_states: tensor of shape (batch, input_seq_len,
                            units), outputs of the encoder

        Returns:
            y, s
            y: tensor of shape (batch, vocab), output word as a one
               hot vector in the target vocabulary
            s: tensor of shape (batch, units), new decoder hidden
               state
        """
        context, weights = self.attention(s_prev, hidden_states)

        x = self.embedding(x)

        context = tf.expand_dims(context, 1)
        x = tf.concat([context, x], axis=-1)

        outputs, s = self.gru(x, initial_state=s_prev)

        outputs = tf.reshape(outputs, (-1, outputs.shape[2]))

        y = self.F(outputs)

        return y, s
