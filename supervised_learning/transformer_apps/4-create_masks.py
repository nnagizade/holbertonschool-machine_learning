#!/usr/bin/env python3
"""Creates all masks for training/validation"""
import tensorflow as tf


def create_masks(inputs, target):
    """Creates all masks for training/validation

    Args:
        inputs: a tf.Tensor of shape (batch_size, seq_len_in) containing
            the input sentence
        target: a tf.Tensor of shape (batch_size, seq_len_out) containing
            the target sentence

    Returns:
        encoder_mask, combined_mask, decoder_mask
    """
    seq_len_out = tf.shape(target)[1]

    encoder_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    encoder_mask = encoder_mask[:, tf.newaxis, tf.newaxis, :]

    decoder_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    decoder_mask = decoder_mask[:, tf.newaxis, tf.newaxis, :]

    look_ahead_mask = 1 - tf.linalg.band_part(
        tf.ones((seq_len_out, seq_len_out)), -1, 0)

    decoder_target_padding_mask = tf.cast(
        tf.math.equal(target, 0), tf.float32)
    decoder_target_padding_mask = decoder_target_padding_mask[
        :, tf.newaxis, tf.newaxis, :]

    combined_mask = tf.maximum(look_ahead_mask, decoder_target_padding_mask)

    return encoder_mask, combined_mask, decoder_mask
