#!/usr/bin/env python3
"""Loads and preps a dataset for machine translation"""
import tensorflow as tf
import transformers
from setup import load_pt2en


class Dataset:
    """Loads and preps the ted_hrlr_translate/pt_to_en dataset"""

    def __init__(self, batch_size, max_len):
        """Class constructor

        Args:
            batch_size: the batch size for training/validation
            max_len: the maximum number of tokens allowed per example
                sentence
        """
        self.data_train = load_pt2en('train')
        self.data_valid = load_pt2en('validation')
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train)
        self.data_train = self.data_train.map(self.tf_encode)
        self.data_valid = self.data_valid.map(self.tf_encode)

        def filter_max_len(pt, en):
            return tf.logical_and(
                tf.size(pt) <= max_len, tf.size(en) <= max_len)

        self.data_train = self.data_train.filter(filter_max_len)
        self.data_train = self.data_train.cache()
        self.data_train = self.data_train.shuffle(20000)
        self.data_train = self.data_train.padded_batch(batch_size)
        self.data_train = self.data_train.prefetch(
            tf.data.experimental.AUTOTUNE)

        self.data_valid = self.data_valid.filter(filter_max_len)
        self.data_valid = self.data_valid.padded_batch(batch_size)

    def tokenize_dataset(self, data):
        """Creates sub-word tokenizers for our dataset

        Args:
            data: a tf.data.Dataset whose examples are formatted as a
                tuple (pt, en)

        Returns:
            tokenizer_pt, tokenizer_en
        """
        tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            'neuralmind/bert-base-portuguese-cased')
        tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            'bert-base-uncased')

        def pt_sentences():
            for pt, en in data.as_numpy_iterator():
                yield pt.decode('utf-8')

        def en_sentences():
            for pt, en in data.as_numpy_iterator():
                yield en.decode('utf-8')

        tokenizer_pt = tokenizer_pt.train_new_from_iterator(
            pt_sentences(), vocab_size=2 ** 13)
        tokenizer_en = tokenizer_en.train_new_from_iterator(
            en_sentences(), vocab_size=2 ** 13)

        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """Encodes a translation into tokens

        Args:
            pt: a tf.Tensor containing the Portuguese sentence
            en: a tf.Tensor containing the corresponding English sentence

        Returns:
            pt_tokens, en_tokens
                pt_tokens: a list containing the Portuguese tokens
                en_tokens: a list containing the English tokens
        """
        vocab_size_pt = self.tokenizer_pt.vocab_size
        vocab_size_en = self.tokenizer_en.vocab_size

        pt_ids = self.tokenizer_pt.encode(
            pt.numpy().decode('utf-8'), add_special_tokens=False)
        en_ids = self.tokenizer_en.encode(
            en.numpy().decode('utf-8'), add_special_tokens=False)

        pt_tokens = [vocab_size_pt] + pt_ids + [vocab_size_pt + 1]
        en_tokens = [vocab_size_en] + en_ids + [vocab_size_en + 1]

        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """Acts as a tensorflow wrapper for the encode instance method

        Args:
            pt: a tf.Tensor containing the Portuguese sentence
            en: a tf.Tensor containing the corresponding English sentence

        Returns:
            pt_result, en_result: tf.Tensor results of the encode method
        """
        pt_result, en_result = tf.py_function(
            func=self.encode, inp=[pt, en], Tout=[tf.int64, tf.int64])
        pt_result.set_shape([None])
        en_result.set_shape([None])

        return pt_result, en_result
