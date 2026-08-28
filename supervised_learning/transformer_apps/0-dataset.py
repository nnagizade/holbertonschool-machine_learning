#!/usr/bin/env python3
"""
Dataset module for loading and preprocessing dataset for machine translation.
"""
import transformers
from setup import load_pt2en


class Dataset:
    """
    Class Dataset that loads and preps a dataset for machine translation.
    """

    def __init__(self):
        """
        Constructor for Dataset class.
        """
        self.data_train = load_pt2en('train')
        self.data_valid = load_pt2en('validation')
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

    def tokenize_dataset(self, data):
        """
        Creates sub-word tokenizers for the dataset.

        Args:
            data: tf.data.Dataset formatted as (pt, en) tuple pairs.

        Returns:
            tokenizer_pt: Portuguese tokenizer
            tokenizer_en: English tokenizer
        """
        tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            'neuralmind/bert-base-portuguese-cased'
        )
        tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            'bert-base-uncased'
        )

        def pt_iterator():
            for pt_batch, _ in data.batch(1000):
                yield [text.decode('utf-8') for text in pt_batch.numpy()]

        def en_iterator():
            for _, en_batch in data.batch(1000):
                yield [text.decode('utf-8') for text in en_batch.numpy()]

        tokenizer_pt = tokenizer_pt.train_new_from_iterator(
            pt_iterator(), vocab_size=2**13
        )
        tokenizer_en = tokenizer_en.train_new_from_iterator(
            en_iterator(), vocab_size=2**13
        )

        return tokenizer_pt, tokenizer_en
