#!/usr/bin/env python3
"""Loads and preps a dataset for machine translation"""
import transformers
from setup import load_pt2en


class Dataset:
    """Loads and preps the ted_hrlr_translate/pt_to_en dataset"""

    def __init__(self):
        """Class constructor"""
        self.data_train = load_pt2en('train')
        self.data_valid = load_pt2en('validation')
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train)

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
