#!/usr/bin/env python3
"""
Dataset module to load and preprocess dataset for machine translation.
"""
import transformers
from setup import load_pt2en


class Dataset:
    """
    Class Dataset that loads and preps a dataset for machine translation.
    """

    def __init__(self):
        """
        Constructor method for Dataset class.
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
            data: tf.data.Dataset whose examples are formatted as (pt, en)

        Returns:
            tokenizer_pt: Portuguese tokenizer
            tokenizer_en: English tokenizer
        """
        pt_texts = []
        en_texts = []
        for pt, en in data:
            pt_texts.append(pt.numpy().decode('utf-8'))
            en_texts.append(en.numpy().decode('utf-8'))

        raw_tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            'neuralmind/bert-base-portuguese-cased'
        )
        raw_tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            'bert-base-uncased'
        )

        tokenizer_pt = raw_tokenizer_pt.train_new_from_iterator(
            pt_texts, vocab_size=2**13
        )
        tokenizer_en = raw_tokenizer_en.train_new_from_iterator(
            en_texts, vocab_size=2**13
        )

        return tokenizer_pt, tokenizer_en
