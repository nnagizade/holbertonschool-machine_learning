#!/usr/bin/env python3
"""
Dataset module for loading and preprocessing dataset for machine translation.
"""
import transformers
from setup import load_pt2en


class Dataset:
    """
    Class that loads and preps a dataset for machine translation.
    """

    def __init__(self):
        """
        Class constructor that initializes train/validation splits and
        creates the Portuguese and English tokenizers from the training data.
        """
        self.data_train = load_pt2en('train')
        self.data_valid = load_pt2en('validation')
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

    def tokenize_dataset(self, data):
        """
        Creates sub-word tokenizers for the dataset using pre-trained BERT
        tokenizers trained on the input dataset with a max vocabulary size.

        Args:
            data: a tf.data.Dataset formatted as (pt, en) tensor pairs.

        Returns:
            tokenizer_pt: Portuguese tokenizer created from training set
            tokenizer_en: English tokenizer created from training set
        """
        raw_tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            'neuralmind/bert-base-portuguese-cased'
        )
        raw_tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            'bert-base-uncased'
        )

        pt_texts = (pt.numpy().decode('utf-8') for pt, _ in data)
        en_texts = (en.numpy().decode('utf-8') for _, en in data)

        tokenizer_pt = raw_tokenizer_pt.train_new_from_iterator(
            pt_texts, vocab_size=2**13
        )
        tokenizer_en = raw_tokenizer_en.train_new_from_iterator(
            en_texts, vocab_size=2**13
        )

        return tokenizer_pt, tokenizer_en
