#!/usr/bin/env python3
"""Calculates the cumulative n-gram BLEU score for a sentence"""
import numpy as np


def n_gram_precision(references, sentence, n):
    """Calculates the clipped n-gram precision for a sentence"""
    def get_ngrams(seq, n):
        """Builds a list of n-grams from a sequence of words"""
        return [tuple(seq[i:i + n]) for i in range(len(seq) - n + 1)]

    sentence_ngrams = get_ngrams(sentence, n)
    total = len(sentence_ngrams)
    if total == 0:
        return 0

    sentence_counts = {}
    for gram in sentence_ngrams:
        sentence_counts[gram] = sentence_counts.get(gram, 0) + 1

    max_ref_counts = {}
    for reference in references:
        ref_ngrams = get_ngrams(reference, n)
        ref_counts = {}
        for gram in ref_ngrams:
            ref_counts[gram] = ref_counts.get(gram, 0) + 1
        for gram, count in ref_counts.items():
            if gram not in max_ref_counts or count > max_ref_counts[gram]:
                max_ref_counts[gram] = count

    clipped_count = 0
    for gram, count in sentence_counts.items():
        clipped_count += min(count, max_ref_counts.get(gram, 0))

    return clipped_count / total


def cumulative_bleu(references, sentence, n):
    """Calculates the cumulative n-gram BLEU score for a sentence.

    Args:
        references: list of reference translations
            each reference translation is a list of the words in
            the translation
        sentence: list containing the model proposed sentence
        n: size of the largest n-gram to use for evaluation

    Returns:
        the cumulative n-gram BLEU score
    """
    precisions = [n_gram_precision(references, sentence, i)
                  for i in range(1, n + 1)]

    if min(precisions) == 0:
        geo_mean = 0
    else:
        log_sum = sum(np.log(p) for p in precisions) / n
        geo_mean = np.exp(log_sum)

    sentence_len = len(sentence)
    ref_lens = [len(reference) for reference in references]
    closest_ref_len = min(
        ref_lens, key=lambda ref_len: (abs(ref_len - sentence_len), ref_len)
    )

    if sentence_len > closest_ref_len:
        bp = 1
    else:
        bp = np.exp(1 - (closest_ref_len / sentence_len))

    return bp * geo_mean
