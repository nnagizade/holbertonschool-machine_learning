#!/usr/bin/env python3
"""LSA module using Gensim."""
import gensim


def lsa(sentences, num_topics):
    """Calculates LSA on a given corpus of sentences."""
    # 1. Create the dictionary from tokenized sentences
    dictionary = gensim.corpora.Dictionary(sentences)

    # 2. Build Bag-of-Words corpus
    corpus = [dictionary.doc2bow(text) for text in sentences]

    # 3. Apply TF-IDF model transformation
    tfidf = gensim.models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]

    # 4. Train LsiModel with the TF-IDF corpus and dictionary
    lsi_model = gensim.models.LsiModel(
        corpus_tfidf,
        id2word=dictionary,
        num_topics=num_topics
    )

    # 5. Return the projection matrix U
    return lsi_model.projection.u
