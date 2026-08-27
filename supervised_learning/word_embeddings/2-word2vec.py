from gensim.models import Word2Vec

def train_word2vec(sentences, vector_size=100, min_count=5, window=5, 
                   negative=5, cbow_mean=True, training_algorithm=0, 
                   learn_rate=0.025, epochs=30, seed=0, workers=1):
    
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        min_count=min_count,
        window=window,
        negative=negative,
        cbow_mean=int(cbow_mean),
        sg=training_algorithm,
        alpha=learn_rate,
        epochs=epochs,
        seed=seed,         # Ensure seed parameter is dynamically passed
        workers=workers
    )
    return model
