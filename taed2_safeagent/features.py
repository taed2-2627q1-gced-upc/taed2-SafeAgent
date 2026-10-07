from functools import partial

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def character_ngrams(text, ngram_range):
    return [
        text[start : start + size]
        for size in range(ngram_range[0], ngram_range[1] + 1)
        for start in range(len(text) - size + 1)
    ]


def make_vectorizer(settings):
    settings = dict(settings)
    ngram_range = tuple(settings.pop("ngram_range"))
    if len(ngram_range) != 2 or not 1 <= ngram_range[0] <= ngram_range[1]:
        raise ValueError("Invalid character range")
    return TfidfVectorizer(
        analyzer=partial(character_ngrams, ngram_range=ngram_range),
        dtype=np.float32,
        **settings,
    )
