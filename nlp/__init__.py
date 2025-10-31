"""
NLP utilities for Vietnamese ASR system.
Includes phonetic processing, Word2Vec training, and FAISS indexing.
"""

from .phonetic import (
    strip_diacritics,
    telex_encode_syllable,
    vn_soundex,
    phonetic_tokens,
    simple_tokenize,
)

__all__ = [
    "strip_diacritics",
    "telex_encode_syllable",
    "vn_soundex",
    "phonetic_tokens",
    "simple_tokenize",
]

