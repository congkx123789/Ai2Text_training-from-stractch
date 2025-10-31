"""
Language model utilities for Vietnamese ASR system.
Includes N-best rescoring with semantic and phonetic embeddings.
"""

from .rescoring import rescore_nbest

__all__ = ["rescore_nbest"]

