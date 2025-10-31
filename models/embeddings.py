"""
Embedding utilities for Vietnamese ASR system.
Provides wrappers for Word2Vec, Phon2Vec, and pre-trained embeddings.
"""

import os
import torch
import torch.nn as nn
from typing import Optional, Dict, List
from gensim.models import KeyedVectors, Word2Vec
import numpy as np


class EmbeddingLoader:
    """Utility class to load and manage embeddings."""
    
    @staticmethod
    def load_word2vec(path: str, mmap: str = "r") -> Optional[KeyedVectors]:
        """Load Word2Vec embeddings from file."""
        if not os.path.exists(path):
            return None
        try:
            return KeyedVectors.load(path, mmap=mmap)
        except Exception as e:
            print(f"Error loading Word2Vec from {path}: {e}")
            return None
    
    @staticmethod
    def load_phon2vec(path: str, mmap: str = "r") -> Optional[KeyedVectors]:
        """Load Phon2Vec embeddings from file."""
        if not os.path.exists(path):
            return None
        try:
            return KeyedVectors.load(path, mmap=mmap)
        except Exception as e:
            print(f"Error loading Phon2Vec from {path}: {e}")
            return None
    
    @staticmethod
    def get_word_embedding(kv: KeyedVectors, word: str) -> Optional[np.ndarray]:
        """Get embedding vector for a word."""
        if kv is None or word not in kv:
            return None
        return kv.get_vector(word)
    
    @staticmethod
    def get_sentence_embedding(kv: KeyedVectors, words: List[str]) -> np.ndarray:
        """Get average embedding for a sentence."""
        if kv is None:
            return np.zeros((kv.vector_size,) if kv else (256,), dtype=np.float32)
        
        vecs = []
        for word in words:
            if word in kv:
                vecs.append(kv.get_vector(word))
        
        if not vecs:
            return np.zeros((kv.vector_size,), dtype=np.float32)
        
        return np.mean(vecs, axis=0)


class EmbeddingWrapper(nn.Module):
    """
    PyTorch wrapper for pre-trained embeddings.
    Converts embeddings to PyTorch tensors and provides lookup functionality.
    """
    
    def __init__(
        self,
        embedding_path: Optional[str] = None,
        vocab_size: int = 1000,
        embedding_dim: int = 256,
        padding_idx: int = 0,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Load pre-trained embeddings if provided
        if embedding_path and os.path.exists(embedding_path):
            self._load_pretrained(embedding_path)
        else:
            # Initialize random embeddings
            self.embedding = nn.Embedding(
                vocab_size,
                embedding_dim,
                padding_idx=padding_idx
            )
    
    def _load_pretrained(self, path: str):
        """Load pre-trained embeddings from file."""
        try:
            kv = KeyedVectors.load(path, mmap="r")
            # Create embedding layer
            embedding_matrix = np.zeros((self.vocab_size, self.embedding_dim))
            
            # Map words to indices (simplified - assumes vocab mapping exists)
            for i, word in enumerate(kv.index_to_key[:self.vocab_size]):
                if i < self.vocab_size:
                    embedding_matrix[i] = kv.get_vector(word)
            
            self.embedding = nn.Embedding.from_pretrained(
                torch.FloatTensor(embedding_matrix),
                freeze=False,
                padding_idx=0
            )
        except Exception as e:
            print(f"Error loading pre-trained embeddings: {e}")
            # Fallback to random initialization
            self.embedding = nn.Embedding(
                self.vocab_size,
                self.embedding_dim,
                padding_idx=0
            )
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            token_ids: Token indices (batch, seq_len)
            
        Returns:
            embeddings: Embedding vectors (batch, seq_len, embedding_dim)
        """
        return self.embedding(token_ids)


class SubwordTokenizer:
    """
    Simple subword tokenizer for Vietnamese.
    Uses character-level tokenization as a baseline.
    """
    
    def __init__(self, vocab: Optional[Dict[str, int]] = None):
        self.vocab = vocab or {}
        self.idx_to_token = {idx: token for token, idx in self.vocab.items()}
        
        # Special tokens
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        
        if not self.vocab:
            self._build_default_vocab()
    
    def _build_default_vocab(self):
        """Build default Vietnamese character vocabulary."""
        # Vietnamese alphabet
        chars = list("abcdefghijklmnopqrstuvwxyz")
        chars.extend(["ă", "â", "đ", "ê", "ô", "ơ", "ư"])
        chars.extend([c.upper() for c in chars])
        chars.extend([str(i) for i in range(10)])
        chars.extend([self.pad_token, self.unk_token, self.bos_token, self.eos_token])
        
        self.vocab = {char: i for i, char in enumerate(chars)}
        self.idx_to_token = {idx: token for token, idx in self.vocab.items()}
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        tokens = []
        for char in text.lower():
            if char in self.vocab:
                tokens.append(self.vocab[char])
            else:
                tokens.append(self.vocab.get(self.unk_token, 0))
        return tokens
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
        tokens = []
        for idx in token_ids:
            if idx in self.idx_to_token:
                token = self.idx_to_token[idx]
                if token not in [self.pad_token, self.bos_token, self.eos_token]:
                    tokens.append(token)
        return "".join(tokens)
    
    def __len__(self):
        return len(self.vocab)

