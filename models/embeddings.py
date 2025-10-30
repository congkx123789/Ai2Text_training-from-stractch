"""
Advanced embedding system for Vietnamese ASR with contextual word embeddings.
Includes Word2Vec, contextual embeddings, and subword tokenization support.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, List, Tuple, Union
import math
from pathlib import Path
import pickle
import json


class PositionalEncoding(nn.Module):
    """Enhanced positional encoding with learnable components."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        
        # Fixed sinusoidal encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
        # Learnable positional embeddings
        self.learnable_pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        
        # Mixing parameter
        self.alpha = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add mixed positional encoding."""
        seq_len = x.size(1)
        
        # Mix fixed and learnable positional encodings
        fixed_pe = self.pe[:, :seq_len, :]
        learnable_pe = self.learnable_pe[:, :seq_len, :]
        
        mixed_pe = self.alpha * fixed_pe + (1 - self.alpha) * learnable_pe
        
        x = x + mixed_pe
        return self.dropout(x)


class ContextualEmbedding(nn.Module):
    """Contextual embedding layer with self-attention."""
    
    def __init__(self, vocab_size: int, d_model: int, num_heads: int = 8, 
                 num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)
        
        # Transformer layers for contextualization
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, token_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with contextual embeddings."""
        # Token embeddings
        embeddings = self.token_embedding(token_ids)
        embeddings = embeddings * math.sqrt(self.d_model)  # Scale embeddings
        
        # Add positional encoding
        embeddings = self.pos_encoding(embeddings)
        
        # Create attention mask for transformer
        if attention_mask is not None:
            # Convert to transformer format (True = masked)
            src_key_padding_mask = ~attention_mask.bool()
        else:
            src_key_padding_mask = None
        
        # Apply transformer layers
        contextualized = self.transformer(
            embeddings, 
            src_key_padding_mask=src_key_padding_mask
        )
        
        # Layer normalization and dropout
        contextualized = self.layer_norm(contextualized)
        contextualized = self.dropout(contextualized)
        
        return contextualized


class Word2VecEmbedding(nn.Module):
    """Word2Vec-style embedding with skip-gram training capability."""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 300):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Center word embeddings (input)
        self.center_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # Context word embeddings (output)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # Initialize embeddings
        self._init_embeddings()
    
    def _init_embeddings(self):
        """Initialize embeddings with Xavier uniform."""
        nn.init.xavier_uniform_(self.center_embeddings.weight)
        nn.init.xavier_uniform_(self.context_embeddings.weight)
    
    def forward(self, center_words: torch.Tensor) -> torch.Tensor:
        """Get center word embeddings."""
        return self.center_embeddings(center_words)
    
    def skip_gram_loss(self, center_words: torch.Tensor, 
                       context_words: torch.Tensor,
                       negative_samples: torch.Tensor) -> torch.Tensor:
        """Compute skip-gram loss with negative sampling."""
        # Get embeddings
        center_embeds = self.center_embeddings(center_words)  # (batch, dim)
        context_embeds = self.context_embeddings(context_words)  # (batch, dim)
        neg_embeds = self.context_embeddings(negative_samples)  # (batch, num_neg, dim)
        
        # Positive score
        pos_score = torch.sum(center_embeds * context_embeds, dim=1)  # (batch,)
        pos_loss = F.logsigmoid(pos_score)
        
        # Negative scores
        neg_score = torch.bmm(neg_embeds, center_embeds.unsqueeze(2)).squeeze(2)  # (batch, num_neg)
        neg_loss = torch.sum(F.logsigmoid(-neg_score), dim=1)  # (batch,)
        
        # Total loss
        loss = -(pos_loss + neg_loss).mean()
        
        return loss


class CrossModalAttention(nn.Module):
    """Cross-modal attention between audio and text embeddings."""
    
    def __init__(self, audio_dim: int, text_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.audio_dim = audio_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        
        # Projection layers
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, audio_dim)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(audio_dim)
    
    def forward(self, audio_features: torch.Tensor, 
                text_embeddings: torch.Tensor,
                text_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply cross-modal attention."""
        # Project to common dimension
        audio_proj = self.audio_proj(audio_features)  # (batch, audio_len, hidden)
        text_proj = self.text_proj(text_embeddings)   # (batch, text_len, hidden)
        
        # Cross attention: audio queries, text keys/values
        if text_mask is not None:
            key_padding_mask = ~text_mask.bool()
        else:
            key_padding_mask = None
        
        attended_audio, _ = self.attention(
            query=audio_proj,
            key=text_proj,
            value=text_proj,
            key_padding_mask=key_padding_mask
        )
        
        # Project back to audio dimension
        enhanced_audio = self.output_proj(attended_audio)
        
        # Residual connection and layer norm
        enhanced_audio = self.layer_norm(audio_features + enhanced_audio)
        
        return enhanced_audio


class EmbeddingFusion(nn.Module):
    """Fusion layer for combining different types of embeddings."""
    
    def __init__(self, embedding_dims: List[int], output_dim: int):
        super().__init__()
        self.embedding_dims = embedding_dims
        self.output_dim = output_dim
        
        # Individual projection layers
        self.projections = nn.ModuleList([
            nn.Linear(dim, output_dim) for dim in embedding_dims
        ])
        
        # Attention weights for fusion
        self.attention_weights = nn.Parameter(torch.ones(len(embedding_dims)))
        
        # Final projection
        self.final_proj = nn.Linear(output_dim, output_dim)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(output_dim)
    
    def forward(self, embeddings: List[torch.Tensor]) -> torch.Tensor:
        """Fuse multiple embeddings."""
        if len(embeddings) != len(self.embedding_dims):
            raise ValueError("Number of embeddings must match embedding_dims")
        
        # Project all embeddings to common dimension
        projected = []
        for i, (embed, proj) in enumerate(zip(embeddings, self.projections)):
            projected.append(proj(embed))
        
        # Weighted fusion
        weights = F.softmax(self.attention_weights, dim=0)
        fused = sum(w * embed for w, embed in zip(weights, projected))
        
        # Final projection and normalization
        fused = self.final_proj(fused)
        fused = self.layer_norm(fused)
        
        return fused


class SubwordTokenizer:
    """Subword tokenizer for Vietnamese using BPE-like approach."""
    
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.char_to_id = {}
        self.id_to_char = {}
        self.subword_to_id = {}
        self.id_to_subword = {}
        self.merge_rules = []
        
        # Special tokens
        self.special_tokens = {
            '<pad>': 0,
            '<unk>': 1,
            '<sos>': 2,
            '<eos>': 3,
            '<blank>': 4,
            '<mask>': 5
        }
        
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.blank_token_id = 4
        self.mask_token_id = 5
    
    def encode(self, text: str) -> List[int]:
        """Encode text to subword IDs."""
        words = text.split()
        token_ids = []
        
        for word in words:
            # Apply BPE merges
            word_tokens = list(word)
            word_tokens[-1] += '</w>'
            
            # Apply merge rules
            for pair in self.merge_rules:
                i = 0
                while i < len(word_tokens) - 1:
                    if (word_tokens[i], word_tokens[i + 1]) == pair:
                        word_tokens = (word_tokens[:i] + 
                                     [''.join(pair)] + 
                                     word_tokens[i + 2:])
                    else:
                        i += 1
            
            # Convert to IDs
            for token in word_tokens:
                token_id = self.subword_to_id.get(token, self.unk_token_id)
                token_ids.append(token_id)
        
        return token_ids
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode subword IDs to text."""
        tokens = []
        for token_id in token_ids:
            token = self.id_to_subword.get(token_id, '<unk>')
            if skip_special_tokens and token in self.special_tokens:
                continue
            tokens.append(token)
        
        # Join and clean up
        text = ''.join(tokens)
        text = text.replace('</w>', ' ')
        return text.strip()
    
    def __len__(self) -> int:
        return len(self.subword_to_id)


class PretrainedEmbeddingLoader:
    """Utility class for loading pre-trained embeddings."""
    
    @staticmethod
    def load_word2vec(path: str, vocab: Dict[str, int], 
                      embedding_dim: int = 300) -> torch.Tensor:
        """Load Word2Vec embeddings from file."""
        embedding_matrix = torch.randn(len(vocab), embedding_dim) * 0.02
        
        try:
            # Try to load from various formats
            if path.endswith('.txt'):
                with open(path, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == embedding_dim + 1:
                            word = parts[0]
                            if word in vocab:
                                vector = torch.tensor([float(x) for x in parts[1:]])
                                embedding_matrix[vocab[word]] = vector
        except Exception as e:
            print(f"Error loading embeddings: {e}")
        
        return embedding_matrix


if __name__ == "__main__":
    # Test the embedding components
    print("Testing embedding components...")
    
    # Test ContextualEmbedding
    vocab_size = 1000
    d_model = 256
    
    contextual_embed = ContextualEmbedding(vocab_size, d_model)
    
    # Dummy input
    batch_size = 2
    seq_len = 10
    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    embeddings = contextual_embed(token_ids)
    print(f"Contextual embeddings shape: {embeddings.shape}")
    
    # Test Word2VecEmbedding
    w2v_embed = Word2VecEmbedding(vocab_size, embedding_dim=300)
    
    center_words = torch.randint(0, vocab_size, (batch_size,))
    embeddings = w2v_embed(center_words)
    print(f"Word2Vec embeddings shape: {embeddings.shape}")
    
    print("All embedding components working!")
