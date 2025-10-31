"""
Enhanced ASR model with contextual embeddings and cross-modal attention.
Extends BaseASR with Word2Vec auxiliary training and cross-modal attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.asr_base import BaseASR
from models.embeddings import EmbeddingWrapper, SubwordTokenizer
from typing import Optional, Tuple


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention between audio and text features.
    Audio features attend to text context.
    """
    
    def __init__(self, audio_dim: int, text_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.audio_dim = audio_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(audio_dim, hidden_dim)
        self.k_proj = nn.Linear(text_dim, hidden_dim)
        self.v_proj = nn.Linear(text_dim, hidden_dim)
        
        self.scale = hidden_dim ** -0.5
        self.dropout = nn.Dropout(0.1)
        
    def forward(
        self,
        audio_features: torch.Tensor,
        text_features: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            audio_features: Audio features (batch, audio_len, audio_dim)
            text_features: Text features (batch, text_len, text_dim)
            text_mask: Text mask (batch, text_len)
            
        Returns:
            enhanced_audio: Enhanced audio features (batch, audio_len, hidden_dim)
        """
        batch_size, audio_len, _ = audio_features.shape
        text_len = text_features.shape[1]
        
        # Project to query, key, value
        q = self.q_proj(audio_features)  # (batch, audio_len, hidden_dim)
        k = self.k_proj(text_features)  # (batch, text_len, hidden_dim)
        v = self.v_proj(text_features)  # (batch, text_len, hidden_dim)
        
        # Attention scores
        scores = torch.bmm(q, k.transpose(1, 2)) * self.scale  # (batch, audio_len, text_len)
        
        # Apply mask if provided
        if text_mask is not None:
            mask = text_mask.unsqueeze(1)  # (batch, 1, text_len)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Weighted sum
        enhanced_audio = torch.bmm(attn_weights, v)  # (batch, audio_len, hidden_dim)
        
        return enhanced_audio


class ContextualTextEncoder(nn.Module):
    """
    Transformer-based text encoder for contextual embeddings.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 8,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, 1000, embedding_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, token_ids: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            token_ids: Token indices (batch, seq_len)
            mask: Attention mask (batch, seq_len)
            
        Returns:
            text_features: Encoded text features (batch, seq_len, embedding_dim)
        """
        seq_len = token_ids.shape[1]
        
        # Embedding + positional encoding
        x = self.embedding(token_ids)  # (batch, seq_len, embedding_dim)
        x = x + self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x)
        
        # Create transformer mask (inverted for PyTorch)
        if mask is not None:
            transformer_mask = (mask == 0)  # True for padding
        else:
            transformer_mask = None
        
        # Transformer encoding
        text_features = self.transformer(x, src_key_padding_mask=transformer_mask)
        
        return text_features


class EnhancedASR(BaseASR):
    """
    Enhanced ASR model with contextual embeddings and cross-modal attention.
    
    Features:
    - Base LSTM encoder-decoder
    - Contextual text encoder (Transformer)
    - Cross-modal attention (audio-text)
    - Optional Word2Vec auxiliary training
    """
    
    def __init__(
        self,
        input_dim: int = 80,
        hidden_dim: int = 256,
        num_layers: int = 3,
        vocab_size: int = 100,
        dropout: float = 0.2,
        bidirectional: bool = True,
        use_cross_modal: bool = True,
        use_word2vec: bool = False,
        embedding_dim: int = 256,
        text_vocab_size: int = 1000,
    ):
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            vocab_size=vocab_size,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        
        self.use_cross_modal = use_cross_modal
        self.use_word2vec = use_word2vec
        
        # Text encoder for contextual embeddings
        if use_cross_modal:
            encoder_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
            self.text_encoder = ContextualTextEncoder(
                vocab_size=text_vocab_size,
                embedding_dim=embedding_dim,
                num_layers=2,
                num_heads=8,
                hidden_dim=512,
                dropout=dropout,
            )
            
            # Cross-modal attention
            self.cross_attention = CrossModalAttention(
                audio_dim=encoder_output_dim,
                text_dim=embedding_dim,
                hidden_dim=hidden_dim,
            )
            
            # Fusion layer
            self.fusion = nn.Linear(encoder_output_dim + hidden_dim, encoder_output_dim)
        
        # Word2Vec projection for auxiliary training
        if use_word2vec:
            self.word2vec_proj = nn.Linear(encoder_output_dim, embedding_dim)
    
    def forward(
        self,
        audio_features: torch.Tensor,
        text_tokens: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        training_mode: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            audio_features: Audio features (batch, time, input_dim)
            text_tokens: Text token IDs for context (batch, text_len) - only used during training
            text_mask: Text attention mask (batch, text_len)
            training_mode: Whether to use text context (training only)
            
        Returns:
            logits: CTC logits (batch, time, vocab_size)
            word2vec_output: Word2Vec projection (batch, time, embedding_dim) or None
        """
        # Encode audio
        encoder_output, _ = self.encoder(audio_features)
        encoder_output = self.dropout(encoder_output)
        
        # Cross-modal attention (training only)
        if self.use_cross_modal and training_mode and text_tokens is not None:
            # Encode text context
            text_features = self.text_encoder(text_tokens, mask=text_mask)
            
            # Cross-modal attention
            enhanced_audio = self.cross_attention(
                encoder_output,
                text_features,
                text_mask
            )
            
            # Fuse original and enhanced features
            fused = torch.cat([encoder_output, enhanced_audio], dim=-1)
            encoder_output = self.fusion(fused)
        
        # Decode to vocabulary
        logits = self.decoder(encoder_output)
        
        # Word2Vec auxiliary output
        word2vec_output = None
        if self.use_word2vec:
            word2vec_output = self.word2vec_proj(encoder_output)
        
        return logits, word2vec_output

