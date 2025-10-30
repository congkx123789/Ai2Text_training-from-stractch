"""
Enhanced ASR model with contextual embeddings and cross-modal attention.
Integrates advanced word embeddings for better Vietnamese speech recognition.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, List
from pathlib import Path

from .embeddings import (
    ContextualEmbedding, 
    Word2VecEmbedding, 
    CrossModalAttention,
    EmbeddingFusion,
    SubwordTokenizer,
    PretrainedEmbeddingLoader
)
from .asr_base import (
    ConvSubsampling,
    MultiHeadAttention,
    FeedForward,
    EncoderLayer
)


class EnhancedASREncoder(nn.Module):
    """Enhanced ASR encoder with contextual embeddings and cross-modal attention."""
    
    def __init__(self, 
                 input_dim: int,
                 d_model: int,
                 num_layers: int,
                 num_heads: int,
                 d_ff: int,
                 vocab_size: int,
                 use_cross_modal: bool = True,
                 use_contextual_embeddings: bool = True,
                 dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.use_cross_modal = use_cross_modal
        self.use_contextual_embeddings = use_contextual_embeddings
        
        # Audio processing layers
        self.conv_subsampling = ConvSubsampling(1, d_model // 4)
        
        # Calculate input dimension after subsampling
        subsampled_dim = (input_dim // 4) * (d_model // 4)
        self.audio_proj = nn.Linear(subsampled_dim, d_model)
        
        # Audio encoder layers
        self.audio_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Text embedding components (if using cross-modal attention)
        if self.use_contextual_embeddings:
            self.text_embeddings = ContextualEmbedding(
                vocab_size=vocab_size,
                d_model=d_model,
                num_heads=num_heads,
                num_layers=2,  # Fewer layers for text
                dropout=dropout
            )
        
        # Cross-modal attention layers
        if self.use_cross_modal:
            self.cross_modal_layers = nn.ModuleList([
                CrossModalAttention(d_model, d_model, d_model)
                for _ in range(num_layers // 2)  # Apply every other layer
            ])
        
        # Layer normalization
        self.norm = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, 
                audio_features: torch.Tensor,
                audio_lengths: Optional[torch.Tensor] = None,
                text_tokens: Optional[torch.Tensor] = None,
                text_lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Enhanced forward pass with optional text context."""
        # Process audio
        x = self.conv_subsampling(audio_features)
        x = self.audio_proj(x)
        x = self.dropout(x)
        
        # Get text embeddings if available
        text_embeddings = None
        text_mask = None
        
        if (self.use_contextual_embeddings and 
            text_tokens is not None and 
            self.training):  # Only use text during training
            
            # Create text attention mask
            if text_lengths is not None:
                batch_size, max_len = text_tokens.shape
                text_mask = torch.arange(max_len, device=text_tokens.device).unsqueeze(0) < text_lengths.unsqueeze(1)
            
            # Get contextual text embeddings
            text_embeddings = self.text_embeddings(text_tokens, text_mask)
        
        # Apply encoder layers with optional cross-modal attention
        cross_modal_idx = 0
        for i, layer in enumerate(self.audio_layers):
            # Standard self-attention
            x = layer(x)
            
            # Apply cross-modal attention every other layer
            if (self.use_cross_modal and 
                text_embeddings is not None and 
                i % 2 == 1 and 
                cross_modal_idx < len(self.cross_modal_layers)):
                
                x = self.cross_modal_layers[cross_modal_idx](
                    x, text_embeddings, text_mask
                )
                cross_modal_idx += 1
        
        x = self.norm(x)
        
        # Update audio lengths after subsampling
        if audio_lengths is not None:
            audio_lengths = (audio_lengths / 4).long()
        
        return x, audio_lengths


class EnhancedASRModel(nn.Module):
    """Complete enhanced ASR model with advanced embeddings."""
    
    def __init__(self,
                 input_dim: int,
                 vocab_size: int,
                 d_model: int = 256,
                 num_encoder_layers: int = 6,
                 num_heads: int = 4,
                 d_ff: int = 1024,
                 dropout: float = 0.1,
                 use_cross_modal: bool = True,
                 use_contextual_embeddings: bool = True,
                 use_embedding_fusion: bool = True,
                 pretrained_embeddings_path: Optional[str] = None):
        """Initialize enhanced ASR model."""
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.use_cross_modal = use_cross_modal
        self.use_contextual_embeddings = use_contextual_embeddings
        
        # Enhanced encoder
        self.encoder = EnhancedASREncoder(
            input_dim=input_dim,
            d_model=d_model,
            num_layers=num_encoder_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            vocab_size=vocab_size,
            use_cross_modal=use_cross_modal,
            use_contextual_embeddings=use_contextual_embeddings,
            dropout=dropout
        )
        
        # Output projection
        self.decoder = nn.Linear(d_model, vocab_size)
        
        # Word2Vec component for auxiliary training
        if use_contextual_embeddings:
            self.word2vec = Word2VecEmbedding(vocab_size, embedding_dim=300)
    
    def forward(self, 
                audio_features: torch.Tensor,
                audio_lengths: Optional[torch.Tensor] = None,
                text_tokens: Optional[torch.Tensor] = None,
                text_lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through enhanced ASR model."""
        # Encode with optional text context
        encoded, audio_lengths = self.encoder(
            audio_features, audio_lengths, text_tokens, text_lengths
        )
        
        # Decode
        logits = self.decoder(encoded)
        
        return logits, audio_lengths
    
    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def get_num_trainable_params(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test enhanced ASR model
    print("Testing Enhanced ASR Model...")
    
    # Model parameters
    input_dim = 80
    vocab_size = 1000
    d_model = 256
    batch_size = 2
    audio_len = 100
    text_len = 20
    
    # Create model
    model = EnhancedASRModel(
        input_dim=input_dim,
        vocab_size=vocab_size,
        d_model=d_model,
        num_encoder_layers=4,
        num_heads=4,
        d_ff=512,
        use_cross_modal=True,
        use_contextual_embeddings=True,
        use_embedding_fusion=True
    )
    
    print(f"Model parameters: {model.get_num_trainable_params():,}")
    
    # Test forward pass
    audio_features = torch.randn(batch_size, audio_len, input_dim)
    audio_lengths = torch.tensor([audio_len, audio_len // 2])
    text_tokens = torch.randint(0, vocab_size, (batch_size, text_len))
    text_lengths = torch.tensor([text_len, text_len // 2])
    
    # Forward pass
    logits, output_lengths = model(audio_features, audio_lengths, text_tokens, text_lengths)
    
    print(f"Input audio shape: {audio_features.shape}")
    print(f"Input text shape: {text_tokens.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Output lengths: {output_lengths}")
    
    print("Enhanced ASR model test completed!")
