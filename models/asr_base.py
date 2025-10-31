"""
Base ASR model for Vietnamese speech recognition.
Uses LSTM encoder with CTC decoder for end-to-end training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseASR(nn.Module):
    """
    Base ASR model with LSTM encoder and CTC decoder.
    
    Architecture:
    - Input: Mel spectrogram features (batch, time, n_mels)
    - Encoder: Multi-layer bidirectional LSTM
    - Decoder: Linear projection to vocabulary size
    - Output: CTC logits (batch, time, vocab_size)
    """
    
    def __init__(
        self,
        input_dim: int = 80,  # Mel spectrogram features
        hidden_dim: int = 256,
        num_layers: int = 3,
        vocab_size: int = 100,
        dropout: float = 0.2,
        bidirectional: bool = True,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.bidirectional = bidirectional
        
        # Encoder: Multi-layer bidirectional LSTM
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )
        
        # Calculate encoder output dimension
        encoder_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # Decoder: Linear projection to vocabulary
        self.decoder = nn.Linear(encoder_output_dim, vocab_size)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, lengths=None):
        """
        Forward pass.
        
        Args:
            x: Input features (batch, time, input_dim)
            lengths: Actual lengths of sequences (batch,)
            
        Returns:
            logits: CTC logits (batch, time, vocab_size)
        """
        # Encode audio features
        # x: (batch, time, input_dim)
        encoder_output, _ = self.encoder(x)
        # encoder_output: (batch, time, hidden_dim * num_directions)
        
        # Apply dropout
        encoder_output = self.dropout(encoder_output)
        
        # Decode to vocabulary logits
        logits = self.decoder(encoder_output)
        # logits: (batch, time, vocab_size)
        
        return logits
    
    def get_num_params(self):
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class CTCASR(BaseASR):
    """
    CTC-based ASR model (alias for BaseASR for clarity).
    Uses Connectionist Temporal Classification for training.
    """
    pass

