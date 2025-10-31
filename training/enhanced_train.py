"""
Enhanced ASR training with embeddings and multi-task learning.
Supports cross-modal attention and Word2Vec auxiliary training.
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.enhanced_asr import EnhancedASR
from models.embeddings import SubwordTokenizer
from training.train import CTCLoss, train_epoch, validate
from utils.logger import setup_logger


class Word2VecLoss(nn.Module):
    """
    Word2Vec skip-gram loss for auxiliary training.
    """
    
    def __init__(self, embedding_dim: int = 256, num_negatives: int = 5):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_negatives = num_negatives
    
    def forward(
        self,
        word2vec_output: torch.Tensor,
        context_words: torch.Tensor,
        negative_words: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute Word2Vec skip-gram loss.
        
        Args:
            word2vec_output: Model word2vec projection (batch, time, embedding_dim)
            context_words: Context word indices (batch, time)
            negative_words: Negative samples (batch, time, num_negatives)
            
        Returns:
            loss: Word2Vec loss
        """
        # Simplified Word2Vec loss (negative sampling)
        # For production, use proper negative sampling
        batch_size, seq_len, embed_dim = word2vec_output.shape
        
        # Reshape for loss computation
        word2vec_output = word2vec_output.reshape(-1, embed_dim)  # (batch*time, embed_dim)
        context_words = context_words.reshape(-1)  # (batch*time,)
        
        # Simple dot product loss
        # In practice, use proper negative sampling
        loss = nn.functional.cross_entropy(
            word2vec_output,
            context_words,
            reduction='mean'
        )
        
        return loss


def train_enhanced_epoch(
    model: EnhancedASR,
    dataloader: DataLoader,
    ctc_criterion: nn.Module,
    w2v_criterion: Optional[nn.Module],
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    logger,
    ctc_weight: float = 1.0,
    w2v_weight: float = 0.1,
    tokenizer: Optional[SubwordTokenizer] = None,
):
    """Train enhanced model for one epoch."""
    model.train()
    total_ctc_loss = 0.0
    total_w2v_loss = 0.0
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, batch in enumerate(dataloader):
        audio_features = batch['audio'].to(device)
        targets = batch['targets'].to(device)
        audio_lengths = batch['audio_lengths'].to(device)
        target_lengths = batch['target_lengths'].to(device)
        
        # Prepare text tokens for cross-modal attention
        text_tokens = None
        text_mask = None
        if model.use_cross_modal and 'transcripts' in batch:
            transcripts = batch['transcripts']
            if tokenizer is not None:
                text_tokens_list = [tokenizer.encode(t) for t in transcripts]
                max_len = max(len(t) for t in text_tokens_list)
                text_tokens = torch.zeros(len(transcripts), max_len, dtype=torch.long)
                text_mask = torch.zeros(len(transcripts), max_len, dtype=torch.bool)
                for i, tokens in enumerate(text_tokens_list):
                    text_tokens[i, :len(tokens)] = torch.tensor(tokens)
                    text_mask[i, :len(tokens)] = True
                text_tokens = text_tokens.to(device)
                text_mask = text_mask.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits, word2vec_output = model(
            audio_features,
            text_tokens=text_tokens,
            text_mask=text_mask,
            training_mode=True,
        )
        
        # CTC loss
        ctc_loss = ctc_criterion(logits, targets, audio_lengths, target_lengths)
        total_loss = ctc_weight * ctc_loss
        
        # Word2Vec auxiliary loss
        w2v_loss_value = 0.0
        if model.use_word2vec and word2vec_output is not None and w2v_criterion is not None:
            if 'context_words' in batch:
                context_words = batch['context_words'].to(device)
                w2v_loss = w2v_criterion(word2vec_output, context_words)
                w2v_loss_value = w2v_loss.item()
                total_loss = total_loss + w2v_weight * w2v_loss
        
        # Backward pass
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
        # Accumulate losses
        total_ctc_loss += ctc_loss.item()
        total_w2v_loss += w2v_loss_value
        num_batches += 1
        
        if batch_idx % 10 == 0:
            logger.info(
                f"Epoch {epoch}, Batch {batch_idx}/{len(dataloader)}, "
                f"CTC Loss: {ctc_loss.item():.4f}, "
                f"W2V Loss: {w2v_loss_value:.4f}, "
                f"Total Loss: {total_loss.item():.4f}"
            )
    
    avg_ctc_loss = total_ctc_loss / num_batches if num_batches > 0 else 0.0
    avg_w2v_loss = total_w2v_loss / num_batches if num_batches > 0 else 0.0
    
    return avg_ctc_loss, avg_w2v_loss


def main():
    parser = argparse.ArgumentParser(description="Train enhanced ASR model")
    parser.add_argument("--config", type=str, default="configs/enhanced.yaml")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup logger
    logger = setup_logger("enhanced_train", config.get('log_dir', 'logs'))
    
    # Device
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Model config
    model_config = config['model']
    model = EnhancedASR(
        input_dim=model_config.get('input_dim', 80),
        hidden_dim=model_config.get('hidden_dim', 256),
        num_layers=model_config.get('num_layers', 3),
        vocab_size=model_config.get('vocab_size', 100),
        dropout=model_config.get('dropout', 0.2),
        bidirectional=model_config.get('bidirectional', True),
        use_cross_modal=model_config.get('use_cross_modal', True),
        use_word2vec=model_config.get('use_word2vec', False),
        embedding_dim=model_config.get('embedding_dim', 256),
        text_vocab_size=model_config.get('text_vocab_size', 1000),
    ).to(device)
    
    logger.info(f"Model parameters: {model.get_num_params():,}")
    logger.info(f"Cross-modal attention: {model.use_cross_modal}")
    logger.info(f"Word2Vec auxiliary: {model.use_word2vec}")
    
    # Tokenizer for text encoding
    tokenizer = SubwordTokenizer() if model.use_cross_modal else None
    
    # Loss functions
    ctc_criterion = CTCLoss(blank_idx=0)
    w2v_criterion = Word2VecLoss(embedding_dim=model_config.get('embedding_dim', 256)) if model.use_word2vec else None
    
    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training'].get('lr', 0.001),
        weight_decay=config['training'].get('weight_decay', 1e-5),
    )
    
    # Loss weights
    ctc_weight = config['training'].get('ctc_weight', 1.0)
    w2v_weight = config['training'].get('w2v_weight', 0.1)
    
    # Resume from checkpoint
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        logger.info(f"Resumed from epoch {start_epoch}")
    
    # Data loaders (placeholder)
    train_loader = None  # TODO: Implement
    val_loader = None  # TODO: Implement
    
    if train_loader is None:
        logger.warning("No training dataloader available. Training skipped.")
        return
    
    # Training loop
    num_epochs = config['training'].get('num_epochs', 10)
    checkpoint_dir = config.get('checkpoint_dir', 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, num_epochs):
        logger.info(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Train
        ctc_loss, w2v_loss = train_enhanced_epoch(
            model, train_loader, ctc_criterion, w2v_criterion,
            optimizer, device, epoch, logger,
            ctc_weight=ctc_weight, w2v_weight=w2v_weight,
            tokenizer=tokenizer,
        )
        logger.info(f"Train CTC Loss: {ctc_loss:.4f}, W2V Loss: {w2v_loss:.4f}")
        
        # Validate
        if val_loader is not None:
            # Use basic validation for now
            from training.train import validate
            val_loss = validate(model, val_loader, ctc_criterion, device, logger)
            logger.info(f"Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'config': config,
                }
                torch.save(checkpoint, os.path.join(checkpoint_dir, 'best_enhanced_model.pt'))
                logger.info("Saved best model")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config,
        }
        torch.save(checkpoint, os.path.join(checkpoint_dir, f'enhanced_checkpoint_epoch_{epoch+1}.pt'))
    
    logger.info("Enhanced training completed!")


if __name__ == "__main__":
    main()

