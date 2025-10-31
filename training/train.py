"""
Basic ASR training script with CTC loss.
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

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.asr_base import BaseASR
from training.dataset import ASRDataset  # Will be implemented if needed
from utils.logger import setup_logger
from utils.metrics import MetricsLogger


class CTCLoss(nn.Module):
    """CTC Loss wrapper."""
    
    def __init__(self, blank_idx: int = 0, reduction: str = 'mean'):
        super().__init__()
        self.ctc_loss = nn.CTCLoss(blank=blank_idx, reduction=reduction, zero_infinity=True)
        self.blank_idx = blank_idx
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute CTC loss.
        
        Args:
            logits: Model output (batch, time, vocab_size)
            targets: Target sequences (batch, max_target_len)
            input_lengths: Actual input lengths (batch,)
            target_lengths: Actual target lengths (batch,)
            
        Returns:
            loss: CTC loss
        """
        # Transpose logits for CTC: (time, batch, vocab_size)
        logits = logits.transpose(0, 1)
        
        # Compute loss
        loss = self.ctc_loss(logits, targets, input_lengths, target_lengths)
        
        return loss


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    logger,
):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, batch in enumerate(dataloader):
        # Move batch to device
        audio_features = batch['audio'].to(device)  # (batch, time, features)
        targets = batch['targets'].to(device)  # (batch, max_target_len)
        audio_lengths = batch['audio_lengths'].to(device)  # (batch,)
        target_lengths = batch['target_lengths'].to(device)  # (batch,)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(audio_features, lengths=audio_lengths)  # (batch, time, vocab_size)
        
        # Compute loss
        loss = criterion(logits, targets, audio_lengths, target_lengths)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
        # Accumulate loss
        total_loss += loss.item()
        num_batches += 1
        
        if batch_idx % 10 == 0:
            logger.info(f"Epoch {epoch}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    logger,
):
    """Validate model."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            audio_features = batch['audio'].to(device)
            targets = batch['targets'].to(device)
            audio_lengths = batch['audio_lengths'].to(device)
            target_lengths = batch['target_lengths'].to(device)
            
            # Forward pass
            logits = model(audio_features, lengths=audio_lengths)
            
            # Compute loss
            loss = criterion(logits, targets, audio_lengths, target_lengths)
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description="Train basic ASR model")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Config file")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup logger
    logger = setup_logger("train", config.get('log_dir', 'logs'))
    
    # Device
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Model
    model_config = config['model']
    model = BaseASR(
        input_dim=model_config.get('input_dim', 80),
        hidden_dim=model_config.get('hidden_dim', 256),
        num_layers=model_config.get('num_layers', 3),
        vocab_size=model_config.get('vocab_size', 100),
        dropout=model_config.get('dropout', 0.2),
        bidirectional=model_config.get('bidirectional', True),
    ).to(device)
    
    logger.info(f"Model parameters: {model.get_num_params():,}")
    
    # Loss and optimizer
    criterion = CTCLoss(blank_idx=0)
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training'].get('lr', 0.001),
        weight_decay=config['training'].get('weight_decay', 1e-5),
    )
    
    # Resume from checkpoint
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        logger.info(f"Resumed from epoch {start_epoch}")
    
    # Data loaders (placeholder - will need actual dataset implementation)
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
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch, logger)
        logger.info(f"Train Loss: {train_loss:.4f}")
        
        # Validate
        if val_loader is not None:
            val_loss = validate(model, val_loader, criterion, device, logger)
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
                torch.save(checkpoint, os.path.join(checkpoint_dir, 'best_model.pt'))
                logger.info("Saved best model")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config,
        }
        torch.save(checkpoint, os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt'))
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main()

