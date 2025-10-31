"""
Enhanced ASR evaluation with N-best rescoring.
Integrates semantic and phonetic embeddings for better hypothesis selection.
"""

import os
import sys
import argparse
import yaml
import torch
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.enhanced_asr import EnhancedASR
from models.embeddings import EmbeddingLoader
from language_model.rescoring import rescore_nbest
from utils.logger import setup_logger
from utils.metrics import compute_wer, compute_cer


def decode_ctc(logits: torch.Tensor, blank_idx: int = 0) -> List[List[int]]:
    """
    Simple CTC decoding (greedy).
    
    Args:
        logits: Model output (batch, time, vocab_size)
        blank_idx: Blank token index
        
    Returns:
        decoded: List of decoded sequences
    """
    batch_size = logits.shape[0]
    decoded = []
    
    for i in range(batch_size):
        # Greedy decoding: take argmax at each time step
        predictions = torch.argmax(logits[i], dim=-1)  # (time,)
        
        # Remove blanks and collapse repeats
        sequence = []
        prev = None
        for pred in predictions:
            if pred != blank_idx and pred != prev:
                sequence.append(pred.item())
            prev = pred
        
        decoded.append(sequence)
    
    return decoded


def beam_search_decode(
    logits: torch.Tensor,
    beam_size: int = 5,
    blank_idx: int = 0,
) -> List[List[Dict[str, Any]]]:
    """
    Beam search decoding for N-best list generation.
    
    Args:
        logits: Model output (batch, time, vocab_size)
        beam_size: Beam width
        blank_idx: Blank token index
        
    Returns:
        nbest_lists: List of N-best hypotheses for each sample
    """
    batch_size = logits.shape[0]
    nbest_lists = []
    
    for i in range(batch_size):
        seq_logits = logits[i]  # (time, vocab_size)
        
        # Simple beam search
        beams = [{'sequence': [], 'score': 0.0}]
        
        for t in range(seq_logits.shape[0]):
            new_beams = []
            
            # Get top-k predictions at this time step
            top_probs, top_indices = torch.topk(
                torch.softmax(seq_logits[t], dim=-1),
                k=min(beam_size * 2, seq_logits.shape[1])
            )
            
            for beam in beams:
                for prob, idx in zip(top_probs, top_indices):
                    idx = idx.item()
                    
                    # CTC: collapse if same as last token
                    new_seq = beam['sequence'].copy()
                    if len(new_seq) == 0 or new_seq[-1] != idx:
                        if idx != blank_idx:
                            new_seq.append(idx)
                    
                    new_score = beam['score'] + torch.log(prob + 1e-9).item()
                    new_beams.append({
                        'sequence': new_seq,
                        'score': new_score
                    })
            
            # Keep top beam_size beams
            beams = sorted(new_beams, key=lambda x: x['score'], reverse=True)[:beam_size]
        
        # Convert to N-best format
        nbest = []
        for beam in beams:
            nbest.append({
                'text': ' '.join(str(t) for t in beam['sequence']),  # Simplified - use proper tokenizer
                'sequence': beam['sequence'],
                'am_score': beam['score'],
            })
        
        nbest_lists.append(nbest)
    
    return nbest_lists


def evaluate_enhanced(
    model: EnhancedASR,
    dataloader: Any,
    device: torch.device,
    logger: Any,
    use_rescoring: bool = True,
    semantic_kv_path: Optional[str] = None,
    phon_kv_path: Optional[str] = None,
    beam_size: int = 5,
    context_text: Optional[str] = None,
):
    """
    Evaluate enhanced model with optional N-best rescoring.
    
    Args:
        model: Enhanced ASR model
        dataloader: Validation dataloader
        device: Device
        logger: Logger
        use_rescoring: Whether to use N-best rescoring
        semantic_kv_path: Path to semantic embeddings
        phon_kv_path: Path to phonetic embeddings
        beam_size: Beam search width
        context_text: Optional context text for biasing
    """
    model.eval()
    
    # Load embeddings for rescoring
    semantic_kv = None
    phon_kv = None
    if use_rescoring:
        loader = EmbeddingLoader()
        if semantic_kv_path:
            semantic_kv = loader.load_word2vec(semantic_kv_path)
            logger.info(f"Loaded semantic embeddings: {semantic_kv is not None}")
        if phon_kv_path:
            phon_kv = loader.load_phon2vec(phon_kv_path)
            logger.info(f"Loaded phonetic embeddings: {phon_kv is not None}")
    
    total_wer = 0.0
    total_cer = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            audio_features = batch['audio'].to(device)
            references = batch.get('references', [])
            
            # Forward pass
            logits, _ = model(audio_features, training_mode=False)
            
            # Beam search for N-best
            nbest_lists = beam_search_decode(logits, beam_size=beam_size)
            
            # Rescore if enabled
            if use_rescoring and (semantic_kv is not None or phon_kv is not None):
                for i, nbest in enumerate(nbest_lists):
                    # Convert to rescore format
                    nbest_format = [
                        {
                            'text': hyp['text'],
                            'am_score': hyp['am_score'],
                            'lm_score': 0.0,  # No LM score in this version
                        }
                        for hyp in nbest
                    ]
                    
                    # Rescore
                    rescored = rescore_nbest(
                        nbest_format,
                        semantic_kv,
                        phon_kv,
                        context_text=context_text,
                        alpha=1.0,
                        beta=0.0,
                        gamma=0.5,
                        delta=0.5,
                    )
                    
                    # Update with rescored scores
                    for j, hyp in enumerate(rescored):
                        if j < len(nbest):
                            nbest[j]['re_score'] = hyp.get('re_score', hyp['am_score'])
                            nbest[j]['text'] = hyp['text']
                
                # Sort by rescore score
                for nbest in nbest_lists:
                    nbest.sort(key=lambda x: x.get('re_score', x['am_score']), reverse=True)
            
            # Evaluate with best hypothesis
            for i, nbest in enumerate(nbest_lists):
                if i < len(references):
                    best_hyp = nbest[0]['text']  # Simplified - use tokenizer
                    ref = references[i]
                    
                    # Compute WER and CER (simplified - assumes proper tokenization)
                    wer = compute_wer(ref, best_hyp)
                    cer = compute_cer(ref, best_hyp)
                    
                    total_wer += wer
                    total_cer += cer
                    num_samples += 1
            
            if batch_idx % 10 == 0:
                logger.info(f"Processed {batch_idx} batches")
    
    avg_wer = total_wer / num_samples if num_samples > 0 else 0.0
    avg_cer = total_cer / num_samples if num_samples > 0 else 0.0
    
    logger.info(f"Evaluation Results:")
    logger.info(f"  WER: {avg_wer:.4f}")
    logger.info(f"  CER: {avg_cer:.4f}")
    logger.info(f"  Samples: {num_samples}")
    
    return avg_wer, avg_cer


def main():
    parser = argparse.ArgumentParser(description="Evaluate enhanced ASR model")
    parser.add_argument("--config", type=str, default="configs/enhanced.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use_rescoring", action="store_true", help="Use N-best rescoring")
    parser.add_argument("--semantic_kv", type=str, default="models/embeddings/word2vec.kv")
    parser.add_argument("--phon_kv", type=str, default="models/embeddings/phon2vec.kv")
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--context", type=str, default=None, help="Context text for biasing")
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup logger
    logger = setup_logger("enhanced_evaluate", config.get('log_dir', 'logs'))
    
    # Device
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Load model
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
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Loaded checkpoint from {args.checkpoint}")
    
    # Data loader (placeholder)
    val_loader = None  # TODO: Implement
    
    if val_loader is None:
        logger.warning("No validation dataloader available. Evaluation skipped.")
        return
    
    # Evaluate
    wer, cer = evaluate_enhanced(
        model,
        val_loader,
        device,
        logger,
        use_rescoring=args.use_rescoring,
        semantic_kv_path=args.semantic_kv if args.use_rescoring else None,
        phon_kv_path=args.phon_kv if args.use_rescoring else None,
        beam_size=args.beam_size,
        context_text=args.context,
    )
    
    logger.info("Evaluation completed!")
    logger.info(f"Final WER: {wer:.4f}, CER: {cer:.4f}")


if __name__ == "__main__":
    main()

