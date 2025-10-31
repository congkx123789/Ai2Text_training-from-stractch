# Complete Integration Guide

## Overview

This guide provides step-by-step instructions for integrating and using the Vietnamese ASR system with embeddings support.

## Step 1: Installation

### Install Dependencies

```bash
pip install -r requirements.txt
```

This will install all required packages including:
- PyTorch for deep learning
- Gensim for Word2Vec/Phon2Vec
- FAISS for vector search
- Unidecode for text normalization
- And other dependencies

## Step 2: Database Setup

### Initialize Database

```bash
python init_db_simple.py
```

This creates the basic database structure.

### Apply Embeddings Migration

```bash
sqlite3 database/asr_training.db < database/migrations/001_add_embedding_tables.sql
```

This adds the embedding tables for Word2Vec and Phon2Vec.

## Step 3: Prepare Data

### Create Sample Data

The project includes a sample data CSV structure:
- `data/sample_data.csv` - CSV with columns: file_path, transcript, split, speaker_id

### Prepare Audio Data

1. Place audio files in `data/raw/`
2. Create CSV file with audio paths and transcripts
3. Run data preparation script:
```bash
python scripts/prepare_data.py --csv your_data.csv --auto_split
```

## Step 4: Build Embeddings

### Train Word2Vec and Phon2Vec

```bash
python scripts/build_embeddings.py --db database/asr_training.db --config configs/embeddings.yaml
```

This will:
1. Train Word2Vec embeddings from transcripts
2. Train Phon2Vec embeddings from phonetic tokens
3. Export embeddings to database
4. Build FAISS indexes

Output:
- `models/embeddings/word2vec.model` and `.kv`
- `models/embeddings/phon2vec.model` and `.kv`
- `models/embeddings/*.faiss` indexes

## Step 5: Training

### Basic Training

```bash
python training/train.py --config configs/default.yaml
```

This trains the base ASR model with CTC loss.

### Enhanced Training

```bash
python training/enhanced_train.py --config configs/enhanced.yaml
```

This trains the enhanced model with:
- Cross-modal attention
- Optional Word2Vec auxiliary training
- Multi-task learning

### Resume Training

```bash
python training/enhanced_train.py --config configs/enhanced.yaml --resume checkpoints/enhanced_checkpoint_epoch_5.pt
```

## Step 6: Evaluation

### Basic Evaluation

```bash
python training/evaluate.py --checkpoint checkpoints/best_model.pt
```

### Enhanced Evaluation with Rescoring

```bash
python training/enhanced_evaluate.py \
    --checkpoint checkpoints/best_enhanced_model.pt \
    --use_rescoring \
    --semantic_kv models/embeddings/word2vec.kv \
    --phon_kv models/embeddings/phon2vec.kv \
    --beam_size 5 \
    --context "đặt bánh sinh nhật"
```

This evaluates with:
- N-best beam search
- Semantic and phonetic rescoring
- Optional context biasing

## Step 7: N-best Rescoring (Standalone)

### Rescore N-best List

Create a JSON file with N-best hypotheses:
```json
[
  {"text": "toi muon dat ban", "am_score": -12.5, "lm_score": -1.1},
  {"text": "toi muon dat banh", "am_score": -12.8, "lm_score": -1.0}
]
```

Run rescoring:
```bash
python scripts/rescore_nbest.py \
    --nbest nbest.json \
    --context "đặt bánh gato sinh nhật" \
    --alpha 1.0 --beta 0.0 --gamma 0.5 --delta 0.5
```

## Configuration

### Default Config (`configs/default.yaml`)

Basic training configuration:
- Model architecture
- Training hyperparameters
- Data paths

### Enhanced Config (`configs/enhanced.yaml`)

Enhanced training with:
- Cross-modal attention settings
- Word2Vec auxiliary training
- Embeddings paths
- Rescoring weights

### Embeddings Config (`configs/embeddings.yaml`)

Embedding training parameters:
- Vector size
- Window size
- Epochs
- FAISS parameters

## Integration Tips

### 1. Custom Dataset

Implement your dataset class following PyTorch Dataset interface:
- Load audio files
- Extract mel spectrograms
- Tokenize transcripts
- Return batches

### 2. Custom Tokenizer

Implement Vietnamese tokenization:
- Character-level or subword
- Handle Vietnamese diacritics
- Vocabulary management

### 3. Audio Preprocessing

Configure audio processing:
- Sample rate: 16kHz
- Mel spectrogram: 80 filters
- Normalization

### 4. Contextual Biasing

Use context text for domain-specific biasing:
- Product names
- Brand names
- Domain-specific terms

## Troubleshooting

### Common Issues

1. **Out of Memory**
   - Reduce batch size in config
   - Disable cross-modal attention
   - Use CPU for smaller models

2. **Tokenizer Not Found**
   - Tokenizer is created automatically
   - Check models/ directory

3. **Embeddings Mismatch**
   - Rebuild embeddings: `python scripts/build_embeddings.py`
   - Check vocabulary alignment

4. **Import Errors**
   - Verify all dependencies installed
   - Check Python path
   - Ensure __init__.py files exist

## Performance Tips

1. Use GPU for training
2. Enable mixed precision training (if supported)
3. Use FAISS GPU (if available) for large-scale search
4. Cache embeddings for faster rescoring
5. Optimize beam search width vs accuracy

## Next Steps

- Add streaming inference
- Implement distributed training
- Add model quantization
- Optimize for mobile deployment
- Add more evaluation metrics

