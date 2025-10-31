# How to Run Vietnamese ASR Project

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Initialize Database

```bash
python init_db_simple.py
```

### 3. Run Basic Demo

```bash
python simple_demo.py
```

This creates a simple ASR model and demonstrates basic functionality.

## Complete Workflow

### Step 1: Setup

1. Install dependencies (see above)
2. Initialize database
3. Apply embeddings migration:
```bash
sqlite3 database/asr_training.db < database/migrations/001_add_embedding_tables.sql
```

### Step 2: Prepare Data

1. Create CSV file with audio paths and transcripts:
```csv
file_path,transcript,split,speaker_id
data/raw/audio1.wav,xin chào việt nam,train,speaker_01
data/raw/audio2.wav,tôi là sinh viên,train,speaker_02
```

2. Place audio files in `data/raw/`

3. Prepare data:
```bash
python scripts/prepare_data.py --csv your_data.csv --auto_split
```

### Step 3: Build Embeddings (Optional for Enhanced Features)

```bash
python scripts/build_embeddings.py --db database/asr_training.db --config configs/embeddings.yaml
```

This creates:
- Word2Vec embeddings
- Phon2Vec embeddings
- FAISS indexes

### Step 4: Train Model

#### Basic Training

```bash
python training/train.py --config configs/default.yaml
```

#### Enhanced Training (with embeddings)

```bash
python training/enhanced_train.py --config configs/enhanced.yaml
```

#### Resume Training

```bash
python training/enhanced_train.py --config configs/enhanced.yaml --resume checkpoints/enhanced_checkpoint_epoch_5.pt
```

### Step 5: Evaluate Model

#### Basic Evaluation

```bash
python training/evaluate.py --checkpoint checkpoints/best_model.pt
```

#### Enhanced Evaluation (with rescoring)

```bash
python training/enhanced_evaluate.py \
    --checkpoint checkpoints/best_enhanced_model.pt \
    --use_rescoring \
    --semantic_kv models/embeddings/word2vec.kv \
    --phon_kv models/embeddings/phon2vec.kv \
    --beam_size 5
```

## Running Project Script

Use the integrated project runner:

```bash
python run_project.py
```

This script:
1. Checks dependencies
2. Creates sample data
3. Initializes database
4. Tests basic model
5. Runs training demo

## Command Line Options

### Training

```bash
python training/train.py --config configs/default.yaml --device cuda
python training/enhanced_train.py --config configs/enhanced.yaml --resume checkpoint.pt
```

### Evaluation

```bash
python training/enhanced_evaluate.py \
    --checkpoint model.pt \
    --use_rescoring \
    --semantic_kv embeddings/word2vec.kv \
    --phon_kv embeddings/phon2vec.kv \
    --beam_size 5 \
    --context "context text"
```

### Embeddings

```bash
# Build embeddings
python scripts/build_embeddings.py --db database/asr_training.db --config configs/embeddings.yaml

# Prepare embeddings
python scripts/prepare_embeddings.py --mode sample --vocab_size 2000
python scripts/prepare_embeddings.py --mode prepare --embedding_path embeddings.txt
python scripts/prepare_embeddings.py --mode create --db_path database/asr_training.db

# Rescore N-best
python scripts/rescore_nbest.py --nbest nbest.json --context "context text"
```

## Configuration Files

### Default Config

Edit `configs/default.yaml` for basic training:
- Model architecture
- Training hyperparameters
- Data paths

### Enhanced Config

Edit `configs/enhanced.yaml` for enhanced features:
- Cross-modal attention
- Word2Vec training
- Embeddings paths

### Embeddings Config

Edit `configs/embeddings.yaml` for embeddings:
- Vector size
- Training epochs
- FAISS parameters

## Output Files

### Checkpoints

- `checkpoints/best_model.pt` - Best basic model
- `checkpoints/best_enhanced_model.pt` - Best enhanced model
- `checkpoints/checkpoint_epoch_N.pt` - Epoch checkpoints

### Embeddings

- `models/embeddings/word2vec.model` - Word2Vec model
- `models/embeddings/word2vec.kv` - KeyedVectors
- `models/embeddings/phon2vec.model` - Phon2Vec model
- `models/embeddings/phon2vec.kv` - KeyedVectors
- `models/embeddings/*.faiss` - FAISS indexes

### Logs

- `logs/train_*.log` - Training logs
- `logs/enhanced_train_*.log` - Enhanced training logs
- `logs/enhanced_evaluate_*.log` - Evaluation logs

## Troubleshooting

### Common Errors

1. **ModuleNotFoundError**
   - Run: `pip install -r requirements.txt`

2. **CUDA out of memory**
   - Reduce batch size in config
   - Use CPU: `--device cpu`

3. **Database not found**
   - Run: `python init_db_simple.py`

4. **Config file not found**
   - Check configs/ directory
   - Verify file paths

### Getting Help

- Check logs in `logs/` directory
- Review configuration files
- See `COMPLETE_INTEGRATION_GUIDE.md` for detailed instructions

## Examples

### Example 1: Basic Training

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Initialize database
python init_db_simple.py

# 3. Train basic model
python training/train.py --config configs/default.yaml
```

### Example 2: Enhanced Training with Embeddings

```bash
# 1. Setup (as above)

# 2. Build embeddings
python scripts/build_embeddings.py --db database/asr_training.db --config configs/embeddings.yaml

# 3. Train enhanced model
python training/enhanced_train.py --config configs/enhanced.yaml

# 4. Evaluate with rescoring
python training/enhanced_evaluate.py \
    --checkpoint checkpoints/best_enhanced_model.pt \
    --use_rescoring \
    --semantic_kv models/embeddings/word2vec.kv \
    --phon_kv models/embeddings/phon2vec.kv
```

## Notes

- Ensure audio files are in WAV format, 16kHz
- Transcripts should be in Vietnamese
- GPU recommended for training
- Check disk space for embeddings and checkpoints

