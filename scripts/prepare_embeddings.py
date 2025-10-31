"""
Embedding preparation utilities for Vietnamese ASR.
Supports creating, preparing, and managing embeddings for the ASR system.
"""

import argparse
import os
import torch
import numpy as np
from pathlib import Path
import sqlite3

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.embeddings import EmbeddingLoader, EmbeddingWrapper
from nlp.word2vec_trainer import train_word2vec, export_to_sqlite as export_w2v
from nlp.phon2vec_trainer import train_phon2vec, export_to_sqlite as export_p2v


def create_sample_embeddings(vocab_size: int = 2000, embedding_dim: int = 256, output_path: str = "models/vietnamese_embeddings.pt"):
    """
    Create sample embeddings for testing.
    
    Args:
        vocab_size: Vocabulary size
        embedding_dim: Embedding dimension
        output_path: Output file path
    """
    print(f"Creating sample embeddings (vocab_size={vocab_size}, dim={embedding_dim})...")
    
    # Create random embeddings
    embeddings = torch.randn(vocab_size, embedding_dim)
    
    # Normalize
    embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(embeddings, output_path)
    print(f"Saved sample embeddings to {output_path}")


def prepare_from_pretrained(embedding_path: str, embedding_type: str = "word2vec", output_path: str = "models/vietnamese_embeddings.pt"):
    """
    Prepare embeddings from pre-trained model.
    
    Args:
        embedding_path: Path to pre-trained embeddings
        embedding_type: Type of embeddings (word2vec, fasttext, etc.)
        output_path: Output file path
    """
    print(f"Loading pre-trained embeddings from {embedding_path}...")
    
    loader = EmbeddingLoader()
    
    if embedding_type == "word2vec":
        kv = loader.load_word2vec(embedding_path)
    else:
        print(f"Unsupported embedding type: {embedding_type}")
        return
    
    if kv is None:
        print("Failed to load embeddings")
        return
    
    # Convert to PyTorch format
    vocab_size = len(kv.index_to_key)
    embedding_dim = kv.vector_size
    
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    
    for i, word in enumerate(kv.index_to_key):
        embedding_matrix[i] = kv.get_vector(word)
    
    embeddings = torch.FloatTensor(embedding_matrix)
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(embeddings, output_path)
    print(f"Saved embeddings to {output_path} (vocab_size={vocab_size}, dim={embedding_dim})")


def create_from_database(db_path: str, output_dir: str = "models/embeddings"):
    """
    Create embeddings from database transcripts.
    
    Args:
        db_path: Path to database
        output_dir: Output directory
    """
    print(f"Creating embeddings from database {db_path}...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Train Word2Vec
    print("Training Word2Vec...")
    w2v_path = train_word2vec(
        db_path=db_path,
        out_dir=output_dir,
        vector_size=256,
        window=5,
        min_count=2,
        workers=4,
        epochs=10,
    )
    print(f"Word2Vec saved to {w2v_path}")
    
    # Export to database
    kv_path = os.path.join(output_dir, "word2vec.kv")
    if os.path.exists(kv_path):
        print("Exporting Word2Vec to database...")
        export_w2v(kv_path, db_path, table="WordEmbeddings")
    
    # Train Phon2Vec
    print("Training Phon2Vec...")
    p2v_path = train_phon2vec(
        db_path=db_path,
        out_dir=output_dir,
        vector_size=128,
        window=5,
        min_count=2,
        workers=4,
        epochs=10,
        telex=True,
        tone=True,
    )
    print(f"Phon2Vec saved to {p2v_path}")
    
    # Export to database
    kv_path = os.path.join(output_dir, "phon2vec.kv")
    if os.path.exists(kv_path):
        print("Exporting Phon2Vec to database...")
        export_p2v(kv_path, db_path, table="PronunciationEmbeddings")
    
    print("Embedding creation completed!")


def main():
    parser = argparse.ArgumentParser(description="Prepare embeddings for Vietnamese ASR")
    parser.add_argument("--mode", type=str, required=True, choices=["sample", "prepare", "create"],
                       help="Mode: sample (create sample), prepare (from pre-trained), create (from database)")
    parser.add_argument("--vocab_size", type=int, default=2000, help="Vocabulary size (for sample mode)")
    parser.add_argument("--embedding_path", type=str, help="Path to pre-trained embeddings (for prepare mode)")
    parser.add_argument("--embedding_type", type=str, default="word2vec", help="Embedding type (for prepare mode)")
    parser.add_argument("--db_path", type=str, default="database/asr_training.db", help="Database path (for create mode)")
    parser.add_argument("--output_path", type=str, default="models/vietnamese_embeddings.pt", help="Output path")
    parser.add_argument("--output_dir", type=str, default="models/embeddings", help="Output directory (for create mode)")
    
    args = parser.parse_args()
    
    if args.mode == "sample":
        create_sample_embeddings(
            vocab_size=args.vocab_size,
            embedding_dim=256,
            output_path=args.output_path
        )
    elif args.mode == "prepare":
        if not args.embedding_path:
            print("Error: --embedding_path is required for prepare mode")
            return
        prepare_from_pretrained(
            embedding_path=args.embedding_path,
            embedding_type=args.embedding_type,
            output_path=args.output_path
        )
    elif args.mode == "create":
        create_from_database(
            db_path=args.db_path,
            output_dir=args.output_dir
        )


if __name__ == "__main__":
    main()

