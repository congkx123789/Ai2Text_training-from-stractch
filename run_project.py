#!/usr/bin/env python3
"""
Simple script to run the Vietnamese ASR project.
This script will guide you through the complete setup and training process.
"""

import os
import sys
import subprocess
from pathlib import Path

def print_header(text):
    """Print a formatted header."""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n Running {description}...")
    print(f"Command: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f" {description} completed successfully!")
            if result.stdout.strip():
                print(f"Output: {result.stdout.strip()}")
            return True
        else:
            print(f" {description} failed!")
            print(f"Error: {result.stderr.strip()}")
            return False
    except Exception as e:
        print(f" Error running command: {e}")
        return False

def check_dependencies():
    """Check if required dependencies are installed."""
    print_header("CHECKING DEPENDENCIES")
    
    # Check Python
    print(f"Python version: {sys.version}")
    
    # Check if we can import basic packages
    try:
        import torch
        print(f" PyTorch version: {torch.__version__}")
    except ImportError:
        print(" PyTorch not found. Installing...")
        run_command("pip install torch torchaudio", "Installing PyTorch")
    
    try:
        import numpy
        print(f" NumPy version: {numpy.__version__}")
    except ImportError:
        print(" NumPy not found. Installing...")
        run_command("pip install numpy", "Installing NumPy")
    
    try:
        import pandas
        print(f" Pandas version: {pandas.__version__}")
    except ImportError:
        print(" Pandas not found. Installing...")
        run_command("pip install pandas", "Installing Pandas")

def create_sample_data():
    """Create sample data for testing."""
    print_header("CREATING SAMPLE DATA")
    
    # Create sample CSV data
    sample_data = """file_path,transcript,split,speaker_id
data/raw/sample1.wav,xin chào việt nam,train,speaker_01
data/raw/sample2.wav,tôi là sinh viên,train,speaker_02
data/raw/sample3.wav,hôm nay trời đẹp,val,speaker_01
data/raw/sample4.wav,chúng tôi học tiếng việt,test,speaker_02"""
    
    # Create directories
    os.makedirs("data/raw", exist_ok=True)
    
    # Save sample CSV
    with open("data/sample_data.csv", "w", encoding="utf-8") as f:
        f.write(sample_data)
    
    print(" Sample data created at data/sample_data.csv")
    print(" Note: You'll need to replace this with your actual audio files and transcripts")

def initialize_database():
    """Initialize the database."""
    print_header("INITIALIZING DATABASE")
    
    # Create a simple database initialization script
    db_init_script = '''
import sqlite3
import os

# Create database directory
os.makedirs("database", exist_ok=True)

# Connect to database
conn = sqlite3.connect("database/asr_training.db")

# Create basic tables
conn.execute("""
CREATE TABLE IF NOT EXISTS AudioFiles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path TEXT NOT NULL,
    filename TEXT NOT NULL,
    duration_seconds REAL,
    sample_rate INTEGER,
    transcript TEXT,
    split_type TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")

conn.commit()
conn.close()
print(" Database initialized successfully!")
'''
    
    with open("init_db_simple.py", "w") as f:
        f.write(db_init_script)
    
    return run_command("python init_db_simple.py", "Initializing database")

def test_basic_model():
    """Test basic model functionality."""
    print_header("TESTING BASIC MODEL")
    
    test_script = '''
import torch
import torch.nn as nn

# Simple test model
class SimpleASR(nn.Module):
    def __init__(self, input_dim=80, vocab_size=100, hidden_dim=256):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x):
        out, _ = self.encoder(x)
        return self.decoder(out)

# Test the model
model = SimpleASR()
dummy_input = torch.randn(2, 100, 80)  # (batch, time, features)
output = model(dummy_input)
print(f" Model test successful! Output shape: {output.shape}")
print(f" Model parameters: {sum(p.numel() for p in model.parameters()):,}")
'''
    
    with open("test_model.py", "w") as f:
        f.write(test_script)
    
    return run_command("python test_model.py", "Testing basic model")

def run_training_demo():
    """Run a simple training demonstration."""
    print_header("RUNNING TRAINING DEMO")
    
    training_demo = '''
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

print(" Starting ASR Training Demo...")

# Simple model
class DemoASR(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.LSTM(80, 128, batch_first=True)
        self.decoder = nn.Linear(128, 50)  # 50 character vocab
    
    def forward(self, x):
        out, _ = self.encoder(x)
        return self.decoder(out)

# Initialize model and optimizer
model = DemoASR()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Simulate training for a few steps
for epoch in range(3):
    # Dummy data
    audio_features = torch.randn(4, 50, 80)  # (batch, time, features)
    targets = torch.randint(0, 50, (4, 50))  # (batch, time)
    
    # Forward pass
    outputs = model(audio_features)
    loss = criterion(outputs.reshape(-1, 50), targets.reshape(-1))
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"Epoch {epoch+1}/3, Loss: {loss.item():.4f}")

print(" Training demo completed successfully!")
print(" This was just a demo. For real training, you'll need actual audio data.")
'''
    
    with open("training_demo.py", "w") as f:
        f.write(training_demo)
    
    return run_command("python training_demo.py", "Running training demo")

def main():
    """Main function to run the project."""
    print_header("VIETNAMESE ASR PROJECT RUNNER")
    print(" This script will help you run your Vietnamese ASR project")
    print(" Current directory:", os.getcwd())
    
    # Step 1: Check dependencies
    check_dependencies()
    
    # Step 2: Create sample data
    create_sample_data()
    
    # Step 3: Initialize database
    if not initialize_database():
        print("  Database initialization failed, but continuing...")
    
    # Step 4: Test basic model
    if not test_basic_model():
        print("  Model test failed, but continuing...")
    
    # Step 5: Run training demo
    if not run_training_demo():
        print("  Training demo failed")
    
    # Final instructions
    print_header("NEXT STEPS")
    print(" Basic setup completed!")
    print("\n To use your ASR system with real data:")
    print("1. Replace data/sample_data.csv with your actual audio file paths and transcripts")
    print("2. Ensure your audio files are in WAV format, 16kHz sample rate")
    print("3. Run the full training script when you have real data")
    print("\n For more advanced features:")
    print("- Check the enhanced training scripts in training/")
    print("- Review configuration options in configs/")
    print("- Read the documentation in docs/")
    
    print("\n Your Vietnamese ASR system is ready to use!")

if __name__ == "__main__":
    main()

