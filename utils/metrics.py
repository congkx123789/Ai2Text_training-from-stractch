"""
Metrics utilities for Vietnamese ASR evaluation.
"""

import torch
from typing import List, Dict
import numpy as np


class MetricsLogger:
    """Logger for tracking training metrics."""
    
    def __init__(self):
        self.metrics = {}
    
    def update(self, name: str, value: float):
        """Update metric value."""
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)
    
    def get_mean(self, name: str) -> float:
        """Get mean value of metric."""
        if name not in self.metrics or len(self.metrics[name]) == 0:
            return 0.0
        return np.mean(self.metrics[name])
    
    def reset(self):
        """Reset all metrics."""
        self.metrics = {}


def compute_wer(reference: str, hypothesis: str) -> float:
    """
    Compute Word Error Rate (WER).
    
    Args:
        reference: Reference text
        hypothesis: Hypothesis text
        
    Returns:
        wer: Word error rate (0-1)
    """
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()
    
    if len(ref_words) == 0:
        return 1.0 if len(hyp_words) > 0 else 0.0
    
    # Simple Levenshtein distance for words
    # For production, use a proper edit distance library
    dp = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]
    
    for i in range(len(ref_words) + 1):
        dp[i][0] = i
    for j in range(len(hyp_words) + 1):
        dp[0][j] = j
    
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    
    edit_distance = dp[len(ref_words)][len(hyp_words)]
    wer = edit_distance / len(ref_words)
    
    return wer


def compute_cer(reference: str, hypothesis: str) -> float:
    """
    Compute Character Error Rate (CER).
    
    Args:
        reference: Reference text
        hypothesis: Hypothesis text
        
    Returns:
        cer: Character error rate (0-1)
    """
    ref_chars = list(reference.lower().replace(" ", ""))
    hyp_chars = list(hypothesis.lower().replace(" ", ""))
    
    if len(ref_chars) == 0:
        return 1.0 if len(hyp_chars) > 0 else 0.0
    
    # Simple Levenshtein distance for characters
    dp = [[0] * (len(hyp_chars) + 1) for _ in range(len(ref_chars) + 1)]
    
    for i in range(len(ref_chars) + 1):
        dp[i][0] = i
    for j in range(len(hyp_chars) + 1):
        dp[0][j] = j
    
    for i in range(1, len(ref_chars) + 1):
        for j in range(1, len(hyp_chars) + 1):
            if ref_chars[i-1] == hyp_chars[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    
    edit_distance = dp[len(ref_chars)][len(hyp_chars)]
    cer = edit_distance / len(ref_chars)
    
    return cer

