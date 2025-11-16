# -*- coding: utf-8 -*-
"""
Pure CNN Gomoku Training Package

A complete pure CNN training pipeline for Gomoku/TicTacToe position evaluation 
and move prediction WITHOUT reinforcement learning components.

Key Components:
- config: Configuration management
- cnn_model: Pure CNN architecture (no MCTS/RL)  
- data_pipeline: CSV loading + tactical analysis
- train: Main training script with Mac M4 optimization
- play: Human vs AI gameplay
- evaluate: Model evaluation + visualization
- convert_ttt_data: TicTacToe data converter

Usage:
    from cnn_gomoku_pure import Config, GomokuCNN
    
    config = Config(board_width=5, use_slap='replace')
    model = GomokuCNN(config)
"""

__version__ = "1.0.0"
__author__ = "GitHub Copilot + Pure CNN Training Pipeline"

from .config import Config
from .cnn_model import GomokuCNN, SimpleCNN, ResNetCNN
from .data_pipeline import (
    GomokuDataset, 
    StreamingGomokuDataset, 
    create_dataloader,
    analyze_position,
    grid_to_state
)

__all__ = [
    'Config',
    'GomokuCNN', 
    'SimpleCNN',
    'ResNetCNN', 
    'GomokuDataset',
    'StreamingGomokuDataset',
    'create_dataloader',
    'analyze_position', 
    'grid_to_state'
]