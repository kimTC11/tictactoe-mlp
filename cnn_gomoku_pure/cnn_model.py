# -*- coding: utf-8 -*-
"""
Pure CNN Architecture for Gomoku Position Evaluation and Move Prediction

Extracted and simplified from policy10a.py, removing all reinforcement learning
components (MCTS, policy-value dual head AlphaZero style). 

Focus on supervised learning:
- Position Evaluation: Win/Lose/Draw prediction
- Move Prediction: Best move prediction 
- Slap Integration: Input canonicalization for better generalization

NO PolicyValueNet, NO MCTS, NO self-play components.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Optional, Tuple, Union
import os
import sys

# Import slap canonicalization from local file
try:
    from .slap6b import slap, unslap
except ImportError:
    try:
        from slap6b import slap, unslap
    except ImportError:
        print("Warning: slap6b not found, creating minimal slap functions")
        def slap(x):
            """Minimal slap implementation - returns largest lexicographic rotation."""
            if x.ndim == 4:  # Batch dimension
                results = []
                for i in range(x.shape[0]):
                    slapped, flip, rot = slap(x[i])
                    results.append((slapped, flip, rot))
                return results
            
            variants = []
            # No flip
            for k in range(4):
                variants.append((np.rot90(x, k=k, axes=(-2, -1)), 'no_flip', k))
            # Horizontal flip 
            x_flip = np.flip(x, axis=-1)
            for k in range(4):
                variants.append((np.rot90(x_flip, k=k, axes=(-2, -1)), 'flip', k))
            
            # Return lexicographically largest
            best = max(variants, key=lambda v: v[0].tolist())
            return best[0], best[1], best[2]
        
        def unslap(x, flip, rot):
            """Reverse slap transformation."""
            y = np.rot90(x, k=-rot, axes=(-2, -1))
            if flip == 'flip':
                y = np.flip(y, axis=-1)
            return y


class SimpleCNN(nn.Module):
    """Simple CNN for Gomoku without ResNet blocks."""
    
    def __init__(self, board_width: int, board_height: int, 
                 in_channels: int = 4, num_filters: int = 32,
                 dropout: float = 0.2, model_type: str = 'dual_head'):
        super().__init__()
        self.board_width = board_width
        self.board_height = board_height
        self.num_filters = num_filters
        self.model_type = model_type
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels, num_filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(num_filters, num_filters * 2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(num_filters * 2, num_filters * 4, kernel_size=3, padding=1)
        
        self.dropout = nn.Dropout(dropout)
        
        # Calculate flattened size after convolutions
        conv_out_size = num_filters * 4 * board_width * board_height
        
        # Value head (position evaluation: win/lose/draw)
        if model_type in ['value_only', 'dual_head']:
            self.value_conv = nn.Conv2d(num_filters * 4, 2, kernel_size=1)
            self.value_fc1 = nn.Linear(2 * board_width * board_height, 64)
            self.value_fc2 = nn.Linear(64, 3)  # [win, draw, lose] probabilities
        
        # Move prediction head (best move)
        if model_type in ['move_prediction', 'dual_head']:
            self.move_conv = nn.Conv2d(num_filters * 4, 4, kernel_size=1)
            move_fc_input = 4 * board_width * board_height
            self.move_fc1 = nn.Linear(move_fc_input, 128)
            self.move_fc2 = nn.Linear(128, board_width * board_height)
        
    def forward(self, x):
        # Convolutional feature extraction
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        outputs = {}
        
        # Value head
        if self.model_type in ['value_only', 'dual_head']:
            x_val = F.relu(self.value_conv(x))
            x_val = x_val.view(x_val.size(0), -1)  # Flatten
            x_val = self.dropout(x_val)
            x_val = F.relu(self.value_fc1(x_val))
            x_val = self.dropout(x_val)
            x_val = self.value_fc2(x_val)
            outputs['value'] = F.softmax(x_val, dim=1)  # [win, draw, lose] probabilities
        
        # Move prediction head
        if self.model_type in ['move_prediction', 'dual_head']:
            x_move = F.relu(self.move_conv(x))
            x_move = x_move.view(x_move.size(0), -1)  # Flatten
            x_move = self.dropout(x_move)
            x_move = F.relu(self.move_fc1(x_move))
            x_move = self.dropout(x_move)
            x_move = self.move_fc2(x_move)
            outputs['move'] = F.log_softmax(x_move, dim=1)  # Move probabilities
        
        return outputs


class ResidualBlock(nn.Module):
    """Residual block for deeper CNN."""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


class ResNetCNN(nn.Module):
    """ResNet-based CNN for Gomoku."""
    
    def __init__(self, board_width: int, board_height: int,
                 in_channels: int = 4, num_filters: int = 32,
                 num_blocks: int = 2, dropout: float = 0.2,
                 model_type: str = 'dual_head'):
        super().__init__()
        self.board_width = board_width
        self.board_height = board_height
        self.model_type = model_type
        
        # Initial convolution
        self.conv1 = nn.Conv2d(in_channels, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters)
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_filters, num_filters) for _ in range(num_blocks)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
        # Value head
        if model_type in ['value_only', 'dual_head']:
            self.value_conv = nn.Conv2d(num_filters, 1, kernel_size=1)
            self.value_bn = nn.BatchNorm2d(1)
            self.value_fc1 = nn.Linear(board_width * board_height, 64)
            self.value_fc2 = nn.Linear(64, 3)  # [win, draw, lose]
        
        # Move prediction head
        if model_type in ['move_prediction', 'dual_head']:
            self.move_conv = nn.Conv2d(num_filters, 2, kernel_size=1)
            self.move_bn = nn.BatchNorm2d(2)
            self.move_fc1 = nn.Linear(2 * board_width * board_height, 128)
            self.move_fc2 = nn.Linear(128, board_width * board_height)
    
    def forward(self, x):
        # Initial convolution
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Residual blocks
        for block in self.res_blocks:
            x = block(x)
        
        outputs = {}
        
        # Value head
        if self.model_type in ['value_only', 'dual_head']:
            x_val = F.relu(self.value_bn(self.value_conv(x)))
            x_val = x_val.view(x_val.size(0), -1)
            x_val = self.dropout(x_val)
            x_val = F.relu(self.value_fc1(x_val))
            x_val = self.dropout(x_val)
            x_val = self.value_fc2(x_val)
            outputs['value'] = F.softmax(x_val, dim=1)
        
        # Move prediction head
        if self.model_type in ['move_prediction', 'dual_head']:
            x_move = F.relu(self.move_bn(self.move_conv(x)))
            x_move = x_move.view(x_move.size(0), -1)
            x_move = self.dropout(x_move)
            x_move = F.relu(self.move_fc1(x_move))
            x_move = self.dropout(x_move)
            x_move = self.move_fc2(x_move)
            outputs['move'] = F.log_softmax(x_move, dim=1)
        
        return outputs


class GomokuCNN:
    """Pure CNN wrapper for Gomoku training."""
    
    def __init__(self, config, model_file: Optional[str] = None):
        self.config = config
        self.device = config.get_device()
        
        # Create model
        if config.num_ResBlock > 0:
            self.model = ResNetCNN(
                board_width=config.board_width,
                board_height=config.board_height,
                in_channels=config.in_channels,
                num_filters=config.num_filters,
                num_blocks=config.num_ResBlock,
                dropout=config.dropout,
                model_type=config.default_model_type
            ).to(self.device)
        else:
            self.model = SimpleCNN(
                board_width=config.board_width,
                board_height=config.board_height,
                in_channels=config.in_channels,
                num_filters=config.num_filters,
                dropout=config.dropout,
                model_type=config.default_model_type
            ).to(self.device)
        
        # Optimizer
        if config.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=config.learning_rate,
                momentum=0.9,
                weight_decay=config.weight_decay
            )
        elif config.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
        elif config.optimizer == 'AdamW':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
        
        # Load model if provided
        if model_file and os.path.exists(model_file):
            self.load(model_file)
    
    def apply_slap(self, states: np.ndarray) -> np.ndarray:
        """Apply slap canonicalization to input states."""
        if self.config.use_slap == 'none':
            return states
        
        if self.config.use_slap == 'replace':
            # Canonicalize each state
            slapped_states = []
            for state in states:
                slapped, _, _ = slap(state)
                slapped_states.append(slapped)
            return np.array(slapped_states)
        
        elif self.config.use_slap == 'add':
            # Double input: original + canonical
            slapped_states = []
            for state in states:
                slapped, _, _ = slap(state)
                # Concatenate along channel dimension
                combined = np.concatenate([state, slapped], axis=0)
                slapped_states.append(combined)
            return np.array(slapped_states)
        
        return states
    
    def predict(self, states: np.ndarray) -> dict:
        """Predict on batch of states."""
        self.model.eval()
        with torch.no_grad():
            # Apply slap
            states = self.apply_slap(states)
            
            # Convert to tensor
            if self.config.use_slap == 'add':
                # Double channel dimension
                states_tensor = torch.FloatTensor(states).to(self.device)
            else:
                states_tensor = torch.FloatTensor(states).to(self.device)
            
            # Forward pass
            outputs = self.model(states_tensor)
            
            # Convert to numpy
            results = {}
            for key, value in outputs.items():
                results[key] = value.cpu().numpy()
            
            return results
    
    def save(self, filepath: str, metadata: Optional[dict] = None):
        """Save model and metadata."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.__dict__,
            'metadata': metadata or {}
        }, filepath)
    
    def load(self, filepath: str):
        """Load model from file."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    def summary(self):
        """Print model summary."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"🧠 Pure CNN Model Summary:")
        print(f"   Architecture: {'ResNet' if self.config.num_ResBlock > 0 else 'Simple CNN'}")
        print(f"   Model Type: {self.config.default_model_type}")
        print(f"   Total Parameters: {total_params:,}")
        print(f"   Trainable Parameters: {trainable_params:,}")
        print(f"   Device: {self.device}")
        print(f"   Slap Mode: {self.config.use_slap}")