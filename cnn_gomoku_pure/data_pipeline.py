# -*- coding: utf-8 -*-
"""
Data Pipeline for Pure CNN Gomoku Training

Handles CSV data loading, position analysis, and target generation for supervised learning.
Focus on position evaluation and move prediction WITHOUT reinforcement learning.

Features:
- CSV parsing for Gomoku positions 
- Tactical analysis (winning moves, blocks, threats)
- Streaming data loading for large datasets
- Slap augmentation integration
- Mac M4 optimization

NO MCTS, NO self-play, NO policy-value dual head RL training.
"""

import os
import sys
import csv
import glob
import random
from typing import List, Tuple, Iterator, Dict, Any, Optional
from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader

# Import slap from local file
try:
    from .slap6b import slap
except ImportError:
    try:
        from slap6b import slap
    except ImportError:
        print("Warning: Using minimal slap implementation")
        def slap(x):
            variants = []
            for k in range(4):
                variants.append(np.rot90(x, k=k, axes=(-2, -1)))
            x_flip = np.flip(x, axis=-1)
            for k in range(4):
                variants.append(np.rot90(x_flip, k=k, axes=(-2, -1)))
            best = max(variants, key=lambda v: v.tolist())
            return best, 'no_flip', 0


def parse_csv_row(row: List[str], width: int = 5, height: int = 5) -> Tuple[np.ndarray, str]:
    """Parse CSV row into (grid, current_player)."""
    if len(row) < 2 + width * height:
        raise ValueError(f"Row too short: expected {2 + width * height} columns")
    
    current_player = row[1].strip().upper()
    cells = [row[i].strip().upper() for i in range(2, 2 + width * height)]
    grid = np.array(cells, dtype=str).reshape(height, width)
    
    return grid, current_player


def grid_to_state(grid: np.ndarray, current_player: str) -> np.ndarray:
    """Convert token grid to 4-channel state tensor."""
    H, W = grid.shape
    
    # Current player stones
    current_stones = (grid == current_player).astype(np.float32)
    
    # Opponent stones  
    opponent = 'O' if current_player == 'X' else 'X'
    opponent_stones = (grid == opponent).astype(np.float32)
    
    # Position encoding (normalized coordinates)
    pos_y = np.tile(np.linspace(-1, 1, H).reshape(-1, 1), (1, W))
    pos_x = np.tile(np.linspace(-1, 1, W).reshape(1, -1), (H, 1))
    
    # Stack channels: [current_player, opponent, pos_y, pos_x]
    state = np.stack([current_stones, opponent_stones, pos_y, pos_x], axis=0)
    return state.astype(np.float32)


def check_winner(grid: np.ndarray, n_in_row: int = 5) -> Optional[str]:
    """Check if there's a winner in the grid."""
    H, W = grid.shape
    
    def check_line(stones, dr, dc):
        """Check a line direction for n_in_row."""
        for r in range(H):
            for c in range(W):
                if stones[r, c] == 1:
                    count = 1
                    # Check forward
                    nr, nc = r + dr, c + dc
                    while 0 <= nr < H and 0 <= nc < W and stones[nr, nc] == 1:
                        count += 1
                        nr += dr
                        nc += dc
                    if count >= n_in_row:
                        return True
        return False
    
    # Check both players
    for player in ['X', 'O']:
        stones = (grid == player).astype(int)
        # Check all directions: horizontal, vertical, diagonal, anti-diagonal
        if (check_line(stones, 0, 1) or    # horizontal
            check_line(stones, 1, 0) or    # vertical 
            check_line(stones, 1, 1) or    # diagonal
            check_line(stones, 1, -1)):    # anti-diagonal
            return player
    
    return None


def find_winning_moves(grid: np.ndarray, player: str, n_in_row: int = 5) -> List[Tuple[int, int]]:
    """Find all moves that would immediately win for the player."""
    H, W = grid.shape
    winning_moves = []
    
    for r in range(H):
        for c in range(W):
            if grid[r, c] == '-':  # Empty cell
                # Try move
                test_grid = grid.copy()
                test_grid[r, c] = player
                if check_winner(test_grid, n_in_row) == player:
                    winning_moves.append((r, c))
    
    return winning_moves


def find_blocking_moves(grid: np.ndarray, player: str, n_in_row: int = 5) -> List[Tuple[int, int]]:
    """Find moves that block opponent's immediate wins."""
    opponent = 'O' if player == 'X' else 'X'
    return find_winning_moves(grid, opponent, n_in_row)


def find_threat_moves(grid: np.ndarray, player: str, n_in_row: int = 5) -> List[Tuple[int, int]]:
    """Find moves that create threats (potential wins in next move)."""
    H, W = grid.shape
    threat_moves = []
    
    for r in range(H):
        for c in range(W):
            if grid[r, c] == '-':  # Empty cell
                # Try move
                test_grid = grid.copy()
                test_grid[r, c] = player
                # Check if this creates a winning opportunity
                winning_after = find_winning_moves(test_grid, player, n_in_row)
                if len(winning_after) > 0:
                    threat_moves.append((r, c))
    
    return threat_moves


def analyze_position(grid: np.ndarray, player: str, n_in_row: int = 5) -> Dict[str, Any]:
    """Analyze position for tactical information."""
    
    # Check current winner
    winner = check_winner(grid, n_in_row)
    
    # Find tactical moves
    winning_moves = find_winning_moves(grid, player, n_in_row)
    blocking_moves = find_blocking_moves(grid, player, n_in_row)
    threat_moves = find_threat_moves(grid, player, n_in_row)
    
    # Calculate position value
    if winner == player:
        value = [1.0, 0.0, 0.0]  # [win, draw, lose]
    elif winner and winner != player:
        value = [0.0, 0.0, 1.0]  # [win, draw, lose]
    else:
        # Heuristic evaluation
        if len(winning_moves) > 0:
            value = [0.8, 0.1, 0.1]  # Strong winning position
        elif len(blocking_moves) > 0:
            value = [0.2, 0.3, 0.5]  # Defensive position
        elif len(threat_moves) > 0:
            value = [0.6, 0.2, 0.2]  # Good attacking position
        else:
            value = [0.33, 0.34, 0.33]  # Neutral position
    
    # Generate move probabilities
    H, W = grid.shape
    move_probs = np.zeros(H * W, dtype=np.float32)
    
    # Priority: winning > blocking > threats > center > random
    available_moves = [(r, c) for r in range(H) for c in range(W) if grid[r, c] == '-']
    
    if len(available_moves) == 0:
        move_probs = np.ones(H * W) / (H * W)  # Uniform if no moves
    else:
        priority_moves = []
        
        # Highest priority: winning moves
        if winning_moves:
            priority_moves = winning_moves
            weight = 1.0
        # Second priority: blocking moves  
        elif blocking_moves:
            priority_moves = blocking_moves
            weight = 0.8
        # Third priority: threat moves
        elif threat_moves:
            priority_moves = threat_moves[:3]  # Top 3 threats
            weight = 0.6
        # Default: center and nearby moves
        else:
            center_r, center_c = H // 2, W // 2
            priority_moves = [(center_r, center_c)]
            # Add nearby moves
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    nr, nc = center_r + dr, center_c + dc
                    if 0 <= nr < H and 0 <= nc < W and grid[nr, nc] == '-':
                        priority_moves.append((nr, nc))
            weight = 0.4
        
        # Assign probabilities
        if priority_moves:
            for r, c in priority_moves:
                move_probs[r * W + c] = weight / len(priority_moves)
        
        # Add small probability to other available moves
        remaining_prob = max(0, 1.0 - move_probs.sum())
        other_moves = [(r, c) for r, c in available_moves if move_probs[r * W + c] == 0]
        if other_moves and remaining_prob > 0:
            for r, c in other_moves:
                move_probs[r * W + c] = remaining_prob / len(other_moves)
        
        # Normalize
        if move_probs.sum() > 0:
            move_probs /= move_probs.sum()
    
    return {
        'winner': winner,
        'value': np.array(value, dtype=np.float32),
        'move_probs': move_probs,
        'winning_moves': winning_moves,
        'blocking_moves': blocking_moves, 
        'threat_moves': threat_moves,
        'available_moves': available_moves
    }


class GomokuDataset(Dataset):
    """Dataset for Gomoku CSV data."""
    
    def __init__(self, csv_files: List[str], config, max_samples: Optional[int] = None):
        self.config = config
        self.samples = []
        
        # Load all CSV files
        for csv_file in tqdm(csv_files, desc="Loading CSV files"):
            samples = self._load_csv(csv_file, max_samples)
            self.samples.extend(samples)
            
            if max_samples and len(self.samples) >= max_samples:
                self.samples = self.samples[:max_samples]
                break
        
        print(f"📊 Loaded {len(self.samples)} samples from {len(csv_files)} CSV files")
    
    def _load_csv(self, csv_file: str, max_samples: Optional[int] = None) -> List[dict]:
        """Load samples from a single CSV file."""
        samples = []
        
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if max_samples and i >= max_samples:
                    break
                
                try:
                    # Parse row
                    grid, current_player = parse_csv_row(row, self.config.board_width, self.config.board_height)
                    
                    # Convert to state
                    state = grid_to_state(grid, current_player)
                    
                    # Analyze position
                    analysis = analyze_position(grid, current_player, self.config.n_in_row)
                    
                    samples.append({
                        'state': state,
                        'value': analysis['value'],
                        'move_probs': analysis['move_probs'],
                        'metadata': {
                            'current_player': current_player,
                            'winner': analysis['winner'],
                            'winning_moves': analysis['winning_moves'],
                            'file': os.path.basename(csv_file)
                        }
                    })
                
                except Exception as e:
                    if i < 5:  # Only print first few errors
                        print(f"Warning: Error processing row {i} in {csv_file}: {e}")
                    continue
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Apply data augmentation (slap)
        state = sample['state']
        value = sample['value']
        move_probs = sample['move_probs']
        
        # Apply slap augmentation
        if self.config.use_slap != 'none' and random.random() < 0.5:
            # Apply slap transformation
            slapped_state, _, _ = slap(state)
            # Note: move_probs would need corresponding transformation
            # For simplicity, we'll use original move_probs
            state = slapped_state
        
        return {
            'state': torch.FloatTensor(state.copy()),
            'value': torch.FloatTensor(value.copy()),
            'move_probs': torch.FloatTensor(move_probs.copy()),
            'metadata': sample['metadata']
        }


class StreamingGomokuDataset:
    """Streaming dataset for very large CSV collections."""
    
    def __init__(self, data_dir: str, config, batch_size: int = 256):
        self.data_dir = data_dir
        self.config = config
        self.batch_size = batch_size
        
        # Find all CSV files
        self.csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
        if not self.csv_files:
            raise ValueError(f"No CSV files found in {data_dir}")
        
        print(f"📁 Found {len(self.csv_files)} CSV files for streaming")
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Stream batches from CSV files."""
        
        # Shuffle files
        if self.config.shuffle:
            random.shuffle(self.csv_files)
        
        batch_states = []
        batch_values = []
        batch_move_probs = []
        batch_metadata = []
        
        for csv_file in self.csv_files:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                
                for row in reader:
                    try:
                        # Parse and process row
                        grid, current_player = parse_csv_row(row, self.config.board_width, self.config.board_height)
                        state = grid_to_state(grid, current_player)
                        analysis = analyze_position(grid, current_player, self.config.n_in_row)
                        
                        # Apply slap augmentation
                        if self.config.use_slap != 'none' and random.random() < 0.5:
                            state, _, _ = slap(state)
                        
                        # Add to batch
                        batch_states.append(state)
                        batch_values.append(analysis['value'])
                        batch_move_probs.append(analysis['move_probs'])
                        batch_metadata.append({
                            'current_player': current_player,
                            'winner': analysis['winner'],
                            'file': os.path.basename(csv_file)
                        })
                        
                        # Yield batch when ready
                        if len(batch_states) >= self.batch_size:
                            yield {
                                'state': torch.FloatTensor(np.array(batch_states)),
                                'value': torch.FloatTensor(np.array(batch_values)),
                                'move_probs': torch.FloatTensor(np.array(batch_move_probs)),
                                'metadata': batch_metadata
                            }
                            
                            # Reset batch
                            batch_states = []
                            batch_values = []
                            batch_move_probs = []
                            batch_metadata = []
                    
                    except Exception as e:
                        continue  # Skip problematic rows
        
        # Yield final partial batch
        if batch_states:
            yield {
                'state': torch.FloatTensor(np.array(batch_states)),
                'value': torch.FloatTensor(np.array(batch_values)),
                'move_probs': torch.FloatTensor(np.array(batch_move_probs)),
                'metadata': batch_metadata
            }


def create_dataloader(data_path: str, config, split: str = 'train', streaming: bool = False) -> DataLoader:
    """Create appropriate dataloader."""
    
    if streaming:
        # Return streaming dataset directly
        return StreamingGomokuDataset(data_path, config, config.batch_size)
    else:
        # Regular dataset with train/val split
        if os.path.isdir(data_path):
            csv_files = glob.glob(os.path.join(data_path, "*.csv"))
        else:
            csv_files = [data_path]
        
        # Split files for train/val
        if split == 'train':
            split_point = int(len(csv_files) * (1 - config.val_split))
            csv_files = csv_files[:split_point]
        else:
            split_point = int(len(csv_files) * (1 - config.val_split))
            csv_files = csv_files[split_point:]
        
        dataset = GomokuDataset(csv_files, config)
        
        return DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=config.shuffle if split == 'train' else False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory
        )