# -*- coding: utf-8 -*-
"""
Configuration for Pure CNN Gomoku Training (No Reinforcement Learning)

This project focuses on supervised CNN training for Gomoku position evaluation
and move prediction, with slap canonicalization for data augmentation.
NO MCTS, NO self-play, NO reinforcement learning components.
"""

class Config:
    """Configuration for pure CNN Gomoku training."""
    
    # == Board Settings ==
    board_width: int = 5        # 5x5 Gomoku by default
    board_height: int = 5       # Can be extended to larger boards
    n_in_row: int = 5          # Win condition
    
    # == CNN Architecture ==
    in_channels: int = 4        # [current_player, opponent, position_info, bias]
    num_filters: int = 32       # Base number of CNN filters
    num_ResBlock: int = 2       # ResNet blocks (0 for simple CNN)
    dropout: float = 0.2        # Dropout rate
    extra_fc: bool = True       # Extra fully connected layer
    
    # == Slap Configuration ==
    use_slap: str = 'replace'   # 'replace', 'add', or 'none'
                                # 'replace': canonicalize input states
                                # 'add': dual-slice input (original + canonical)
                                # 'none': no slap augmentation
    
    # == Training Hyperparameters ==
    batch_size: int = 256       # Training batch size (512 for Mac M4)
    learning_rate: float = 0.001
    weight_decay: float = 1e-4  # L2 regularization
    optimizer: str = 'Adam'     # 'Adam', 'SGD', 'AdamW'
    
    # == Training Control ==
    epochs: int = 50
    val_split: float = 0.15     # Validation split ratio
    early_stopping: int = 10    # Stop if no improvement for N epochs
    
    # == Mac M4 Optimization ==
    use_mps: bool = True        # Use MPS acceleration on Mac
    pin_memory: bool = True     # Pin memory for faster GPU transfer
    num_workers: int = 2        # DataLoader workers (Mac M4 optimized)
    
    # == Logging & Checkpoints ==
    checkpoint_every: int = 5   # Save checkpoint every N epochs
    log_every: int = 100        # Log progress every N batches
    save_best_only: bool = True # Only save best validation models
    
    # == Data Pipeline ==
    streaming: bool = False     # Use streaming for large datasets (70GB+)
    buffer_size: int = 10000    # Buffer size for streaming
    shuffle: bool = True        # Shuffle training data
    
    # == Model Output ==
    model_types = {
        'value_only': 'Position evaluation only (win/lose/draw)',
        'move_prediction': 'Next best move prediction', 
        'dual_head': 'Both position evaluation + move prediction'
    }
    default_model_type: str = 'dual_head'
    
    def __init__(self, **kwargs):
        """Allow runtime configuration override."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: Unknown config parameter '{key}'")
    
    def get_device(self):
        """Get optimal device for Mac M4."""
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and self.use_mps:
            return torch.device('mps')
        else:
            return torch.device('cpu')
    
    def summary(self):
        """Print configuration summary."""
        print("🎯 Pure CNN Gomoku Configuration:")
        print(f"   Board: {self.board_width}x{self.board_height}, {self.n_in_row}-in-a-row")
        print(f"   CNN: {self.num_filters} filters, {self.num_ResBlock} ResBlocks")
        print(f"   Slap: {self.use_slap}")
        print(f"   Training: {self.batch_size} batch, {self.learning_rate} lr, {self.epochs} epochs")
        print(f"   Device: {self.get_device()}")
        print(f"   Model Type: {self.default_model_type}")


# Import torch here to avoid circular imports
import torch