#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pure CNN Training for Gomoku - No Reinforcement Learning

Supervised training on CSV data for position evaluation and move prediction.
Optimized for Mac M4 with large datasets and graceful shutdown handling.

NO MCTS, NO self-play, NO reinforcement learning components.

Usage:
    # Basic training
    uv run train.py --data data/csv_files --epochs 20
    
    # Mac M4 optimized for large datasets  
    uv run train.py --data /path/to/70GB/data --epochs 50 --streaming --batch-size 512
    
    # Custom model configuration
    uv run train.py --data data --model-type dual_head --resnet-blocks 3 --slap replace
"""

import os
import sys
import argparse
import signal
import time
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Local imports
from config import Config
from cnn_model import GomokuCNN
from data_pipeline import create_dataloader


class TrainingManager:
    """Manage training with graceful shutdown and checkpointing."""
    
    def __init__(self, config):
        self.config = config
        self.interrupted = False
        self.model = None
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C and other termination signals."""
        print(f"\n🛑 Received signal {signum}. Initiating graceful shutdown...")
        self.interrupted = True
    
    def save_checkpoint(self, model, optimizer, epoch, train_loss, val_loss, is_best=False):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'best_val_loss': self.best_val_loss,
            'config': self.config.__dict__,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        
        # Save checkpoint
        checkpoint_path = f"checkpoints/checkpoint_epoch_{epoch}.pt"
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            model.save(f"models/best_model_epoch_{epoch}.pt", {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'timestamp': time.time()
            })
        
        print(f"💾 Checkpoint saved: {checkpoint_path}")
    
    def save_emergency_model(self, model):
        """Emergency save when interrupted."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        emergency_path = f"models/emergency_save_{timestamp}.pt"
        os.makedirs("models", exist_ok=True)
        
        model.save(emergency_path, {
            'emergency_save': True,
            'epoch': self.current_epoch,
            'timestamp': time.time(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        })
        
        print(f"🚨 Emergency model saved: {emergency_path}")
        return emergency_path
    
    def load_checkpoint(self, model, optimizer, checkpoint_path):
        """Load training checkpoint to resume."""
        if not os.path.exists(checkpoint_path):
            print(f"❌ Checkpoint not found: {checkpoint_path}")
            return 0
        
        checkpoint = torch.load(checkpoint_path, map_location=model.device)
        
        model.model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        
        print(f"📂 Resumed from epoch {self.current_epoch}, best val loss: {self.best_val_loss:.4f}")
        return self.current_epoch


def calculate_loss(outputs, targets, model_type):
    """Calculate appropriate loss based on model type."""
    losses = {}
    total_loss = 0
    
    if 'value' in outputs:
        # Classification loss for position evaluation
        value_loss = F.cross_entropy(outputs['value'], targets['value'].argmax(dim=1))
        losses['value_loss'] = value_loss
        total_loss += value_loss
    
    if 'move' in outputs:
        # Cross-entropy loss for move prediction
        move_loss = F.kl_div(outputs['move'], targets['move_probs'], reduction='batchmean')
        losses['move_loss'] = move_loss  
        total_loss += move_loss
    
    losses['total_loss'] = total_loss
    return losses


def train_epoch(model, dataloader, optimizer, manager, epoch, device):
    """Train for one epoch."""
    model.model.train()
    
    running_loss = 0.0
    running_value_loss = 0.0
    running_move_loss = 0.0
    num_batches = 0
    
    # Use tqdm for progress tracking
    if hasattr(dataloader, '__iter__'):  # Streaming dataset
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    else:
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        if manager.interrupted:
            print("🛑 Training interrupted by user")
            break
        
        # Move to device
        states = batch['state'].to(device)
        values = batch['value'].to(device)
        move_probs = batch['move_probs'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model.model(states)
        
        # Calculate loss
        targets = {'value': values, 'move_probs': move_probs}
        losses = calculate_loss(outputs, targets, model.config.default_model_type)
        
        # Backward pass
        losses['total_loss'].backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Update running losses
        running_loss += losses['total_loss'].item()
        if 'value_loss' in losses:
            running_value_loss += losses['value_loss'].item()
        if 'move_loss' in losses:
            running_move_loss += losses['move_loss'].item()
        num_batches += 1
        
        # Update progress bar
        avg_loss = running_loss / num_batches
        pbar.set_postfix({
            'loss': f'{avg_loss:.4f}',
            'val_loss': f'{running_value_loss/num_batches:.4f}' if running_value_loss > 0 else 'N/A',
            'move_loss': f'{running_move_loss/num_batches:.4f}' if running_move_loss > 0 else 'N/A'
        })
        
        # Log periodically
        if batch_idx % manager.config.log_every == 0 and batch_idx > 0:
            print(f"Batch {batch_idx}: loss={avg_loss:.4f}")
    
    return running_loss / max(num_batches, 1)


def validate(model, dataloader, manager, device):
    """Validate model."""
    model.model.eval()
    
    running_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            if manager.interrupted:
                break
            
            # Move to device
            states = batch['state'].to(device)
            values = batch['value'].to(device)
            move_probs = batch['move_probs'].to(device)
            
            # Forward pass
            outputs = model.model(states)
            
            # Calculate loss
            targets = {'value': values, 'move_probs': move_probs}
            losses = calculate_loss(outputs, targets, model.config.default_model_type)
            
            running_loss += losses['total_loss'].item()
            num_batches += 1
    
    return running_loss / max(num_batches, 1)


def main():
    parser = argparse.ArgumentParser(description="Pure CNN Gomoku Training")
    
    # Data arguments
    parser.add_argument('--data', required=True, help='Path to CSV data directory or file')
    parser.add_argument('--streaming', action='store_true', help='Use streaming for large datasets')
    
    # Model arguments
    parser.add_argument('--model-type', choices=['value_only', 'move_prediction', 'dual_head'], 
                       default='dual_head', help='Type of model to train')
    parser.add_argument('--resnet-blocks', type=int, default=2, help='Number of ResNet blocks')
    parser.add_argument('--filters', type=int, default=32, help='Base number of CNN filters')
    parser.add_argument('--slap', choices=['none', 'replace', 'add'], default='replace',
                       help='Slap canonicalization mode')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--optimizer', choices=['Adam', 'SGD', 'AdamW'], default='Adam')
    
    # System arguments
    parser.add_argument('--device', choices=['auto', 'cpu', 'mps', 'cuda'], default='auto')
    parser.add_argument('--resume', help='Checkpoint path to resume from')
    parser.add_argument('--output', default='models/final_model.pt', help='Output model path')
    
    args = parser.parse_args()
    
    # Create config
    config = Config(
        default_model_type=args.model_type,
        num_ResBlock=args.resnet_blocks,
        num_filters=args.filters,
        use_slap=args.slap,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        optimizer=args.optimizer,
        streaming=args.streaming
    )
    
    # Set device
    if args.device != 'auto':
        if args.device == 'mps':
            config.use_mps = True
        elif args.device == 'cpu':
            config.use_mps = False
    
    device = config.get_device()
    print(f"🖥️  Using device: {device}")
    
    # Print configuration
    config.summary()
    
    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    
    # Initialize model
    model = GomokuCNN(config)
    model.summary()
    
    # Create training manager
    manager = TrainingManager(config)
    manager.model = model
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        start_epoch = manager.load_checkpoint(model, model.optimizer, args.resume) + 1
    
    # Create data loaders
    print("📊 Loading training data...")
    train_loader = create_dataloader(args.data, config, 'train', config.streaming)
    
    if not config.streaming:
        print("📊 Loading validation data...")
        val_loader = create_dataloader(args.data, config, 'val', False)
    else:
        val_loader = None
        print("⚠️  Streaming mode: no separate validation set")
    
    # Training loop
    print(f"🚀 Starting training from epoch {start_epoch}")
    
    no_improve_count = 0
    
    for epoch in range(start_epoch, config.epochs):
        if manager.interrupted:
            break
        
        manager.current_epoch = epoch
        
        # Training
        print(f"\n📈 Epoch {epoch + 1}/{config.epochs}")
        train_loss = train_epoch(model, train_loader, model.optimizer, manager, epoch + 1, device)
        manager.train_losses.append(train_loss)
        
        # Validation (if not streaming)
        val_loss = None
        if val_loader is not None:
            val_loss = validate(model, val_loader, manager, device)
            manager.val_losses.append(val_loss)
            
            # Check for improvement
            is_best = val_loss < manager.best_val_loss
            if is_best:
                manager.best_val_loss = val_loss
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} {'🏆' if is_best else ''}")
            
            # Early stopping
            if no_improve_count >= config.early_stopping:
                print(f"🛑 Early stopping after {no_improve_count} epochs without improvement")
                break
        else:
            print(f"Train Loss: {train_loss:.4f}")
            is_best = epoch % 5 == 0  # Save every 5 epochs in streaming mode
        
        # Save checkpoint
        if (epoch + 1) % config.checkpoint_every == 0:
            manager.save_checkpoint(model, model.optimizer, epoch + 1, train_loss, val_loss, is_best)
        
        if manager.interrupted:
            break
    
    # Final save
    if manager.interrupted:
        # Emergency save
        emergency_path = manager.save_emergency_model(model)
        print(f"🚨 Training interrupted. Model saved to {emergency_path}")
    else:
        # Normal completion
        model.save(args.output, {
            'final_model': True,
            'epochs_trained': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'timestamp': time.time()
        })
        print(f"✅ Training completed. Final model saved to {args.output}")
    
    # Save training history
    history = {
        'train_losses': manager.train_losses,
        'val_losses': manager.val_losses,
        'best_val_loss': manager.best_val_loss,
        'config': config.__dict__
    }
    
    history_path = "logs/training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"📈 Training history saved to {history_path}")


if __name__ == '__main__':
    main()