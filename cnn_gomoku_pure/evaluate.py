#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Evaluation and Visualization for Pure CNN Gomoku

Evaluate trained models on test data and generate visualizations.
NO reinforcement learning components - pure supervised evaluation.

Usage:
    # Evaluate model on test data
    uv run evaluate.py --model models/best_model.pt --data test_data.csv
    
    # Generate detailed analysis
    uv run evaluate.py --model models/best_model.pt --data test_data.csv --detailed
    
    # Compare multiple models
    uv run evaluate.py --models models/model1.pt models/model2.pt --data test_data.csv
"""

import os
import sys
import argparse
import numpy as np
import torch
import json
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional

from config import Config
from cnn_model import GomokuCNN
from data_pipeline import create_dataloader, GomokuDataset


def evaluate_model(model: GomokuCNN, dataloader, device: str) -> Dict:
    """Evaluate model on dataset."""
    model.model.eval()
    
    total_samples = 0
    total_loss = 0.0
    
    # Metrics for value prediction
    value_correct = 0
    value_samples = 0
    
    # Metrics for move prediction  
    move_top1_correct = 0
    move_top3_correct = 0
    move_samples = 0
    
    predictions = {'values': [], 'moves': [], 'targets': []}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            states = batch['state'].to(device)
            target_values = batch['value'].to(device)
            target_moves = batch['move_probs'].to(device)
            
            # Forward pass
            outputs = model.model(states)
            
            batch_size = states.size(0)
            total_samples += batch_size
            
            # Value head evaluation
            if 'value' in outputs:
                value_pred = outputs['value']
                value_target = target_values.argmax(dim=1)
                
                # Accuracy
                value_pred_class = value_pred.argmax(dim=1)
                value_correct += (value_pred_class == value_target).sum().item()
                value_samples += batch_size
                
                # Store predictions
                predictions['values'].extend(value_pred.cpu().numpy())
                
            # Move head evaluation
            if 'move' in outputs:
                move_pred = outputs['move']
                move_target = target_moves.argmax(dim=1)
                
                # Top-1 accuracy
                move_pred_class = move_pred.argmax(dim=1)
                move_top1_correct += (move_pred_class == move_target).sum().item()
                
                # Top-3 accuracy
                _, top3_pred = move_pred.topk(3, dim=1)
                move_top3_correct += sum([move_target[i] in top3_pred[i] for i in range(batch_size)])
                move_samples += batch_size
                
                # Store predictions
                predictions['moves'].extend(move_pred.cpu().numpy())
            
            # Store targets
            predictions['targets'].extend({
                'value': target_values.cpu().numpy(),
                'move': target_moves.cpu().numpy()
            })
    
    # Calculate metrics
    metrics = {
        'total_samples': total_samples,
        'value_accuracy': value_correct / max(value_samples, 1),
        'move_top1_accuracy': move_top1_correct / max(move_samples, 1),
        'move_top3_accuracy': move_top3_correct / max(move_samples, 1),
        'value_samples': value_samples,
        'move_samples': move_samples
    }
    
    return metrics, predictions


def plot_training_history(history_path: str, output_dir: str):
    """Plot training loss curves."""
    if not os.path.exists(history_path):
        print(f"⚠️  Training history not found: {history_path}")
        return
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Training loss
    epochs = range(1, len(history['train_losses']) + 1)
    axes[0].plot(epochs, history['train_losses'], label='Train Loss', marker='o')
    if history['val_losses']:
        axes[0].plot(epochs[:len(history['val_losses'])], history['val_losses'], 
                    label='Validation Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training History')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Loss distribution
    axes[1].hist(history['train_losses'], bins=20, alpha=0.7, label='Train Loss')
    if history['val_losses']:
        axes[1].hist(history['val_losses'], bins=20, alpha=0.7, label='Val Loss')
    axes[1].set_xlabel('Loss Value')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Loss Distribution')
    axes[1].legend()
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'training_history.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Training history plot saved: {plot_path}")


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                         labels: List[str], output_path: str):
    """Plot confusion matrix."""
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title('Position Value Prediction Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Confusion matrix saved: {output_path}")


def plot_move_accuracy_heatmap(predictions: List[np.ndarray], targets: List[np.ndarray],
                              board_size: int, output_path: str):
    """Plot heatmap of move prediction accuracy by position."""
    
    # Aggregate predictions by board position
    position_correct = np.zeros((board_size, board_size))
    position_total = np.zeros((board_size, board_size))
    
    for pred, target in zip(predictions, targets):
        pred_move = np.argmax(pred)
        target_move = np.argmax(target['move'])
        
        # Convert to board coordinates
        target_r, target_c = target_move // board_size, target_move % board_size
        
        position_total[target_r, target_c] += 1
        if pred_move == target_move:
            position_correct[target_r, target_c] += 1
    
    # Calculate accuracy (avoid division by zero)
    accuracy_map = np.where(position_total > 0, 
                           position_correct / position_total, 
                           np.nan)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(accuracy_map, annot=True, fmt='.2f', cmap='RdYlBu_r',
                cbar_kws={'label': 'Accuracy'}, vmin=0, vmax=1)
    plt.title(f'Move Prediction Accuracy by Board Position ({board_size}x{board_size})')
    plt.xlabel('Column')
    plt.ylabel('Row')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Move accuracy heatmap saved: {output_path}")


def analyze_model_predictions(model: GomokuCNN, test_samples: List[Dict], 
                             output_dir: str, num_samples: int = 10):
    """Analyze individual predictions in detail."""
    
    model.model.eval()
    analysis_dir = os.path.join(output_dir, 'sample_analysis')
    os.makedirs(analysis_dir, exist_ok=True)
    
    with open(os.path.join(analysis_dir, 'detailed_analysis.txt'), 'w') as f:
        f.write("Detailed Model Analysis\n")
        f.write("=" * 50 + "\n\n")
        
        for i, sample in enumerate(test_samples[:num_samples]):
            state = sample['state'].unsqueeze(0)  # Add batch dim
            
            with torch.no_grad():
                pred = model.model(state)
            
            f.write(f"Sample {i+1}:\n")
            f.write(f"Current Player: {sample['metadata']['current_player']}\n")
            f.write(f"Actual Winner: {sample['metadata']['winner']}\n")
            
            if 'value' in pred:
                value_probs = pred['value'][0].cpu().numpy()
                f.write(f"Value Prediction: Win={value_probs[0]:.3f}, Draw={value_probs[1]:.3f}, Lose={value_probs[2]:.3f}\n")
            
            if 'move' in pred:
                move_probs = pred['move'][0].cpu().numpy()
                top_moves = np.argsort(move_probs)[-3:][::-1]  # Top 3 moves
                f.write("Top Move Predictions:\n")
                for j, move_idx in enumerate(top_moves):
                    r, c = move_idx // model.config.board_width, move_idx % model.config.board_width
                    prob = np.exp(move_probs[move_idx])  # Convert from log prob
                    f.write(f"  {j+1}. ({r},{c}): {prob:.3f}\n")
            
            f.write("\n" + "-" * 30 + "\n\n")
    
    print(f"📋 Detailed analysis saved: {analysis_dir}/detailed_analysis.txt")


def compare_models(model_paths: List[str], test_data: str, config: Config) -> Dict:
    """Compare multiple models on the same test data."""
    
    results = {}
    
    # Create test dataloader
    test_loader = create_dataloader(test_data, config, 'val', streaming=False)
    device = config.get_device()
    
    for model_path in model_paths:
        print(f"\n🔍 Evaluating {os.path.basename(model_path)}...")
        
        # Load model
        model = GomokuCNN(config)
        model.load(model_path)
        
        # Evaluate
        metrics, predictions = evaluate_model(model, test_loader, device)
        
        results[model_path] = {
            'metrics': metrics,
            'predictions': predictions
        }
        
        print(f"✅ {os.path.basename(model_path)}:")
        print(f"   Value Accuracy: {metrics['value_accuracy']:.3f}")
        print(f"   Move Top-1 Accuracy: {metrics['move_top1_accuracy']:.3f}")
        print(f"   Move Top-3 Accuracy: {metrics['move_top3_accuracy']:.3f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Pure CNN Gomoku Models")
    
    # Model arguments
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument('--model', help='Single model to evaluate')
    model_group.add_argument('--models', nargs='+', help='Multiple models to compare')
    
    # Data arguments
    parser.add_argument('--data', required=True, help='Test data CSV file or directory')
    parser.add_argument('--output-dir', default='evaluation_results', 
                       help='Output directory for results')
    
    # Evaluation options
    parser.add_argument('--detailed', action='store_true', 
                       help='Generate detailed analysis and visualizations')
    parser.add_argument('--max-samples', type=int, 
                       help='Maximum test samples to evaluate')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine models to evaluate
    if args.model:
        model_paths = [args.model]
    else:
        model_paths = args.models
    
    # Verify model files exist
    for model_path in model_paths:
        if not os.path.exists(model_path):
            print(f"❌ Model not found: {model_path}")
            return
    
    # Create config (load from first model)
    checkpoint = torch.load(model_paths[0], map_location='cpu')
    if 'config' in checkpoint:
        config_dict = checkpoint['config']
        config = Config(**config_dict)
    else:
        print("⚠️  No config found in model, using default")
        config = Config()
    
    print(f"🎯 Evaluating on device: {config.get_device()}")
    
    if len(model_paths) == 1:
        # Single model evaluation
        model_path = model_paths[0]
        print(f"🔍 Evaluating model: {os.path.basename(model_path)}")
        
        # Load model
        model = GomokuCNN(config)
        model.load(model_path)
        
        # Create test dataloader
        test_loader = create_dataloader(args.data, config, 'val', streaming=False)
        device = config.get_device()
        
        # Evaluate
        metrics, predictions = evaluate_model(model, test_loader, device)
        
        # Print results
        print("\n📊 Evaluation Results:")
        print(f"Total Samples: {metrics['total_samples']}")
        if metrics['value_samples'] > 0:
            print(f"Value Accuracy: {metrics['value_accuracy']:.3f} ({metrics['value_samples']} samples)")
        if metrics['move_samples'] > 0:
            print(f"Move Top-1 Accuracy: {metrics['move_top1_accuracy']:.3f}")
            print(f"Move Top-3 Accuracy: {metrics['move_top3_accuracy']:.3f}")
            print(f"Move Samples: {metrics['move_samples']}")
        
        # Save results
        results_path = os.path.join(args.output_dir, 'evaluation_results.json')
        with open(results_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"💾 Results saved: {results_path}")
        
        # Detailed analysis
        if args.detailed:
            print("\n🔬 Generating detailed analysis...")
            
            # Plot training history
            history_path = "logs/training_history.json"
            if os.path.exists(history_path):
                plot_training_history(history_path, args.output_dir)
            
            # Generate visualizations if we have predictions
            if metrics['value_samples'] > 0 and predictions['values']:
                # Confusion matrix for value predictions
                y_true = [np.argmax(target['value']) for target in predictions['targets']]
                y_pred = [np.argmax(pred) for pred in predictions['values']]
                
                cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
                plot_confusion_matrix(y_true, y_pred, ['Win', 'Draw', 'Lose'], cm_path)
            
            if metrics['move_samples'] > 0 and predictions['moves']:
                # Move accuracy heatmap
                heatmap_path = os.path.join(args.output_dir, 'move_accuracy_heatmap.png')
                plot_move_accuracy_heatmap(predictions['moves'], predictions['targets'],
                                         config.board_width, heatmap_path)
            
            # Detailed sample analysis
            test_dataset = GomokuDataset([args.data] if os.path.isfile(args.data) else 
                                       [f for f in os.listdir(args.data) if f.endswith('.csv')], 
                                       config, max_samples=100)
            analyze_model_predictions(model, test_dataset.samples, args.output_dir)
    
    else:
        # Multiple model comparison
        print(f"🔄 Comparing {len(model_paths)} models...")
        results = compare_models(model_paths, args.data, config)
        
        # Save comparison results
        comparison_path = os.path.join(args.output_dir, 'model_comparison.json')
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for model_path, result in results.items():
            serializable_results[os.path.basename(model_path)] = result['metrics']
        
        with open(comparison_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"💾 Comparison results saved: {comparison_path}")
    
    print(f"\n✅ Evaluation complete. Results saved to {args.output_dir}")


if __name__ == '__main__':
    main()