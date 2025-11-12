#!/usr/bin/env python3
"""
TicTacToe Neural Network Training Script (with Visualization)
Author: Kim Tuan Cuong Nguyen

Train a feedforward neural network (MLP) to predict optimal TicTacToe moves.
Includes accuracy/loss visualization.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import logging
import time
from datetime import datetime


# =====================
# Neural Network Model
# =====================
class TicTacToeNet(nn.Module):
    """Feedforward Neural Network for TicTacToe move prediction"""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 10)  # Output: 10 possible moves
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x


# =====================
# Logging Setup
# =====================
def setup_logging():
    """Setup detailed logging for training process"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('tictactoe_training.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def explain_crossentropy_loss(outputs, targets, step_num=0):
    """
    Detailed explanation of CrossEntropyLoss calculation
    """
    print(f"\n🔍 Loss Calculation Explanation (Step {step_num}):")
    print("=" * 60)
    
    # Take first sample for detailed explanation
    sample_output = outputs[0].detach()
    sample_target = targets[0].item()
    
    print(f"📋 Sample Input:")
    print(f"   Raw logits (model output): {sample_output.numpy()}")
    print(f"   Target move: {sample_target}")
    
    # Step 1: Softmax calculation
    softmax_probs = F.softmax(sample_output, dim=0)
    print(f"\n📊 Step 1 - Softmax Probabilities:")
    for i, prob in enumerate(softmax_probs):
        marker = "👉" if i == sample_target else "  "
        print(f"   {marker} Position {i}: {prob.item():.4f}")
    
    # Step 2: Cross-entropy calculation
    target_prob = softmax_probs[sample_target]
    loss_value = -torch.log(target_prob)
    
    print(f"\n🧮 Step 2 - Cross-Entropy Loss:")
    print(f"   Target probability: {target_prob.item():.4f}")
    print(f"   Loss = -log({target_prob.item():.4f}) = {loss_value.item():.4f}")
    
    # Batch loss
    batch_loss = F.cross_entropy(outputs, targets)
    print(f"\n📈 Batch Loss (average): {batch_loss.item():.4f}")
    print("=" * 60)

def visualize_board_state(board_features, target_move, predicted_move=None):
    """
    Visualize the board state for better understanding
    """
    print(f"\n🎮 Board State Visualization:")
    print("-" * 30)
    
    # Extract board positions (first 9 features)
    board = board_features[:9].reshape(3, 3)
    symbols = {-1: '_', 0: 'O', 1: 'X'}
    
    for i in range(3):
        row_str = []
        for j in range(3):
            pos = i * 3 + j + 1  # 1-based position
            symbol = symbols[int(board[i, j].item())]
            if pos == target_move:
                symbol = f"[{symbol}]"  # Highlight target position
            elif predicted_move and pos == predicted_move:
                symbol = f"({symbol})"  # Highlight predicted position
            row_str.append(f"{symbol:^3}")
        print(" | ".join(row_str))
        if i < 2:
            print("-" * 13)
    
    current_player = "X" if board_features[9].item() == 1 else "O"
    print(f"\n📍 Current player: {current_player}")
    print(f"🎯 Target move: Position {target_move}")
    if predicted_move:
        print(f"🤖 Predicted move: Position {predicted_move}")

# =====================
# Enhanced Training Function
# =====================
def train_tictactoe_model(csv_path="tictactoemoves.csv", num_epochs=100, batch_size=64, lr=1e-3, 
                         verbose_loss=False, log_every=10):
    # Setup logging
    logger = setup_logging()
    start_time = time.time()
    
    print("🚀 === TicTacToe Neural Network Training ===")
    logger.info("Training session started")
    
    # 1️⃣ Load dataset
    print(f"\n📂 Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    X = df[['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'player']].values.astype(np.float32)
    y = df['move'].values.astype(np.int64)
    
    # Fix target labels: map -1 to 0, and shift others by +1
    # Original: -1, 1, 2, 3, 4, 5, 6, 7, 8, 9
    # New:       0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    y = np.where(y == -1, 0, y)  # Map -1 to 0 (no move)
    # Note: moves 1-9 stay the same since they're already in valid range

    print(f"✅ Dataset loaded: {len(df)} samples")
    print(f"📊 Target range: {y.min()} to {y.max()}")
    print(f"📈 Feature shape: {X.shape}")
    
    # Log dataset statistics
    logger.info(f"Dataset: {len(df)} samples, feature shape: {X.shape}")
    logger.info(f"Target distribution: {np.bincount(y)}")
    
    # Show sample data explanation
    print(f"\n📋 Data Format Explanation:")
    print(f"   • Features (C1-C9): Board positions (-1=empty, 0=O, 1=X)")
    print(f"   • Feature (player): Current player (0=O, 1=X)")  
    print(f"   • Target (move): Optimal move (0=no move, 1-9=positions)")
    
    # Show first few samples
    print(f"\n🔍 Sample data (first 3 rows):")
    for i in range(min(3, len(df))):
        board_state = X[i][:9]
        current_player = "X" if X[i][9] == 1 else "O"  
        target_move = y[i]
        print(f"   Sample {i+1}: Board={board_state} | Player={current_player} | Target={target_move}")

    # 2️⃣ Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train)
    X_test = torch.tensor(X_test)
    y_train = torch.tensor(y_train)
    y_test = torch.tensor(y_test)

    # 3️⃣ Create dataloaders
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # 4️⃣ Model, loss, optimizer setup
    print(f"\n🧠 Creating neural network model...")
    model = TicTacToeNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    print(f"✅ Model architecture:")
    print(model)
    
    print(f"\n📚 Loss Function: CrossEntropyLoss")
    print(f"   • Combines Softmax + Negative Log-Likelihood")
    print(f"   • Formula: Loss = -log(softmax(output)[target_class])")
    print(f"   • Penalizes wrong predictions exponentially")
    
    print(f"\n🎯 Optimizer: Adam")
    print(f"   • Learning rate: {lr}")
    print(f"   • Weight decay: 1e-5 (L2 regularization)")
    
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    print(f"\n🚀 Starting training for {num_epochs} epochs...")

    # For plotting and logging
    train_losses = []
    train_accs = []
    test_accs = []
    batch_count = 0

    # 5️⃣ Enhanced Training Loop
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_idx, (Xb, yb) in enumerate(train_loader):
            batch_count += 1
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(Xb)
            loss = criterion(outputs, yb)
            
            # Detailed loss explanation for first few batches
            if verbose_loss and batch_count <= 3:
                explain_crossentropy_loss(outputs, yb, batch_count)
                
                # Show board visualization for first sample in batch
                if batch_count == 1:
                    _, predicted = torch.max(outputs, 1)
                    visualize_board_state(Xb[0], yb[0].item(), predicted[0].item())
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Accumulate metrics
            running_loss += loss.item() * Xb.size(0)
            
            # Training accuracy calculation
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == yb).sum().item()
            total_train += yb.size(0)
            
            # Log detailed batch info occasionally
            if verbose_loss and batch_idx % 50 == 0:
                batch_loss = loss.item()
                batch_acc = 100 * (predicted == yb).sum().item() / yb.size(0)
                print(f"   Batch {batch_idx}: Loss={batch_loss:.4f}, Acc={batch_acc:.2f}%")

        # Calculate epoch metrics
        epoch_time = time.time() - epoch_start_time
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = 100 * correct_train / total_train

        # Evaluate on test set
        model.eval()
        correct_test = 0
        total_test = 0
        test_loss_total = 0
        
        with torch.no_grad():
            for Xb, yb in test_loader:
                outputs = model(Xb)
                test_loss_total += criterion(outputs, yb).item() * Xb.size(0)
                _, predicted = torch.max(outputs, 1)
                total_test += yb.size(0)
                correct_test += (predicted == yb).sum().item()
        
        test_acc = 100 * correct_test / total_test
        test_loss = test_loss_total / len(test_loader.dataset)

        # Store metrics
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        test_accs.append(test_acc)
        
        # Enhanced logging
        logger.info(f"Epoch {epoch+1}: Train Loss={epoch_loss:.6f}, Train Acc={epoch_acc:.2f}%, Test Loss={test_loss:.6f}, Test Acc={test_acc:.2f}%, Time={epoch_time:.2f}s")

        # Progress reporting
        if (epoch + 1) % log_every == 0:
            print(f"\n📊 Epoch [{epoch + 1:3d}/{num_epochs}] Report:")
            print(f"   🔥 Train Loss: {epoch_loss:.6f} | Train Accuracy: {epoch_acc:.2f}%")
            print(f"   🎯 Test Loss:  {test_loss:.6f} | Test Accuracy:  {test_acc:.2f}%") 
            print(f"   ⏱️  Time: {epoch_time:.2f}s | Total: {time.time()-start_time:.1f}s")
            
            # Loss interpretation
            if epoch_loss > 2.0:
                print(f"   💡 Loss Analysis: High loss - model still learning basic patterns")
            elif epoch_loss > 1.0:
                print(f"   💡 Loss Analysis: Medium loss - model making progress") 
            elif epoch_loss > 0.5:
                print(f"   💡 Loss Analysis: Low loss - model performing well")
            else:
                print(f"   💡 Loss Analysis: Very low loss - model highly confident")
            
            # Show prediction confidence for a sample
            if verbose_loss and len(test_loader.dataset) > 0:
                sample_idx = 0
                sample_x = X_test[sample_idx:sample_idx+1]
                sample_y = y_test[sample_idx]
                
                with torch.no_grad():
                    sample_output = model(sample_x)
                    sample_probs = F.softmax(sample_output, dim=1)[0]
                    _, sample_pred = torch.max(sample_output, 1)
                
                print(f"   🔍 Sample Prediction Confidence:")
                print(f"      Target: {sample_y.item()}, Predicted: {sample_pred.item()}")
                print(f"      Confidence: {sample_probs[sample_pred].item():.3f}")
                top3_probs, top3_indices = torch.topk(sample_probs, 3)
                print(f"      Top 3 moves: {[(idx.item(), prob.item()) for idx, prob in zip(top3_indices, top3_probs)]}")

    # 6️⃣ Visualization
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Training Acc')
    plt.plot(test_accs, label='Test Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Curve')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # 7️⃣ Save model with detailed info
    total_time = time.time() - start_time
    model_path = "TicTacToe_Model.pth"
    torch.save(model.state_dict(), model_path)
    
    print(f"\n🎉 Training Complete!")
    print(f"   📁 Model saved to: {model_path}")
    print(f"   📊 Final Training Accuracy: {train_accs[-1]:.2f}%")
    print(f"   📊 Final Test Accuracy: {test_accs[-1]:.2f}%")
    print(f"   ⏱️  Total Training Time: {total_time:.1f} seconds")
    print(f"   📈 Final Loss: {train_losses[-1]:.6f}")
    
    # Log final results
    logger.info(f"Training completed in {total_time:.1f}s")
    logger.info(f"Final metrics - Train Acc: {train_accs[-1]:.2f}%, Test Acc: {test_accs[-1]:.2f}%, Loss: {train_losses[-1]:.6f}")
    
    # Performance analysis
    if test_accs[-1] > 90:
        print(f"   🏆 Excellent performance! Model is highly accurate.")
    elif test_accs[-1] > 75:
        print(f"   ✅ Good performance! Model is reasonably accurate.")
    elif test_accs[-1] > 60:
        print(f"   ⚠️  Moderate performance. Consider more training or data.")
    else:
        print(f"   ❌ Poor performance. Model needs significant improvement.")

    return model


# =====================
# Test Model Function
# =====================
def test_model_state(model_path="TicTacToe_Model.pth"):
    """Load trained model and test with custom input"""
    model = TicTacToeNet()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Example board state
    test_state = torch.tensor([[1, 0, -1, -1, -1, -1, -1, -1, -1, 0]], dtype=torch.float32)
    with torch.no_grad():
        output = model(test_state)
        _, predicted_move = torch.max(output, 1)
        print("\n=== Model Test ===")
        print(f"Board state: {test_state[0][:9].tolist()}")
        print(f"Current player: {'O (0)' if test_state[0][9] == 0 else 'X (1)'}")
        print(f"Predicted move index: {predicted_move.item()}")
        
        # Note: Model outputs 0-9 where 0=no move, 1-9=board positions
        if predicted_move.item() == 0:
            print("Predicted: No move (game ended)")
        else:
            print(f"Predicted: Move to position {predicted_move.item()}")
        print(f"Raw logits: {output.tolist()}")


if __name__ == "__main__":
    try:
        print("🎮 TicTacToe Neural Network Training with Detailed Logging")
        print("=" * 60)
        
        # Training with detailed explanations
        model = train_tictactoe_model(
            csv_path="tictactoemoves.csv",
            num_epochs=50,  # Reduced for demonstration
            batch_size=64,
            lr=1e-3,
            verbose_loss=True,  # Enable detailed loss explanations
            log_every=5         # Log every 5 epochs
        )
        
        print("\n🧪 Testing trained model...")
        test_model_state("TicTacToe_Model.pth")

    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Make sure 'tictactoemoves.csv' exists in the current directory.")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()