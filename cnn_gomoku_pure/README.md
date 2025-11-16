# Pure CNN Gomoku Training 🎯

**Complete pure CNN training pipeline for Gomoku/TicTacToe position evaluation and move prediction.**

**NO reinforcement learning, NO MCTS, NO self-play** - just supervised CNN training with slap canonicalization.

## 🚀 Features

### ✅ **Pure Supervised Learning**
- Position evaluation: Win/Draw/Lose classification
- Move prediction: Best move suggestions  
- Tactical analysis: Winning moves, blocks, threats

### ✅ **Advanced CNN Architecture**
- Simple CNN or ResNet backbone
- Dual-head design (value + move prediction)
- Dropout regularization & batch normalization

### ✅ **Slap Canonicalization Integration**
- `replace`: Canonical input normalization
- `add`: Dual-slice input (original + canonical)
- Data augmentation for better generalization

### ✅ **Mac M4 Optimization**
- MPS GPU acceleration
- Streaming data loading for 70GB+ datasets
- Memory efficient processing
- Graceful shutdown handling

### ✅ **Production Ready**
- Checkpoint/resume system
- Model evaluation & visualization
- Human vs AI gameplay
- Data format conversion utilities

---

## 📦 Installation

```bash
# Clone and setup
git clone <repo>
cd cnn_gomoku_pure

# Install dependencies (Mac M4 optimized)
pip install -r requirements.txt

# For Mac M4 MPS support:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

---

## 📊 Data Preparation

### **Option 1: Convert TicTacToe Data**

```bash
# Analyze your TicTacToe CSV format
uv run convert_ttt_data.py --input /Users/kimcuong/source/python/TicTacToeNN/data/tictactoe_games.csv --analyze-only

# Convert to training format
uv run convert_ttt_data.py \
    --input /Users/kimcuong/source/python/TicTacToeNN/data/tictactoe_games.csv \
    --output data/tictactoe_converted.csv \
    --board-size 3
```

### **Option 2: Use Gomoku CSV Data**
Expected format: `[board_string, current_player, cell_0, cell_1, ..., cell_24]`
- `current_player`: "X" or "O"  
- `cell_i`: "X", "O", or "-"

---

## 🏃‍♂️ Training

### **Basic Training**
```bash
# Quick training on small dataset
uv run train.py --data data/tictactoe_converted.csv --epochs 20

# Custom model configuration
uv run train.py \
    --data data/tictactoe_converted.csv \
    --model-type dual_head \
    --resnet-blocks 2 \
    --filters 32 \
    --slap replace \
    --epochs 50 \
    --batch-size 256
```

### **Mac M4 Large Dataset Training**
```bash
# Streaming mode for 70GB+ datasets
uv run train.py \
    --data /path/to/large/dataset \
    --streaming \
    --batch-size 512 \
    --epochs 100 \
    --output models/m4_optimized.pt
```

### **Resume Training**
```bash
# Resume from checkpoint
uv run train.py \
    --data data/my_data.csv \
    --resume checkpoints/checkpoint_epoch_10.pt \
    --epochs 50
```

---

## 🎮 Play Against AI

```bash
# Auto-detect best model, you go first
uv run play.py --first human

# AI goes first, show position evaluation
uv run play.py --show-eval

# Use specific model
uv run play.py --model models/best_model.pt --first human

# Larger board
uv run play.py --board-size 8 --first human
```

**Gameplay Example:**
```
🎮 Pure CNN Gomoku - 5x5
🎯 Goal: Get 5 in a row
👤 Human: X, 🤖 AI: O

   0 1 2 3 4
  ───────────
0 │ · · · · ·
1 │ · · X · ·
2 │ · · · · ·
3 │ · · · · ·
4 │ · · · · ·

🧠 AI thinks: Win=0.234, Draw=0.456, Lose=0.310
💡 AI suggests: (2,2):0.234, (1,1):0.198, (3,2):0.156

Your move (row,col): 2,2
```

---

## 📈 Model Evaluation

### **Single Model Evaluation**
```bash
# Basic evaluation
uv run evaluate.py --model models/best_model.pt --data test_data.csv

# Detailed analysis with visualizations
uv run evaluate.py \
    --model models/best_model.pt \
    --data test_data.csv \
    --detailed \
    --output-dir evaluation_results
```

### **Compare Multiple Models**
```bash
# Compare different architectures
uv run evaluate.py \
    --models models/simple_cnn.pt models/resnet.pt models/dual_head.pt \
    --data test_data.csv
```

**Generated Visualizations:**
- Training loss curves
- Confusion matrix for position evaluation
- Move prediction accuracy heatmap
- Detailed sample analysis

---

## 🏗️ Project Structure

```
cnn_gomoku_pure/
├── config.py              # Configuration management
├── cnn_model.py           # Pure CNN architecture (no RL)
├── data_pipeline.py       # CSV loading + tactical analysis
├── train.py              # Main training script
├── play.py               # Human vs AI gameplay
├── evaluate.py           # Model evaluation + visualization
├── convert_ttt_data.py   # TicTacToe data converter
├── requirements.txt      # Dependencies
└── README.md            # This file

data/                    # Training data
├── tictactoe_converted.csv
└── your_gomoku_data.csv

models/                  # Trained models
├── best_model.pt
├── final_model.pt
└── emergency_save_*.pt

logs/                    # Training logs
├── training_history.json
└── loss_plots.png

checkpoints/             # Training checkpoints
├── checkpoint_epoch_5.pt
└── checkpoint_epoch_10.pt

evaluation_results/      # Evaluation outputs
├── confusion_matrix.png
├── move_accuracy_heatmap.png
└── detailed_analysis.txt
```

---

## ⚙️ Configuration Options

### **Model Types**
- `value_only`: Position evaluation only (Win/Draw/Lose)
- `move_prediction`: Best move prediction only
- `dual_head`: Both position evaluation + move prediction ⭐

### **Slap Modes**
- `none`: No slap augmentation
- `replace`: Canonical input normalization ⭐
- `add`: Dual-slice input (original + canonical)

### **CNN Architecture**
- `num_ResBlock`: 0 = Simple CNN, >0 = ResNet depth
- `num_filters`: Base CNN filter count (32, 64, 128)
- `dropout`: Regularization (0.1-0.3)

---

## 🧪 Examples

### **TicTacToe 3x3 Training**
```bash
# 1. Convert your TicTacToe data
uv run convert_ttt_data.py \
    --input /Users/kimcuong/source/python/TicTacToeNN/data/tictactoe_games.csv \
    --output data/ttt_converted.csv \
    --board-size 3

# 2. Train CNN model
uv run train.py \
    --data data/ttt_converted.csv \
    --model-type dual_head \
    --resnet-blocks 1 \
    --slap replace \
    --epochs 30 \
    --batch-size 128 \
    --output models/tictactoe_cnn.pt

# 3. Play against trained AI
uv run play.py \
    --model models/tictactoe_cnn.pt \
    --board-size 3 \
    --first human \
    --show-eval

# 4. Evaluate model performance
uv run evaluate.py \
    --model models/tictactoe_cnn.pt \
    --data data/ttt_converted.csv \
    --detailed
```

### **Gomoku 5x5 Training**
```bash
# Train on Gomoku data
uv run train.py \
    --data data/gomoku_5x5.csv \
    --model-type dual_head \
    --resnet-blocks 2 \
    --filters 64 \
    --slap replace \
    --epochs 100 \
    --batch-size 256 \
    --output models/gomoku_5x5.pt
```

---

## 🔧 Advanced Usage

### **Custom Data Format**
If your CSV has a custom format, specify column mappings:

```bash
uv run convert_ttt_data.py \
    --input your_custom.csv \
    --output data/converted.csv \
    --board-cols 0 1 2 3 4 5 6 7 8 \
    --player-col 9 \
    --outcome-col 10 \
    --board-size 3
```

### **Streaming for Large Datasets**
For datasets > 10GB:

```bash
uv run train.py \
    --data /path/to/huge/dataset/ \
    --streaming \
    --batch-size 512 \
    --epochs 200 \
    --checkpoint-every 1000
```

### **Mac M4 Memory Optimization**
```bash
# Optimal settings for Mac M4 24GB RAM
uv run train.py \
    --data large_dataset/ \
    --streaming \
    --batch-size 512 \
    --device mps \
    --epochs 50
```

---

## 🤔 Troubleshooting

### **Common Issues**

**"ModuleNotFoundError: slap6b"**
```bash
# Slap functions are included in cnn_model.py, no external dependency needed
```

**"CUDA out of memory"**
```bash
# Reduce batch size or use streaming
uv run train.py --data your_data.csv --batch-size 128 --streaming
```

**"CSV format not recognized"**
```bash
# Analyze your data first
uv run convert_ttt_data.py --input your_data.csv --analyze-only
```

### **Performance Tips**

- **Use slap canonicalization** (`--slap replace`) for better generalization
- **Start with dual_head model** for both position + move learning  
- **Use ResNet** (`--resnet-blocks 2`) for deeper learning on larger datasets
- **Enable streaming** for datasets > 1GB to avoid memory issues

---

## 🎯 Key Differences from RL Approach

| Aspect | Pure CNN (This Project) | AlphaZero/RL Approach |
|--------|-------------------------|----------------------|
| **Training** | Supervised on labeled data | Self-play + reinforcement learning |
| **Data** | CSV positions + outcomes | Generated through gameplay |
| **Architecture** | CNN only | CNN + MCTS |
| **Speed** | Fast inference | Slower (MCTS simulations) |
| **Data Requirements** | Needs labeled dataset | Generates own data |
| **Complexity** | Simple, interpretable | Complex, harder to debug |

---

## 📄 License

MIT License - feel free to use in your projects!

---

## 🙋‍♂️ Getting Started

1. **Convert your TicTacToe data:**
   ```bash
   uv run convert_ttt_data.py --input /Users/kimcuong/source/python/TicTacToeNN/data/tictactoe_games.csv --output data/my_data.csv --board-size 3
   ```

2. **Train CNN model:**
   ```bash
   uv run train.py --data data/my_data.csv --epochs 20
   ```

3. **Play against AI:**
   ```bash
   uv run play.py --first human
   ```

That's it! 🎉 Pure CNN Gomoku training without any reinforcement learning complexity.