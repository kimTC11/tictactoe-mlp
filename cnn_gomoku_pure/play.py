#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Human vs AI Gameplay for Pure CNN Gomoku

Simple gameplay interface without MCTS - just pure CNN predictions.
Supports position evaluation and move prediction models.

Usage:
    # Play against CNN (you go first)
    uv run play.py --model models/final_model.pt --first human
    
    # AI goes first
    uv run play.py --model models/final_model.pt
    
    # Load best available model automatically
    uv run play.py --first human
"""

import os
import sys
import argparse
import numpy as np
import torch
from typing import Optional, Tuple, List

from config import Config
from cnn_model import GomokuCNN
from data_pipeline import grid_to_state, analyze_position


class GomokuGame:
    """Simple Gomoku game logic."""
    
    def __init__(self, width: int = 5, height: int = 5, n_in_row: int = 5):
        self.width = width
        self.height = height
        self.n_in_row = n_in_row
        self.reset()
    
    def reset(self):
        """Reset game to initial state."""
        self.board = np.full((self.height, self.width), '-', dtype=str)
        self.current_player = 'X'
        self.move_count = 0
        self.winner = None
        self.game_over = False
    
    def get_available_moves(self) -> List[Tuple[int, int]]:
        """Get list of available moves."""
        moves = []
        for r in range(self.height):
            for c in range(self.width):
                if self.board[r, c] == '-':
                    moves.append((r, c))
        return moves
    
    def is_valid_move(self, row: int, col: int) -> bool:
        """Check if move is valid."""
        return (0 <= row < self.height and 
                0 <= col < self.width and 
                self.board[row, col] == '-')
    
    def make_move(self, row: int, col: int) -> bool:
        """Make a move. Return True if successful."""
        if not self.is_valid_move(row, col):
            return False
        
        self.board[row, col] = self.current_player
        self.move_count += 1
        
        # Check for winner
        if self._check_winner(row, col):
            self.winner = self.current_player
            self.game_over = True
        elif len(self.get_available_moves()) == 0:
            self.winner = 'Draw'
            self.game_over = True
        else:
            # Switch players
            self.current_player = 'O' if self.current_player == 'X' else 'X'
        
        return True
    
    def _check_winner(self, last_row: int, last_col: int) -> bool:
        """Check if the last move resulted in a win."""
        player = self.board[last_row, last_col]
        
        # Check all directions from the last move
        directions = [
            (0, 1),   # horizontal
            (1, 0),   # vertical
            (1, 1),   # diagonal
            (1, -1)   # anti-diagonal
        ]
        
        for dr, dc in directions:
            count = 1  # Count the placed stone
            
            # Check forward direction
            r, c = last_row + dr, last_col + dc
            while (0 <= r < self.height and 0 <= c < self.width and 
                   self.board[r, c] == player):
                count += 1
                r += dr
                c += dc
            
            # Check backward direction
            r, c = last_row - dr, last_col - dc
            while (0 <= r < self.height and 0 <= c < self.width and 
                   self.board[r, c] == player):
                count += 1
                r -= dr
                c -= dc
            
            if count >= self.n_in_row:
                return True
        
        return False
    
    def display(self):
        """Display the current board."""
        print("\n   " + " ".join(f"{i}" for i in range(self.width)))
        print("  " + "─" * (self.width * 2 + 1))
        
        for r in range(self.height):
            row_display = f"{r} │"
            for c in range(self.width):
                cell = self.board[r, c]
                if cell == '-':
                    cell = '·'
                row_display += f" {cell}"
            print(row_display)
        
        print(f"\nCurrent player: {self.current_player}")
        if self.winner:
            if self.winner == 'Draw':
                print("🤝 Game ended in a draw!")
            else:
                print(f"🏆 {self.winner} wins!")


class CNNPlayer:
    """AI player using pure CNN (no MCTS)."""
    
    def __init__(self, model_path: str, config: Config):
        self.config = config
        self.model = GomokuCNN(config)
        self.model.load(model_path)
        self.model.model.eval()
        
        print(f"🤖 CNN AI loaded from {model_path}")
        self.model.summary()
    
    def get_move(self, game: GomokuGame) -> Tuple[int, int]:
        """Get AI's move using CNN prediction."""
        
        # Convert game board to CNN input
        state = grid_to_state(game.board, game.current_player)
        state_batch = np.expand_dims(state, axis=0)  # Add batch dimension
        
        # Get CNN predictions
        with torch.no_grad():
            predictions = self.model.predict(state_batch)
        
        # Get available moves
        available_moves = game.get_available_moves()
        if not available_moves:
            return None
        
        # Use move predictions if available
        if 'move' in predictions:
            move_probs = predictions['move'][0]  # Remove batch dimension
            
            # Mask unavailable moves
            masked_probs = np.full(game.width * game.height, -np.inf)
            for r, c in available_moves:
                masked_probs[r * game.width + c] = move_probs[r * game.width + c]
            
            # Select best move
            best_move_idx = np.argmax(masked_probs)
            best_row = best_move_idx // game.width
            best_col = best_move_idx % game.width
            
            return (best_row, best_col)
        
        # Fallback: use position evaluation to pick move
        elif 'value' in predictions:
            best_move = None
            best_eval = -np.inf
            
            # Try each available move and evaluate resulting position
            for r, c in available_moves:
                # Simulate move
                test_board = game.board.copy()
                test_board[r, c] = game.current_player
                
                # Get position evaluation
                test_state = grid_to_state(test_board, game.current_player)
                test_batch = np.expand_dims(test_state, axis=0)
                
                with torch.no_grad():
                    test_pred = self.model.predict(test_batch)
                
                # Value format: [win_prob, draw_prob, lose_prob]
                win_prob = test_pred['value'][0][0]  # Probability of winning
                
                if win_prob > best_eval:
                    best_eval = win_prob
                    best_move = (r, c)
            
            return best_move if best_move else available_moves[0]
        
        # Last resort: random move
        import random
        return random.choice(available_moves)
    
    def get_position_evaluation(self, game: GomokuGame) -> dict:
        """Get CNN's evaluation of current position."""
        state = grid_to_state(game.board, game.current_player)
        state_batch = np.expand_dims(state, axis=0)
        
        with torch.no_grad():
            predictions = self.model.predict(state_batch)
        
        result = {}
        if 'value' in predictions:
            value_probs = predictions['value'][0]
            result['position_eval'] = {
                'win_probability': float(value_probs[0]),
                'draw_probability': float(value_probs[1]) if len(value_probs) > 1 else 0.0,
                'lose_probability': float(value_probs[2]) if len(value_probs) > 2 else 0.0
            }
        
        if 'move' in predictions:
            move_probs = predictions['move'][0]
            # Get top 3 move suggestions
            available_moves = game.get_available_moves()
            if available_moves:
                move_suggestions = []
                for r, c in available_moves:
                    prob = move_probs[r * game.width + c]
                    move_suggestions.append(((r, c), float(prob)))
                
                move_suggestions.sort(key=lambda x: x[1], reverse=True)
                result['move_suggestions'] = move_suggestions[:3]
        
        return result


def auto_find_best_model() -> Optional[str]:
    """Automatically find the best available model."""
    model_paths = [
        "models/final_model.pt",
        "models/best_model_epoch_*.pt",
        "models/emergency_save_*.pt"
    ]
    
    import glob
    for pattern in model_paths:
        matches = glob.glob(pattern)
        if matches:
            # Return the most recent one
            return max(matches, key=os.path.getmtime)
    
    return None


def get_human_move(game: GomokuGame) -> Tuple[int, int]:
    """Get move input from human player."""
    while True:
        try:
            move_input = input(f"\nYour move (row,col): ").strip()
            if ',' in move_input:
                row_str, col_str = move_input.split(',', 1)
                row, col = int(row_str.strip()), int(col_str.strip())
            else:
                print("❌ Please enter move as 'row,col' (e.g., '2,3')")
                continue
            
            if game.is_valid_move(row, col):
                return (row, col)
            else:
                print("❌ Invalid move. Try again.")
                
        except (ValueError, KeyboardInterrupt):
            print("❌ Invalid input. Enter row,col (e.g., '2,3') or Ctrl+C to quit")


def main():
    parser = argparse.ArgumentParser(description="Play Gomoku vs Pure CNN AI")
    parser.add_argument('--model', help='Path to trained model file')
    parser.add_argument('--first', choices=['human', 'ai'], default='ai', 
                       help='Who goes first')
    parser.add_argument('--board-size', type=int, default=5, 
                       help='Board size (5 for 5x5)')
    parser.add_argument('--show-eval', action='store_true',
                       help='Show AI position evaluation each turn')
    
    args = parser.parse_args()
    
    # Find model
    model_path = args.model
    if not model_path:
        model_path = auto_find_best_model()
        if not model_path:
            print("❌ No trained model found. Train a model first:")
            print("   uv run train.py --data your_data.csv --epochs 20")
            return
        else:
            print(f"🎯 Auto-detected model: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return
    
    # Create config
    config = Config(
        board_width=args.board_size,
        board_height=args.board_size,
        n_in_row=5  # Always 5 in a row for Gomoku
    )
    
    # Initialize game
    game = GomokuGame(args.board_size, args.board_size, 5)
    ai_player = CNNPlayer(model_path, config)
    
    print(f"\n🎮 Pure CNN Gomoku - {args.board_size}x{args.board_size}")
    print(f"🎯 Goal: Get {game.n_in_row} in a row")
    print(f"👤 Human: X, 🤖 AI: O")
    print(f"🎲 {args.first.capitalize()} goes first")
    
    # Determine starting player
    if args.first == 'human':
        game.current_player = 'X'
        human_symbol = 'X'
        ai_symbol = 'O'
    else:
        game.current_player = 'O'  # AI starts
        human_symbol = 'O'
        ai_symbol = 'X'
    
    # Game loop
    while not game.game_over:
        game.display()
        
        if game.current_player == human_symbol:
            # Human turn
            print(f"\n👤 Your turn ({human_symbol})")
            
            if args.show_eval:
                eval_info = ai_player.get_position_evaluation(game)
                if 'position_eval' in eval_info:
                    pe = eval_info['position_eval']
                    print(f"🧠 AI thinks: Win={pe['win_probability']:.3f}, Draw={pe['draw_probability']:.3f}, Lose={pe['lose_probability']:.3f}")
                if 'move_suggestions' in eval_info:
                    print("💡 AI suggests:", ", ".join([f"({r},{c}):{p:.3f}" for (r,c), p in eval_info['move_suggestions'][:3]]))
            
            try:
                row, col = get_human_move(game)
                game.make_move(row, col)
            except KeyboardInterrupt:
                print("\n👋 Thanks for playing!")
                break
                
        else:
            # AI turn
            print(f"\n🤖 AI turn ({ai_symbol})")
            
            row, col = ai_player.get_move(game)
            if row is not None:
                print(f"🤖 AI plays: ({row}, {col})")
                game.make_move(row, col)
            else:
                print("🤖 AI couldn't find a move")
                break
    
    # Final display
    if not game.game_over:
        return  # Interrupted
        
    game.display()
    
    if game.winner == 'Draw':
        print("\n🤝 Game ended in a draw!")
    elif game.winner == human_symbol:
        print(f"\n🎉 Congratulations! You ({human_symbol}) won!")
    else:
        print(f"\n🤖 AI ({ai_symbol}) won! Better luck next time.")


if __name__ == '__main__':
    main()