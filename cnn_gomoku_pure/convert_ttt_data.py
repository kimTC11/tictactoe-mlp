#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Converter for TicTacToe → Pure CNN Training Format

Convert TicTacToe CSV data to the format expected by the Pure CNN training pipeline.
Handles different CSV formats and board sizes.

Usage:
    # Convert TicTacToe CSV to Gomoku format
    uv run convert_ttt_data.py \
        --input /Users/kimcuong/source/python/TicTacToeNN/data/tictactoe_games.csv \
        --output data/converted_ttt.csv \
        --board-size 3

    # Process multiple files
    uv run convert_ttt_data.py \
        --input-dir /Users/kimcuong/source/python/TicTacToeNN/data \
        --output-dir data/converted \
        --board-size 3
"""

import os
import csv
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm


def analyze_ttt_csv(csv_path: str, max_rows: int = 100):
    """Analyze TicTacToe CSV structure."""
    print(f"🔍 Analyzing CSV structure: {os.path.basename(csv_path)}")
    
    with open(csv_path, 'r') as f:
        # Read first few lines to understand format
        sample_lines = [next(f) for _ in range(min(10, max_rows))]
    
    print("📋 First few lines:")
    for i, line in enumerate(sample_lines[:5]):
        print(f"  {i+1}: {line.strip()}")
    
    # Try to detect format using pandas
    try:
        df = pd.read_csv(csv_path, nrows=max_rows)
        print(f"📊 Detected columns: {list(df.columns)}")
        print(f"📏 Shape: {df.shape}")
        
        # Show sample data
        print("🎯 Sample data:")
        print(df.head())
        
        return df.columns.tolist(), df.shape[1]
    
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return None, None


def detect_ttt_format(csv_path: str) -> dict:
    """Detect TicTacToe CSV format automatically."""
    
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        first_row = next(reader)
        second_row = next(reader) if reader else None
    
    num_cols = len(first_row)
    
    # Common TicTacToe formats:
    formats = {
        'standard_10': {
            'description': '9 board cells + outcome',
            'board_cols': list(range(9)),
            'outcome_col': 9,
            'player_col': None,
            'expected_cols': 10
        },
        'standard_11': {
            'description': '9 board cells + current player + outcome',
            'board_cols': list(range(9)), 
            'player_col': 9,
            'outcome_col': 10,
            'expected_cols': 11
        },
        'with_header_10': {
            'description': 'Header + 9 board cells + outcome',
            'board_cols': list(range(1, 10)),
            'outcome_col': 10,
            'player_col': None,
            'expected_cols': 11
        },
        'custom': {
            'description': 'Custom format - needs manual specification',
            'board_cols': None,
            'outcome_col': None,
            'player_col': None,
            'expected_cols': num_cols
        }
    }
    
    # Try to match format
    for format_name, format_info in formats.items():
        if format_info['expected_cols'] == num_cols:
            print(f"🎯 Detected format: {format_name} - {format_info['description']}")
            return format_info
    
    print(f"⚠️  Unknown format with {num_cols} columns")
    return formats['custom']


def convert_ttt_row(row, format_info, board_size=3):
    """Convert single TicTacToe row to Gomoku format."""
    
    # Extract board state
    if format_info['board_cols'] is None:
        # Assume first N columns are board
        board_cells = row[:board_size*board_size]
    else:
        board_cells = [row[i] for i in format_info['board_cols']]
    
    # Extract current player
    if format_info['player_col'] is not None:
        current_player = row[format_info['player_col']].strip().upper()
    else:
        # Try to infer from board state or outcome
        x_count = sum(1 for cell in board_cells if str(cell).strip().upper() == 'X')
        o_count = sum(1 for cell in board_cells if str(cell).strip().upper() == 'O')
        
        if x_count == o_count:
            current_player = 'X'  # X goes first
        elif x_count == o_count + 1:
            current_player = 'O'  # O's turn
        else:
            current_player = 'X'  # Default
    
    # Extract outcome
    if format_info['outcome_col'] is not None:
        outcome = str(row[format_info['outcome_col']]).strip().upper()
    else:
        outcome = 'DRAW'  # Default
    
    # Normalize board cells
    normalized_board = []
    for cell in board_cells:
        cell_str = str(cell).strip().upper()
        if cell_str in ['X', '1', 'x']:
            normalized_board.append('X')
        elif cell_str in ['O', '2', 'o', '0']:
            normalized_board.append('O')
        else:
            normalized_board.append('-')  # Empty
    
    # Ensure we have exactly board_size^2 cells
    while len(normalized_board) < board_size * board_size:
        normalized_board.append('-')
    
    normalized_board = normalized_board[:board_size*board_size]
    
    # Create output row in expected format:
    # [board_string, current_player, cell_0, cell_1, ..., cell_n]
    board_string = ''.join(normalized_board)
    output_row = [board_string, current_player] + normalized_board
    
    return output_row, outcome


def convert_ttt_csv(input_path: str, output_path: str, board_size: int = 3, 
                   format_info: dict = None, max_rows: int = None):
    """Convert TicTacToe CSV to Gomoku training format."""
    
    print(f"🔄 Converting {os.path.basename(input_path)} → {os.path.basename(output_path)}")
    
    if format_info is None:
        format_info = detect_ttt_format(input_path)
    
    converted_rows = []
    skipped_rows = 0
    
    with open(input_path, 'r') as f:
        reader = csv.reader(f)
        
        # Skip header if present
        first_row = next(reader)
        
        # Check if first row looks like header
        if any(isinstance(cell, str) and not cell.strip().upper() in ['X', 'O', '-', '1', '2', '0', 'DRAW', 'WIN', 'LOSS'] 
               for cell in first_row):
            print("📋 Skipping header row")
        else:
            # Process first row as data
            try:
                converted_row, outcome = convert_ttt_row(first_row, format_info, board_size)
                converted_rows.append(converted_row)
            except Exception as e:
                print(f"⚠️  Skipping problematic row: {e}")
                skipped_rows += 1
        
        # Process remaining rows
        for row_idx, row in enumerate(tqdm(reader, desc="Converting rows")):
            if max_rows and len(converted_rows) >= max_rows:
                break
                
            try:
                converted_row, outcome = convert_ttt_row(row, format_info, board_size)
                converted_rows.append(converted_row)
            except Exception as e:
                skipped_rows += 1
                if row_idx < 10:  # Only print first few errors
                    print(f"⚠️  Row {row_idx}: {e}")
    
    # Write converted data
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header comment
        writer.writerow([f"# Converted from {os.path.basename(input_path)} - {len(converted_rows)} rows"])
        
        # Write data rows
        for row in converted_rows:
            writer.writerow(row)
    
    print(f"✅ Converted {len(converted_rows)} rows to {output_path}")
    if skipped_rows > 0:
        print(f"⚠️  Skipped {skipped_rows} problematic rows")
    
    return len(converted_rows)


def main():
    parser = argparse.ArgumentParser(description="Convert TicTacToe CSV to Pure CNN format")
    
    # Input/output
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input', help='Single CSV file to convert')
    input_group.add_argument('--input-dir', help='Directory of CSV files to convert')
    
    parser.add_argument('--output', help='Output CSV file (for single input)')
    parser.add_argument('--output-dir', default='data/converted', 
                       help='Output directory (for directory input)')
    
    # Conversion options
    parser.add_argument('--board-size', type=int, default=3, help='Board size (3 for TicTacToe)')
    parser.add_argument('--max-rows', type=int, help='Maximum rows to convert per file')
    parser.add_argument('--analyze-only', action='store_true', 
                       help='Only analyze CSV structure without converting')
    
    # Format specification (for custom formats)
    parser.add_argument('--board-cols', nargs='+', type=int,
                       help='Column indices for board cells (0-indexed)')
    parser.add_argument('--player-col', type=int, help='Column index for current player')
    parser.add_argument('--outcome-col', type=int, help='Column index for game outcome')
    
    args = parser.parse_args()
    
    # Prepare format info if specified
    format_info = None
    if args.board_cols or args.player_col is not None or args.outcome_col is not None:
        format_info = {
            'description': 'Custom user-specified format',
            'board_cols': args.board_cols,
            'player_col': args.player_col,
            'outcome_col': args.outcome_col,
            'expected_cols': None
        }
    
    if args.input:
        # Single file conversion
        if args.analyze_only:
            analyze_ttt_csv(args.input)
        else:
            output_path = args.output or f"data/converted_{os.path.basename(args.input)}"
            convert_ttt_csv(args.input, output_path, args.board_size, format_info, args.max_rows)
    
    elif args.input_dir:
        # Directory conversion
        input_dir = Path(args.input_dir)
        csv_files = list(input_dir.glob("*.csv"))
        
        if not csv_files:
            print(f"❌ No CSV files found in {input_dir}")
            return
        
        print(f"📁 Found {len(csv_files)} CSV files in {input_dir}")
        
        if args.analyze_only:
            for csv_file in csv_files:
                analyze_ttt_csv(str(csv_file))
                print()
        else:
            os.makedirs(args.output_dir, exist_ok=True)
            
            total_converted = 0
            for csv_file in csv_files:
                output_path = os.path.join(args.output_dir, f"converted_{csv_file.name}")
                rows_converted = convert_ttt_csv(str(csv_file), output_path, args.board_size, 
                                               format_info, args.max_rows)
                total_converted += rows_converted
            
            print(f"\n🎉 Total converted: {total_converted} rows across {len(csv_files)} files")
            print(f"📂 Output directory: {args.output_dir}")


if __name__ == '__main__':
    main()