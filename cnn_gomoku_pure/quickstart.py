#!/usr/bin/env python3
"""
Quick Start Script for Pure CNN Gomoku

This script demonstrates how to quickly get started with the package.
"""

import os
import sys

def main():
    print("🎯 Pure CNN Gomoku - Quick Start")
    print("=" * 40)
    
    print("\n📋 Available Commands:")
    print("1. Convert TicTacToe data:")
    print("   python convert_ttt_data.py --input your_data.csv --output data/converted.csv --board-size 3")
    
    print("\n2. Train CNN model:")
    print("   python train.py --data data/converted.csv --epochs 20 --model-type dual_head")
    
    print("\n3. Play against AI:")
    print("   python play.py --first human --board-size 3")
    
    print("\n4. Evaluate model:")
    print("   python evaluate.py --model models/final_model.pt --data data/test.csv --detailed")
    
    print("\n🚀 For detailed instructions, see README.md")
    
    # Check if we have data
    data_dir = "data"
    if os.path.exists(data_dir):
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        if csv_files:
            print(f"\n📊 Found {len(csv_files)} CSV files in data/:")
            for csv_file in csv_files[:5]:  # Show first 5
                print(f"   • {csv_file}")
            if len(csv_files) > 5:
                print(f"   ... and {len(csv_files) - 5} more")
        else:
            print("\n⚠️  No CSV files found in data/. Add your training data first.")
    else:
        print("\n📁 Data directory will be created when you add training data.")
    
    # Check if we have models
    models_dir = "models"
    if os.path.exists(models_dir):
        model_files = [f for f in os.listdir(models_dir) if f.endswith(('.pt', '.model'))]
        if model_files:
            print(f"\n🧠 Found {len(model_files)} trained models:")
            for model_file in model_files[:3]:
                print(f"   • {model_file}")
        else:
            print("\n🎯 No trained models found. Train your first model!")
    
    print("\n✨ Happy training!")

if __name__ == "__main__":
    main()
