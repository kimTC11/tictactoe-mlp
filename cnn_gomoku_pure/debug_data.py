#!/usr/bin/env python3
"""Debug script for data loading issues."""

import os
import sys
sys.path.append('.')

from config import Config
from data_pipeline import GomokuDataset
import glob

def main():
    print("🔧 Debug: Data Loading Test")
    
    # Create config
    config = Config()
    print(f"📋 Config: Board {config.board_width}x{config.board_height}, n_in_row={config.n_in_row}")
    
    # Find CSV files
    data_path = "data/training_data"
    csv_files = glob.glob(os.path.join(data_path, "*.csv"))
    print(f"📁 Found CSV files: {csv_files}")
    
    if not csv_files:
        print("❌ No CSV files found!")
        return
    
    # Test dataset creation
    print("\n🧪 Testing dataset creation...")
    try:
        dataset = GomokuDataset(csv_files, config, max_samples=10)  # Limit for testing
        print(f"✅ Dataset created with {len(dataset)} samples")
        
        if len(dataset) > 0:
            # Test first sample
            sample = dataset[0]
            print(f"📊 Sample 0:")
            print(f"  State shape: {sample['state'].shape}")
            print(f"  Value shape: {sample['value'].shape}")
            print(f"  Move probs shape: {sample['move_probs'].shape}")
            print(f"  Metadata: {sample['metadata']}")
        
    except Exception as e:
        print(f"❌ Error creating dataset: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()