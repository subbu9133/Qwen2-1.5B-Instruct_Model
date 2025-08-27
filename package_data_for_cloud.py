#!/usr/bin/env python3
"""
Package processed data for Google Colab or Kaggle upload.
This script creates a zip file with your processed datasets.
"""

import os
import zipfile
import shutil
from pathlib import Path

def package_data():
    """Package processed data for cloud training."""
    
    # Source and destination paths
    source_dir = "data/processed/combined_amazon_twitter"
    package_name = "sentiment_analysis_data.zip"
    
    # Check if source directory exists
    if not os.path.exists(source_dir):
        print(f"❌ Source directory not found: {source_dir}")
        return False
    
    # Files to include
    files_to_package = [
        "train.jsonl",
        "validation.jsonl", 
        "test.jsonl"
    ]
    
    print("📦 Packaging data for cloud training...")
    
    # Create zip file
    with zipfile.ZipFile(package_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_name in files_to_package:
            file_path = os.path.join(source_dir, file_name)
            if os.path.exists(file_path):
                zipf.write(file_path, file_name)
                print(f"✅ Added {file_name}")
                
                # Show file size
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                print(f"   Size: {size_mb:.1f} MB")
            else:
                print(f"⚠️  Warning: {file_name} not found")
    
    # Check final package size
    if os.path.exists(package_name):
        package_size_mb = os.path.getsize(package_name) / (1024 * 1024)
        print(f"\n📦 Package created: {package_name}")
        print(f"📏 Total size: {package_size_mb:.1f} MB")
        
        # Show data statistics
        show_data_stats(source_dir)
        
        return True
    else:
        print("❌ Failed to create package")
        return False

def show_data_stats(source_dir):
    """Show statistics about the packaged data."""
    import json
    
    print("\n📊 Dataset Statistics:")
    
    files = ["train.jsonl", "validation.jsonl", "test.jsonl"]
    total_examples = 0
    
    for file_name in files:
        file_path = os.path.join(source_dir, file_name)
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                count = sum(1 for line in f)
                total_examples += count
                print(f"  {file_name}: {count:,} examples")
    
    print(f"  Total: {total_examples:,} examples")
    
    # Show sample data format
    train_file = os.path.join(source_dir, "train.jsonl")
    if os.path.exists(train_file):
        print("\n📝 Sample data format:")
        with open(train_file, 'r', encoding='utf-8') as f:
            sample = json.loads(f.readline())
            print(f"  Text: {sample['text'][:100]}...")
            print(f"  Label: {sample['label']} ({sample.get('sentiment', 'N/A')})")
            print(f"  Source: {sample.get('dataset_source', 'N/A')}")

def create_upload_instructions():
    """Create instructions for uploading to cloud platforms."""
    
    instructions = """
# 🚀 Cloud Training Instructions

## Google Colab Setup

1. **Upload files to Colab:**
   - Go to https://colab.research.google.com/
   - Create a new notebook or upload the provided notebook
   - Upload `sentiment_analysis_data.zip`
   - Extract: `!unzip sentiment_analysis_data.zip`

2. **Enable GPU:**
   - Runtime → Change runtime type
   - Hardware accelerator: GPU (T4 free, A100/V100 paid)

3. **Run the notebook:**
   - Execute cells in order
   - Training will be ~10x faster on GPU

## Kaggle Setup

1. **Create new notebook:**
   - Go to https://www.kaggle.com/code
   - Create new notebook
   - Settings → Accelerator: GPU P100 (free)

2. **Add dataset:**
   - Upload `sentiment_analysis_data.zip` as a dataset
   - Add the dataset to your notebook
   - Access files at `/kaggle/input/your-dataset-name/`

3. **Run training:**
   - Copy notebook cells
   - Adjust file paths if needed
   - Start training

## Expected Performance

- **CPU (local)**: 4-6 hours
- **T4 GPU (Colab free)**: 30-45 minutes  
- **P100 GPU (Kaggle free)**: 20-30 minutes
- **A100 GPU (Colab Pro)**: 10-15 minutes

## Model Size After Training
- LoRA adapters: ~50-100 MB
- Full model checkpoint: ~3-6 GB
- Compressed for download: ~1-2 GB
"""
    
    with open("CLOUD_TRAINING_INSTRUCTIONS.md", "w", encoding="utf-8") as f:
        f.write(instructions)
    
    print("📋 Created CLOUD_TRAINING_INSTRUCTIONS.md")

if __name__ == "__main__":
    print("🎯 Packaging sentiment analysis data for cloud training\n")
    
    success = package_data()
    if success:
        create_upload_instructions()
        print("\n🎉 Ready for cloud training!")
        print("\nNext steps:")
        print("1. Upload sentiment_analysis_data.zip to Google Colab or Kaggle")
        print("2. Upload sentiment_analysis_training.ipynb")
        print("3. Follow instructions in CLOUD_TRAINING_INSTRUCTIONS.md")
        print("4. Enable GPU and start training!")
    else:
        print("\n❌ Packaging failed. Please check your data files.")
