#!/usr/bin/env python3
"""
Amazon + Twitter Combined Dataset Processing Script

This script processes both Amazon and Twitter datasets and combines them
for robust sentiment analysis training.

Author: ML Project Team
Date: 2024
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_preprocessing.amazon_processor import AmazonReviewsProcessor
from data_preprocessing.twitter_processor import TwitterSentimentProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/amazon_twitter_processing.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def combine_datasets(amazon_files: dict, twitter_files: dict, 
                    output_dir: str = "data/processed") -> dict:
    """Combine Amazon and Twitter datasets."""
    logger.info("Combining Amazon and Twitter datasets...")
    
    combined_files = {}
    
    for split in ['train', 'validation', 'test']:
        amazon_file = amazon_files.get(split)
        twitter_file = twitter_files.get(split)
        
        combined_examples = []
        
        # Load Amazon examples
        if amazon_file and os.path.exists(amazon_file):
            with open(amazon_file, 'r', encoding='utf-8') as f:
                for line in f:
                    example = json.loads(line.strip())
                    example['dataset_source'] = 'amazon'
                    combined_examples.append(example)
            logger.info(f"Loaded {len([ex for ex in combined_examples if ex['dataset_source'] == 'amazon'])} Amazon {split} examples")
        
        # Load Twitter examples
        if twitter_file and os.path.exists(twitter_file):
            with open(twitter_file, 'r', encoding='utf-8') as f:
                for line in f:
                    example = json.loads(line.strip())
                    example['dataset_source'] = 'twitter'
                    combined_examples.append(example)
            twitter_count = len([ex for ex in combined_examples if ex['dataset_source'] == 'twitter'])
            logger.info(f"Loaded {twitter_count} Twitter {split} examples")
        
        # Balance between Amazon and Twitter
        amazon_examples = [ex for ex in combined_examples if ex['dataset_source'] == 'amazon']
        twitter_examples = [ex for ex in combined_examples if ex['dataset_source'] == 'twitter']
        
        # Use equal amounts from each source
        min_count = min(len(amazon_examples), len(twitter_examples))
        if min_count > 0:
            import numpy as np
            balanced_amazon = np.random.choice(amazon_examples, min_count, replace=False).tolist()
            balanced_twitter = np.random.choice(twitter_examples, min_count, replace=False).tolist()
            
            balanced_combined = balanced_amazon + balanced_twitter
            np.random.shuffle(balanced_combined)
        else:
            balanced_combined = combined_examples
        
        # Save combined dataset
        combined_output_dir = os.path.join(output_dir, "combined_amazon_twitter")
        os.makedirs(combined_output_dir, exist_ok=True)
        
        combined_file = os.path.join(combined_output_dir, f"{split}.jsonl")
        with open(combined_file, 'w', encoding='utf-8') as f:
            for example in balanced_combined:
                json.dump(example, f, ensure_ascii=False)
                f.write('\n')
        
        combined_files[split] = combined_file
        logger.info(f"Saved {len(balanced_combined)} combined {split} examples to {combined_file}")
    
    # Generate combined statistics
    all_examples = []
    for split_file in combined_files.values():
        with open(split_file, 'r', encoding='utf-8') as f:
            for line in f:
                all_examples.append(json.loads(line.strip()))
    
    stats = generate_combined_stats(all_examples)
    stats_file = os.path.join(combined_output_dir, "combined_stats.json")
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    combined_files['stats'] = stats_file
    
    logger.info(f"Combined dataset created with {len(all_examples)} total examples")
    return combined_files

def generate_combined_stats(examples: list) -> dict:
    """Generate statistics for combined dataset."""
    stats = {
        'total_examples': len(examples),
        'amazon_examples': len([ex for ex in examples if ex['dataset_source'] == 'amazon']),
        'twitter_examples': len([ex for ex in examples if ex['dataset_source'] == 'twitter']),
        'label_distribution': {},
        'sentiment_distribution': {},
        'source_distribution': {},
        'avg_text_length': {},
        'datasets_used': ['amazon_reviews', 'twitter_sentiment']
    }
    
    amazon_lengths = []
    twitter_lengths = []
    
    for example in examples:
        # Label distribution
        label = str(example['label'])
        stats['label_distribution'][label] = stats['label_distribution'].get(label, 0) + 1
        
        # Sentiment distribution
        sentiment = example['sentiment']
        stats['sentiment_distribution'][sentiment] = stats['sentiment_distribution'].get(sentiment, 0) + 1
        
        # Source distribution
        source = example['dataset_source']
        stats['source_distribution'][source] = stats['source_distribution'].get(source, 0) + 1
        
        # Text lengths by source
        if source == 'amazon':
            amazon_lengths.append(example['length'])
        else:
            twitter_lengths.append(example['length'])
    
    if amazon_lengths:
        stats['avg_text_length']['amazon'] = sum(amazon_lengths) / len(amazon_lengths)
    if twitter_lengths:
        stats['avg_text_length']['twitter'] = sum(twitter_lengths) / len(twitter_lengths)
    
    return stats

def update_training_config(combined_files: dict):
    """Update training configuration to use combined Amazon + Twitter dataset."""
    import yaml
    
    config_file = "config/training_config.yaml"
    
    try:
        # Load existing config
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Update data file paths
        config['data']['train_file'] = combined_files['train']
        config['data']['validation_file'] = combined_files['validation'] 
        config['data']['test_file'] = combined_files['test']
        
        # Update run name
        config['training']['run_name'] = "sentiment-analysis-amazon-twitter"
        
        # Add sentiment analysis specific settings
        config['task'] = 'sentiment_analysis'
        config['num_labels'] = 2  # Binary classification
        
        # Save updated config
        backup_file = config_file.replace('.yaml', '_backup.yaml')
        if os.path.exists(config_file):
            os.rename(config_file, backup_file)
        
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        logger.info(f"Updated training configuration: {config_file}")
        logger.info(f"Backup saved as: {backup_file}")
        
    except Exception as e:
        logger.error(f"Error updating training configuration: {e}")

def main():
    """Main function for Amazon + Twitter processing."""
    parser = argparse.ArgumentParser(description="Process Amazon and Twitter datasets for sentiment analysis")
    
    # Amazon options
    parser.add_argument("--amazon-source", choices=["huggingface", "csv"], default="huggingface")
    parser.add_argument("--amazon-dataset", default="amazon_polarity")
    parser.add_argument("--amazon-file", help="Path to Amazon CSV file")
    parser.add_argument("--amazon-sample-size", type=int, default=50000)
    
    # Twitter options  
    parser.add_argument("--twitter-type", choices=["sentiment140", "generic"], default="sentiment140")
    parser.add_argument("--twitter-file", help="Path to Twitter dataset file")
    parser.add_argument("--twitter-sample-size", type=int, default=50000)
    
    # General options
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--update-config", action="store_true", help="Update training config")
    
    args = parser.parse_args()
    
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    
    try:
        logger.info("🚀 Starting Amazon + Twitter sentiment analysis processing...")
        
        # Process Amazon dataset
        logger.info("📦 Processing Amazon dataset...")
        amazon_processor = AmazonReviewsProcessor()
        amazon_files = amazon_processor.process_full_pipeline(
            source=args.amazon_source,
            dataset_name=args.amazon_dataset,
            file_path=args.amazon_file,
            sample_size=args.amazon_sample_size,
            output_dir=os.path.join(args.output_dir, "amazon")
        )
        
        # Process Twitter dataset
        logger.info("🐦 Processing Twitter dataset...")
        twitter_processor = TwitterSentimentProcessor()
        if not args.twitter_file:
            logger.error("Twitter file path is required!")
            twitter_processor.download_dataset_info()
            return
        
        twitter_files = twitter_processor.process_full_pipeline(
            dataset_type=args.twitter_type,
            file_path=args.twitter_file,
            sample_size=args.twitter_sample_size,
            output_dir=os.path.join(args.output_dir, "twitter")
        )
        
        # Combine datasets
        logger.info("🔗 Combining Amazon and Twitter datasets...")
        combined_files = combine_datasets(amazon_files, twitter_files, args.output_dir)
        
        # Update training configuration
        if args.update_config:
            logger.info("⚙️ Updating training configuration...")
            update_training_config(combined_files)
        
        # Print summary
        print("\n" + "="*70)
        print("🎉 AMAZON + TWITTER SENTIMENT ANALYSIS SETUP COMPLETED")
        print("="*70)
        print("📁 Generated Files:")
        for file_type, file_path in combined_files.items():
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path) / 1024  # KB
                print(f"   {file_type:12}: {file_path} ({file_size:.1f} KB)")
        
        # Load and display stats
        stats_file = combined_files.get('stats')
        if stats_file and os.path.exists(stats_file):
            with open(stats_file, 'r') as f:
                stats = json.load(f)
            
            print(f"\n📊 Dataset Statistics:")
            print(f"   Total examples: {stats['total_examples']}")
            print(f"   Amazon examples: {stats['amazon_examples']}")
            print(f"   Twitter examples: {stats['twitter_examples']}")
            print(f"   Label distribution: {stats['label_distribution']}")
            print(f"   Sentiment distribution: {stats['sentiment_distribution']}")
        
        print(f"\n🚀 Next Steps:")
        print("1. Review the combined dataset statistics")
        print("2. Train sentiment analysis model:")
        print("   python scripts/train_model.py --task sentiment_analysis")
        print("3. Evaluate model performance:")
        print("   python scripts/evaluate_model.py")
        
        if not args.update_config:
            print("\n💡 Tip: Use --update-config to automatically update training configuration")
        
        print("="*70)
        
    except Exception as e:
        logger.error(f"❌ Processing failed: {e}")
        raise

if __name__ == "__main__":
    main()
