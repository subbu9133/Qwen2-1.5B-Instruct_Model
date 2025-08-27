#!/usr/bin/env python3
"""
Amazon Reviews Dataset Processor for Sentiment Analysis

This module processes Amazon product reviews and converts them 
to sentiment analysis format.

Author: ML Project Team
Date: 2024
"""

import os
import pandas as pd
import numpy as np
import re
import json
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AmazonReviewsProcessor:
    """Processor for Amazon Reviews dataset."""
    
    def __init__(self, data_dir: str = "data/raw"):
        """Initialize the processor."""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Sentiment mapping from star ratings
        self.rating_to_sentiment = {
            1: 'very_negative',
            2: 'negative', 
            3: 'neutral',
            4: 'positive',
            5: 'very_positive'
        }
        
        # Binary mapping (for binary classification)
        self.rating_to_binary = {
            1: 0, 2: 0, 3: 0,  # Negative (1-3 stars)
            4: 1, 5: 1         # Positive (4-5 stars)
        }
        
        logger.info("Amazon Reviews processor initialized")
    
    def download_dataset_info(self) -> None:
        """Provide information on how to download Amazon dataset."""
        print("📦 Amazon Reviews Dataset Download Instructions:")
        print("=" * 60)
        print("1. Via Hugging Face (Recommended):")
        print("   pip install datasets")
        print("   from datasets import load_dataset")
        print("   dataset = load_dataset('amazon_polarity')")
        print()
        print("2. Via Kaggle:")
        print("   - Go to: https://www.kaggle.com/datasets/bittlingmayer/amazonreviews")
        print("   - Download and extract to: data/raw/")
        print()
        print("3. Via direct download:")
        print("   - Amazon Product Reviews: https://jmcauley.ucsd.edu/data/amazon/")
        print("=" * 60)
    
    def load_from_huggingface(self, dataset_name: str = "amazon_polarity", 
                             sample_size: Optional[int] = None) -> pd.DataFrame:
        """Load Amazon dataset from Hugging Face."""
        try:
            from datasets import load_dataset
            
            logger.info(f"Loading {dataset_name} from Hugging Face...")
            dataset = load_dataset(dataset_name)
            
            # Convert to pandas DataFrame
            train_df = pd.DataFrame(dataset['train'])
            test_df = pd.DataFrame(dataset['test'])
            
            # Combine train and test
            df = pd.concat([train_df, test_df], ignore_index=True)
            
            # Sample if requested
            if sample_size and len(df) > sample_size:
                df = df.sample(n=sample_size, random_state=42)
            
            logger.info(f"Loaded {len(df)} Amazon reviews")
            return df
            
        except ImportError:
            logger.error("Please install datasets: pip install datasets")
            raise
        except Exception as e:
            logger.error(f"Error loading from Hugging Face: {e}")
            raise
    
    def load_from_csv(self, file_path: str, sample_size: Optional[int] = None) -> pd.DataFrame:
        """Load Amazon dataset from CSV file."""
        try:
            logger.info(f"Loading Amazon reviews from {file_path}")
            
            # Try different encodings
            for encoding in ['utf-8', 'latin-1', 'iso-8859-1']:
                try:
                    df = pd.read_csv(file_path, encoding=encoding, nrows=sample_size)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise ValueError("Could not decode the CSV file with any encoding")
            
            logger.info(f"Loaded {len(df)} Amazon reviews from CSV")
            return df
            
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            raise
    
    def preprocess_review_text(self, text: str) -> str:
        """Clean and preprocess Amazon review text."""
        if pd.isna(text):
            return ""
        
        text = str(text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove very short reviews
        if len(text) < 10:
            return ""
        
        return text
    
    def convert_to_sentiment_format(self, df: pd.DataFrame, 
                                   text_column: str = 'content',
                                   rating_column: str = 'label',
                                   binary_classification: bool = True) -> List[Dict]:
        """Convert Amazon reviews to sentiment analysis format."""
        logger.info("Converting Amazon reviews to sentiment format...")
        
        examples = []
        processed_count = 0
        
        for _, row in df.iterrows():
            try:
                # Get text and rating
                text = self.preprocess_review_text(row[text_column])
                if not text:
                    continue
                
                # Check if rating is valid
                if pd.isna(row[rating_column]) or row[rating_column] is None:
                    continue
                    
                rating = int(row[rating_column])
                
                # Map rating to sentiment
                if binary_classification:
                    sentiment_label = self.rating_to_binary.get(rating)
                    if sentiment_label is None:
                        continue  # Skip unknown ratings
                    sentiment_text = 'positive' if sentiment_label == 1 else 'negative'
                else:
                    sentiment_text = self.rating_to_sentiment.get(rating, 'neutral')
                    sentiment_label = rating - 1  # 0-4 scale
                
                # Create example
                example = {
                    'text': text,
                    'label': sentiment_label,
                    'sentiment': sentiment_text,
                    'original_rating': rating,
                    'source': 'amazon_reviews',
                    'confidence': 0.9,  # High confidence for star ratings
                    'length': len(text)
                }
                
                examples.append(example)
                processed_count += 1
                
                if processed_count % 10000 == 0:
                    logger.info(f"Processed {processed_count} Amazon reviews...")
                
            except Exception as e:
                logger.warning(f"Error processing Amazon review: {e}")
                continue
        
        logger.info(f"Converted {len(examples)} Amazon reviews to sentiment format")
        return examples
    
    def process_full_pipeline(self, source: str = "huggingface",
                             dataset_name: str = "amazon_polarity",
                             file_path: str = None,
                             sample_size: Optional[int] = 100000,
                             binary_classification: bool = True,
                             output_dir: str = "data/processed") -> Dict[str, str]:
        """Run the complete Amazon reviews processing pipeline."""
        logger.info("Starting Amazon reviews processing pipeline...")
        
        # Load dataset
        if source == "huggingface":
            df = self.load_from_huggingface(dataset_name, sample_size)
        elif source == "csv" and file_path:
            df = self.load_from_csv(file_path, sample_size)
        else:
            raise ValueError("Invalid source or missing file_path for CSV")
        
        # Convert to sentiment format
        examples = self.convert_to_sentiment_format(df, binary_classification=binary_classification)
        
        # Split into train/val/test
        train_examples, temp_examples = train_test_split(
            examples, test_size=0.3, random_state=42, stratify=[ex['label'] for ex in examples]
        )
        val_examples, test_examples = train_test_split(
            temp_examples, test_size=0.5, random_state=42, stratify=[ex['label'] for ex in temp_examples]
        )
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Export datasets
        output_files = {}
        
        for split_name, split_examples in [
            ('train', train_examples),
            ('validation', val_examples), 
            ('test', test_examples)
        ]:
            output_file = os.path.join(output_dir, f"amazon_{split_name}.jsonl")
            self._export_to_jsonl(split_examples, output_file)
            output_files[split_name] = output_file
        
        # Generate statistics
        stats = self._generate_stats(examples, train_examples, val_examples, test_examples)
        stats_file = os.path.join(output_dir, "amazon_stats.json")
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        output_files['stats'] = stats_file
        
        logger.info("Amazon reviews processing pipeline completed!")
        return output_files
    
    def _export_to_jsonl(self, examples: List[Dict], output_path: str) -> None:
        """Export examples to JSONL format."""
        with open(output_path, 'w', encoding='utf-8') as f:
            for example in examples:
                json.dump(example, f, ensure_ascii=False)
                f.write('\n')
        logger.info(f"Exported {len(examples)} examples to {output_path}")
    
    def _generate_stats(self, all_examples: List[Dict], 
                       train_examples: List[Dict],
                       val_examples: List[Dict], 
                       test_examples: List[Dict]) -> Dict:
        """Generate dataset statistics."""
        stats = {
            'total_examples': len(all_examples),
            'train_examples': len(train_examples),
            'validation_examples': len(val_examples),
            'test_examples': len(test_examples),
            'source': 'amazon_reviews',
            'label_distribution': {},
            'avg_text_length': np.mean([ex['length'] for ex in all_examples]),
            'sentiment_distribution': {}
        }
        
        # Calculate distributions
        for example in all_examples:
            label = example['label']
            sentiment = example['sentiment']
            
            stats['label_distribution'][str(label)] = stats['label_distribution'].get(str(label), 0) + 1
            stats['sentiment_distribution'][sentiment] = stats['sentiment_distribution'].get(sentiment, 0) + 1
        
        return stats

def main():
    """Main function for Amazon reviews processing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process Amazon reviews for sentiment analysis")
    parser.add_argument("--source", choices=["huggingface", "csv"], default="huggingface")
    parser.add_argument("--dataset-name", default="amazon_polarity")
    parser.add_argument("--file-path", help="Path to CSV file if using csv source")
    parser.add_argument("--sample-size", type=int, default=100000)
    parser.add_argument("--binary", action="store_true", help="Use binary classification")
    parser.add_argument("--output-dir", default="data/processed")
    
    args = parser.parse_args()
    
    try:
        processor = AmazonReviewsProcessor()
        
        if args.source == "huggingface":
            processor.download_dataset_info()
        
        output_files = processor.process_full_pipeline(
            source=args.source,
            dataset_name=args.dataset_name,
            file_path=args.file_path,
            sample_size=args.sample_size,
            binary_classification=args.binary,
            output_dir=args.output_dir
        )
        
        print("\n" + "="*50)
        print("AMAZON REVIEWS PROCESSING COMPLETED")
        print("="*50)
        for file_type, file_path in output_files.items():
            print(f"{file_type}: {file_path}")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise

if __name__ == "__main__":
    main()
