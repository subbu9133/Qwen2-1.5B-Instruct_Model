#!/usr/bin/env python3
"""
Twitter Dataset Processor for Sentiment Analysis

This module processes Twitter sentiment datasets and converts them 
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

class TwitterSentimentProcessor:
    """Processor for Twitter sentiment datasets."""
    
    def __init__(self, data_dir: str = "data/raw"):
        """Initialize the processor."""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Common Twitter sentiment mappings
        self.sentiment_mappings = {
            'sentiment140': {0: 'negative', 4: 'positive'},  # Sentiment140 format
            'binary': {0: 'negative', 1: 'positive'},        # Standard binary
            'ternary': {0: 'negative', 1: 'neutral', 2: 'positive'}  # Three-class
        }
        
        logger.info("Twitter sentiment processor initialized")
    
    def download_dataset_info(self) -> None:
        """Provide information on how to download Twitter datasets."""
        print("🐦 Twitter Sentiment Dataset Download Instructions:")
        print("=" * 60)
        print("1. Sentiment140 Dataset:")
        print("   - Kaggle: https://www.kaggle.com/datasets/kazanova/sentiment140")
        print("   - Size: 1.6M tweets")
        print("   - Format: CSV with 6 columns")
        print()
        print("2. Twitter Sentiment Analysis Dataset:")
        print("   - Kaggle: https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis")
        print("   - Size: Various sizes")
        print("   - Format: CSV")
        print()
        print("3. Other Twitter datasets:")
        print("   - SemEval Twitter tasks")
        print("   - Custom scraped datasets (follow Twitter ToS)")
        print("=" * 60)
    
    def load_sentiment140(self, file_path: str, sample_size: Optional[int] = None) -> pd.DataFrame:
        """Load Sentiment140 dataset."""
        try:
            logger.info(f"Loading Sentiment140 dataset from {file_path}")
            
            # Sentiment140 column names
            columns = ['target', 'id', 'date', 'flag', 'user', 'text']
            
            # Sample both negative and positive tweets
            if sample_size:
                # Get half negative (first part) and half positive (last part)
                half_size = sample_size // 2
                
                # Read negative tweets (first half of dataset)
                df_negative = pd.read_csv(
                    file_path,
                    encoding='latin-1',
                    header=None,
                    names=columns,
                    nrows=half_size
                )
                
                # Read positive tweets (skip to second half of dataset)
                # Sentiment140 has ~800K negative and ~800K positive tweets
                skip_to_positive = 800000
                df_positive = pd.read_csv(
                    file_path,
                    encoding='latin-1',
                    header=None,
                    names=columns,
                    skiprows=range(1, skip_to_positive),
                    nrows=half_size
                )
                
                # Combine both
                df = pd.concat([df_negative, df_positive], ignore_index=True)
                # Shuffle the combined dataset
                df = df.sample(frac=1, random_state=42).reset_index(drop=True)
            else:
                df = pd.read_csv(
                    file_path,
                    encoding='latin-1',
                    header=None,
                    names=columns,
                    nrows=sample_size
                )
            
            logger.info(f"Loaded {len(df)} tweets from Sentiment140")
            return df
            
        except Exception as e:
            logger.error(f"Error loading Sentiment140: {e}")
            raise
    
    def load_twitter_csv(self, file_path: str, text_column: str = 'text',
                        sentiment_column: str = 'sentiment',
                        sample_size: Optional[int] = None) -> pd.DataFrame:
        """Load generic Twitter sentiment CSV."""
        try:
            logger.info(f"Loading Twitter dataset from {file_path}")
            
            # Try different encodings
            for encoding in ['utf-8', 'latin-1', 'iso-8859-1']:
                try:
                    df = pd.read_csv(file_path, encoding=encoding, nrows=sample_size)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise ValueError("Could not decode the CSV file")
            
            # Validate required columns
            if text_column not in df.columns:
                raise ValueError(f"Text column '{text_column}' not found")
            if sentiment_column not in df.columns:
                raise ValueError(f"Sentiment column '{sentiment_column}' not found")
            
            logger.info(f"Loaded {len(df)} tweets from CSV")
            return df
            
        except Exception as e:
            logger.error(f"Error loading Twitter CSV: {e}")
            raise
    
    def preprocess_tweet_text(self, text: str) -> str:
        """Clean and preprocess tweet text."""
        if pd.isna(text):
            return ""
        
        text = str(text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '[URL]', text, flags=re.MULTILINE)
        
        # Replace user mentions with placeholder
        text = re.sub(r'@\w+', '[USER]', text)
        
        # Replace hashtags but keep the text
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Remove excessive punctuation
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        text = re.sub(r'[.]{2,}', '...', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove very short tweets
        if len(text) < 5:
            return ""
        
        return text
    
    def convert_sentiment140_format(self, df: pd.DataFrame) -> List[Dict]:
        """Convert Sentiment140 format to sentiment analysis format."""
        logger.info("Converting Sentiment140 to sentiment format...")
        
        examples = []
        processed_count = 0
        
        for _, row in df.iterrows():
            try:
                # Preprocess text
                clean_text = self.preprocess_tweet_text(row['text'])
                if not clean_text:
                    continue
                
                # Map sentiment (0=negative, 4=positive in Sentiment140)
                original_sentiment = row['target']
                if original_sentiment == 0:
                    sentiment_label = 0  # negative
                    sentiment_text = 'negative'
                elif original_sentiment == 4:
                    sentiment_label = 1  # positive  
                    sentiment_text = 'positive'
                else:
                    continue  # Skip neutral/unknown
                
                example = {
                    'text': clean_text,
                    'label': sentiment_label,
                    'sentiment': sentiment_text,
                    'original_sentiment': original_sentiment,
                    'source': 'twitter_sentiment140',
                    'confidence': 0.8,  # Medium confidence for crowdsourced labels
                    'length': len(clean_text),
                    'tweet_id': row['id']
                }
                
                examples.append(example)
                processed_count += 1
                
                if processed_count % 10000 == 0:
                    logger.info(f"Processed {processed_count} tweets...")
                
            except Exception as e:
                logger.warning(f"Error processing tweet: {e}")
                continue
        
        logger.info(f"Converted {len(examples)} tweets to sentiment format")
        return examples
    
    def convert_generic_format(self, df: pd.DataFrame, 
                              text_column: str = 'text',
                              sentiment_column: str = 'sentiment',
                              mapping_type: str = 'binary') -> List[Dict]:
        """Convert generic Twitter CSV to sentiment format."""
        logger.info("Converting generic Twitter dataset to sentiment format...")
        
        examples = []
        sentiment_mapping = self.sentiment_mappings.get(mapping_type, self.sentiment_mappings['binary'])
        
        for _, row in df.iterrows():
            try:
                clean_text = self.preprocess_tweet_text(row[text_column])
                if not clean_text:
                    continue
                
                # Map sentiment
                original_sentiment = row[sentiment_column]
                
                # Handle string sentiments
                if isinstance(original_sentiment, str):
                    original_sentiment = original_sentiment.lower()
                    if original_sentiment in ['positive', 'pos', '1']:
                        sentiment_label = 1
                        sentiment_text = 'positive'
                    elif original_sentiment in ['negative', 'neg', '0']:
                        sentiment_label = 0
                        sentiment_text = 'negative'
                    elif original_sentiment in ['neutral']:
                        if mapping_type == 'ternary':
                            sentiment_label = 1  # neutral as middle class
                            sentiment_text = 'neutral'
                        else:
                            continue  # Skip neutral in binary classification
                    else:
                        continue
                else:
                    # Numeric sentiment
                    if original_sentiment in sentiment_mapping:
                        sentiment_text = sentiment_mapping[original_sentiment]
                        sentiment_label = 0 if sentiment_text == 'negative' else (2 if sentiment_text == 'neutral' else 1)
                    else:
                        continue
                
                example = {
                    'text': clean_text,
                    'label': sentiment_label,
                    'sentiment': sentiment_text,
                    'original_sentiment': original_sentiment,
                    'source': 'twitter_generic',
                    'confidence': 0.7,
                    'length': len(clean_text)
                }
                
                examples.append(example)
                
            except Exception as e:
                logger.warning(f"Error processing tweet: {e}")
                continue
        
        logger.info(f"Converted {len(examples)} tweets to sentiment format")
        return examples
    
    def process_full_pipeline(self, dataset_type: str = "sentiment140",
                             file_path: str = None,
                             text_column: str = 'text',
                             sentiment_column: str = 'sentiment',
                             sample_size: Optional[int] = 100000,
                             output_dir: str = "data/processed") -> Dict[str, str]:
        """Run the complete Twitter processing pipeline."""
        logger.info("Starting Twitter sentiment processing pipeline...")
        
        # Load dataset based on type
        if dataset_type == "sentiment140":
            if not file_path:
                raise ValueError("file_path required for Sentiment140 dataset")
            df = self.load_sentiment140(file_path, sample_size)
            examples = self.convert_sentiment140_format(df)
        elif dataset_type == "generic":
            if not file_path:
                raise ValueError("file_path required for generic dataset")
            df = self.load_twitter_csv(file_path, text_column, sentiment_column, sample_size)
            examples = self.convert_generic_format(df, text_column, sentiment_column)
        else:
            raise ValueError(f"Unknown dataset_type: {dataset_type}")
        
        # Balance the dataset
        examples = self._balance_dataset(examples)
        
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
            output_file = os.path.join(output_dir, f"twitter_{split_name}.jsonl")
            self._export_to_jsonl(split_examples, output_file)
            output_files[split_name] = output_file
        
        # Generate statistics
        stats = self._generate_stats(examples, train_examples, val_examples, test_examples)
        stats_file = os.path.join(output_dir, "twitter_stats.json")
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        output_files['stats'] = stats_file
        
        logger.info("Twitter sentiment processing pipeline completed!")
        return output_files
    
    def _balance_dataset(self, examples: List[Dict], max_ratio: float = 0.6) -> List[Dict]:
        """Balance positive/negative examples."""
        # Group by sentiment
        positive = [ex for ex in examples if ex['label'] == 1]
        negative = [ex for ex in examples if ex['label'] == 0]
        
        logger.info(f"Original: {len(positive)} positive, {len(negative)} negative")
        
        # Balance to avoid extreme imbalance
        min_count = min(len(positive), len(negative))
        max_count = int(min_count / (1 - max_ratio) * max_ratio)
        
        if len(positive) > max_count:
            positive = np.random.choice(positive, max_count, replace=False).tolist()
        if len(negative) > max_count:
            negative = np.random.choice(negative, max_count, replace=False).tolist()
        
        balanced = positive + negative
        np.random.shuffle(balanced)
        
        logger.info(f"Balanced: {len(positive)} positive, {len(negative)} negative")
        return balanced
    
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
            'source': 'twitter_sentiment',
            'label_distribution': {},
            'sentiment_distribution': {},
            'avg_text_length': np.mean([ex['length'] for ex in all_examples])
        }
        
        # Calculate distributions
        for example in all_examples:
            label = example['label']
            sentiment = example['sentiment']
            
            stats['label_distribution'][str(label)] = stats['label_distribution'].get(str(label), 0) + 1
            stats['sentiment_distribution'][sentiment] = stats['sentiment_distribution'].get(sentiment, 0) + 1
        
        return stats

def main():
    """Main function for Twitter sentiment processing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process Twitter sentiment datasets")
    parser.add_argument("--dataset-type", choices=["sentiment140", "generic"], default="sentiment140")
    parser.add_argument("--file-path", required=True, help="Path to dataset file")
    parser.add_argument("--text-column", default="text", help="Name of text column")
    parser.add_argument("--sentiment-column", default="sentiment", help="Name of sentiment column")
    parser.add_argument("--sample-size", type=int, default=100000)
    parser.add_argument("--output-dir", default="data/processed")
    
    args = parser.parse_args()
    
    try:
        processor = TwitterSentimentProcessor()
        processor.download_dataset_info()
        
        output_files = processor.process_full_pipeline(
            dataset_type=args.dataset_type,
            file_path=args.file_path,
            text_column=args.text_column,
            sentiment_column=args.sentiment_column,
            sample_size=args.sample_size,
            output_dir=args.output_dir
        )
        
        print("\n" + "="*50)
        print("TWITTER SENTIMENT PROCESSING COMPLETED")
        print("="*50)
        for file_type, file_path in output_files.items():
            print(f"{file_type}: {file_path}")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise

if __name__ == "__main__":
    main()
