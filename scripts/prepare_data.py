#!/usr/bin/env python3
"""
Data Preparation Script for Qwen3 Content Moderation

This script prepares training data for the content moderation model,
including data collection, cleaning, augmentation, and splitting.

Author: ML Project Team
Date: 2024
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_preprocessing.data_collector import ContentModerationDataCollector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/data_preparation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def load_external_datasets(collector: ContentModerationDataCollector) -> None:
    """Load external datasets if available."""
    

    # Load combined datasets if available
    combined_dir = Path("data/processed/combined")
    if combined_dir.exists():
        for jsonl_file in combined_dir.glob("*.jsonl"):
            try:
                logger.info(f"Loading combined data: {jsonl_file}...")
                collector.load_from_jsonl(str(jsonl_file))
            except Exception as e:
                logger.warning(f"Could not load {jsonl_file}: {e}")
    
    # Example: Load HatEval dataset (if available)
    hateval_path = "data/raw/hateval_dataset.csv"
    if os.path.exists(hateval_path):
        try:
            logger.info("Loading HatEval dataset...")
            collector.load_from_csv(
                hateval_path,
                content_column="text",
                label_columns=["hate_speech"],
                severity_column="severity"
            )
        except Exception as e:
            logger.warning(f"Could not load HatEval dataset: {e}")
    
    # Example: Load Founta harassment dataset (if available)
    founta_path = "data/raw/founta_dataset.csv"
    if os.path.exists(founta_path):
        try:
            logger.info("Loading Founta harassment dataset...")
            collector.load_from_csv(
                founta_path,
                content_column="text",
                label_columns=["harassment"],
                severity_column="severity"
            )
        except Exception as e:
            logger.warning(f"Could not load Founta dataset: {e}")
    
    # Load any JSONL files in raw data directory
    raw_data_dir = Path("data/raw")
    if raw_data_dir.exists():
        for jsonl_file in raw_data_dir.glob("*.jsonl"):
            if jsonl_file.name.startswith("sample"):
                continue  # Skip sample files
            
            try:
                logger.info(f"Loading {jsonl_file}...")
                collector.load_from_jsonl(str(jsonl_file))
            except Exception as e:
                logger.warning(f"Could not load {jsonl_file}: {e}")

def create_synthetic_training_data(collector: ContentModerationDataCollector, 
                                 num_safe_examples: int = 5000) -> None:
    """Create synthetic training data for content moderation."""
    logger.info("Creating synthetic training data...")
    
    # Generate safe content examples
    collector.create_safe_content_examples(num_examples=num_safe_examples)
    
    # Generate harmful content examples
    collector.create_harmful_content_examples()
    
    # Add more specific examples for each category
    _add_sexually_explicit_examples(collector)
    _add_dangerous_content_examples(collector)
    _add_hate_speech_examples(collector)
    _add_harassment_examples(collector)
    _add_misinformation_examples(collector)
    _add_privacy_violation_examples(collector)
    _add_illegal_activity_examples(collector)

def _add_sexually_explicit_examples(collector: ContentModerationDataCollector) -> None:
    """Add sexually explicit content examples."""
    examples = [
        {
            "content": "This content contains explicit sexual descriptions that are not appropriate for general audiences.",
            "severity": "high",
            "explanation": "Contains explicit sexual content inappropriate for minors."
        },
        {
            "content": "Nude images and adult content available here for viewing.",
            "severity": "high", 
            "explanation": "References to explicit visual content."
        },
        {
            "content": "Adult entertainment services and explicit content for mature audiences only.",
            "severity": "medium",
            "explanation": "Adult content reference with age restriction notice."
        }
    ]
    
    for example in examples:
        collector.add_example(
            content=example["content"],
            labels={"sexually_explicit": 1},
            severity=example["severity"],
            source="synthetic_explicit",
            explanation=example["explanation"]
        )

def _add_dangerous_content_examples(collector: ContentModerationDataCollector) -> None:
    """Add dangerous content examples."""
    examples = [
        {
            "content": "Instructions for creating harmful substances that could cause injury or death.",
            "severity": "critical",
            "explanation": "Contains instructions for creating dangerous materials."
        },
        {
            "content": "How to harm yourself when feeling depressed - detailed methods and techniques.",
            "severity": "critical",
            "explanation": "Promotes self-harm and provides dangerous instructions."
        },
        {
            "content": "Ways to cause maximum damage to property and people using everyday items.",
            "severity": "high",
            "explanation": "Instructions for causing harm to others and property damage."
        },
        {
            "content": "Dangerous challenge that has resulted in serious injuries - here's how to do it.",
            "severity": "high",
            "explanation": "Promotes dangerous activities that could cause harm."
        }
    ]
    
    for example in examples:
        collector.add_example(
            content=example["content"],
            labels={"dangerous_content": 1},
            severity=example["severity"],
            source="synthetic_dangerous",
            explanation=example["explanation"]
        )

def _add_hate_speech_examples(collector: ContentModerationDataCollector) -> None:
    """Add hate speech examples."""
    examples = [
        {
            "content": "Those people are inferior and don't deserve equal rights in our society.",
            "severity": "high",
            "explanation": "Contains discriminatory language targeting a group of people."
        },
        {
            "content": "I hate all members of that religion and think they should be banned.",
            "severity": "high",
            "explanation": "Religious discrimination and hate speech."
        },
        {
            "content": "People of that race are naturally less intelligent and more violent.",
            "severity": "critical",
            "explanation": "Racist stereotyping and discriminatory claims."
        },
        {
            "content": "Those immigrants are ruining our country and taking our jobs.",
            "severity": "medium",
            "explanation": "Xenophobic sentiment targeting immigrants."
        }
    ]
    
    for example in examples:
        collector.add_example(
            content=example["content"],
            labels={"hate_speech": 1},
            severity=example["severity"],
            source="synthetic_hate",
            explanation=example["explanation"]
        )

def _add_harassment_examples(collector: ContentModerationDataCollector) -> None:
    """Add harassment examples."""
    examples = [
        {
            "content": "You're worthless and everyone hates you. You should just disappear forever.",
            "severity": "high",
            "explanation": "Targeted harassment with threats and personal attacks."
        },
        {
            "content": "I'm going to make your life miserable and follow you everywhere online.",
            "severity": "critical",
            "explanation": "Stalking threats and harassment."
        },
        {
            "content": "You're so stupid and ugly, no wonder nobody wants to be around you.",
            "severity": "medium",
            "explanation": "Personal attacks and bullying behavior."
        },
        {
            "content": "I'll ruin your reputation and make sure everyone knows what a loser you are.",
            "severity": "high",
            "explanation": "Threats to damage reputation and social standing."
        }
    ]
    
    for example in examples:
        collector.add_example(
            content=example["content"],
            labels={"harassment": 1},
            severity=example["severity"],
            source="synthetic_harassment",
            explanation=example["explanation"]
        )

def _add_misinformation_examples(collector: ContentModerationDataCollector) -> None:
    """Add misinformation examples."""
    examples = [
        {
            "content": "Vaccines contain dangerous microchips that control your mind and track your location.",
            "severity": "high",
            "explanation": "Medical misinformation that could harm public health."
        },
        {
            "content": "Climate change is a hoax created by scientists to get research funding.",
            "severity": "medium",
            "explanation": "Science denial and conspiracy theory."
        },
        {
            "content": "Drinking bleach can cure various diseases and boost your immune system.",
            "severity": "critical",
            "explanation": "Dangerous medical misinformation that could cause harm."
        },
        {
            "content": "The election was stolen through widespread fraud despite no evidence.",
            "severity": "medium",
            "explanation": "Political misinformation undermining democratic institutions."
        }
    ]
    
    for example in examples:
        collector.add_example(
            content=example["content"],
            labels={"misinformation": 1},
            severity=example["severity"],
            source="synthetic_misinfo",
            explanation=example["explanation"]
        )

def _add_privacy_violation_examples(collector: ContentModerationDataCollector) -> None:
    """Add privacy violation examples."""
    examples = [
        {
            "content": "Here's John Smith's address: 123 Main St, and his phone number: 555-0123.",
            "severity": "high",
            "explanation": "Sharing private personal information without consent."
        },
        {
            "content": "I found Sarah's social security number and credit card details online.",
            "severity": "critical",
            "explanation": "Sharing sensitive financial and identity information."
        },
        {
            "content": "Let me post these private photos of my ex without their permission.",
            "severity": "high",
            "explanation": "Non-consensual sharing of private images."
        },
        {
            "content": "Here are the login credentials for my colleague's work account.",
            "severity": "high",
            "explanation": "Unauthorized sharing of access credentials."
        }
    ]
    
    for example in examples:
        collector.add_example(
            content=example["content"],
            labels={"privacy_violation": 1},
            severity=example["severity"],
            source="synthetic_privacy",
            explanation=example["explanation"]
        )

def _add_illegal_activity_examples(collector: ContentModerationDataCollector) -> None:
    """Add illegal activity examples."""
    examples = [
        {
            "content": "Where to buy illegal drugs online and how to avoid getting caught.",
            "severity": "critical",
            "explanation": "Facilitating illegal drug transactions."
        },
        {
            "content": "How to hack into someone's computer and steal their personal files.",
            "severity": "critical",
            "explanation": "Instructions for cybercrime and hacking."
        },
        {
            "content": "Best methods for money laundering and hiding illegal profits.",
            "severity": "critical",
            "explanation": "Instructions for financial crimes."
        },
        {
            "content": "How to create fake IDs and documents that look completely authentic.",
            "severity": "high",
            "explanation": "Instructions for document fraud and identity crimes."
        }
    ]
    
    for example in examples:
        collector.add_example(
            content=example["content"],
            labels={"illegal_activity": 1},
            severity=example["severity"],
            source="synthetic_illegal",
            explanation=example["explanation"]
        )

def validate_data_quality(collector: ContentModerationDataCollector) -> None:
    """Validate the quality of prepared data."""
    logger.info("Validating data quality...")
    
    total_examples = len(collector.examples)
    if total_examples == 0:
        raise ValueError("No examples found in the dataset!")
    
    # Check category distribution
    category_counts = {}
    for example in collector.examples:
        primary_cat = collector._get_primary_category(example)
        category_counts[primary_cat] = category_counts.get(primary_cat, 0) + 1
    
    logger.info(f"Category distribution: {category_counts}")
    
    # Check for minimum examples per category
    min_examples_per_category = 10
    for category, count in category_counts.items():
        if count < min_examples_per_category:
            logger.warning(f"Category '{category}' has only {count} examples (minimum recommended: {min_examples_per_category})")
    
    # Check content quality
    empty_content = sum(1 for ex in collector.examples if not ex.content.strip())
    if empty_content > 0:
        logger.warning(f"Found {empty_content} examples with empty content")
    
    # Check label consistency
    invalid_labels = 0
    for example in collector.examples:
        label_sum = sum(example.labels.values())
        if label_sum == 0:
            invalid_labels += 1
    
    if invalid_labels > 0:
        logger.warning(f"Found {invalid_labels} examples with no positive labels")
    
    logger.info(f"Data quality validation completed. Total examples: {total_examples}")

def main():
    """Main data preparation function."""
    parser = argparse.ArgumentParser(description="Prepare training data for content moderation")
    parser.add_argument(
        "--num-safe-examples",
        type=int,
        default=5000,
        help="Number of safe content examples to generate"
    )
    parser.add_argument(
        "--augmentation-factor",
        type=float,
        default=0.1,
        help="Data augmentation factor (0.1 = 10% more examples)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Output directory for processed datasets"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Training set ratio"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation set ratio"
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Test set ratio"
    )
    
    args = parser.parse_args()
    
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    
    try:
        logger.info("Starting data preparation process...")
        
        # Initialize data collector
        collector = ContentModerationDataCollector()
        
        # Load external datasets
        load_external_datasets(collector)
        
        # Create synthetic training data
        create_synthetic_training_data(
            collector, 
            num_safe_examples=args.num_safe_examples
        )
        
        # Balance dataset
        logger.info("Balancing dataset...")
        collector.balance_dataset()
        
        # Augment data
        logger.info(f"Augmenting data with factor {args.augmentation_factor}...")
        collector.augment_data(augmentation_factor=args.augmentation_factor)
        
        # Validate data quality
        validate_data_quality(collector)
        
        # Export datasets
        logger.info("Exporting processed datasets...")
        collector.export_datasets(
            output_dir=args.output_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio
        )
        
        logger.info("Data preparation completed successfully!")
        
        # Print summary
        stats_file = os.path.join(args.output_dir, "dataset_stats.json")
        if os.path.exists(stats_file):
            with open(stats_file, 'r') as f:
                stats = json.load(f)
            
            print("\n" + "="*50)
            print("DATASET PREPARATION SUMMARY")
            print("="*50)
            print(f"Total examples: {stats['total_examples']}")
            print(f"Training examples: {stats['train_examples']}")
            print(f"Validation examples: {stats['validation_examples']}")
            print(f"Test examples: {stats['test_examples']}")
            print("\nCategory distribution:")
            for category, count in stats['distribution'].items():
                print(f"  {category}: {count}")
            print("="*50)
        
    except Exception as e:
        logger.error(f"Data preparation failed: {e}")
        raise

if __name__ == "__main__":
    main()
