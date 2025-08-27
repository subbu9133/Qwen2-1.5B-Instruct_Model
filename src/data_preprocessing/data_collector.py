#!/usr/bin/env python3
"""
Content Moderation Data Collection and Preprocessing

This module handles data collection, preprocessing, and augmentation
for training content moderation models.

Author: ML Project Team
Date: 2024
"""

import os
import json
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import re
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ContentExample:
    """Data structure for content moderation examples."""
    content: str
    labels: Dict[str, int]
    severity: str
    source: str
    explanation: Optional[str] = None
    confidence: float = 1.0

class ContentModerationDataCollector:
    """Data collector and preprocessor for content moderation."""
    
    def __init__(self, config_path: str = "config/model_config.yaml"):
        """Initialize the data collector."""
        self.config = self._load_config(config_path)
        self.categories = list(self.config['capabilities']['content_categories'].keys())
        self.examples = []
        
        logger.info(f"Initialized data collector with categories: {self.categories}")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            raise
    
    def add_example(self, content: str, labels: Dict[str, int], 
                   severity: str, source: str, explanation: str = None,
                   confidence: float = 1.0) -> None:
        """Add a new content example."""
        # Validate labels
        for category in labels:
            if category not in self.categories:
                logger.warning(f"Unknown category: {category}")
        
        # Ensure all categories are present
        complete_labels = {cat: 0 for cat in self.categories}
        complete_labels.update(labels)
        
        example = ContentExample(
            content=content,
            labels=complete_labels,
            severity=severity,
            source=source,
            explanation=explanation,
            confidence=confidence
        )
        
        self.examples.append(example)
        logger.debug(f"Added example from {source}: {len(content)} chars")
    
    def create_safe_content_examples(self, num_examples: int = 1000) -> None:
        """Generate safe content examples."""
        safe_examples = [
            "The weather is beautiful today with clear skies and sunshine.",
            "I enjoy reading books about science and technology.",
            "My favorite recipe includes fresh vegetables and herbs.",
            "Exercise is important for maintaining good health.",
            "The movie was entertaining with great special effects.",
            "I'm learning a new programming language this month.",
            "The museum exhibition showcased ancient artifacts.",
            "Gardening is a relaxing hobby that connects us with nature.",
            "The conference discussed sustainable energy solutions.",
            "Classical music helps me focus while studying.",
            "The local bakery serves delicious fresh bread daily.",
            "Photography captures beautiful moments in time.",
            "The hiking trail offers stunning mountain views.",
            "Education opens doors to new opportunities.",
            "The art gallery featured contemporary paintings.",
            "Cooking brings families together for shared meals.",
            "The library is a quiet place for research and reading.",
            "Travel broadens our understanding of different cultures.",
            "Technology has revolutionized how we communicate.",
            "The concert featured talented musicians from around the world.",
        ]
        
        # Expand with variations
        extended_examples = []
        for base_example in safe_examples:
            extended_examples.append(base_example)
            
            # Add variations
            variations = [
                f"Today, {base_example.lower()}",
                f"I think {base_example.lower()}",
                f"Generally speaking, {base_example.lower()}",
                f"In my opinion, {base_example.lower()}",
                f"It's worth noting that {base_example.lower()}",
            ]
            extended_examples.extend(variations[:2])  # Add 2 variations per example
        
        # Sample the requested number
        selected_examples = np.random.choice(
            extended_examples, 
            min(num_examples, len(extended_examples)), 
            replace=False
        )
        
        for content in selected_examples:
            self.add_example(
                content=content,
                labels={"safe_content": 1},
                severity="safe",
                source="generated_safe",
                explanation="This content is safe and appropriate for all audiences."
            )
        
        logger.info(f"Generated {len(selected_examples)} safe content examples")
    
    def create_harmful_content_examples(self) -> None:
        """Create examples of harmful content for training."""
        
        # Hate speech examples (mild, for educational purposes)
        hate_examples = [
            "I really dislike that group of people and think they're inferior.",
            "Those people don't belong in our country.",
            "I hate it when certain types of people get special treatment.",
        ]
        
        for content in hate_examples:
            self.add_example(
                content=content,
                labels={"hate_speech": 1},
                severity="medium",
                source="generated_hate",
                explanation="Contains discriminatory language targeting groups of people."
            )
        
        # Harassment examples
        harassment_examples = [
            "You're stupid and worthless, nobody likes you.",
            "I'm going to make your life miserable.",
            "You should just disappear, everyone would be happier.",
        ]
        
        for content in harassment_examples:
            self.add_example(
                content=content,
                labels={"harassment": 1},
                severity="high",
                source="generated_harassment",
                explanation="Contains targeted harassment and bullying language."
            )
        
        # Dangerous content examples
        dangerous_examples = [
            "Here's how to hurt yourself when you're feeling sad.",
            "Instructions for making dangerous substances at home.",
            "Ways to cause maximum damage with common items.",
        ]
        
        for content in dangerous_examples:
            self.add_example(
                content=content,
                labels={"dangerous_content": 1},
                severity="critical",
                source="generated_dangerous",
                explanation="Contains instructions for dangerous or harmful activities."
            )
        
        # Sexually explicit examples (very mild placeholders)
        explicit_examples = [
            "Inappropriate sexual content description here.",
            "Explicit adult material reference here.",
            "Sexual content not suitable for minors here.",
        ]
        
        for content in explicit_examples:
            self.add_example(
                content=content,
                labels={"sexually_explicit": 1},
                severity="high",
                source="generated_explicit",
                explanation="Contains sexually explicit content inappropriate for general audiences."
            )
        
        # Privacy violation examples
        privacy_examples = [
            "Here's someone's personal address and phone number: 123 Main St, 555-0123",
            "I found John's social security number: 123-45-6789",
            "Let me share this person's private information without permission.",
        ]
        
        for content in privacy_examples:
            self.add_example(
                content=content,
                labels={"privacy_violation": 1},
                severity="high",
                source="generated_privacy",
                explanation="Contains or attempts to share private personal information."
            )
        
        logger.info("Generated harmful content examples for training")
    
    def load_from_csv(self, file_path: str, content_column: str = "content",
                     label_columns: List[str] = None, severity_column: str = "severity") -> None:
        """Load data from CSV file."""
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded CSV with {len(df)} rows from {file_path}")
            
            if label_columns is None:
                label_columns = [col for col in df.columns if col in self.categories]
            
            for _, row in df.iterrows():
                # Extract labels
                labels = {}
                for col in label_columns:
                    if col in row:
                        labels[col] = int(row[col])
                
                # Get severity
                severity = row.get(severity_column, "medium")
                
                self.add_example(
                    content=str(row[content_column]),
                    labels=labels,
                    severity=severity,
                    source=f"csv_{Path(file_path).stem}"
                )
            
        except Exception as e:
            logger.error(f"Error loading CSV {file_path}: {e}")
            raise
    
    def load_from_jsonl(self, file_path: str) -> None:
        """Load data from JSONL file."""
        try:
            count = 0
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line.strip())
                    self.add_example(
                        content=data['content'],
                        labels=data['labels'],
                        severity=data.get('severity', 'medium'),
                        source=f"jsonl_{Path(file_path).stem}",
                        explanation=data.get('explanation'),
                        confidence=data.get('confidence', 1.0)
                    )
                    count += 1
            
            logger.info(f"Loaded {count} examples from JSONL file: {file_path}")
            
        except Exception as e:
            logger.error(f"Error loading JSONL {file_path}: {e}")
            raise
    
    def balance_dataset(self, target_ratio: Dict[str, float] = None) -> None:
        """Balance the dataset by category."""
        if target_ratio is None:
            # Default: 70% safe, 30% harmful distributed across categories
            target_ratio = {
                "safe_content": 0.7,
                "sexually_explicit": 0.05,
                "dangerous_content": 0.05,
                "hate_speech": 0.05,
                "harassment": 0.05,
                "misinformation": 0.03,
                "privacy_violation": 0.03,
                "illegal_activity": 0.04
            }
        
        # Count current examples by primary category
        category_counts = {cat: 0 for cat in self.categories}
        for example in self.examples:
            # Find primary category (first non-zero label)
            primary_cat = "safe_content"
            for cat, label in example.labels.items():
                if label == 1 and cat != "safe_content":
                    primary_cat = cat
                    break
            category_counts[primary_cat] += 1
        
        logger.info(f"Current category distribution: {category_counts}")
        
        total_examples = len(self.examples)
        target_counts = {cat: int(total_examples * ratio) 
                        for cat, ratio in target_ratio.items()}
        
        logger.info(f"Target category distribution: {target_counts}")
        
        # Resample categories that are over-represented
        new_examples = []
        for cat in self.categories:
            cat_examples = [ex for ex in self.examples 
                          if self._get_primary_category(ex) == cat]
            
            target_count = target_counts.get(cat, 0)
            if len(cat_examples) > target_count and target_count > 0:
                # Downsample
                sampled = resample(cat_examples, n_samples=target_count, 
                                 random_state=42, replace=False)
                new_examples.extend(sampled)
            elif len(cat_examples) < target_count:
                # Upsample
                if len(cat_examples) > 0:
                    sampled = resample(cat_examples, n_samples=target_count,
                                     random_state=42, replace=True)
                    new_examples.extend(sampled)
            else:
                new_examples.extend(cat_examples)
        
        self.examples = new_examples
        logger.info(f"Balanced dataset to {len(self.examples)} examples")
    
    def _get_primary_category(self, example: ContentExample) -> str:
        """Get the primary category for an example."""
        for cat, label in example.labels.items():
            if label == 1 and cat != "safe_content":
                return cat
        return "safe_content"
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove special characters (but keep basic punctuation)
        text = re.sub(r'[^\w\s.,!?;:()-]', '', text)
        
        # Normalize quotes
        text = re.sub(r'["""]', '"', text)
        text = re.sub(r"[''']", "'", text)
        
        return text
    
    def augment_data(self, augmentation_factor: float = 0.2) -> None:
        """Augment data with variations."""
        original_count = len(self.examples)
        target_augmented = int(original_count * augmentation_factor)
        
        # Simple augmentation techniques
        augmented_examples = []
        
        for _ in range(target_augmented):
            # Select random example
            original = np.random.choice(self.examples)
            
            # Create variation
            augmented_content = self._create_variation(original.content)
            
            augmented_example = ContentExample(
                content=augmented_content,
                labels=original.labels.copy(),
                severity=original.severity,
                source=f"{original.source}_augmented",
                explanation=original.explanation,
                confidence=original.confidence * 0.9  # Slightly lower confidence
            )
            
            augmented_examples.append(augmented_example)
        
        self.examples.extend(augmented_examples)
        logger.info(f"Augmented dataset with {len(augmented_examples)} examples")
    
    def _create_variation(self, text: str) -> str:
        """Create a variation of the text."""
        # Simple augmentation techniques
        variations = [
            lambda t: t.replace(".", "!"),
            lambda t: t.replace("I ", "I really "),
            lambda t: t.replace("is ", "seems "),
            lambda t: t.replace("The ", "This "),
            lambda t: f"Honestly, {t.lower()}",
            lambda t: f"Obviously, {t.lower()}",
        ]
        
        # Apply random variation
        variation_func = np.random.choice(variations)
        return variation_func(text)
    
    def export_datasets(self, output_dir: str, train_ratio: float = 0.8,
                       val_ratio: float = 0.1, test_ratio: float = 0.1) -> None:
        """Export datasets to train/validation/test splits."""
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert to DataFrame for easier splitting
        data_rows = []
        for example in self.examples:
            row = {
                "content": example.content,
                "labels": example.labels,
                "severity": example.severity,
                "source": example.source,
                "explanation": example.explanation,
                "confidence": example.confidence
            }
            data_rows.append(row)
        
        # Split data
        train_data, temp_data = train_test_split(
            data_rows, test_size=(1 - train_ratio), random_state=42,
            stratify=[self._get_primary_category_from_labels(row["labels"]) 
                     for row in data_rows]
        )
        
        val_size = val_ratio / (val_ratio + test_ratio)
        val_data, test_data = train_test_split(
            temp_data, test_size=(1 - val_size), random_state=42,
            stratify=[self._get_primary_category_from_labels(row["labels"]) 
                     for row in temp_data]
        )
        
        # Export to JSONL files
        self._export_to_jsonl(train_data, os.path.join(output_dir, "train.jsonl"))
        self._export_to_jsonl(val_data, os.path.join(output_dir, "validation.jsonl"))
        self._export_to_jsonl(test_data, os.path.join(output_dir, "test.jsonl"))
        
        # Export statistics
        stats = {
            "total_examples": len(self.examples),
            "train_examples": len(train_data),
            "validation_examples": len(val_data),
            "test_examples": len(test_data),
            "categories": self.categories,
            "distribution": self._get_distribution_stats()
        }
        
        with open(os.path.join(output_dir, "dataset_stats.json"), 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Exported datasets to {output_dir}")
        logger.info(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    def _get_primary_category_from_labels(self, labels: Dict[str, int]) -> str:
        """Get primary category from labels dict."""
        for cat, label in labels.items():
            if label == 1 and cat != "safe_content":
                return cat
        return "safe_content"
    
    def _export_to_jsonl(self, data: List[Dict], file_path: str) -> None:
        """Export data to JSONL format."""
        with open(file_path, 'w', encoding='utf-8') as f:
            for row in data:
                json.dump(row, f, ensure_ascii=False)
                f.write('\n')
    
    def _get_distribution_stats(self) -> Dict[str, int]:
        """Get distribution statistics."""
        stats = {cat: 0 for cat in self.categories}
        for example in self.examples:
            primary_cat = self._get_primary_category(example)
            stats[primary_cat] += 1
        return stats

def main():
    """Main function for data collection and preprocessing."""
    logging.basicConfig(level=logging.INFO)
    
    # Initialize collector
    collector = ContentModerationDataCollector()
    
    # Generate synthetic data
    collector.create_safe_content_examples(num_examples=2000)
    collector.create_harmful_content_examples()
    
    # Load any existing data files
    data_dir = Path("data/raw")
    if data_dir.exists():
        for file_path in data_dir.glob("*.jsonl"):
            if file_path.name != "sample_data.jsonl":  # Skip our sample
                try:
                    collector.load_from_jsonl(str(file_path))
                except Exception as e:
                    logger.warning(f"Could not load {file_path}: {e}")
    
    # Balance and augment data
    collector.balance_dataset()
    collector.augment_data(augmentation_factor=0.1)
    
    # Export processed datasets
    collector.export_datasets("data/processed")
    
    logger.info("Data collection and preprocessing completed!")

if __name__ == "__main__":
    main()
