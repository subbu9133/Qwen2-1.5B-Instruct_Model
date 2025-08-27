#!/usr/bin/env python3
"""
Qwen3 Content Moderation Model Training Script

This script fine-tunes a Qwen3 model for content moderation tasks,
combining capabilities from ShieldGemma-2b and Llama-Guard-4-12B.

Author: ML Project Team
Date: 2024
"""

import os
import sys
import yaml
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import transformers
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from datasets import Dataset, load_dataset
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    PeftModel,
    prepare_model_for_kbit_training
)
import bitsandbytes as bnb
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import wandb
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ContentModerationExample:
    """Data structure for content moderation examples."""
    content: str
    labels: Dict[str, int]
    severity: str
    explanation: Optional[str] = None

class ContentModerationDataset:
    """Dataset class for content moderation training data."""
    
    def __init__(self, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.categories = [
            'sexually_explicit',
            'dangerous_content', 
            'hate_speech',
            'harassment',
            'misinformation',
            'privacy_violation',
            'illegal_activity',
            'safe_content'
        ]
    
    def load_data(self, file_path: str) -> List[ContentModerationExample]:
        """Load data from JSONL file."""
        examples = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line.strip())
                    # Handle both content moderation and sentiment analysis formats
                    if 'content' in data and 'labels' in data:
                        # Content moderation format
                        example = ContentModerationExample(
                            content=data['content'],
                            labels=data['labels'],
                            severity=data.get('severity', 'low'),
                            explanation=data.get('explanation')
                        )
                    elif 'text' in data and 'label' in data:
                        # Sentiment analysis format
                        example = ContentModerationExample(
                            content=data['text'],
                            labels=[data['label']],  # Convert single label to list
                            severity=data.get('severity', 'low'),
                            explanation=data.get('explanation')
                        )
                    else:
                        logger.warning(f"Skipping malformed data entry: {data}")
                        continue
                    examples.append(example)
            logger.info(f"Loaded {len(examples)} examples from {file_path}")
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {e}")
            raise
        
        return examples
    
    def create_prompt(self, example: ContentModerationExample) -> str:
        """Create training prompt for both content moderation and sentiment analysis."""
        
        # Check if this is sentiment analysis data (labels is a list) or content moderation (labels is a dict)
        if isinstance(example.labels, list):
            # Sentiment analysis format
            prompt = f"""<|im_start|>system
You are a sentiment analysis AI. Analyze the following text and classify the sentiment as positive or negative.
<|im_end|>
<|im_start|>user
Text to analyze: {example.content}
<|im_end|>
<|im_start|>assistant
"""
            # For sentiment analysis: 0 = negative, 1 = positive
            sentiment_label = example.labels[0] if example.labels else 0
            classification = "POSITIVE" if sentiment_label == 1 else "NEGATIVE"
            
            response = f"""Sentiment: {classification}
Confidence: 0.9
Reasoning: Sentiment analysis completed based on text content.
<|im_end|>"""
        
        else:
            # Content moderation format (original)
            prompt = f"""<|im_start|>system
You are a content moderation AI. Analyze the following content and classify it according to safety categories. Provide your reasoning and classification.
<|im_end|>
<|im_start|>user
Content to analyze: {example.content}
<|im_end|>
<|im_start|>assistant
"""
            
            # Create response
            unsafe_categories = [cat for cat, label in example.labels.items() if label == 1 and cat != 'safe_content']
            
            if unsafe_categories:
                classification = "UNSAFE"
                categories_str = ", ".join(unsafe_categories)
            else:
                classification = "SAFE"
                categories_str = "safe_content"
            
            response = f"""Classification: {classification}
Categories: {categories_str}
Severity: {example.severity}
Confidence: 0.9
Reasoning: {example.explanation or "Content analysis completed based on safety guidelines."}
<|im_end|>"""
        
        return prompt + response
    
    def tokenize_examples(self, examples: List[ContentModerationExample]) -> Dataset:
        """Tokenize examples for training."""
        prompts = [self.create_prompt(example) for example in examples]
        
        # Tokenize with consistent padding
        tokenized = self.tokenizer(
            prompts,
            truncation=True,
            padding=True,  # Enable padding
            max_length=self.max_length,
            return_tensors=None
        )
        
        # Create labels (same as input_ids for causal LM)
        labels = []
        for input_ids in tokenized["input_ids"]:
            # Copy input_ids to labels
            label_ids = input_ids.copy()
            labels.append(label_ids)
        
        tokenized["labels"] = labels
        
        return Dataset.from_dict(tokenized)

class ContentModerationTrainer:
    """Main trainer class for content moderation model."""
    
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.tokenizer = None
        self.model = None
        self.dataset_handler = None
        
        # Set up device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize wandb if configured
        if self.config.get('wandb', {}).get('project'):
            wandb.init(
                project=self.config['wandb']['project'],
                entity=self.config['wandb'].get('entity'),
                tags=self.config['wandb'].get('tags', []),
                notes=self.config['wandb'].get('notes', ''),
                config=self.config
            )
    
    def _load_config(self, config_path: str) -> Dict:
        """Load training configuration."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {e}")
            raise
    
    def setup_model_and_tokenizer(self):
        """Set up model and tokenizer with LoRA configuration."""
        model_name = self.config['model']['name']
        logger.info(f"Loading model and tokenizer: {model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=True
        )
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with quantization if specified
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": getattr(torch, self.config['model']['torch_dtype']),
            "device_map": self.config['model']['device_map']
        }
        
        if self.config['model'].get('load_in_4bit', False):
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["quantization_config"] = bnb_config
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        
        # Prepare model for k-bit training if using quantization
        if self.config['model'].get('load_in_4bit', False):
            self.model = prepare_model_for_kbit_training(self.model)
        
        # Set up LoRA
        lora_config = LoraConfig(
            r=self.config['lora']['r'],
            lora_alpha=self.config['lora']['lora_alpha'],
            target_modules=self.config['lora']['target_modules'],
            lora_dropout=self.config['lora']['lora_dropout'],
            bias=self.config['lora']['bias'],
            task_type=TaskType.CAUSAL_LM,
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        # Ensure model is in training mode and gradients are enabled
        self.model.train()
        for param in self.model.parameters():
            if param.requires_grad:
                param.requires_grad = True
        
        # Initialize dataset handler
        self.dataset_handler = ContentModerationDataset(
            self.tokenizer,
            self.config['model']['max_sequence_length']
        )
        
        logger.info("Model and tokenizer setup completed")
    
    def load_and_prepare_data(self) -> Tuple[Dataset, Dataset]:
        """Load and prepare training and validation datasets."""
        logger.info("Loading and preparing datasets...")
        
        # Load training data
        train_examples = self.dataset_handler.load_data(
            self.config['data']['train_file']
        )
        
        # Load validation data
        val_examples = self.dataset_handler.load_data(
            self.config['data']['validation_file']
        )
        
        # Limit samples if specified
        if self.config['data'].get('max_train_samples'):
            train_examples = train_examples[:self.config['data']['max_train_samples']]
        
        if self.config['data'].get('max_eval_samples'):
            val_examples = val_examples[:self.config['data']['max_eval_samples']]
        
        # Tokenize datasets
        train_dataset = self.dataset_handler.tokenize_examples(train_examples)
        val_dataset = self.dataset_handler.tokenize_examples(val_examples)
        
        logger.info(f"Training dataset size: {len(train_dataset)}")
        logger.info(f"Validation dataset size: {len(val_dataset)}")
        
        return train_dataset, val_dataset
    
    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics."""
        predictions, labels = eval_pred
        
        # For causal LM, we need to decode and analyze the predictions
        # This is a simplified version - you might want to implement more sophisticated metrics
        
        # Calculate perplexity
        losses = []
        for pred, label in zip(predictions, labels):
            # Skip padding tokens
            mask = label != -100
            if mask.sum() > 0:
                loss = nn.CrossEntropyLoss()(
                    torch.tensor(pred[mask]),
                    torch.tensor(label[mask])
                )
                losses.append(loss.item())
        
        avg_loss = np.mean(losses) if losses else float('inf')
        perplexity = np.exp(avg_loss)
        
        return {
            "perplexity": perplexity,
            "loss": avg_loss
        }
    
    def train(self):
        """Execute the training process."""
        logger.info("Starting training process...")
        
        # Load data
        train_dataset, val_dataset = self.load_and_prepare_data()
        
        # Set up training arguments
        training_args = TrainingArguments(
            output_dir=self.config['training']['output_dir'],
            num_train_epochs=self.config['training']['num_train_epochs'],
            per_device_train_batch_size=self.config['training']['per_device_train_batch_size'],
            per_device_eval_batch_size=self.config['training']['per_device_eval_batch_size'],
            gradient_accumulation_steps=self.config['training']['gradient_accumulation_steps'],
            learning_rate=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay'],
            warmup_ratio=self.config['training']['warmup_ratio'],
            lr_scheduler_type=self.config['training']['lr_scheduler_type'],
            logging_steps=self.config['training']['logging_steps'],
            eval_strategy=self.config['training']['evaluation_strategy'],
            eval_steps=self.config['training']['eval_steps'],
            save_strategy=self.config['training']['save_strategy'],
            save_steps=self.config['training']['save_steps'],
            save_total_limit=self.config['training']['save_total_limit'],
            metric_for_best_model=self.config['training']['metric_for_best_model'],
            greater_is_better=self.config['training']['greater_is_better'],
            load_best_model_at_end=self.config['training']['load_best_model_at_end'],
            dataloader_pin_memory=self.config['training']['dataloader_pin_memory'],
            remove_unused_columns=self.config['training']['remove_unused_columns'],
            fp16=self.config['training']['fp16'],
            bf16=self.config['training']['bf16'],
            gradient_checkpointing=self.config['training']['gradient_checkpointing'],
            report_to=self.config['training']['report_to'],
            run_name=self.config['training']['run_name'],
            max_grad_norm=self.config['training']['max_grad_norm'],
            optim=self.config['training']['optim'],
        )
        
        # Set up data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # We're doing causal LM, not masked LM
        )
        
        # Set up callbacks
        callbacks = []
        if self.config['training'].get('early_stopping_patience'):
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=self.config['training']['early_stopping_patience'],
                    early_stopping_threshold=self.config['training']['early_stopping_threshold']
                )
            )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=callbacks,
        )
        
        # Start training
        logger.info("Starting model training...")
        train_result = trainer.train()
        
        # Save the final model
        logger.info("Saving final model...")
        trainer.save_model()
        trainer.save_state()
        
        # Log training results
        logger.info(f"Training completed. Final loss: {train_result.training_loss}")
        
        return trainer, train_result
    
    def save_model(self, trainer, output_path: str):
        """Save the trained model and tokenizer."""
        logger.info(f"Saving model to {output_path}")
        
        # Create output directory
        os.makedirs(output_path, exist_ok=True)
        
        # Save model and tokenizer
        trainer.save_model(output_path)
        self.tokenizer.save_pretrained(output_path)
        
        # Save configuration
        config_path = os.path.join(output_path, "training_config.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        logger.info("Model saved successfully")

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Qwen3 Content Moderation Model")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/training_config.yaml",
        help="Path to training configuration file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/final/qwen3-content-moderation",
        help="Output directory for the trained model"
    )
    
    args = parser.parse_args()
    
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    
    try:
        # Initialize trainer
        trainer_instance = ContentModerationTrainer(args.config)
        
        # Setup model and tokenizer
        trainer_instance.setup_model_and_tokenizer()
        
        # Train the model
        trainer, train_result = trainer_instance.train()
        
        # Save the final model
        trainer_instance.save_model(trainer, args.output_dir)
        
        logger.info("Training pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise
    
    finally:
        # Finish wandb run
        if wandb.run is not None:
            wandb.finish()

if __name__ == "__main__":
    main()
