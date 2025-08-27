#!/usr/bin/env python3
"""
Model Evaluation Script for Qwen3 Content Moderation

This script evaluates the fine-tuned Qwen3 content moderation model
on various metrics and generates comprehensive evaluation reports.

Author: ML Project Team
Date: 2024
"""

import os
import sys
import json
import yaml
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re

import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/evaluation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ContentModerationEvaluator:
    """Evaluator for content moderation models."""
    
    def __init__(self, model_path: str, config_path: str = "config/model_config.yaml"):
        """Initialize the evaluator."""
        self.model_path = model_path
        self.config = self._load_config(config_path)
        self.categories = list(self.config['capabilities']['content_categories'].keys())
        
        # Load model and tokenizer
        self.tokenizer = None
        self.model = None
        self._load_model()
        
        # Evaluation results
        self.results = {}
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            raise
    
    def _load_model(self):
        """Load the fine-tuned model and tokenizer."""
        logger.info(f"Loading model from {self.model_path}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            
            # Check if it's a PEFT model
            if os.path.exists(os.path.join(self.model_path, "adapter_config.json")):
                logger.info("Loading PEFT adapter...")
                # If base model path is in config, load it first
                base_model_path = self._get_base_model_path()
                if base_model_path:
                    base_model = AutoModelForCausalLM.from_pretrained(
                        base_model_path,
                        torch_dtype=torch.bfloat16,
                        device_map="auto",
                        trust_remote_code=True
                    )
                    self.model = PeftModel.from_pretrained(base_model, self.model_path)
            
            self.model.eval()
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def _get_base_model_path(self) -> Optional[str]:
        """Get base model path from adapter config."""
        adapter_config_path = os.path.join(self.model_path, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            with open(adapter_config_path, 'r') as f:
                adapter_config = json.load(f)
                return adapter_config.get("base_model_name_or_path")
        return None
    
    def load_test_data(self, test_file: str) -> List[Dict]:
        """Load test data from JSONL file."""
        test_examples = []
        try:
            with open(test_file, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line.strip())
                    test_examples.append(data)
            
            logger.info(f"Loaded {len(test_examples)} test examples")
            return test_examples
            
        except Exception as e:
            logger.error(f"Error loading test data: {e}")
            raise
    
    def create_prompt(self, content: str) -> str:
        """Create evaluation prompt for content."""
        prompt = f"""<|im_start|>system
You are a content moderation AI. Analyze the following content and classify it according to safety categories. Provide your reasoning and classification.
<|im_end|>
<|im_start|>user
Content to analyze: {content}
<|im_end|>
<|im_start|>assistant
"""
        return prompt
    
    def generate_response(self, prompt: str, max_new_tokens: int = 256) -> str:
        """Generate model response for given prompt."""
        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.model.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.1,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # Decode response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the assistant's response
            assistant_start = full_response.find("<|im_start|>assistant\n")
            if assistant_start != -1:
                response = full_response[assistant_start + len("<|im_start|>assistant\n"):]
                # Remove any trailing special tokens
                response = response.replace("<|im_end|>", "").strip()
                return response
            else:
                return full_response[len(prompt):].strip()
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return ""
    
    def parse_model_response(self, response: str) -> Dict:
        """Parse model response to extract classification information."""
        result = {
            'classification': 'UNKNOWN',
            'categories': [],
            'severity': 'unknown',
            'confidence': 0.0,
            'reasoning': ''
        }
        
        try:
            # Extract classification
            class_match = re.search(r'Classification:\s*(SAFE|UNSAFE)', response, re.IGNORECASE)
            if class_match:
                result['classification'] = class_match.group(1).upper()
            
            # Extract categories
            cat_match = re.search(r'Categories:\s*([^\n]+)', response, re.IGNORECASE)
            if cat_match:
                categories_str = cat_match.group(1)
                # Split by comma and clean
                categories = [cat.strip() for cat in categories_str.split(',')]
                result['categories'] = [cat for cat in categories if cat in self.categories]
            
            # Extract severity
            sev_match = re.search(r'Severity:\s*(\w+)', response, re.IGNORECASE)
            if sev_match:
                result['severity'] = sev_match.group(1).lower()
            
            # Extract confidence
            conf_match = re.search(r'Confidence:\s*([\d.]+)', response, re.IGNORECASE)
            if conf_match:
                result['confidence'] = float(conf_match.group(1))
            
            # Extract reasoning
            reason_match = re.search(r'Reasoning:\s*([^\n]+)', response, re.IGNORECASE)
            if reason_match:
                result['reasoning'] = reason_match.group(1).strip()
        
        except Exception as e:
            logger.warning(f"Error parsing response: {e}")
        
        return result
    
    def evaluate_example(self, example: Dict) -> Dict:
        """Evaluate a single example."""
        content = example['content']
        true_labels = example['labels']
        
        # Generate model response
        prompt = self.create_prompt(content)
        response = self.generate_response(prompt)
        parsed_response = self.parse_model_response(response)
        
        # Determine ground truth
        true_unsafe_categories = [cat for cat, label in true_labels.items() 
                                if label == 1 and cat != 'safe_content']
        true_classification = 'UNSAFE' if true_unsafe_categories else 'SAFE'
        
        # Determine predicted classification
        pred_classification = parsed_response['classification']
        pred_categories = parsed_response['categories']
        
        return {
            'content': content,
            'true_classification': true_classification,
            'pred_classification': pred_classification,
            'true_categories': true_unsafe_categories,
            'pred_categories': pred_categories,
            'true_labels': true_labels,
            'confidence': parsed_response['confidence'],
            'severity': parsed_response['severity'],
            'reasoning': parsed_response['reasoning'],
            'raw_response': response
        }
    
    def evaluate_dataset(self, test_data: List[Dict]) -> List[Dict]:
        """Evaluate the entire test dataset."""
        logger.info("Starting dataset evaluation...")
        
        evaluation_results = []
        
        for i, example in enumerate(tqdm(test_data, desc="Evaluating")):
            try:
                result = self.evaluate_example(example)
                evaluation_results.append(result)
                
                if (i + 1) % 100 == 0:
                    logger.info(f"Evaluated {i + 1}/{len(test_data)} examples")
                    
            except Exception as e:
                logger.error(f"Error evaluating example {i}: {e}")
                continue
        
        logger.info(f"Evaluation completed. {len(evaluation_results)} examples processed.")
        return evaluation_results
    
    def calculate_metrics(self, evaluation_results: List[Dict]) -> Dict:
        """Calculate comprehensive evaluation metrics."""
        logger.info("Calculating evaluation metrics...")
        
        # Binary classification metrics (SAFE vs UNSAFE)
        true_binary = [1 if r['true_classification'] == 'UNSAFE' else 0 
                      for r in evaluation_results]
        pred_binary = [1 if r['pred_classification'] == 'UNSAFE' else 0 
                      for r in evaluation_results]
        
        binary_accuracy = accuracy_score(true_binary, pred_binary)
        binary_precision, binary_recall, binary_f1, _ = precision_recall_fscore_support(
            true_binary, pred_binary, average='binary'
        )
        
        # Multi-label classification metrics
        category_metrics = {}
        for category in self.categories:
            if category == 'safe_content':
                continue
                
            true_cat = [1 if category in r['true_categories'] else 0 
                       for r in evaluation_results]
            pred_cat = [1 if category in r['pred_categories'] else 0 
                       for r in evaluation_results]
            
            if sum(true_cat) > 0:  # Only calculate if category exists in test set
                cat_accuracy = accuracy_score(true_cat, pred_cat)
                cat_precision, cat_recall, cat_f1, _ = precision_recall_fscore_support(
                    true_cat, pred_cat, average='binary', zero_division=0
                )
                
                category_metrics[category] = {
                    'accuracy': cat_accuracy,
                    'precision': cat_precision,
                    'recall': cat_recall,
                    'f1': cat_f1,
                    'support': sum(true_cat)
                }
        
        # Overall metrics
        metrics = {
            'binary_classification': {
                'accuracy': binary_accuracy,
                'precision': binary_precision,
                'recall': binary_recall,
                'f1': binary_f1
            },
            'category_classification': category_metrics,
            'total_examples': len(evaluation_results)
        }
        
        # Calculate confusion matrix
        cm = confusion_matrix(true_binary, pred_binary)
        metrics['confusion_matrix'] = cm.tolist()
        
        return metrics
    
    def generate_classification_report(self, evaluation_results: List[Dict]) -> str:
        """Generate detailed classification report."""
        true_binary = [r['true_classification'] for r in evaluation_results]
        pred_binary = [r['pred_classification'] for r in evaluation_results]
        
        return classification_report(
            true_binary, pred_binary,
            target_names=['SAFE', 'UNSAFE'],
            digits=4
        )
    
    def analyze_errors(self, evaluation_results: List[Dict]) -> Dict:
        """Analyze classification errors."""
        logger.info("Analyzing classification errors...")
        
        errors = {
            'false_positives': [],  # Predicted UNSAFE, actually SAFE
            'false_negatives': [],  # Predicted SAFE, actually UNSAFE
            'category_errors': {}   # Wrong category predictions
        }
        
        for result in evaluation_results:
            true_class = result['true_classification']
            pred_class = result['pred_classification']
            
            if true_class == 'SAFE' and pred_class == 'UNSAFE':
                errors['false_positives'].append(result)
            elif true_class == 'UNSAFE' and pred_class == 'SAFE':
                errors['false_negatives'].append(result)
            
            # Category-level errors
            true_cats = set(result['true_categories'])
            pred_cats = set(result['pred_categories'])
            
            if true_cats != pred_cats:
                for cat in self.categories:
                    if cat == 'safe_content':
                        continue
                    
                    if cat not in errors['category_errors']:
                        errors['category_errors'][cat] = {
                            'false_positives': 0,
                            'false_negatives': 0
                        }
                    
                    if cat in pred_cats and cat not in true_cats:
                        errors['category_errors'][cat]['false_positives'] += 1
                    elif cat in true_cats and cat not in pred_cats:
                        errors['category_errors'][cat]['false_negatives'] += 1
        
        return errors
    
    def create_visualizations(self, metrics: Dict, output_dir: str) -> None:
        """Create evaluation visualizations."""
        logger.info("Creating evaluation visualizations...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Confusion matrix
        cm = np.array(metrics['confusion_matrix'])
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['SAFE', 'UNSAFE'],
                   yticklabels=['SAFE', 'UNSAFE'])
        plt.title('Confusion Matrix - Binary Classification')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300)
        plt.close()
        
        # Category performance
        if metrics['category_classification']:
            categories = list(metrics['category_classification'].keys())
            f1_scores = [metrics['category_classification'][cat]['f1'] for cat in categories]
            
            plt.figure(figsize=(12, 6))
            bars = plt.bar(categories, f1_scores)
            plt.title('F1 Score by Category')
            plt.xlabel('Category')
            plt.ylabel('F1 Score')
            plt.xticks(rotation=45, ha='right')
            
            # Add value labels on bars
            for bar, score in zip(bars, f1_scores):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{score:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'category_f1_scores.png'), dpi=300)
            plt.close()
        
        logger.info(f"Visualizations saved to {output_dir}")
    
    def export_results(self, evaluation_results: List[Dict], metrics: Dict, 
                      errors: Dict, output_dir: str) -> None:
        """Export evaluation results to files."""
        logger.info("Exporting evaluation results...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save detailed results
        results_file = os.path.join(output_dir, 'evaluation_results.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
        
        # Save metrics
        metrics_file = os.path.join(output_dir, 'metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save error analysis
        errors_file = os.path.join(output_dir, 'error_analysis.json')
        with open(errors_file, 'w', encoding='utf-8') as f:
            json.dump(errors, f, indent=2, ensure_ascii=False)
        
        # Save classification report
        report = self.generate_classification_report(evaluation_results)
        report_file = os.path.join(output_dir, 'classification_report.txt')
        with open(report_file, 'w') as f:
            f.write(report)
        
        # Create summary CSV
        summary_data = []
        for result in evaluation_results:
            summary_data.append({
                'content': result['content'][:100] + '...' if len(result['content']) > 100 else result['content'],
                'true_classification': result['true_classification'],
                'pred_classification': result['pred_classification'],
                'correct': result['true_classification'] == result['pred_classification'],
                'confidence': result['confidence'],
                'severity': result['severity']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_file = os.path.join(output_dir, 'evaluation_summary.csv')
        summary_df.to_csv(summary_file, index=False, encoding='utf-8')
        
        logger.info(f"Results exported to {output_dir}")
    
    def run_evaluation(self, test_file: str, output_dir: str = "outputs/evaluation") -> Dict:
        """Run complete evaluation pipeline."""
        logger.info("Starting complete evaluation pipeline...")
        
        # Load test data
        test_data = self.load_test_data(test_file)
        
        # Evaluate dataset
        evaluation_results = self.evaluate_dataset(test_data)
        
        if not evaluation_results:
            raise ValueError("No evaluation results generated!")
        
        # Calculate metrics
        metrics = self.calculate_metrics(evaluation_results)
        
        # Analyze errors
        errors = self.analyze_errors(evaluation_results)
        
        # Create visualizations
        self.create_visualizations(metrics, output_dir)
        
        # Export results
        self.export_results(evaluation_results, metrics, errors, output_dir)
        
        # Log summary
        logger.info("=" * 50)
        logger.info("EVALUATION SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Total examples evaluated: {metrics['total_examples']}")
        logger.info(f"Binary accuracy: {metrics['binary_classification']['accuracy']:.4f}")
        logger.info(f"Binary F1 score: {metrics['binary_classification']['f1']:.4f}")
        logger.info(f"False positives: {len(errors['false_positives'])}")
        logger.info(f"False negatives: {len(errors['false_negatives'])}")
        logger.info("=" * 50)
        
        return {
            'evaluation_results': evaluation_results,
            'metrics': metrics,
            'errors': errors
        }

def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate Qwen3 Content Moderation Model")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the trained model"
    )
    parser.add_argument(
        "--test-file",
        type=str,
        default="data/processed/test.jsonl",
        help="Path to test data file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/evaluation",
        help="Output directory for evaluation results"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/model_config.yaml",
        help="Path to model configuration file"
    )
    
    args = parser.parse_args()
    
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    
    try:
        # Initialize evaluator
        evaluator = ContentModerationEvaluator(args.model_path, args.config)
        
        # Run evaluation
        results = evaluator.run_evaluation(args.test_file, args.output_dir)
        
        logger.info("Evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise

if __name__ == "__main__":
    main()
