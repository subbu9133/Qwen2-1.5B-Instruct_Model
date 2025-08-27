# 🛡️ Qwen3-4B Content Moderation Model

A fine-tuned Qwen3-4B model that combines the content moderation capabilities of **ShieldGemma-2B** and **Llama-Guard-4-12B** into a single, efficient model.

## 🎯 Project Overview

This project fine-tunes the Qwen3-4B model to replicate and enhance the content moderation capabilities of:
- **Google's ShieldGemma-2B**: Specialized in detecting harmful content across multiple categories
- **Meta's Llama-Guard-4-12B**: Advanced safety classification with detailed reasoning

## 🚀 Key Features

### Content Categories (Combined from both models)
- **Sexually Explicit Content**: Nudity, sexual acts, inappropriate material
- **Dangerous Content**: Violence, self-harm, dangerous activities
- **Hate Speech**: Discriminatory language, racism, xenophobia
- **Harassment**: Bullying, stalking, targeted abuse
- **Misinformation**: False claims, conspiracy theories, misleading content
- **Privacy Violation**: Doxxing, personal information exposure
- **Illegal Activity**: Drug promotion, fraud, terrorism, trafficking
- **Safe Content**: Appropriate, educational, informative content

### Model Capabilities
- **Binary Classification**: SAFE vs UNSAFE
- **Multi-Category Detection**: Identify specific harmful categories
- **Severity Assessment**: Low, Medium, High, Critical levels
- **Confidence Scoring**: 0.0-1.0 confidence for each classification
- **Detailed Reasoning**: Explain why content was classified as unsafe
- **Real-time Processing**: Fast inference for production use

## 🏗️ Architecture

- **Base Model**: Qwen3-4B (4 billion parameters)
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Input Format**: Chat template with system/user/assistant roles
- **Output Format**: Structured JSON with classification details
- **Model Size**: ~2.5GB (compressed), ~8GB (full precision)

## 📊 Performance Targets

- **Accuracy**: >95% on content moderation tasks
- **Precision**: >90% for unsafe content detection
- **Recall**: >85% for harmful content identification
- **False Positive Rate**: <5%
- **Inference Speed**: <100ms per request
- **Throughput**: >1000 requests/second

## 🛠️ Installation

```bash
# Clone the repository
git clone <repository-url>
cd Qwen3_4B_Content_Moderation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup environment
bash scripts/setup_environment.sh

## 📊 Viewing Results and Performance

To view the comprehensive results including charts, confusion matrices, and performance visualizations:

### Option 1: Direct File Access
- Navigate to the `Binary results/` directory for binary classification results
- Check `Sentiment_Qwen2_ model results/` for detailed model performance analysis
- Open PDF files to view charts and visualizations

### Option 2: Generate Results
```bash
# Run the complete testing script to generate new results
python "Binary results/sentiment_analysis_complete_testing (1).py"

# View results in the generated output files
```

## 📁 Project Structure

```
Qwen3_4B_Content_Moderation/
├── config/                     # Configuration files
│   ├── model_config.yaml      # Model architecture config
│   ├── training_config.yaml   # Training parameters
│   └── deployment_config.yaml # Deployment settings
├── src/                       # Source code
│   ├── data_preprocessing/    # Data processing modules
│   ├── training/             # Training utilities
│   ├── evaluation/           # Evaluation scripts
│   └── inference/            # Inference and deployment
├── scripts/                   # Main execution scripts
├── data/                      # Datasets and processed data
├── models/                    # Model checkpoints and final models
├── notebooks/                 # Jupyter notebooks for training
├── logs/                      # Training and evaluation logs
└── examples/                  # Usage examples and demos
```

## 🚀 Quick Start

### 1. Data Preparation
```bash
python scripts/prepare_data.py --config config/training_config.yaml
```

### 2. Model Training
```bash
python scripts/train_model.py --config config/training_config.yaml
```

### 3. Model Evaluation
```bash
python scripts/evaluate_model.py --model-path models/final/qwen3-4b-content-moderation
```

### 4. Model Deployment
```bash
python scripts/deploy_model.py --model-path models/final/qwen3-4b-content-moderation
```

## 📚 Training Data

The model is trained on a combination of:
- **Content Moderation Datasets**: Hate speech, harassment, toxicity detection
- **Safety Datasets**: Harmful content identification
- **Synthetic Data**: Generated examples for edge cases
- **Balanced Dataset**: Equal representation of safe and unsafe content

## 🔧 Configuration

### Training Configuration
- **Learning Rate**: 2e-4
- **Batch Size**: 2 (per device)
- **Gradient Accumulation**: 8 steps
- **Epochs**: 3
- **LoRA Rank**: 16
- **LoRA Alpha**: 32

### Model Configuration
- **Max Sequence Length**: 2048 tokens
- **Precision**: bfloat16
- **Device**: Auto (GPU/CPU)
- **Quantization**: Optional 4-bit for memory efficiency

## 📈 Evaluation Metrics

- **Classification Metrics**: Accuracy, Precision, Recall, F1
- **Safety Metrics**: False positive rate, False negative rate
- **Performance Metrics**: Latency, throughput, memory usage
- **Category-specific Metrics**: Per-category detection rates

## 🌐 API Usage

### Single Content Moderation
```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/moderate",
    json={
        "content": "Text to moderate",
        "include_reasoning": True
    },
    headers={"X-API-Key": "your-api-key"}
)

result = response.json()
print(f"Classification: {result['classification']}")
print(f"Categories: {result['categories']}")
print(f"Confidence: {result['confidence']}")
```

### Batch Processing
```python
response = requests.post(
    "http://localhost:8000/api/v1/moderate/batch",
    json={
        "contents": ["Text 1", "Text 2", "Text 3"],
        "include_reasoning": False
    }
)
```

## 🚀 Cloud Training

### Google Colab
1. Upload training data and notebook
2. Enable GPU runtime
3. Run training cells
4. Download trained model

### Kaggle
1. Create new notebook with GPU
2. Add dataset
3. Run training pipeline
4. Export model

## 📊 Model Comparison

| Feature | Qwen3-4B (Ours) | ShieldGemma-2B | Llama-Guard-4-12B |
|---------|------------------|-----------------|-------------------|
| Model Size | 4B | 2B | 4-12B |
| Categories | 8 | 6 | 7 |
| Severity Levels | 4 | 3 | 3 |
| Reasoning | ✅ | ✅ | ✅ |
| Confidence | ✅ | ✅ | ✅ |
| Training Cost | Low | Medium | High |
| Inference Speed | Fast | Fast | Medium |

## 📈 Results and Performance

### Sentiment Analysis Results
The model has been extensively tested and evaluated on various datasets. Detailed results including performance metrics, confusion matrices, and analysis charts can be found in the following files:

#### 📋 Binary Classification Results
- **Complete Results Report**: `Binary results/Qwen2_Binary_Sentiment_Analysis_Results_20250826_032309.docx`
- **Results Summary**: `Binary results/Code and results.pdf`
- **Test Cases**: `Binary results/test_cases_20250826_032309.txt`
- **Performance Data**: `Binary results/sentiment_analysis_results_20250826_032309.csv`

#### 📊 Model Performance Analysis
- **Detailed Analysis**: `Sentiment_Qwen2_ model results/with Neutral results.pdf`
- **Results Summary**: `Sentiment_Qwen2_ model results/sentiment_analysis_summary_20250821_183948.txt`
- **Performance Metrics**: `Sentiment_Qwen2_ model results/sentiment_analysis_results_20250821_183741.csv`

### Key Performance Highlights
- **Binary Classification Accuracy**: High accuracy on sentiment classification tasks
- **Multi-class Performance**: Robust performance across different sentiment categories
- **Real-world Testing**: Validated on diverse text datasets including social media and product reviews
- **Comprehensive Evaluation**: Includes precision, recall, F1-score, and confusion matrix analysis

> **Note**: To view the detailed results with charts and visualizations, please open the PDF files in the results directories. These contain comprehensive performance analysis, confusion matrices, and detailed evaluation metrics.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Qwen Team**: For the base Qwen3-4B model
- **Google**: For ShieldGemma-2B architecture insights
- **Meta**: For Llama-Guard-4-12B safety concepts
- **Hugging Face**: For the transformers library and model hosting

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-org/qwen3-4b-content-moderation/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/qwen3-4b-content-moderation/discussions)
- **Email**: support@example.com

---

**Note**: This model is designed for content moderation and safety applications. Always use responsibly and in accordance with applicable laws and regulations.
