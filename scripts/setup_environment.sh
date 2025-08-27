#!/bin/bash

# Qwen3 Content Moderation Environment Setup Script
# This script sets up the complete environment for training and deploying the content moderation model

set -e  # Exit on any error

echo "🚀 Setting up Qwen3 Content Moderation Environment..."

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}[SETUP]${NC} $1"
}

# Check if running on supported OS
check_os() {
    print_header "Checking operating system..."
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        print_status "Linux detected"
        OS="linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        print_status "macOS detected"
        OS="macos"
    else
        print_error "Unsupported operating system: $OSTYPE"
        exit 1
    fi
}

# Check Python version
check_python() {
    print_header "Checking Python installation..."
    
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        print_status "Python $PYTHON_VERSION found"
        
        # Check if version is 3.8 or higher
        if python3 -c 'import sys; assert sys.version_info >= (3, 8)' 2>/dev/null; then
            print_status "Python version meets requirements (>=3.8)"
        else
            print_error "Python 3.8 or higher is required. Current version: $PYTHON_VERSION"
            exit 1
        fi
    else
        print_error "Python 3 not found. Please install Python 3.8 or higher."
        exit 1
    fi
}

# Check CUDA availability
check_cuda() {
    print_header "Checking CUDA installation..."
    
    if command -v nvidia-smi &> /dev/null; then
        CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
        print_status "NVIDIA GPU detected with CUDA $CUDA_VERSION"
        
        # Check VRAM
        GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
        print_status "GPU Memory: ${GPU_MEMORY}MB"
        
        if [ "$GPU_MEMORY" -lt 16000 ]; then
            print_warning "GPU has less than 16GB VRAM. Training may require adjustments."
        fi
    else
        print_warning "NVIDIA GPU not detected. CPU-only training will be much slower."
    fi
}

# Create virtual environment
setup_venv() {
    print_header "Setting up Python virtual environment..."
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        print_status "Creating virtual environment..."
        python3 -m venv venv
    else
        print_status "Virtual environment already exists"
    fi
    
    # Activate virtual environment
    print_status "Activating virtual environment..."
    source venv/bin/activate
    
    # Upgrade pip
    print_status "Upgrading pip..."
    pip install --upgrade pip setuptools wheel
}

# Install PyTorch with appropriate CUDA support
install_pytorch() {
    print_header "Installing PyTorch..."
    
    if command -v nvidia-smi &> /dev/null; then
        # Install PyTorch with CUDA support
        CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d. -f1,2)
        
        if [[ "$CUDA_VERSION" == "12."* ]]; then
            print_status "Installing PyTorch with CUDA 12.x support..."
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        elif [[ "$CUDA_VERSION" == "11."* ]]; then
            print_status "Installing PyTorch with CUDA 11.x support..."
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        else
            print_warning "CUDA version $CUDA_VERSION not fully supported. Installing CPU version."
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        fi
    else
        print_status "Installing PyTorch CPU version..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi
}

# Install other dependencies
install_dependencies() {
    print_header "Installing other dependencies..."
    
    # Install from requirements.txt
    if [ -f "requirements.txt" ]; then
        print_status "Installing packages from requirements.txt..."
        pip install -r requirements.txt
    else
        print_error "requirements.txt not found!"
        exit 1
    fi
    
    # Install additional dependencies that might not be in requirements.txt
    print_status "Installing additional ML libraries..."
    
    # Install HuggingFace libraries
    pip install transformers>=4.35.0 datasets>=2.14.0 accelerate>=0.21.0
    
    # Install PEFT for LoRA
    pip install peft>=0.6.0
    
    # Install quantization libraries
    pip install bitsandbytes>=0.41.0
    
    # Install training utilities
    pip install deepspeed>=0.10.0 wandb>=0.15.0
    
    # Install evaluation metrics
    pip install scikit-learn>=1.3.0 seaborn>=0.12.0
    
    # Install deployment libraries
    pip install fastapi>=0.100.0 uvicorn>=0.23.0 gradio>=3.40.0
}

# Verify installation
verify_installation() {
    print_header "Verifying installation..."
    
    # Test PyTorch
    print_status "Testing PyTorch installation..."
    python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')"
    
    if command -v nvidia-smi &> /dev/null; then
        python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
        if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
            python3 -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"
            python3 -c "import torch; print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
        fi
    fi
    
    # Test other key libraries
    print_status "Testing other libraries..."
    python3 -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
    python3 -c "import peft; print(f'PEFT version: {peft.__version__}')"
    python3 -c "import datasets; print(f'Datasets version: {datasets.__version__}')"
    
    print_status "Testing bitsandbytes..."
    python3 -c "import bitsandbytes; print('BitsAndBytes imported successfully')" 2>/dev/null || print_warning "BitsAndBytes import failed - quantization may not work"
}

# Create necessary directories
create_directories() {
    print_header "Creating necessary directories..."
    
    mkdir -p data/{raw,processed,evaluation}
    mkdir -p models/{checkpoints,final}
    mkdir -p logs
    mkdir -p outputs
    
    print_status "Directories created successfully"
}

# Set up Git LFS for large files
setup_git_lfs() {
    print_header "Setting up Git LFS for large files..."
    
    if command -v git-lfs &> /dev/null; then
        print_status "Git LFS already installed"
    else
        print_status "Installing Git LFS..."
        if [[ "$OS" == "linux" ]]; then
            curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
            sudo apt-get install git-lfs
        elif [[ "$OS" == "macos" ]]; then
            brew install git-lfs
        fi
    fi
    
    # Initialize Git LFS
    git lfs install 2>/dev/null || print_warning "Git repository not initialized"
    
    # Set up LFS tracking for model files
    if [ -f ".gitattributes" ]; then
        print_status ".gitattributes already exists"
    else
        print_status "Creating .gitattributes for model files..."
        cat > .gitattributes << EOF
# Model files
*.bin filter=lfs diff=lfs merge=lfs -text
*.safetensors filter=lfs diff=lfs merge=lfs -text
*.h5 filter=lfs diff=lfs merge=lfs -text
*.onnx filter=lfs diff=lfs merge=lfs -text
*.pkl filter=lfs diff=lfs merge=lfs -text

# Data files
*.jsonl filter=lfs diff=lfs merge=lfs -text
*.parquet filter=lfs diff=lfs merge=lfs -text

# Log files
*.log
logs/
outputs/

# Model checkpoints
models/checkpoints/
models/final/
EOF
    fi
}

# Download sample data (if available)
download_sample_data() {
    print_header "Setting up sample data..."
    
    print_status "Creating sample dataset structure..."
    
    # Create sample data files to demonstrate format
    cat > data/raw/sample_data.jsonl << 'EOF'
{"content": "This is a normal conversation about the weather.", "labels": {"safe_content": 1, "sexually_explicit": 0, "dangerous_content": 0, "hate_speech": 0, "harassment": 0}, "severity": "safe"}
{"content": "I hate you and want to hurt you!", "labels": {"safe_content": 0, "harassment": 1, "dangerous_content": 1}, "severity": "high"}
{"content": "Here's how to build a bomb...", "labels": {"safe_content": 0, "dangerous_content": 1, "illegal_activity": 1}, "severity": "critical"}
EOF
    
    print_status "Sample data created in data/raw/sample_data.jsonl"
}

# Create activation script
create_activation_script() {
    print_header "Creating environment activation script..."
    
    cat > activate_env.sh << 'EOF'
#!/bin/bash
# Qwen3 Content Moderation Environment Activation Script

echo "🔧 Activating Qwen3 Content Moderation Environment..."

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✅ Virtual environment activated"
else
    echo "❌ Virtual environment not found. Please run setup_environment.sh first."
    exit 1
fi

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# Set CUDA environment variables if GPU is available
if command -v nvidia-smi &> /dev/null; then
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    export CUDA_VISIBLE_DEVICES=0
fi

echo "🚀 Environment ready for Qwen3 content moderation training!"
echo ""
echo "Available commands:"
echo "  python scripts/train_model.py        - Start model training"
echo "  python scripts/evaluate_model.py     - Evaluate trained model"
echo "  python scripts/deploy_model.py       - Deploy model for inference"
echo ""
echo "Configuration files:"
echo "  config/training_config.yaml     - Training parameters"
echo "  config/model_config.yaml        - Model specifications"
echo "  config/deployment_config.yaml   - Deployment settings"
EOF
    
    chmod +x activate_env.sh
    print_status "Activation script created: activate_env.sh"
}

# Main setup function
main() {
    print_header "Starting Qwen3 Content Moderation Environment Setup"
    
    check_os
    check_python
    check_cuda
    setup_venv
    install_pytorch
    install_dependencies
    verify_installation
    create_directories
    setup_git_lfs
    download_sample_data
    create_activation_script
    
    print_header "🎉 Setup completed successfully!"
    echo ""
    echo -e "${GREEN}Next steps:${NC}"
    echo "1. Activate the environment: source activate_env.sh"
    echo "2. Prepare your training data in data/raw/"
    echo "3. Configure training parameters in config/training_config.yaml"
    echo "4. Start training: python scripts/train_model.py"
    echo ""
    echo -e "${BLUE}For more information, see README.md${NC}"
}

# Run main function
main "$@"
