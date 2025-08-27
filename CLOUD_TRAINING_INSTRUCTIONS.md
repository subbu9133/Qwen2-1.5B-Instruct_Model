
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
