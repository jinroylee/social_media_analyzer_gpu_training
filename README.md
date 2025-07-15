# Social Media Analyzer - GPU Training on RunPod

A serverless AI training pipeline for predicting social media engagement using CLIP (Vision-Language) models with LoRA fine-tuning, deployed on RunPod's GPU infrastructure.

## 🎯 Overview

This repository contains a complete MLOps pipeline that:
- Trains CLIP-based models to predict social media engagement metrics
- Uses LoRA (Low-Rank Adaptation) for efficient fine-tuning
- Loads training data from AWS S3 with lazy loading
- Deploys on RunPod serverless GPU infrastructure
- Tracks experiments with MLflow
- Saves trained models back to S3

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Client/API    │───▶│  RunPod Handler  │───▶│  Training Job   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │    Docker        │    │   S3 Storage    │
                       │  (ECR Registry)  │    │ (Data & Models) │
                       └──────────────────┘    └─────────────────┘
```

## 📦 Components

### Core Files
- **`runpod_handler.py`** - Main serverless endpoint handler for RunPod
- **`runpod_training.py`** - Core training logic with CLIP + LoRA
- **`Dockerfile.runpod`** - Docker image for RunPod deployment
- **`deploy_runpod.sh`** - Automated deployment script

### ML Package (`modelfactory/`)
- **`models/clip_regressor.py`** - CLIP-based engagement prediction model
- **`utils/engagement_dataset.py`** - S3-backed dataset with lazy image loading
- **`utils/mlflow_utils.py`** - MLflow experiment tracking utilities
- **`utils/data_stat.py`** - Data visualization and statistics
- **`collect/tiktok_data_collect.py`** - TikTok data collection pipeline
- **`test.py`** - Model evaluation and testing

### Testing & Validation
- **`test_runpod.py`** - Local and remote endpoint testing
- **`test_runpod_endpoint.py`** - Python-based endpoint validation
- **`test_endpoint.sh`** - Shell-based endpoint testing

## 🚀 Quick Start

### Prerequisites
- Docker installed and running
- AWS CLI configured with ECR permissions
- RunPod account with API key
- S3 bucket with training data in `processed/train.pkl` format

### 1. Deploy to RunPod

```bash
# Build and push Docker image to ECR
./deploy_runpod.sh
```

### 2. Create RunPod Template
1. Go to [RunPod Console](https://www.runpod.io/console)
2. Create new template with your ECR image
3. Configure GPU settings (RTX 4090 or A100 recommended)
4. Create serverless endpoint

### 3. Test Endpoint

```bash
# Set environment variables
export RUNPOD_ENDPOINT_URL="https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync"
export RUNPOD_API_KEY="your-api-key"
export AWS_ACCESS_KEY_ID="your-aws-key"
export AWS_SECRET_ACCESS_KEY="your-aws-secret"
export S3_BUCKET_NAME="your-s3-bucket"
export AWS_REGION="your-region"

# Test with shell script
./test_endpoint.sh

# Or test with Python
python test_runpod_endpoint.py
```

## 📋 API Usage

### Training Request Format

```json
{
  "input": {
    "batch_size": 32,
    "epochs": 10,
    "learning_rate": 1e-4,
    "lora_learning_rate": 1e-3,
    "lora_rank": 8,
    "use_lora": true,
    "s3_bucket": "your-bucket",
    "aws_region": "us-east-1",
    "aws_access_key_id": "your-key",
    "aws_secret_access_key": "your-secret"
  }
}
```

### Response Format

```json
{
  "status": "success",
  "message": "Training completed successfully",
  "training_result": {
    "best_val_mae": 0.1234,
    "final_epoch": 10,
    "s3_model_path": "s3://bucket/models/model_timestamp.pth",
    "s3_config_path": "s3://bucket/models/config_timestamp.json"
  },
  "timestamp": "2024-01-01T12:00:00"
}
```

## 🔧 Configuration

### Hyperparameters
- **batch_size**: Training batch size (default: 32)
- **epochs**: Number of training epochs (default: 10)
- **learning_rate**: Base learning rate (default: 1e-4)
- **lora_learning_rate**: LoRA-specific learning rate (default: 1e-3)
- **lora_rank**: LoRA rank for low-rank adaptation (default: 8)
- **use_lora**: Enable LoRA fine-tuning (default: true)

### AWS Configuration
- **s3_bucket**: S3 bucket for data and model storage
- **aws_region**: AWS region (default: ap-northeast-2)
- **aws_access_key_id**: AWS access key
- **aws_secret_access_key**: AWS secret key

## 🔍 Data Format

Training data should be stored as `processed/train.pkl` in your S3 bucket with this structure:

```python
[
  {
    "image_s3_key": "images/image1.jpg",
    "description": "Social media post text",
    "sentiment": 0.75,  # Float between -1 and 1
    "engagement_score": 1250.0  # Target engagement metric
  },
  # ... more samples
]
```

## 📊 Model Architecture

- **Base Model**: OpenAI CLIP-ViT-Large-Patch14
- **Fine-tuning**: LoRA (Low-Rank Adaptation)
- **Task**: Regression for engagement prediction
- **Inputs**: Images + Text + Sentiment
- **Output**: Engagement score prediction
- **Loss**: Huber Loss for robustness

## 🧪 Testing

### Local Testing
```bash
# Test handler locally (requires GPU)
python test_runpod.py --local

# Test model evaluation
python -m modelfactory.test
```

### Remote Testing
```bash
# Test deployed endpoint
python test_runpod.py --remote YOUR_ENDPOINT_URL --api-key YOUR_API_KEY
```

## 📁 Project Structure

```
social_media_analyzer_gpu_training/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── setup.py                     # Package setup
├── deploy_runpod.sh            # Deployment script
├── Dockerfile.runpod           # Docker image
├── runpod_handler.py           # Main serverless handler
├── runpod_training.py          # Training logic
├── test_*.py                   # Testing scripts
├── test_endpoint.sh            # Shell testing script
└── modelfactory/               # ML package
    ├── __init__.py
    ├── models/
    │   ├── __init__.py
    │   └── clip_regressor.py   # CLIP model
    ├── utils/
    │   ├── __init__.py
    │   ├── engagement_dataset.py  # Dataset handling
    │   ├── mlflow_utils.py        # Experiment tracking
    │   ├── data_stat.py           # Data visualization
    │   └── view_pkl.py            # Pickle utilities
    ├── collect/
    │   └── tiktok_data_collect.py # Data collection
    └── test.py                    # Model testing
```

## 🛠️ Development

### Local Development
```bash
# Install package in development mode
pip install -e .

# Install dependencies
pip install -r requirements.txt
```

### Building Docker Image
```bash
# Build for local testing
docker build -f Dockerfile.runpod -t runpod-gpu-training .

# Build for RunPod (linux/amd64)
docker build --platform linux/amd64 -f Dockerfile.runpod -t runpod-gpu-training .
```

## 📈 Performance Metrics

The model is evaluated using:
- **MAE (Mean Absolute Error)**: Primary metric for engagement prediction
- **R² Score**: Coefficient of determination
- **Spearman Correlation**: Rank correlation with actual engagement
- **Huber Loss**: Training loss for robustness to outliers

## 🔒 Security

- AWS credentials are passed securely through request parameters
- Docker image excludes sensitive files via `.dockerignore`
- Model artifacts are encrypted in S3 with AWS managed keys

## 📝 License

This project is for internal use. Please ensure compliance with TikTok's Terms of Service when collecting data.

---

For detailed API documentation and advanced usage, see the individual component READMEs in the `modelfactory/` directory. 