# ModelFactory - ML Package for Social Media Engagement Prediction

The `modelfactory` package contains the core machine learning components for training CLIP-based models to predict social media engagement metrics.

## 📦 Package Structure

```
modelfactory/
├── __init__.py                    # Package initialization
├── README.md                      # This file
├── test.py                       # Model evaluation and testing
├── models/
│   ├── __init__.py
│   └── clip_regressor.py         # CLIP-based engagement model
├── utils/
│   ├── __init__.py
│   ├── engagement_dataset.py     # S3-backed dataset with lazy loading
│   ├── mlflow_utils.py          # MLflow experiment tracking
│   ├── data_stat.py             # Data visualization utilities
│   └── view_pkl.py              # Pickle file inspection
└── collect/
    └── tiktok_data_collect.py    # TikTok data collection pipeline
```

## 🎯 Core Components

### 1. CLIP Engagement Regressor (`models/clip_regressor.py`)

A PyTorch model that combines CLIP's vision and text encoders with LoRA fine-tuning for engagement prediction.

**Features:**
- Uses OpenAI CLIP-ViT-Large-Patch14 as the base model
- LoRA (Low-Rank Adaptation) for efficient fine-tuning
- Multimodal fusion of image, text, and sentiment features
- Regression head for engagement score prediction

**Usage:**
```python
from modelfactory.models.clip_regressor import CLIPEngagementRegressor

model = CLIPEngagementRegressor(
    use_lora=True,
    lora_rank=8,
    clip_model_name="openai/clip-vit-large-patch14"
)

# Forward pass
outputs = model(pixel_values, input_ids, attention_mask, sentiment)
```

### 2. Engagement Dataset (`utils/engagement_dataset.py`)

A PyTorch Dataset class optimized for large-scale training with S3 storage.

**Features:**
- Lazy loading of images from S3 to minimize memory usage
- Automatic preprocessing with CLIP processor
- Handles missing images gracefully with fallback
- Configurable AWS credentials

**Usage:**
```python
from modelfactory.utils.engagement_dataset import EngagementDataset
from transformers import CLIPProcessor, CLIPTokenizer

processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

dataset = EngagementDataset(data, processor, tokenizer)
```

**Expected Data Format:**
```python
[
    {
        "image_s3_key": "images/post_123.jpg",
        "description": "Check out this amazing product! #beauty #skincare",
        "sentiment": 0.85,  # Float between -1 and 1
        "engagement_score": 1250.0,  # Target metric
        "s3_bucket": "your-bucket",
        "aws_config": {
            "aws_access_key_id": "...",
            "aws_secret_access_key": "...",
            "aws_region": "us-east-1"
        }
    }
]
```

### 3. MLflow Utilities (`utils/mlflow_utils.py`)

Experiment tracking and model management utilities.

**Features:**
- Automatic experiment creation and management
- Hyperparameter and metric logging
- Model artifact storage
- Integration with remote MLflow servers

**Usage:**
```python
from modelfactory.utils.mlflow_utils import MLflowTracker, create_experiment_config

tracker = MLflowTracker(
    experiment_name="social_media_engagement",
    tracking_uri="http://your-mlflow-server",
    artifact_location="s3://your-bucket/mlruns"
)

with tracker.start_run(run_name="experiment_1") as run:
    tracker.log_hyperparameters({"batch_size": 32, "epochs": 10})
    tracker.log_metrics({"mae": 0.123, "r2": 0.85})
```

### 4. Data Statistics (`utils/data_stat.py`)

Visualization and statistical analysis utilities for engagement data.

**Features:**
- Distribution analysis of engagement scores
- Box plots and histograms
- Statistical summaries and outlier detection
- Correlation analysis

**Usage:**
```python
from modelfactory.utils.data_stat import visualize_distribution

scores = [100, 200, 150, 300, 250, 180]
visualize_distribution(scores, title="Engagement Score Distribution")
```

### 5. Data Collection (`collect/tiktok_data_collect.py`)

Automated pipeline for collecting TikTok content and engagement metrics.

**Features:**
- Trending content collection in beauty/cosmetics niche
- Engagement metrics extraction (views, likes, shares, comments)
- Image thumbnail processing (256×256 JPEG)
- Parquet export for ML pipeline integration

**Usage:**
```bash
python -m modelfactory.collect.tiktok_data_collect
```

## 🔧 Configuration

### Environment Variables

```bash
# AWS Configuration
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_REGION="us-east-1"
export S3_BUCKET_NAME="your-bucket"

# MLflow Configuration
export MLFLOW_TRACKING_URI="http://your-mlflow-server"
export MLFLOW_EXPERIMENT_NAME="social_media_engagement"
```

### Model Hyperparameters

```python
{
    "batch_size": 32,           # Training batch size
    "epochs": 10,               # Number of training epochs
    "learning_rate": 1e-4,      # Base learning rate
    "lora_learning_rate": 1e-3, # LoRA-specific learning rate
    "lora_rank": 8,             # LoRA rank (lower = more efficient)
    "use_lora": True,           # Enable LoRA fine-tuning
    "clip_model_name": "openai/clip-vit-large-patch14"
}
```

## 🧪 Testing

### Unit Testing
```bash
# Test individual components
python -c "from modelfactory.models.clip_regressor import CLIPEngagementRegressor; print('✅ Model import successful')"
python -c "from modelfactory.utils.engagement_dataset import EngagementDataset; print('✅ Dataset import successful')"
```

### Model Evaluation
```bash
# Run full model evaluation
python -m modelfactory.test
```

### Data Inspection
```bash
# View pickle data structure
python -m modelfactory.utils.view_pkl
```

## 📊 Model Performance

### Metrics
- **Primary**: Mean Absolute Error (MAE)
- **Secondary**: R² Score, Spearman Correlation
- **Training Loss**: Huber Loss (robust to outliers)

### Typical Performance
- **Training MAE**: ~0.08-0.12
- **Validation MAE**: ~0.10-0.15
- **R² Score**: ~0.75-0.85
- **Spearman Correlation**: ~0.80-0.90

## 🔄 Data Pipeline

```
TikTok API → Data Collection → S3 Storage → Dataset → Training → Model → S3
     ↓              ↓              ↓          ↓         ↓        ↓      ↓
 Raw Posts → Processed Data → train.pkl → Batches → Weights → model.pth
```

## 🛠️ Development

### Installing for Development
```bash
# Clone the repository
git clone <repository-url>
cd social_media_analyzer_gpu_training

# Install in development mode
pip install -e .
```

### Adding New Components

1. **New Model**: Add to `models/` directory
2. **New Utility**: Add to `utils/` directory  
3. **New Data Source**: Add to `collect/` directory
4. **Update Imports**: Modify `__init__.py` files

### Code Style
- Follow PEP 8 conventions
- Use type hints where possible
- Add docstrings for all public functions
- Include logging for debugging

## 🔍 Troubleshooting

### Common Issues

**Import Errors**
```bash
# Ensure package is installed
pip install -e .

# Check Python path
python -c "import sys; print(sys.path)"
```

**S3 Connection Issues**
```bash
# Verify AWS credentials
aws s3 ls s3://your-bucket/

# Check IAM permissions for S3 access
```

**CUDA/GPU Issues**
```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Verify CUDA version compatibility
nvidia-smi
```

**Memory Issues**
- Reduce batch size
- Enable gradient checkpointing
- Use smaller LoRA rank
- Implement data streaming

## 📈 Performance Optimization

### Training Optimization
- Use mixed precision training (AMP)
- Gradient accumulation for large effective batch sizes
- Learning rate scheduling
- Early stopping based on validation metrics

### Memory Optimization
- Lazy loading from S3
- Gradient checkpointing
- Lower precision (fp16)
- Smaller batch sizes

### Speed Optimization
- Data prefetching
- Multiple data loading workers
- Optimized data transforms
- Efficient S3 access patterns

## 🔗 Integration

### With Airflow
```python
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from modelfactory import train_model

def training_task(**context):
    result = train_model(hyperparams, aws_config)
    return result

dag = DAG('social_media_training')
training_op = PythonOperator(
    task_id='train_model',
    python_callable=training_task,
    dag=dag
)
```

### With RunPod
The package is designed to work seamlessly with RunPod serverless infrastructure through the `runpod_handler.py` and `runpod_training.py` interfaces.

---

For more information, see the main [README.md](../README.md) in the repository root. 