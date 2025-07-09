# RunPod Serverless GPU Training

This project has been converted from AWS SageMaker to RunPod Serverless for GPU-accelerated machine learning model training.

## Overview

The RunPod serverless function trains a CLIP-based engagement prediction model using LoRA fine-tuning. The training data is loaded from S3, and the trained model is saved back to S3 storage.

## Files Structure

- `Dockerfile.runpod` - RunPod-compatible Docker image
- `runpod_handler.py` - RunPod serverless handler (entry point)
- `runpod_training.py` - Core training logic adapted for RunPod
- `docker/requirements.txt` - Updated with RunPod dependencies
- `modelfactory/` - Model and utility modules (unchanged)

## Docker Build

```bash
docker build -f Dockerfile.runpod -t your-registry/social-media-trainer .
```

## RunPod Setup

1. **Create a RunPod Template:**
   - Go to RunPod console
   - Create a new template
   - Set the container image to your built Docker image
   - Configure GPU settings (recommend RTX 4090 or A100)
   - Set environment variables if needed

2. **Deploy as Serverless Endpoint:**
   - Create a new serverless endpoint
   - Use the template created above
   - Configure auto-scaling settings
   - Get the endpoint URL

## API Usage

### Request Format

```json
{
  "input": {
    "batch_size": 32,
    "epochs": 10,
    "learning_rate": 1e-4,
    "lora_learning_rate": 1e-3,
    "lora_rank": 8,
    "use_lora": true,
    "s3_bucket": "your-s3-bucket",
    "aws_region": "us-east-1",
    "aws_access_key_id": "your-access-key",
    "aws_secret_access_key": "your-secret-key"
  }
}
```

### Response Format

**Success Response:**
```json
{
  "status": "success",
  "message": "Training completed successfully",
  "hyperparameters": {
    "batch_size": 32,
    "epochs": 10,
    "learning_rate": 1e-4,
    "lora_learning_rate": 1e-3,
    "lora_rank": 8,
    "use_lora": true
  },
  "training_result": {
    "best_val_mae": 0.1234,
    "final_epoch": 10,
    "s3_model_path": "s3://your-bucket/models/runpod_model_20231201_123456.pth",
    "s3_config_path": "s3://your-bucket/models/runpod_config_20231201_123456.json",
    "model_uploaded": true,
    "config_uploaded": true,
    "training_metrics": [...],
    "total_epochs": 10
  },
  "timestamp": "2023-12-01T12:34:56.789Z"
}
```

**Error Response:**
```json
{
  "status": "error",
  "message": "Training failed: [error message]",
  "traceback": "[detailed error traceback]",
  "timestamp": "2023-12-01T12:34:56.789Z"
}
```

## Python Client Example

```python
import requests
import json

# RunPod endpoint URL (replace with your actual endpoint)
endpoint_url = "https://api.runpod.ai/v2/your-endpoint-id/runsync"

# Request payload
payload = {
    "input": {
        "batch_size": 32,
        "epochs": 5,
        "learning_rate": 1e-4,
        "lora_learning_rate": 1e-3,
        "lora_rank": 8,
        "use_lora": True,
        "s3_bucket": "your-s3-bucket",
        "aws_region": "us-east-1",
        "aws_access_key_id": "your-access-key",
        "aws_secret_access_key": "your-secret-key"
    }
}

# Headers
headers = {
    "Authorization": "Bearer your-runpod-api-key",
    "Content-Type": "application/json"
}

# Make request
response = requests.post(endpoint_url, json=payload, headers=headers)
result = response.json()

print(f"Status: {result['status']}")
if result['status'] == 'success':
    print(f"Best validation MAE: {result['training_result']['best_val_mae']}")
    print(f"Model saved to: {result['training_result']['s3_model_path']}")
else:
    print(f"Error: {result['message']}")
```

## cURL Example

```bash
curl -X POST \
  https://api.runpod.ai/v2/your-endpoint-id/runsync \
  -H "Authorization: Bearer your-runpod-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "batch_size": 32,
      "epochs": 5,
      "learning_rate": 1e-4,
      "lora_learning_rate": 1e-3,
      "lora_rank": 8,
      "use_lora": true,
      "s3_bucket": "your-s3-bucket",
      "aws_region": "us-east-1",
      "aws_access_key_id": "your-access-key",
      "aws_secret_access_key": "your-secret-key"
    }
  }'
```

## Required Environment Setup

### S3 Data Structure
Your S3 bucket should contain:
```
your-s3-bucket/
├── processed/
│   └── train.pkl          # Training data in pickle format
└── models/                # Output directory for trained models
    ├── runpod_model_*.pth # Model weights
    └── runpod_config_*.json # Model configuration
```

### Data Format
The `train.pkl` file should contain a list of dictionaries with the following structure:
```python
[
    {
        'image': PIL.Image,          # PIL Image object
        'text': str,                 # Text description
        'sentiment': float,          # Sentiment score
        'label': float              # Engagement score (target)
    },
    ...
]
```

## Hyperparameters

- `batch_size`: Training batch size (default: 32)
- `epochs`: Number of training epochs (default: 10)
- `learning_rate`: Base learning rate (default: 1e-4)
- `lora_learning_rate`: LoRA adapter learning rate (default: 1e-3)
- `lora_rank`: LoRA rank parameter (default: 8)
- `use_lora`: Whether to use LoRA fine-tuning (default: true)

## Features

- **GPU-optimized**: Uses CUDA for faster training
- **LoRA fine-tuning**: Efficient parameter-efficient fine-tuning
- **S3 integration**: Loads data from and saves models to S3
- **MLflow tracking**: Local experiment tracking
- **Automatic model saving**: Best model is automatically saved
- **Comprehensive metrics**: Training and validation metrics tracked
- **Error handling**: Robust error handling with detailed messages

## Cost Optimization

- Uses RunPod's pay-per-use pricing
- Automatically scales down when not in use
- Efficient Docker image with minimal dependencies
- LoRA fine-tuning reduces memory usage

## Monitoring

- Check RunPod console for job status
- Logs are available in the RunPod interface
- MLflow logs are stored locally during training
- Training metrics are returned in the API response

## Troubleshooting

1. **Import errors**: Ensure all dependencies are in requirements.txt
2. **S3 access denied**: Check AWS credentials and bucket permissions
3. **CUDA out of memory**: Reduce batch_size or use smaller LoRA rank
4. **Training data not found**: Verify S3 bucket and file path
5. **Model upload failed**: Check S3 write permissions

## Security Notes

- AWS credentials are passed through the API (consider using IAM roles)
- Use HTTPS for all API calls
- Store RunPod API keys securely
- Consider using AWS STS for temporary credentials 