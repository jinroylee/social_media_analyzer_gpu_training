import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from transformers import CLIPProcessor, CLIPTokenizer
import sys
import os
import json
import pickle
import math
from PIL import Image
from datetime import datetime
import boto3
from botocore.exceptions import ClientError
from io import BytesIO
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import spearmanr
import logging

# Setup logging
logger = logging.getLogger(__name__)

def setup_python_path():
    """Setup Python path for RunPod environment"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
        logger.info(f"Added {current_dir} to Python path")

# Setup path before imports
setup_python_path()

# Import custom modules
try:
    from modelfactory.models.clip_regressor import CLIPEngagementRegressor
    from modelfactory.utils.engagement_dataset import EngagementDataset
    logger.info("Successfully imported modelfactory modules")
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.error(f"Python path: {sys.path}")
    logger.error(f"Current working directory: {os.getcwd()}")
    logger.error(f"Directory contents: {os.listdir('.')}")
    raise

# Try to import MLflow, but make it optional
try:
    from modelfactory.utils.mlflow_utils import MLflowTracker, create_experiment_config
    MLFLOW_AVAILABLE = True
    logger.info("MLflow is available")
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow is not available - experiment tracking will be disabled")
    
    # Create dummy classes for MLflow functionality
    class MLflowTracker:
        def __init__(self, *args, **kwargs):
            pass
        def start_run(self, *args, **kwargs):
            return self
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def set_tags(self, *args, **kwargs):
            pass
        def log_hyperparameters(self, *args, **kwargs):
            pass
        def log_metrics(self, *args, **kwargs):
            pass
    
    def create_experiment_config(**kwargs):
        return kwargs

def load_pkl_from_s3(bucket_name, key, aws_config=None):
    """Load pickle data from S3"""
    try:
        if aws_config:
            s3_client = boto3.client(
                's3',
                aws_access_key_id=aws_config['aws_access_key_id'],
                aws_secret_access_key=aws_config['aws_secret_access_key'],
                region_name=aws_config['aws_region']
            )
        else:
            s3_client = boto3.client('s3')
        
        response = s3_client.get_object(Bucket=bucket_name, Key=key)
        pkl_data = pickle.loads(response['Body'].read())
        return pkl_data
    except ClientError as e:
        logger.error(f"Error loading pickle from S3 {key}: {e}")
        raise

def save_model_to_s3(model, bucket_name, s3_key, aws_config=None):
    """Save PyTorch model state dict to S3"""
    try:
        if aws_config:
            s3_client = boto3.client(
                's3',
                aws_access_key_id=aws_config['aws_access_key_id'],
                aws_secret_access_key=aws_config['aws_secret_access_key'],
                region_name=aws_config['aws_region']
            )
        else:
            s3_client = boto3.client('s3')
        
        # Save model to a BytesIO buffer
        buffer = BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        
        # Upload to S3
        s3_client.put_object(
            Bucket=bucket_name,
            Key=s3_key,
            Body=buffer.getvalue(),
            ContentType='application/octet-stream'
        )
        
        logger.info(f"Model saved to S3: s3://{bucket_name}/{s3_key}")
        return f"s3://{bucket_name}/{s3_key}"
        
    except ClientError as e:
        logger.error(f"Error saving model to S3 {s3_key}: {e}")
        raise

def save_config_to_s3(config, bucket_name, s3_key, aws_config=None):
    """Save configuration dictionary to S3 as JSON"""
    try:
        if aws_config:
            s3_client = boto3.client(
                's3',
                aws_access_key_id=aws_config['aws_access_key_id'],
                aws_secret_access_key=aws_config['aws_secret_access_key'],
                region_name=aws_config['aws_region']
            )
        else:
            s3_client = boto3.client('s3')
        
        # Convert config to JSON
        config_json = json.dumps(config, indent=2, default=str)
        
        # Upload to S3
        s3_client.put_object(
            Bucket=bucket_name,
            Key=s3_key,
            Body=config_json,
            ContentType='application/json'
        )
        
        logger.info(f"Config saved to S3: s3://{bucket_name}/{s3_key}")
        return f"s3://{bucket_name}/{s3_key}"
        
    except ClientError as e:
        logger.error(f"Error saving config to S3 {s3_key}: {e}")
        raise

def create_local_model_artifacts(model, hyperparams):
    """Create model artifacts locally"""
    # Create local model directory
    model_dir = '/app/models'
    os.makedirs(model_dir, exist_ok=True)
    
    # Save PyTorch model
    model_path = os.path.join(model_dir, 'model.pth')
    torch.save(model.state_dict(), model_path)
    
    # Create model configuration
    config = {
        'model_type': 'CLIPEngagementRegressor',
        'use_lora': hyperparams['use_lora'],
        'lora_rank': hyperparams['lora_rank'],
        'clip_model_name': 'openai/clip-vit-large-patch14',
        'hyperparameters': hyperparams,
        'timestamp': datetime.now().isoformat()
    }
    
    # Save config
    config_path = os.path.join(model_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Model artifacts created locally at {model_dir}")
    logger.info(f"Files created: {os.listdir(model_dir)}")
    
    return config

def evaluate(model, dataloader, device):
    """Evaluation function"""
    model.eval()
    predictions = []
    targets = []
    total_loss = 0
    criterion = nn.HuberLoss()
    
    with torch.no_grad():
        for batch in dataloader:
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            sentiment = batch['sentiment'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(pixel_values, input_ids, attention_mask, sentiment)
            loss = criterion(outputs.squeeze(), labels)
            total_loss += loss.item()
            
            predictions.extend(outputs.squeeze().cpu().numpy())
            targets.extend(labels.cpu().numpy())
    
    mae = mean_absolute_error(targets, predictions)
    correlation, _ = spearmanr(targets, predictions)
    r2 = r2_score(targets, predictions)
    avg_loss = total_loss / len(dataloader)
    
    return mae, correlation, r2, avg_loss

def load_and_process_training_data(aws_config):
    """Load training data from S3 and process images"""
    logger.info("Loading training data from S3...")
    
    # Load the raw training data
    data = load_pkl_from_s3(aws_config['s3_bucket'], "processed/train.pkl", aws_config)
    logger.info(f"Loaded {len(data)} training samples")
    
    # Create S3 client for image loading
    s3_client = boto3.client(
        's3',
        aws_access_key_id=aws_config['aws_access_key_id'],
        aws_secret_access_key=aws_config['aws_secret_access_key'],
        region_name=aws_config['aws_region']
    )
    
    # Process each item to load images
    processed_data = []
    logger.info("Processing images from S3...")
    
    for i, item in enumerate(data):
        try:
            # Load image from S3
            image_key = item['thumbnail_path']
            response = s3_client.get_object(Bucket=aws_config['s3_bucket'], Key=image_key)
            image_data = response['Body'].read()
            
            # Convert to PIL Image
            image = Image.open(BytesIO(image_data))
            
            # Create processed item with image data
            processed_item = item.copy()
            processed_item['image'] = image
            processed_data.append(processed_item)
            
            # Log progress every 1000 items
            if (i + 1) % 1000 == 0:
                logger.info(f"Processed {i + 1}/{len(data)} images")
                
        except Exception as e:
            logger.warning(f"Failed to load image {item.get('thumbnail_path', 'unknown')}: {e}")
            # Skip this item if image loading fails
            continue
    
    logger.info(f"Successfully processed {len(processed_data)} samples with images")
    return processed_data

def train_model(hyperparams, aws_config):
    """
    Main training function for RunPod serverless environment.
    
    Args:
        hyperparams: Dictionary of training hyperparameters
        aws_config: Dictionary of AWS configuration
        
    Returns:
        dict: Training results and metrics
    """
    logger.info("Starting RunPod GPU training job...")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"Python path: {sys.path[:3]}...")
    
    logger.info(f"Hyperparameters: {hyperparams}")
    logger.info(f"AWS Config: {aws_config['s3_bucket']}, {aws_config['aws_region']}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create output directories
    os.makedirs('/app/tmp', exist_ok=True)
    os.makedirs('/app/logs', exist_ok=True)
    
    # Initialize MLflow tracker for local logging
    mlflow_tracker = MLflowTracker(
        experiment_name="runpod_social_media_engagement",
        tracking_uri="file:///app/logs/mlruns",
        artifact_location="/app/logs/mlruns"
    )
    
    run_name = f"runpod_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    with mlflow_tracker.start_run(run_name=run_name) as run:
        # Set tags
        mlflow_tracker.set_tags({
            "model_type": "CLIP_LoRA",
            "task": "engagement_prediction",
            "framework": "pytorch",
            "device": str(device),
            "runpod": "true"
        })
        
        # Log hyperparameters
        config = create_experiment_config(**hyperparams)
        mlflow_tracker.log_hyperparameters(config)
        
        # Load processors
        logger.info("Loading CLIP processors...")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        
        # Load data from S3
        data = load_and_process_training_data(aws_config)
        
        # Create dataset
        logger.info("Creating dataset...")
        dataset = EngagementDataset(data, processor, tokenizer)
        
        # Split data
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        logger.info(f"Train size: {train_size}, Validation size: {val_size}")
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=hyperparams['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=hyperparams['batch_size'], shuffle=False)
        
        # Initialize model
        logger.info("Initializing model...")
        model = CLIPEngagementRegressor(
            use_lora=hyperparams['use_lora'], 
            lora_rank=hyperparams['lora_rank']
        ).to(device)
        
        # Setup optimizer
        lora_params = []
        other_params = []
        
        for name, param in model.named_parameters():
            if 'lora_' in name:
                lora_params.append(param)
            else:
                other_params.append(param)
        
        optimizer = torch.optim.AdamW([
            {'params': other_params, 'lr': hyperparams['learning_rate']},
            {'params': lora_params, 'lr': hyperparams['lora_learning_rate']}
        ])
        
        criterion = nn.HuberLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
        
        best_val_mae = float('inf')
        best_model_state = None
        training_metrics = []
        
        logger.info("Starting training loop...")
        
        # Training loop
        for epoch in range(hyperparams['epochs']):
            # Training phase
            model.train()
            train_loss = 0
            train_predictions = []
            train_targets = []
            
            for batch_idx, batch in enumerate(train_loader):
                pixel_values = batch['pixel_values'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                sentiment = batch['sentiment'].to(device)
                labels = batch['label'].to(device)
                
                optimizer.zero_grad()
                outputs = model(pixel_values, input_ids, attention_mask, sentiment)
                loss = criterion(outputs.squeeze(), labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_predictions.extend(outputs.squeeze().detach().cpu().numpy())
                train_targets.extend(labels.cpu().numpy())
                
                if batch_idx % 10 == 0:
                    logger.info(f'Epoch {epoch+1}/{hyperparams["epochs"]}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
            
            # Calculate training metrics
            train_mae = mean_absolute_error(train_targets, train_predictions)
            train_corr, _ = spearmanr(train_targets, train_predictions)
            train_r2 = r2_score(train_targets, train_predictions)
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation
            val_mae, val_corr, val_r2, val_loss = evaluate(model, val_loader, device)
            scheduler.step(val_loss)
            
            # Log metrics
            epoch_metrics = {
                'train_loss': avg_train_loss,
                'train_mae': train_mae,
                'train_correlation': train_corr,
                'train_r2': train_r2,
                'val_loss': val_loss,
                'val_mae': val_mae,
                'val_correlation': val_corr,
                'val_r2': val_r2,
                'learning_rate': optimizer.param_groups[0]['lr']
            }
            
            training_metrics.append(epoch_metrics)
            mlflow_tracker.log_metrics(epoch_metrics, step=epoch)
            
            logger.info(f'Epoch {epoch+1}/{hyperparams["epochs"]}:')
            logger.info(f'  Train - Loss: {avg_train_loss:.4f}, MAE: {train_mae:.4f}, R²: {train_r2:.4f}')
            logger.info(f'  Val - Loss: {val_loss:.4f}, MAE: {val_mae:.4f}, R²: {val_r2:.4f}')
            
            # Save best model
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_model_state = model.state_dict().copy()
                logger.info(f'  New best model! MAE: {val_mae:.4f}')
        
        # Load best model state
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            logger.info(f"Loaded best model with validation MAE: {best_val_mae:.4f}")
        
        # Create model artifacts locally
        logger.info("Creating model artifacts locally...")
        config = create_local_model_artifacts(model, hyperparams)
        
        # Save model to S3
        logger.info("Uploading model to S3...")
        model_s3_key = f"models/runpod_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
        config_s3_key = f"models/runpod_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Upload model to S3
        model_uploaded = save_model_to_s3(model, aws_config['s3_bucket'], model_s3_key, aws_config)
        config_uploaded = save_config_to_s3(config, aws_config['s3_bucket'], config_s3_key, aws_config)
        
        # Log final metrics
        final_metrics = {
            'best_val_mae': best_val_mae,
            'final_epoch': hyperparams['epochs'],
            's3_model_path': f"s3://{aws_config['s3_bucket']}/{model_s3_key}",
            'model_uploaded': model_uploaded,
            'config_uploaded': config_uploaded
        }
        
        mlflow_tracker.log_metrics(final_metrics)
        
        logger.info(f'Training completed! Best validation MAE: {best_val_mae:.4f}')
        logger.info(f'Model uploaded to S3: s3://{aws_config["s3_bucket"]}/{model_s3_key}')
        
        # Return training results
        return {
            'best_val_mae': best_val_mae,
            'final_epoch': hyperparams['epochs'],
            's3_model_path': f"s3://{aws_config['s3_bucket']}/{model_s3_key}",
            's3_config_path': f"s3://{aws_config['s3_bucket']}/{config_s3_key}",
            'model_uploaded': model_uploaded,
            'config_uploaded': config_uploaded,
            'training_metrics': training_metrics[-5:] if len(training_metrics) > 5 else training_metrics,  # Last 5 epochs
            'total_epochs': len(training_metrics)
        } 