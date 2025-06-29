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
from dotenv import load_dotenv
from botocore.exceptions import ClientError
from io import BytesIO
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import spearmanr

# Handle different execution environments
def setup_python_path():
    """Setup Python path for different execution environments"""
    # SageMaker environment
    sagemaker_code_dir = '/opt/ml/code'
    if os.path.exists(sagemaker_code_dir) and sagemaker_code_dir not in sys.path:
        sys.path.insert(0, sagemaker_code_dir)
        print(f"Added {sagemaker_code_dir} to Python path")
    
    # Local development environment
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        print(f"Added {project_root} to Python path")

# Setup path before imports
setup_python_path()

# Now import your custom modules
try:
    from modelfactory.models.clip_regressor import CLIPEngagementRegressor
    from modelfactory.utils.engagement_dataset import EngagementDataset
    from modelfactory.utils.mlflow_utils import MLflowTracker, create_experiment_config
    print("Successfully imported modelfactory modules")
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Python path: {sys.path}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Directory contents: {os.listdir('.')}")
    raise

# SageMaker specific paths
SM_MODEL_DIR = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')
SM_OUTPUT_DATA_DIR = os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output/data')
SM_CHANNEL_TRAINING = os.environ.get('SM_CHANNEL_TRAINING', '/opt/ml/input/data/training')

# Load environment variables (for local testing)
load_dotenv()

def get_hyperparameters():
    """Get hyperparameters from SageMaker or environment variables"""
    return {
        'batch_size': int(os.environ.get('batch_size', '32')),
        'epochs': int(os.environ.get('epochs', '10')),
        'learning_rate': float(os.environ.get('learning_rate', '1e-4')),
        'lora_learning_rate': float(os.environ.get('lora_learning_rate', '1e-3')),
        'lora_rank': int(os.environ.get('lora_rank', '8')),
        'use_lora': os.environ.get('use_lora', 'true').lower() == 'true',
    }

def get_aws_config():
    """Get AWS configuration"""
    return {
        's3_bucket': os.environ.get('S3_BUCKET_NAME', 'socialmediaanalyzer'),
        'aws_region': os.environ.get('AWS_REGION', 'ap-northeast-2'),
    }

def load_pkl_from_s3(bucket_name, key):
    """Load pickle data from S3"""
    try:
        s3_client = boto3.client('s3')
        response = s3_client.get_object(Bucket=bucket_name, Key=key)
        pkl_data = pickle.loads(response['Body'].read())
        return pkl_data
    except ClientError as e:
        print(f"Error loading pickle from S3 {key}: {e}")
        raise

def save_model_to_s3(model, bucket_name, s3_key):
    """Save PyTorch model state dict to S3"""
    try:
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
        
        print(f"Model successfully saved to s3://{bucket_name}/{s3_key}")
        return True
        
    except ClientError as e:
        print(f"Error saving model to S3: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error saving model to S3: {e}")
        return False

def save_config_to_s3(config, bucket_name, s3_key):
    """Save model configuration to S3"""
    try:
        s3_client = boto3.client('s3')
        
        # Convert config to JSON string
        config_json = json.dumps(config, indent=2)
        
        # Upload to S3
        s3_client.put_object(
            Bucket=bucket_name,
            Key=s3_key,
            Body=config_json.encode('utf-8'),
            ContentType='application/json'
        )
        
        print(f"Config successfully saved to s3://{bucket_name}/{s3_key}")
        return True
        
    except ClientError as e:
        print(f"Error saving config to S3: {e}")
        return False

def save_model_artifacts(model, model_dir, hyperparams):
    """Save model artifacts for SageMaker"""
    # Ensure directory exists
    os.makedirs(model_dir, exist_ok=True)
    
    # Save PyTorch model
    model_path = os.path.join(model_dir, 'model.pth')
    torch.save(model.state_dict(), model_path)
    
    # Save model configuration
    config = {
        'model_type': 'CLIPEngagementRegressor',
        'use_lora': hyperparams['use_lora'],
        'lora_rank': hyperparams['lora_rank'],
        'clip_model_name': 'openai/clip-vit-large-patch14',
        'hyperparameters': hyperparams,
        'timestamp': datetime.now().isoformat()
    }
    
    config_path = os.path.join(model_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Model artifacts saved to {model_dir}")
    print(f"Files saved: {os.listdir(model_dir)}")
    
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

def main():
    """Main training function"""
    print("Starting SageMaker training job...")
    print(f"Working directory: {os.getcwd()}")
    print(f"Python path: {sys.path[:3]}...")  # Show first 3 entries
    
    # Get configuration
    hyperparams = get_hyperparameters()
    aws_config = get_aws_config()
    
    print(f"Hyperparameters: {hyperparams}")
    print(f"AWS Config: {aws_config}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directories
    os.makedirs(SM_OUTPUT_DATA_DIR, exist_ok=True)
    os.makedirs(SM_MODEL_DIR, exist_ok=True)
    
    # Initialize MLflow tracker
    mlflow_tracker = MLflowTracker(
        experiment_name="sagemaker_social_media_engagement",
        tracking_uri=f"file://{SM_OUTPUT_DATA_DIR}/mlruns",
        artifact_location=f"{SM_OUTPUT_DATA_DIR}/mlruns"
    )
    
    run_name = f"sagemaker_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    with mlflow_tracker.start_run(run_name=run_name) as run:
        # Set tags
        mlflow_tracker.set_tags({
            "model_type": "CLIP_LoRA",
            "task": "engagement_prediction",
            "framework": "pytorch",
            "device": str(device),
            "sagemaker": "true"
        })
        
        # Log hyperparameters
        config = create_experiment_config(**hyperparams)
        mlflow_tracker.log_hyperparameters(config)
        
        # Load processors
        print("Loading CLIP processors...")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        
        # Load data from S3
        print("Loading training data from S3...")
        data = load_pkl_from_s3(aws_config['s3_bucket'], "processed/train.pkl")
        print(f"Loaded {len(data)} training samples")
        
        # Create dataset
        print("Creating dataset...")
        dataset = EngagementDataset(data, processor, tokenizer)
        
        # Split data
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        print(f"Train size: {train_size}, Validation size: {val_size}")
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=hyperparams['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=hyperparams['batch_size'], shuffle=False)
        
        # Initialize model
        print("Initializing model...")
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
        
        print("Starting training...")
        
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
                    print(f'Epoch {epoch+1}/{hyperparams["epochs"]}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
            
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
            
            mlflow_tracker.log_metrics(epoch_metrics, step=epoch)
            
            print(f'Epoch {epoch+1}/{hyperparams["epochs"]}:')
            print(f'  Train - Loss: {avg_train_loss:.4f}, MAE: {train_mae:.4f}, R²: {train_r2:.4f}')
            print(f'  Val - Loss: {val_loss:.4f}, MAE: {val_mae:.4f}, R²: {val_r2:.4f}')
            
            # Save best model
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_model_state = model.state_dict().copy()
                print(f'  New best model! MAE: {val_mae:.4f}')
        
        # Load best model state
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            print(f"Loaded best model with validation MAE: {best_val_mae:.4f}")
        
        # Save final model artifacts locally
        print("Saving model artifacts locally...")
        config = save_model_artifacts(model, SM_MODEL_DIR, hyperparams)
        
        # Save model to S3
        print("Uploading model to S3...")
        model_s3_key = "models/best_model_lora.pth"
        config_s3_key = "models/best_model_lora_config.json"
        
        # Upload model to S3
        if save_model_to_s3(model, aws_config['s3_bucket'], model_s3_key):
            print(f"✅ Model uploaded to s3://{aws_config['s3_bucket']}/{model_s3_key}")
        else:
            print("❌ Failed to upload model to S3")
        
        # Upload config to S3
        if save_config_to_s3(config, aws_config['s3_bucket'], config_s3_key):
            print(f"✅ Config uploaded to s3://{aws_config['s3_bucket']}/{config_s3_key}")
        else:
            print("❌ Failed to upload config to S3")
        
        # Log final metrics
        mlflow_tracker.log_metrics({
            'best_val_mae': best_val_mae,
            'final_epoch': hyperparams['epochs'],
            's3_model_path': f"s3://{aws_config['s3_bucket']}/{model_s3_key}"
        })
        
        print(f'Training completed! Best validation MAE: {best_val_mae:.4f}')
        print(f'Model saved locally to {SM_MODEL_DIR}')
        print(f'Model uploaded to S3: s3://{aws_config["s3_bucket"]}/{model_s3_key}')
        print(f"Final model directory contents: {os.listdir(SM_MODEL_DIR)}")

if __name__ == "__main__":
    main()
