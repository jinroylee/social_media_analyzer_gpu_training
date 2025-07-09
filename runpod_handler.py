import runpod
import json
import os
import sys
import traceback
from datetime import datetime
import threading
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_python_path():
    """Setup Python path for RunPod environment"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
        logger.info(f"Added {current_dir} to Python path")

# Setup path before imports
setup_python_path()

# Import the training module
from runpod_training import train_model

def handler(event):
    """
    RunPod serverless handler function.
    
    Args:
        event: JSON event data containing training parameters
        
    Returns:
        dict: Response with training status and results
    """
    try:
        logger.info(f"Received event: {event}")
        
        # Extract input parameters
        job_input = event.get('input', {})
        
        # Set default hyperparameters if not provided
        hyperparams = {
            'batch_size': job_input.get('batch_size', 32),
            'epochs': job_input.get('epochs', 10),
            'learning_rate': job_input.get('learning_rate', 1e-4),
            'lora_learning_rate': job_input.get('lora_learning_rate', 1e-3),
            'lora_rank': job_input.get('lora_rank', 8),
            'use_lora': job_input.get('use_lora', True),
        }
        
        # AWS/S3 configuration
        aws_config = {
            's3_bucket': job_input.get('s3_bucket', 'socialmediaanalyzer'),
            'aws_region': job_input.get('aws_region', 'ap-northeast-2'),
            'aws_access_key_id': job_input.get('aws_access_key_id'),
            'aws_secret_access_key': job_input.get('aws_secret_access_key'),
        }
        
        # Validate required AWS credentials
        if not aws_config['aws_access_key_id'] or not aws_config['aws_secret_access_key']:
            return {
                'status': 'error',
                'message': 'AWS credentials (aws_access_key_id and aws_secret_access_key) are required',
                'timestamp': datetime.now().isoformat()
            }
        
        # Set AWS environment variables
        os.environ['AWS_ACCESS_KEY_ID'] = aws_config['aws_access_key_id']
        os.environ['AWS_SECRET_ACCESS_KEY'] = aws_config['aws_secret_access_key']
        os.environ['AWS_DEFAULT_REGION'] = aws_config['aws_region']
        
        logger.info(f"Starting training with hyperparameters: {hyperparams}")
        logger.info(f"AWS config: {aws_config['s3_bucket']}, {aws_config['aws_region']}")
        
        # Start training
        training_result = train_model(hyperparams, aws_config)
        
        # Return success response
        return {
            'status': 'success',
            'message': 'Training completed successfully',
            'hyperparameters': hyperparams,
            'training_result': training_result,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        logger.error(traceback.format_exc())
        
        return {
            'status': 'error',
            'message': f'Training failed: {str(e)}',
            'traceback': traceback.format_exc(),
            'timestamp': datetime.now().isoformat()
        }

if __name__ == "__main__":
    # Start the RunPod serverless handler
    logger.info("Starting RunPod serverless handler...")
    runpod.serverless.start({"handler": handler}) 