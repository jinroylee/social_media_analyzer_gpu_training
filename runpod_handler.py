import runpod
import json
import os
import sys
import traceback
from datetime import datetime
import threading
import logging
import time

# Setup logging with more verbose output
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger(__name__)

def setup_python_path():
    """Setup Python path for RunPod environment"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
        logger.info(f"Added {current_dir} to Python path")

# Setup path before imports
setup_python_path()

# Import the training module with error handling
try:
    from runpod_training import train_model
    logger.info("✅ Successfully imported train_model")
except ImportError as e:
    logger.error(f"❌ Failed to import train_model: {e}")
    logger.error("This will cause handler to fail")
    # Don't exit here, let the handler deal with it
    train_model = None

def handler(event):
    """
    RunPod serverless handler function.
    
    Args:
        event: JSON event data containing training parameters
        
    Returns:
        dict: Response with training status and results
    """
    try:
        logger.info("🔥 HANDLER CALLED!")
        logger.info(f"Received event: {event}")
        
        # Check if training module is available
        if train_model is None:
            return {
                'status': 'error',
                'message': 'Training module not available - import failed',
                'timestamp': datetime.now().isoformat()
            }
        
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
        
        # Debug: Log received credentials (safely)
        logger.info(f"AWS Access Key ID: {aws_config['aws_access_key_id'][:10]}..." if aws_config['aws_access_key_id'] else "None")
        logger.info(f"AWS Secret Key: {'*' * 10}...{aws_config['aws_secret_access_key'][-4:] if aws_config['aws_secret_access_key'] else 'None'}")
        logger.info(f"S3 Bucket: {aws_config['s3_bucket']}")
        logger.info(f"AWS Region: {aws_config['aws_region']}")
        
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

def test_handler():
    """Test the handler function with minimal parameters"""
    logger.info("🧪 Testing handler function...")
    test_event = {
        'input': {
            'batch_size': 8,
            'epochs': 1,
            'aws_access_key_id': 'test',
            'aws_secret_access_key': 'test'
        }
    }
    
    try:
        result = handler(test_event)
        logger.info(f"✅ Handler test result: {result}")
        return True
    except Exception as e:
        logger.error(f"❌ Handler test failed: {e}")
        return False

if __name__ == "__main__":
    # Force output to be unbuffered
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    
    logger.info("=" * 60)
    logger.info("🚀 STARTING RUNPOD SERVERLESS HANDLER")
    logger.info("=" * 60)
    
    # Print environment info
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"Python path: {sys.path[:3]}...")  # Show first 3 entries
    
    # Check if we're in test mode
    if len(sys.argv) > 1 and "--test_input" in sys.argv:
        logger.info("🧪 Test mode detected - RunPod will handle test execution")
    
    try:
        # Setup Python path
        setup_python_path()
        logger.info("✅ Python path setup completed")
        
        # Verify critical imports
        try:
            import torch
            logger.info(f"✅ PyTorch version: {torch.__version__}")
        except ImportError as e:
            logger.error(f"❌ PyTorch import failed: {e}")
            
        try:
            import transformers
            logger.info(f"✅ Transformers version: {transformers.__version__}")
        except ImportError as e:
            logger.error(f"❌ Transformers import failed: {e}")
            
        # Verify RunPod import
        logger.info(f"✅ RunPod version: {getattr(runpod, '__version__', 'unknown')}")
        
        # Check if handler is callable
        if callable(handler):
            logger.info("✅ Handler function is callable")
        else:
            logger.error("❌ Handler function is not callable")
            raise RuntimeError("Handler function is not callable")
        
        # Test the handler function (skip if train_model is None)
        if train_model is not None:
            if not test_handler():
                logger.warning("⚠️ Handler test failed, but continuing...")
        else:
            logger.warning("⚠️ Skipping handler test - train_model not available")
        
        # Start the serverless handler
        logger.info("🔥 STARTING RUNPOD SERVERLESS...")
        logger.info("📡 Handler ready to receive requests")
        logger.info("🌐 Waiting for incoming requests...")
        
        # Add a small delay to ensure logs are flushed
        time.sleep(1)
        
        # This is the critical line - it should block and wait for requests
        logger.info("🚀 Calling runpod.serverless.start() now...")
        runpod.serverless.start({"handler": handler})
        
        # This line should never be reached
        logger.error("❌ RunPod serverless handler exited unexpectedly")
        
    except KeyboardInterrupt:
        logger.info("🛑 Handler interrupted by user")
    except Exception as e:
        logger.error("=" * 60)
        logger.error("❌ HANDLER STARTUP FAILED")
        logger.error("=" * 60)
        logger.error(f"Error: {str(e)}")
        logger.error(f"Type: {type(e).__name__}")
        logger.error("Full traceback:")
        logger.error(traceback.format_exc())
        
        # Keep container running for debugging
        logger.error("Keeping container alive for 600 seconds for debugging...")
        time.sleep(600)
        raise
    
    logger.info("Handler process completed") 