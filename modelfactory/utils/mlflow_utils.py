import mlflow
import mlflow.pytorch
import mlflow.sklearn
import torch
import os
from typing import Dict, Any, Optional
import tempfile
import logging

logger = logging.getLogger(__name__)

class MLflowTracker:
    """MLflow experiment tracking utility for social media engagement prediction."""
    
    def __init__(self, 
                 experiment_name: str = "social_media_engagement_prediction",
                 tracking_uri: Optional[str] = None,
                 artifact_location: Optional[str] = None):
        """
        Initialize MLflow tracker.
        
        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: MLflow tracking server URI (default: local file store)
            artifact_location: Location to store artifacts
        """
        self.experiment_name = experiment_name
        
        tracking_uri = "https://mlflow-server-lb-780614927.ap-northeast-2.elb.amazonaws.com"
        # Set tracking URI (default to local file store)
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        else:
            # Use local directory for MLflow tracking
            mlflow_dir = os.path.join(os.getcwd(), "mlruns")
            mlflow.set_tracking_uri(f"file://{mlflow_dir}")
        
        # Create or get experiment
        try:
            self.experiment_id = mlflow.create_experiment(
                name=experiment_name,
                artifact_location=artifact_location
            )
        except mlflow.exceptions.MlflowException:
            # Experiment already exists
            experiment = mlflow.get_experiment_by_name(experiment_name)
            self.experiment_id = experiment.experiment_id
        
        logger.info(f"Using MLflow experiment: {experiment_name} (ID: {self.experiment_id})")
    
    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None):
        """Start a new MLflow run."""
        return mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=run_name,
            tags=tags
        )
    
    def log_hyperparameters(self, params: Dict[str, Any]):
        """Log hyperparameters to MLflow."""
        for key, value in params.items():
            mlflow.log_param(key, value)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to MLflow."""
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)
    
    def log_model_architecture(self, model: torch.nn.Module):
        """Log model architecture information."""
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        mlflow.log_param("total_parameters", total_params)
        mlflow.log_param("trainable_parameters", trainable_params)
        mlflow.log_param("model_class", model.__class__.__name__)
        
        # Log model structure to a text file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(str(model))
            temp_path = f.name
        
        mlflow.log_artifact(temp_path, "model_architecture.txt")
        os.unlink(temp_path)
    
    def log_pytorch_model(self, 
                         model: torch.nn.Module, 
                         input_example: Optional[Dict[str, torch.Tensor]] = None,
                         model_path: str = "model"):
        """
        Log PyTorch model to MLflow.
        
        Args:
            model: The PyTorch model to log
            input_example: Example input for model signature inference
            model_path: Path within the run's artifact directory
        """
        try:
            # Create signature if input example is provided
            signature = None
            if input_example is not None:
                with torch.no_grad():
                    model.eval()
                    prediction = model(**input_example)
                    signature = mlflow.models.infer_signature(
                        input_example, 
                        prediction.cpu().numpy()
                    )
            
            # Log the model
            mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path=model_path,
                signature=signature,
                input_example=input_example
            )
            
            logger.info(f"Model logged to MLflow at path: {model_path}")
            
        except Exception as e:
            logger.error(f"Error logging model to MLflow: {e}")
    
    def log_model_checkpoint(self, model: torch.nn.Module, checkpoint_name: str):
        """Log model state dict as checkpoint."""
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            torch.save(model.state_dict(), f.name)
            mlflow.log_artifact(f.name, f"checkpoints/{checkpoint_name}")
            os.unlink(f.name)
    
    def log_training_config(self, config: Dict[str, Any]):
        """Log training configuration."""
        for key, value in config.items():
            if isinstance(value, (str, int, float, bool)):
                mlflow.log_param(f"config_{key}", value)
            else:
                mlflow.log_param(f"config_{key}", str(value))
    
    def log_dataset_info(self, train_size: int, val_size: int, test_size: Optional[int] = None):
        """Log dataset information."""
        mlflow.log_param("train_dataset_size", train_size)
        mlflow.log_param("val_dataset_size", val_size)
        if test_size:
            mlflow.log_param("test_dataset_size", test_size)
    
    def log_artifact_from_file(self, file_path: str, artifact_path: Optional[str] = None):
        """Log a file as an artifact."""
        mlflow.log_artifact(file_path, artifact_path)
    
    def log_text_artifact(self, text: str, filename: str, artifact_path: Optional[str] = None):
        """Log text content as an artifact."""
        with tempfile.NamedTemporaryFile(mode='w', suffix=f'_{filename}', delete=False) as f:
            f.write(text)
            temp_path = f.name
        
        mlflow.log_artifact(temp_path, artifact_path)
        os.unlink(temp_path)
    
    def end_run(self):
        """End the current MLflow run."""
        mlflow.end_run()
    
    @staticmethod
    def set_tags(tags: Dict[str, str]):
        """Set tags for the current run."""
        for key, value in tags.items():
            mlflow.set_tag(key, value)
    
    @staticmethod
    def log_artifact(local_path: str, artifact_path: Optional[str] = None):
        """Log an artifact to the current run."""
        mlflow.log_artifact(local_path, artifact_path)


def create_experiment_config(
    batch_size: int,
    epochs: int,
    learning_rate: float,
    lora_learning_rate: float,
    lora_rank: int,
    use_lora: bool,
    model_name: str = "openai/clip-vit-large-patch14"
) -> Dict[str, Any]:
    """Create a configuration dictionary for MLflow logging."""
    return {
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "lora_learning_rate": lora_learning_rate,
        "lora_rank": lora_rank,
        "use_lora": use_lora,
        "base_model": model_name,
        "loss_function": "HuberLoss",
        "optimizer": "AdamW",
        "scheduler": "ReduceLROnPlateau"
    } 