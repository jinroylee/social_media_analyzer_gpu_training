"""
Social Media Analyzer - Model Factory Package
"""

__version__ = "0.1.0"
__author__ = "ML Team"
__description__ = "Social Media Engagement Prediction ML Pipeline"

# Import key components for easier access
try:
    from .models.clip_regressor import CLIPEngagementRegressor
    from .utils.engagement_dataset import EngagementDataset
    from .utils.mlflow_utils import MLflowTracker, create_experiment_config
    
    __all__ = [
        'CLIPEngagementRegressor',
        'EngagementDataset', 
        'MLflowTracker',
        'create_experiment_config'
    ]
except ImportError as e:
    # Handle import errors gracefully during package installation
    print(f"Warning: Some imports failed during package initialization: {e}")
    __all__ = []
