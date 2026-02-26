"""
Machine Learning Models module for Smart Mess Optimization System
Provides demand prediction, forecasting, and pattern recognition capabilities
"""

from app.ml_models.demand_predictor import DemandPredictor
from app.ml_models.data_preprocessor import DataPreprocessor, FeatureEngineer
from app.ml_models.train_model import ModelTrainer, ModelEvaluator
from app.ml_models.model_registry import ModelRegistry
from app.ml_models.ensemble import EnsemblePredictor
from app.ml_models.time_series import TimeSeriesForecaster

__version__ = '1.0.0'
__all__ = [
    # Main predictor
    'DemandPredictor',
    
    # Data processing
    'DataPreprocessor',
    'FeatureEngineer',
    
    # Training and evaluation
    'ModelTrainer',
    'ModelEvaluator',
    
    # Model management
    'ModelRegistry',
    
    # Advanced models
    'EnsemblePredictor',
    'TimeSeriesForecaster'
]

# Package metadata
__author__ = 'Smart Mess Optimization Team'
__description__ = 'ML models for food demand prediction and optimization'
__license__ = 'MIT'

# Model configuration defaults
DEFAULT_MODEL_CONFIG = {
    'xgboost': {
        'n_estimators': 200,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'objective': 'reg:squarederror'
    },
    'random_forest': {
        'n_estimators': 200,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': 42,
        'n_jobs': -1
    },
    'lstm': {
        'lstm_units_1': 64,
        'lstm_units_2': 32,
        'dense_units': 16,
        'dropout_rate': 0.2,
        'epochs': 50,
        'batch_size': 32,
        'validation_split': 0.2,
        'optimizer': 'adam',
        'loss': 'mse'
    },
    'gradient_boosting': {
        'n_estimators': 150,
        'max_depth': 5,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'random_state': 42
    }
}

# Feature columns used in models
DEFAULT_FEATURE_COLUMNS = [
    'day_of_week',
    'month',
    'day',
    'is_weekend',
    'is_holiday',
    'is_exam',
    'temperature',
    'rainfall',
    'humidity',
    'special_event_encoded',
    'meal_type_encoded',
    'dish_name_encoded',
    'lag_1_day',
    'lag_7_day',
    'lag_30_day',
    'rolling_mean_7',
    'rolling_mean_30',
    'day_of_month',
    'week_of_year',
    'season'
]

# Model performance thresholds
PERFORMANCE_THRESHOLDS = {
    'rmse': 50,  # Maximum acceptable RMSE
    'mae': 30,   # Maximum acceptable MAE
    'r2': 0.7,   # Minimum acceptable R²
    'mape': 20   # Maximum acceptable MAPE (%)
}

def get_model_info(model_type: str = None) -> dict:
    """
    Get information about available models
    
    Args:
        model_type: Specific model type to get info for
        
    Returns:
        Dictionary with model information
    """
    model_info = {
        'xgboost': {
            'name': 'XGBoost Regressor',
            'description': 'Gradient boosting framework known for speed and performance',
            'best_for': 'Tabular data with mixed features',
            'training_time': 'Medium',
            'interpretability': 'High',
            'supports_gpu': True
        },
        'random_forest': {
            'name': 'Random Forest Regressor',
            'description': 'Ensemble of decision trees, robust to overfitting',
            'best_for': 'Handling non-linear relationships',
            'training_time': 'Medium',
            'interpretability': 'Medium',
            'supports_gpu': False
        },
        'lstm': {
            'name': 'LSTM Neural Network',
            'description': 'Long Short-Term Memory network for sequence prediction',
            'best_for': 'Time series and sequential data',
            'training_time': 'High',
            'interpretability': 'Low',
            'supports_gpu': True
        },
        'gradient_boosting': {
            'name': 'Gradient Boosting Regressor',
            'description': 'Classic gradient boosting machine',
            'best_for': 'Structured data with clear patterns',
            'training_time': 'Medium',
            'interpretability': 'Medium',
            'supports_gpu': False
        }
    }
    
    if model_type:
        return model_info.get(model_type, {})
    return model_info

def validate_model_config(config: dict) -> bool:
    """
    Validate model configuration
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        True if valid, raises ValueError otherwise
    """
    required_keys = ['n_estimators', 'max_depth', 'learning_rate', 'random_state']
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")
    
    if config['n_estimators'] < 10:
        raise ValueError("n_estimators must be at least 10")
    
    if config['max_depth'] < 1:
        raise ValueError("max_depth must be at least 1")
    
    if config['learning_rate'] <= 0 or config['learning_rate'] > 1:
        raise ValueError("learning_rate must be between 0 and 1")
    
    return True

# Initialize logging for ML models
import logging
ml_logger = logging.getLogger(__name__)
ml_logger.info(f"ML Models module initialized (version {__version__})")
ml_logger.info(f"Available models: {', '.join(DEFAULT_MODEL_CONFIG.keys())}")