import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import joblib
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')

class DemandPredictor:
    def __init__(self, model_type='xgboost'):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = ['day_of_week', 'month', 'temperature', 'rainfall', 
                                'is_weekend', 'is_holiday', 'is_exam', 'special_event']
        
    def prepare_features(self, df):
        """Prepare features for model training/prediction"""
        df = df.copy()
        
        # Extract date features
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df['day_of_week'] = df['date'].dt.dayofweek
            df['month'] = df['date'].dt.month
            df['day'] = df['date'].dt.day
        
        # Encode categorical variables
        categorical_cols = ['meal_type', 'dish_name', 'special_event']
        for col in categorical_cols:
            if col in df.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df[col + '_encoded'] = self.label_encoders[col].fit_transform(df[col].fillna('None'))
                else:
                    df[col + '_encoded'] = self.label_encoders[col].transform(df[col].fillna('None'))
                self.feature_columns.append(col + '_encoded')
        
        # Handle missing values
        df = df.fillna(0)
        
        return df
    
    def build_lstm_model(self, input_shape):
        """Build LSTM model for time series prediction"""
        model = keras.Sequential([
            layers.LSTM(64, return_sequences=True, input_shape=input_shape),
            layers.Dropout(0.2),
            layers.LSTM(32, return_sequences=False),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def train(self, X, y):
        """Train the demand prediction model"""
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        if self.model_type == 'xgboost':
            self.model = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
            self.model.fit(X_scaled, y)
            
        elif self.model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
            self.model.fit(X_scaled, y)
            
        elif self.model_type == 'lstm':
            # Reshape for LSTM (samples, timesteps, features)
            X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
            self.model = self.build_lstm_model((1, X_scaled.shape[1]))
            self.model.fit(X_reshaped, y, epochs=50, batch_size=32, validation_split=0.2, verbose=0)
            
        return self
    
    def predict(self, X):
        """Make predictions"""
        X_scaled = self.scaler.transform(X)
        
        if self.model_type == 'lstm':
            X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
            predictions = self.model.predict(X_reshaped, verbose=0).flatten()
        else:
            predictions = self.model.predict(X_scaled)
            
        return predictions
    
    def predict_next_day(self, current_features, meal_types, dishes):
        """Predict demand for next day for all meals and dishes"""
        predictions = {}
        
        for meal in meal_types:
            predictions[meal] = {}
            for dish in dishes:
                # Prepare features for this combination
                features = current_features.copy()
                features['meal_type'] = meal
                features['dish_name'] = dish
                
                # Create feature dataframe
                feature_df = pd.DataFrame([features])
                prepared_features = self.prepare_features(feature_df)
                
                # Select only the columns used in training
                X_pred = prepared_features[self.feature_columns]
                
                # Predict
                pred = self.predict(X_pred)[0]
                predictions[meal][dish] = max(0, int(pred))  # Ensure non-negative
                
        return predictions
    
    def save_model(self, path):
        """Save model and scaler"""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'model_type': self.model_type
        }, path)
    
    def load_model(self, path):
        """Load model and scaler"""
        data = joblib.load(path)
        self.model = data['model']
        self.scaler = data['scaler']
        self.label_encoders = data['label_encoders']
        self.feature_columns = data['feature_columns']
        self.model_type = data['model_type']