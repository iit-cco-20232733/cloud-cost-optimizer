"""
LSTM Model Service
Handles model loading and predictions
"""
import numpy as np
import joblib
import tensorflow as tf
from tensorflow import keras
from utils.config_loader import config

class LSTMModelService:
    def __init__(self):
        self.lstm_config = config.get_lstm_config()
        self.model_features = config.get_model_features()
        self.instance_mapping = config.get_instance_mapping()
        self.timesteps = self.lstm_config['timesteps']
        
        self.model = None
        self.scaler = None
    
    def load_model(self):
        """Load LSTM model and scaler"""
        if self.model is None:
            try:
                model_path = self.lstm_config['model_path']
                print(f"[MODEL] Loading LSTM model from {model_path}")
                self.model = keras.models.load_model(model_path)
                print("[MODEL] LSTM model loaded successfully")
            except Exception as e:
                print(f"[ERROR] Failed to load model: {e}")
                raise
        
        if self.scaler is None:
            try:
                scaler_path = self.lstm_config['scaler_path']
                print(f"[SCALER] Loading scaler from {scaler_path}")
                self.scaler = joblib.load(scaler_path)
                print("[SCALER] Scaler loaded successfully")
            except Exception as e:
                print(f"[ERROR] Failed to load scaler: {e}")
                raise
    
    def predict(self, metrics):
        """Make prediction for given metrics"""
        try:
            if self.model is None or self.scaler is None:
                self.load_model()
            
            # Extract features in correct order
            features = []
            for feature_name in self.model_features:
                features.append(metrics.get(feature_name, 0))
            
            print(f"[DEBUG] Features: {features}")
            
            # Prepare features array
            features_array = np.array(features).reshape(1, -1)
            
            # Scale features
            features_scaled = self.scaler.transform(features_array)
            
            # Reshape for LSTM: (batch_size=1, timesteps, features)
            features_sequence = np.repeat(features_scaled, self.timesteps, axis=0)
            features_reshaped = features_sequence.reshape(1, self.timesteps, len(features))
            
            # Make prediction
            predictions = self.model.predict(features_reshaped, verbose=0)[0]
            predicted_class = np.argmax(predictions)
            confidence = np.max(predictions)
            
            predicted_instance = self.instance_mapping.get(predicted_class, f'unknown_{predicted_class}')
            
            print(f"[LSTM] Predicted: {predicted_instance} (confidence: {confidence:.3f})")
            
            return {
                'instance_type': predicted_instance,
                'confidence': float(confidence),
                'prediction_code': int(predicted_class),
                'probabilities': [float(p) for p in predictions]
            }
            
        except Exception as e:
            print(f"[ERROR] Prediction failed: {e}")
            return None

# Singleton instance
lstm_service = LSTMModelService()
