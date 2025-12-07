"""
Check LSTM model input shape
"""
import joblib
import tensorflow as tf

# Load model and scaler
model = tf.keras.models.load_model('models/cloud_instance_lstm_model.h5')
scaler = joblib.load('models/scaler.pkl')

print("Model Summary:")
print(model.summary())

print("\n" + "="*60)
print("Scaler Information:")
print(f"Number of features expected: {scaler.n_features_in_}")
print(f"Feature names: {getattr(scaler, 'feature_names_in_', 'Not available')}")

print("\n" + "="*60)
print("Model Input Shape:")
print(f"Expected input shape: {model.input_shape}")
