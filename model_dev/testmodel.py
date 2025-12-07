import numpy as np
import pandas as pd
from tensorflow import keras
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# Load your trained model and scaler
lstm_model = keras.models.load_model('models/lstm_model.h5')
scaler = joblib.load('models/scaler.pkl')

# Instance type mapping (same as in your Flask app)
INSTANCE_TYPE_MAPPING = {
    0: 'c5.2xlarge',
    1: 'c5.large', 
    2: 'c5.xlarge',
    3: 'm5.2xlarge',
    4: 'm5.large',
    5: 'm5.xlarge',
    6: 'r5.large',
    7: 'r5.xlarge',
    8: 't2.medium',
    9: 't2.micro',
    10: 't2.nano',
    11: 't2.small',
    12: 't3.micro',
    13: 't3.nano',
    14: 'x1.large'
}

# CORRECTED: Features that were actually used during training (7 features)
FEATURE_NAMES = [
    'CPU_Utilization_Percent', 'Memory_Utilization_Percent', 
    'Network_In_Mbps', 'Network_Out_Mbps', 'Disk_Usage_Percent', 
    'Response_Time_ms', 'Hourly_Cost_USD'
]

print("🎯 Using the CORRECT 7 features that model was trained with:")
for i, feature in enumerate(FEATURE_NAMES):
    print(f"  [{i}]: {feature}")

def create_test_samples():
    """Create diverse test samples with the CORRECT 7 features"""
    
    test_samples = [
        # Format: [CPU, Memory, Net_In, Net_Out, Disk_Usage, Response_Time, Hourly_Cost]
        {
            'name': 'Compute Heavy',
            'features': [85, 35, 8, 8, 45, 70, 0.17],
            'expected_category': 'Compute Optimized'
        },
        {
            'name': 'Memory Heavy', 
            'features': [25, 90, 6, 6, 55, 85, 0.25],
            'expected_category': 'Memory Optimized'
        },
        {
            'name': 'Light Workload',
            'features': [15, 25, 1, 1, 20, 180, 0.012],
            'expected_category': 'Burstable'
        },
        {
            'name': 'Balanced Workload',
            'features': [50, 65, 12, 12, 50, 95, 0.192],
            'expected_category': 'General Purpose'
        },
        {
            'name': 'Heavy Workload',
            'features': [75, 80, 20, 20, 70, 60, 0.38],
            'expected_category': 'High Performance'
        },
        {
            'name': 'Micro Workload',
            'features': [10, 30, 0.5, 0.5, 15, 200, 0.006],
            'expected_category': 'Minimal'
        },
        {
            'name': 'Network Heavy',
            'features': [40, 50, 45, 50, 35, 90, 0.22],
            'expected_category': 'Network Intensive'
        },
        {
            'name': 'High Response Time',
            'features': [30, 45, 8, 8, 75, 250, 0.18],
            'expected_category': 'Storage/Latency Sensitive'
        },
        {
            'name': 'Cost Optimized',
            'features': [20, 40, 3, 3, 25, 150, 0.008],
            'expected_category': 'Budget Friendly'
        },
        {
            'name': 'High Cost Premium',
            'features': [90, 95, 30, 30, 80, 40, 0.45],
            'expected_category': 'Premium Performance'
        }
    ]
    
    return test_samples

def predict_with_analysis(model, sample_features, scaler, timesteps=5):
    """Make prediction with detailed analysis"""
    
    print(f"Input features: {dict(zip(FEATURE_NAMES, sample_features))}")
    
    # Prepare features (already 7 features, correct count)
    features_array = np.array(sample_features).reshape(1, -1)
    print(f"Features array shape: {features_array.shape}")
    
    # Apply the trained scaler
    try:
        features_scaled = scaler.transform(features_array)
        print(f"Scaled features: {features_scaled[0]}")
    except Exception as e:
        print(f"Scaling error: {e}")
        return None, None, None
    
    # Reshape for LSTM: (batch_size=1, timesteps=5, features=7)
    features_sequence = np.repeat(features_scaled, timesteps, axis=0)
    features_reshaped = features_sequence.reshape(1, timesteps, len(sample_features))
    
    print(f"Reshaped for LSTM: {features_reshaped.shape}")
    
    # Make prediction
    try:
        predictions = model.predict(features_reshaped, verbose=0)[0]
        predicted_class = np.argmax(predictions)
        confidence = np.max(predictions)
        
        print(f"\n📊 Top 5 Predictions:")
        sorted_predictions = sorted(enumerate(predictions), key=lambda x: x[1], reverse=True)
        
        for i, (class_idx, prob) in enumerate(sorted_predictions[:5]):
            instance_name = INSTANCE_TYPE_MAPPING.get(class_idx, f'unknown_{class_idx}')
            marker = '🎯' if class_idx == predicted_class else '  '
            print(f"  {marker} {class_idx:2d}: {instance_name:12s} -> {prob:.4f} ({prob*100:.2f}%)")
        
        predicted_instance = INSTANCE_TYPE_MAPPING.get(predicted_class, f'unknown_{predicted_class}')
        
        return predicted_instance, confidence, predictions
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return None, None, None

def analyze_predictions(results):
    """Analyze the diversity and patterns in predictions"""
    if not results:
        print("No results to analyze")
        return
    
    print(f"\n📈 PREDICTION ANALYSIS")
    print("="*50)
    
    # Count predictions by instance type
    prediction_counts = {}
    for result in results:
        pred = result['predicted_instance']
        prediction_counts[pred] = prediction_counts.get(pred, 0) + 1
    
    print("Instance Type Distribution:")
    for instance, count in sorted(prediction_counts.items()):
        percentage = (count / len(results)) * 100
        print(f"  {instance:12s}: {count:2d} predictions ({percentage:5.1f}%)")
    
    # Confidence analysis
    confidences = [r['confidence'] for r in results]
    print(f"\nConfidence Statistics:")
    print(f"  Average: {np.mean(confidences):.3f}")
    print(f"  Min:     {np.min(confidences):.3f}")
    print(f"  Max:     {np.max(confidences):.3f}")
    print(f"  Std:     {np.std(confidences):.3f}")
    
    # Diversity score
    unique_predictions = len(set([r['predicted_instance'] for r in results]))
    diversity_score = unique_predictions / len(results)
    
    print(f"\nDiversity Analysis:")
    print(f"  Total samples:        {len(results)}")
    print(f"  Unique predictions:   {unique_predictions}")
    print(f"  Diversity score:      {diversity_score:.2f}")
    
    if diversity_score < 0.3:
        print("  ⚠️  Low diversity - model may be biased")
    elif diversity_score > 0.7:
        print("  ✅ Good diversity - model discriminates well")
    else:
        print("  ℹ️  Moderate diversity - acceptable performance")

def main():
    """Main testing function"""
    print("🚀 LSTM MODEL TESTING - CORRECTED VERSION")
    print("="*60)
    
    print(f"✅ Model loaded: {type(lstm_model).__name__}")
    print(f"✅ Model input shape: {lstm_model.input_shape}")
    print(f"✅ Model output shape: {lstm_model.output_shape}")
    print(f"✅ Scaler loaded: {type(scaler).__name__}")
    print(f"✅ Scaler expects {scaler.n_features_in_} features")
    
    # Create and test samples
    test_samples = create_test_samples()
    results = []
    
    for i, sample in enumerate(test_samples):
        print(f"\n{'='*60}")
        print(f"🧪 TEST SAMPLE {i+1}: {sample['name']}")
        print(f"Expected Category: {sample['expected_category']}")
        print('='*60)
        
        predicted_instance, confidence, all_probs = predict_with_analysis(
            lstm_model, sample['features'], scaler
        )
        
        if predicted_instance is not None:
            print(f"\n🎯 FINAL PREDICTION: {predicted_instance}")
            print(f"🔥 CONFIDENCE: {confidence:.4f} ({confidence*100:.2f}%)")
            
            # Determine if prediction makes sense
            confidence_level = "HIGH" if confidence > 0.4 else "MEDIUM" if confidence > 0.25 else "LOW"
            print(f"📊 CONFIDENCE LEVEL: {confidence_level}")
            
            results.append({
                'sample_name': sample['name'],
                'predicted_instance': predicted_instance,
                'confidence': confidence,
                'expected_category': sample['expected_category']
            })
        else:
            print("❌ PREDICTION FAILED")
    
    if results:
        # Summary table
        print(f"\n{'='*60}")
        print("📋 SUMMARY OF ALL PREDICTIONS")
        print('='*60)
        
        for result in results:
            conf_emoji = "🔥" if result['confidence'] > 0.4 else "⚡" if result['confidence'] > 0.25 else "💭"
            print(f"{result['sample_name']:20s} -> {result['predicted_instance']:12s} "
                  f"{conf_emoji} {result['confidence']:.3f}")
        
        # Analysis
        analyze_predictions(results)
        
        print(f"\n✅ Testing completed successfully!")
        print(f"📊 Processed {len(results)} samples with working 7-feature configuration")
    else:
        print("\n❌ All predictions failed - check model compatibility")

if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Make sure these files exist:")
        print("  - models/lstm_model.h5")
        print("  - models/scaler.pkl")
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()