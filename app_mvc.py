"""
Cloud Instance Optimization Flask Application - MVC Architecture
Main application file with clean separation of concerns
"""
from flask import Flask, request, jsonify, send_from_directory
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import services and controllers
from utils.config_loader import config
from services import aws_service, gemini_service, lstm_service
from controllers import analysis_controller

# Initialize Flask app
app_config = config.get_app_config()
app = Flask(__name__, 
           static_folder=app_config['paths']['frontend'], 
           static_url_path='',
           template_folder=app_config['paths']['templates'])

# Global dataset variable
dataset = None
active_month_label = None

def load_dataset():
    """Load dataset from S3"""
    global dataset
    try:
        dataset = aws_service.load_dataset_from_s3()
        if dataset is not None:
            print(f"[DATASET] Loaded {len(dataset)} records")
            if 'Month' in dataset.columns:
                print(f"[DATASET] Months available: {dataset['Month'].unique().tolist()}")
        return dataset
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}")
        return None

# Load model and dataset on startup
print("="*60)
print("Initializing Cloud Instance Optimization System")
print("="*60)

try:
    lstm_service.load_model()
    load_dataset()
    print("[INIT] System initialized successfully")
except Exception as e:
    print(f"[ERROR] Initialization failed: {e}")

print("="*60)

# ===========================================
# STATIC ROUTES
# ===========================================

@app.route('/')
def index():
    """Serve welcome page"""
    return app.send_static_file('index.html')

@app.route('/predict')
def predict_page():
    """Serve prediction page"""
    return app.send_static_file('predict.html')

@app.route('/real-time')
def realtime_page():
    """Serve real-time dashboard"""
    return app.send_static_file('real-time.html')

@app.route('/js/<path:asset_path>')
def static_js(asset_path):
    """Serve JavaScript assets"""
    return send_from_directory(os.path.join(app.static_folder, 'js'), asset_path)

# ===========================================
# API ROUTES - Dataset Operations
# ===========================================

@app.route('/api/get_months')
def get_months():
    """Get available months from dataset"""
    global active_month_label
    if dataset is not None:
        months = dataset['Month'].dropna().unique().tolist()
        if active_month_label and active_month_label not in months:
            months.insert(0, active_month_label)
        elif active_month_label:
            months = [active_month_label] + [m for m in months if m != active_month_label]
        return jsonify({'months': months, 'active_month': active_month_label})
    return jsonify({'months': [], 'active_month': active_month_label})

@app.route('/api/dataset_info')
def get_dataset_info():
    """Get dataset information"""
    if dataset is None:
        return jsonify({'error': 'Dataset not loaded'}), 404
    
    try:
        info = {
            'total_records': len(dataset),
            'columns': list(dataset.columns),
            'months': sorted(dataset['Month'].unique().tolist()) if 'Month' in dataset.columns else [],
            'instance_count': dataset['Instance_ID'].nunique() if 'Instance_ID' in dataset.columns else 0
        }
        return jsonify(info)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ===========================================
# API ROUTES - Predictions
# ===========================================

@app.route('/api/test_single_prediction', methods=['POST'])
def test_single_prediction():
    """Test a single prediction with custom input"""
    try:
        data = request.json
        model_features = config.get_model_features()
        
        # Validate input
        missing_features = [f for f in model_features if f not in data]
        if missing_features:
            return jsonify({
                'error': 'Missing required features',
                'missing_features': missing_features,
                'required_features': model_features
            }), 400
        
        # Extract metrics
        metrics = {feature: data[feature] for feature in model_features}
        
        # Make LSTM prediction
        lstm_prediction = lstm_service.predict(metrics)
        if lstm_prediction is None:
            return jsonify({'error': 'LSTM prediction failed'}), 500
        
        # Get Gemini recommendation
        gemini_recommendation = gemini_service.get_instance_recommendation(metrics)
        
        return jsonify({
            'input_metrics': metrics,
            'prediction': lstm_prediction,
            'predicted_instance_type': lstm_prediction['instance_type'],
            'gemini_recommendation': gemini_recommendation,
            'success': True
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ===========================================
# API ROUTES - Analysis
# ===========================================

@app.route('/api/analyze_month', methods=['POST'])
def analyze_month():
    """Analyze instances for a specific month"""
    global active_month_label
    try:
        data = request.json or {}
        month_name = data.get('month') or active_month_label
        
        if not month_name:
            # Use the most recent month from dataset
            if dataset is not None and 'Month' in dataset.columns:
                months = sorted(dataset['Month'].unique().tolist())
                month_name = months[-1] if months else None
        
        if not month_name:
            return jsonify({'error': 'Month is required and no data available'}), 400
        
        # Get monthly data
        if dataset is None:
            return jsonify({'error': 'Dataset not loaded'}), 404
        
        month_data = dataset[dataset['Month'] == month_name] if 'Month' in dataset.columns else dataset
        
        if len(month_data) == 0:
            return jsonify({'error': f'No data found for {month_name}'}), 404
        
        # Analyze
        result = analysis_controller.analyze_month(month_data)
        
        if result is None:
            return jsonify({'error': 'Analysis failed'}), 500
        
        active_month_label = month_name
        
        # Check if Gemini was available
        gemini_available = any(
            r.get('gemini_prediction', {}).get('confidence') != 'unknown' 
            for r in result['results']
        )
        
        # Calculate total savings
        total_savings = sum(
            r['analysis'].get('potential_monthly_savings', 0) 
            for r in result['results']
            if r['analysis'].get('potential_monthly_savings', 0) > 0
        )
        
        return jsonify({
            'month': month_name,
            'active_month': active_month_label,
            'total_instances': result['total_instances'],
            'status_summary': result['status_counts'],
            'total_potential_savings': round(total_savings, 2),
            'results': result['results'],
            'gemini_available': gemini_available,
            'api_status': 'Gemini API available' if gemini_available else 'Using LSTM predictions'
        })
        
    except Exception as e:
        print(f"[ERROR] Error in analyze_month: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/gemini_metrics', methods=['POST'])
def gemini_metrics():
    """Get Gemini metrics comparison for instances"""
    try:
        payload = request.json or {}
        instance_id = payload.get('instance_id', 'unknown')
        current_type = payload.get('current_type')
        predicted_type = payload.get('predicted_type')
        metrics = payload.get('metrics', {})
        
        if not current_type or not predicted_type:
            return jsonify({
                'error': 'Missing instance types',
                'rows': [],
                'recommendations': [],
                'disclaimer': 'Missing instance type information'
            }), 400
        
        # Get comparison from Gemini
        result = gemini_service.get_instance_comparison(
            instance_id, current_type, predicted_type, metrics
        )
        
        return jsonify(result)
        
    except Exception as e:
        print(f"[ERROR] Gemini metrics failed: {str(e)}")
        return jsonify({
            'error': str(e),
            'rows': [],
            'recommendations': [],
            'disclaimer': 'Gemini service unavailable'
        }), 500

# ===========================================
# API ROUTES - Model Info
# ===========================================

@app.route('/api/model_info')
def get_model_info():
    """Get model configuration information"""
    return jsonify({
        'features': config.get_model_features(),
        'feature_count': len(config.get_model_features()),
        'instance_types': list(config.get_instance_mapping().values()),
        'lstm_config': config.get_lstm_config()
    })

@app.route('/api/instance_types')
def get_instance_types():
    """Get instance type mapping"""
    return jsonify({
        'mapping': config.get_instance_mapping(),
        'note': 'Detailed instance specifications are provided by Gemini AI'
    })

# ===========================================
# MAIN
# ===========================================

if __name__ == '__main__':
    print("="*60)
    print("Cloud Instance Optimization API - MVC Architecture")
    print("="*60)
    print("Available Endpoints:")
    print("  Frontend:")
    print("    GET  /              - Welcome page")
    print("    GET  /predict       - Prediction lab")
    print("    GET  /real-time     - Real-time dashboard")
    print("  API:")
    print("    POST /api/analyze_month          - Analyze month data")
    print("    POST /api/test_single_prediction - Test single prediction")
    print("    POST /api/gemini_metrics         - Get Gemini comparison")
    print("    GET  /api/get_months             - Get available months")
    print("    GET  /api/dataset_info           - Get dataset info")
    print("    GET  /api/model_info             - Get model info")
    print("    GET  /api/instance_types         - Get instance types")
    print("="*60)
    
    app.run(
        debug=app_config['app']['debug'],
        host=app_config['app']['host'],
        port=app_config['app']['port']
    )
