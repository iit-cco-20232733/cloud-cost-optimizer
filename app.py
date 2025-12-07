"""
Cloud Instance Optimization Flask Application - CORRECTED VERSION
Analyzes monthly cloud instance data and provides optimization recommendations
"""
from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import joblib
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
import boto3
from io import StringIO

try:
    from google import genai
    from google.genai import types as genai_types
    GENAI_AVAILABLE = True
except ImportError:
    genai = None
    genai_types = None
    GENAI_AVAILABLE = False
warnings.filterwarnings('ignore')

app = Flask(__name__, static_folder='frontend', static_url_path='', template_folder='templates')

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', 'AIzaSyDOyu_xMOxjIFYnpTYS0owZrsFA3CDqqaU')
GEMINI_MODEL_NAME = 'gemini-flash-lite-latest'  # Stable model with better rate limits
_gemini_client = None

# AWS EC2 Pricing Reference (us-east-1 on-demand, approximate as of 2024)
AWS_PRICING_REFERENCE = {
    't2.nano': 0.0058,
    't2.micro': 0.0116,
    't2.small': 0.023,
    't2.medium': 0.0464,
    't3.nano': 0.0052,
    't3.micro': 0.0104,
    't3.small': 0.0208,
    't3.medium': 0.0416,
    'm5.large': 0.096,
    'm5.xlarge': 0.192,
    'm5.2xlarge': 0.384,
    'c5.large': 0.085,
    'c5.xlarge': 0.17,
    'c5.2xlarge': 0.34,
    'r5.large': 0.126,
    'r5.xlarge': 0.252,
    'r5.2xlarge': 0.504,
    'x1.large': 1.001
}


def get_instance_price(instance_type):
    """Get instance price from reference table"""
    return AWS_PRICING_REFERENCE.get(instance_type, 0)


def validate_and_fix_cost(instance_type, gemini_cost):
    """Validate Gemini cost and fix if incorrect"""
    reference_cost = get_instance_price(instance_type)
    
    # If Gemini cost is 0 or wildly off (>50% difference), use reference
    if gemini_cost == 0:
        return reference_cost
    
    if reference_cost > 0:
        diff_percent = abs(gemini_cost - reference_cost) / reference_cost * 100
        if diff_percent > 50:  # More than 50% off
            print(f"[WARNING] Gemini cost ${gemini_cost:.4f} for {instance_type} differs {diff_percent:.1f}% from reference ${reference_cost:.4f}, using reference")
            return reference_cost
    
    return gemini_cost



def get_gemini_client():
    """Initialise a singleton Gemini client."""
    global _gemini_client
    if not GENAI_AVAILABLE:
        raise RuntimeError('google-genai package not installed. Run `pip install google-genai`.')
    if not GEMINI_API_KEY or GEMINI_API_KEY.startswith('REPLACE_WITH_'):
        raise RuntimeError('Gemini API key not configured. Update GEMINI_API_KEY in app.py or set environment variable.')
    if _gemini_client is None:
        _gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    return _gemini_client


def build_spec_fallback(current_type, predicted_type, metrics_map=None):
    """Provide a minimal fallback response when Gemini is unavailable."""
    metrics_map = metrics_map or {}
    cpu_util = metrics_map.get('CPU_Utilization_Percent', {})
    memory_util = metrics_map.get('Memory_Utilization_Percent', {})
    cost_metric = metrics_map.get('Hourly_Cost_USD', {})

    def format_percent(value):
        if value is None or value == '—':
            return '—'
        try:
            return f"{float(value):.1f}%"
        except (TypeError, ValueError):
            return f"{value}%"

    return {
        'rows': [
            {
                'metric': 'Current Instance',
                'current': current_type,
                'recommended': predicted_type,
                'unit': 'Type'
            },
            {
                'metric': 'CPU Utilization',
                'current': format_percent(cpu_util.get('current')),
                'recommended': '—',
                'unit': '%'
            },
            {
                'metric': 'Memory Utilization',
                'current': format_percent(memory_util.get('current')),
                'recommended': '—',
                'unit': '%'
            },
            {
                'metric': 'Hourly Cost',
                'current': f"${cost_metric.get('current', 0):.4f}" if cost_metric.get('current') else '—',
                'recommended': '—',
                'unit': 'USD/hr'
            },
        ],
        'caption': 'Basic comparison (Gemini unavailable)',
        'disclaimer': 'Gemini AI is currently unavailable. Showing basic metrics only. Detailed specs will be available when Gemini is online.',
        'recommendations': [
            'Gemini AI provides detailed instance specifications and recommendations.',
            'Please check your API configuration or try again later.',
            'Contact support if the issue persists.'
        ],
        'generated': False
    }


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


# Note: Instance configurations, metrics, and pricing are now retrieved from Gemini AI
# Only INSTANCE_TYPE_MAPPING is kept for model class-to-name decoding

# Cache for instance pricing and recommendations to avoid repeated Gemini API calls
_pricing_cache = {}
_recommendation_cache = {}

def get_batch_gemini_recommendations_and_costs(instances_data):
    """Get all Gemini recommendations and costs in a single API call
    
    Args:
        instances_data: List of dicts with 'instance_id', 'current_type', and 'metrics'
    
    Returns:
        dict: {instance_id: {'recommended_type': str, 'current_cost': float, 'recommended_cost': float, 'reasoning': str}}
    """
    if not GENAI_AVAILABLE or not instances_data:
        return {}
    
    try:
        client = get_gemini_client()
        
        # Build batch request payload
        instances_list = []
        for data in instances_data:
            instances_list.append({
                'instance_id': data['instance_id'],
                'current_type': data['current_type'],
                'metrics': data['metrics']
            })
        
        prompt = f"""
You are an AWS EC2 optimization expert with access to current AWS pricing data. For each instance below:
1. Recommend the optimal AWS EC2 instance type based on workload metrics
2. Provide ACCURATE on-demand hourly costs for BOTH the current and recommended instance types
3. Explain your recommendation

CRITICAL PRICING REQUIREMENTS:
- Costs must be actual AWS us-east-1 on-demand pricing (as of 2024)
- Each instance type has a DIFFERENT cost - DO NOT return the same cost for different types
- Example pricing reference:
  * t2.nano: $0.0058/hr, t2.micro: $0.0116/hr, t2.small: $0.023/hr, t2.medium: $0.0464/hr
  * t3.nano: $0.0052/hr, t3.micro: $0.0104/hr, t3.small: $0.0208/hr, t3.medium: $0.0416/hr
  * m5.large: $0.096/hr, m5.xlarge: $0.192/hr, m5.2xlarge: $0.384/hr
  * c5.large: $0.085/hr, c5.xlarge: $0.17/hr, c5.2xlarge: $0.34/hr
  * r5.large: $0.126/hr, r5.xlarge: $0.252/hr, r5.2xlarge: $0.504/hr
  * x1.large: $1.001/hr

Return ONLY valid JSON in this exact format:
{{
  "recommendations": [
    {{
      "instance_id": "string",
      "current_type": "string",
      "recommended_type": "string",
      "current_cost": 0.0,
      "recommended_cost": 0.0,
      "reasoning": "string",
      "confidence": "high|medium|low"
    }}
  ]
}}

RECOMMENDATION GUIDELINES:
- Analyze CPU%, Memory%, Network, Disk usage patterns
- If CPU < 30% AND Memory < 30%: Consider downsizing to save costs
- If CPU > 80% OR Memory > 80%: Recommend upgrade for performance
- If metrics are balanced: Keep current or suggest cost-effective alternative
- Choose from: t2.nano, t2.micro, t2.small, t2.medium, t3.nano, t3.micro, t3.small, t3.medium, m5.large, m5.xlarge, m5.2xlarge, c5.large, c5.xlarge, c5.2xlarge, r5.large, r5.xlarge, r5.2xlarge, x1.large

Instances and their workload metrics:
{json.dumps(instances_list, indent=2)}
"""
        
        contents = [
            genai_types.Content(
                role='user',
                parts=[genai_types.Part.from_text(text=prompt)]
            )
        ]
        
        response = client.models.generate_content(
            model=GEMINI_MODEL_NAME,
            contents=contents,
            config=genai_types.GenerateContentConfig(response_mime_type='application/json')
        )
        
        # Parse response
        raw_text = ''
        if hasattr(response, 'text') and response.text:
            raw_text = response.text
        elif getattr(response, 'candidates', None):
            for candidate in response.candidates:
                candidate_parts = getattr(getattr(candidate, 'content', None), 'parts', [])
                for part in candidate_parts:
                    raw_text += getattr(part, 'text', '')
        
        if raw_text:
            batch_result = json.loads(raw_text)
            
            # Handle both dict and list responses
            if isinstance(batch_result, list):
                batch_result = batch_result[0] if batch_result else {}
            
            if not isinstance(batch_result, dict):
                print(f"[WARNING] Unexpected Gemini batch response format: {type(batch_result)}")
                return {}
            
            # Parse recommendations into a lookup dict
            recommendations = {}
            for rec in batch_result.get('recommendations', []):
                instance_id = rec.get('instance_id')
                current_type = rec.get('current_type', 'unknown')
                recommended_type = rec.get('recommended_type', 'unknown')
                
                if instance_id:
                    # Validate and fix costs using reference pricing
                    current_cost_raw = float(rec.get('current_cost', 0))
                    recommended_cost_raw = float(rec.get('recommended_cost', 0))
                    
                    current_cost = validate_and_fix_cost(current_type, current_cost_raw)
                    recommended_cost = validate_and_fix_cost(recommended_type, recommended_cost_raw)
                    
                    recommendations[instance_id] = {
                        'recommended_type': recommended_type,
                        'current_cost': current_cost,
                        'recommended_cost': recommended_cost,
                        'reasoning': rec.get('reasoning', 'No reasoning provided'),
                        'confidence': rec.get('confidence', 'medium')
                    }
                    print(f"[GEMINI] {instance_id}: {current_type} → {recommended_type} (${current_cost:.4f} → ${recommended_cost:.4f})")
            
            return recommendations
        
        return {}
        
    except json.JSONDecodeError as e:
        print(f"[WARNING] Failed to parse Gemini batch JSON response: {e}")
        return {}
    except Exception as e:
        error_msg = str(e)
        if 'API key expired' in error_msg or 'INVALID_ARGUMENT' in error_msg:
            print(f"[ERROR] Gemini API key issue: {error_msg}")
            print(f"[INFO] Please update GEMINI_API_KEY in app.py or set as environment variable")
        elif '429' in error_msg or 'RESOURCE_EXHAUSTED' in error_msg:
            print(f"[WARNING] Gemini rate limit hit: {error_msg}")
            print(f"[INFO] Using fallback to LSTM predictions and default costs")
        else:
            print(f"[WARNING] Failed to get Gemini batch recommendations: {e}")
        return {}

def get_gemini_instance_recommendation(metrics):
    """Get Gemini's independent instance type recommendation based on workload metrics
    
    Args:
        metrics: Dictionary of instance metrics (CPU, Memory, Network, etc.)
    
    Returns:
        dict: {'instance_type': str, 'reasoning': str, 'confidence': str} or None
    """
    if not GENAI_AVAILABLE:
        return None
    
    # Create cache key from metrics
    cache_key = f"{metrics.get('CPU_Utilization_Percent', 0):.1f}_{metrics.get('Memory_Utilization_Percent', 0):.1f}"
    if cache_key in _recommendation_cache:
        print(f"[CACHE] Using cached Gemini recommendation for {cache_key}")
        return _recommendation_cache[cache_key]
    
    try:
        client = get_gemini_client()
        
        # Format metrics for Gemini
        metrics_text = "\n".join([f"- {k}: {v}" for k, v in metrics.items()])
        
        prompt = f"""
You are an AWS EC2 instance sizing expert. Based on the following workload metrics, recommend the most appropriate AWS EC2 instance type.

Workload Metrics:
{metrics_text}

Return ONLY valid JSON with this exact schema:
{{
    "instance_type": "string",
    "reasoning": "string",
    "confidence": "high|medium|low"
}}

IMPORTANT:
1. Analyze the CPU and Memory utilization patterns
2. Consider network throughput requirements
3. Account for cost efficiency
4. Choose from common AWS instance types: t2.nano, t2.micro, t2.small, t2.medium, t3.nano, t3.micro, m5.large, m5.xlarge, m5.2xlarge, c5.large, c5.xlarge, c5.2xlarge, r5.large, r5.xlarge, x1.large
5. Provide clear reasoning for your recommendation
6. Be conservative - prefer slightly larger instances for stability

Provide your expert recommendation based solely on these metrics.
"""
        
        contents = [
            genai_types.Content(
                role='user',
                parts=[genai_types.Part.from_text(text=prompt)]
            )
        ]
        
        response = client.models.generate_content(
            model=GEMINI_MODEL_NAME,
            contents=contents,
            config=genai_types.GenerateContentConfig(response_mime_type='application/json')
        )
        
        # Parse response
        raw_text = ''
        if hasattr(response, 'text') and response.text:
            raw_text = response.text
        elif getattr(response, 'candidates', None):
            for candidate in response.candidates:
                candidate_parts = getattr(getattr(candidate, 'content', None), 'parts', [])
                for part in candidate_parts:
                    raw_text += getattr(part, 'text', '')
        
        if raw_text:
            recommendation = json.loads(raw_text)
            
            # Handle both dict and list responses
            if isinstance(recommendation, list):
                recommendation = recommendation[0] if recommendation else {}
            
            if not isinstance(recommendation, dict):
                print(f"[WARNING] Unexpected Gemini response format: {type(recommendation)}")
                return None
            
            result = {
                'instance_type': recommendation.get('instance_type', 'unknown'),
                'reasoning': recommendation.get('reasoning', 'No reasoning provided'),
                'confidence': recommendation.get('confidence', 'medium')
            }
            
            # Cache the result
            _recommendation_cache[cache_key] = result
            return result
        
        return None
        
    except json.JSONDecodeError as e:
        print(f"[WARNING] Failed to parse Gemini JSON response: {e}")
        return None
    except Exception as e:
        error_msg = str(e)
        if 'API key expired' in error_msg or 'INVALID_ARGUMENT' in error_msg:
            print(f"[ERROR] Gemini API key issue: {error_msg}")
            print(f"[INFO] Please update GEMINI_API_KEY in app.py or set as environment variable")
        else:
            print(f"[WARNING] Failed to get Gemini recommendation: {e}")
        return None

def get_instance_costs_batch_from_gemini(instance_types):
    """DEPRECATED: Costs are now included in batch recommendations.
    This function is no longer used. Use get_batch_gemini_recommendations_and_costs instead.
    """
    print(f"[WARNING] get_instance_costs_batch_from_gemini is deprecated and should not be called")
    return {}

def get_instance_costs_from_gemini(current_type, predicted_type):
    """DEPRECATED: Use get_batch_gemini_recommendations_and_costs instead.
    Get instance costs from Gemini AI for both current and predicted instance types
    
    Returns:
        tuple: (cost_current, cost_predicted) or (None, None) if Gemini unavailable
    """
    print(f"[INFO] get_instance_costs_from_gemini called but deprecated - use batch recommendations instead")
    # Check cache first
    cache_key = f"{current_type}|{predicted_type}"
    if cache_key in _pricing_cache:
        return _pricing_cache[cache_key]
    
    if not GENAI_AVAILABLE:
        return None, None
    
    try:
        client = get_gemini_client()
        prompt = f"""
You are an AWS EC2 pricing expert. Provide the current on-demand hourly pricing for the following AWS EC2 instance types.

Return ONLY valid JSON with this exact schema:
{{
    "current_instance": "{current_type}",
    "current_cost_usd_per_hour": 0.0,
    "predicted_instance": "{predicted_type}",
    "predicted_cost_usd_per_hour": 0.0
}}

IMPORTANT: Look up the actual AWS EC2 on-demand pricing from your knowledge base for:
1. {current_type}
2. {predicted_type}

Provide accurate hourly costs in USD. Use the latest AWS pricing available in your knowledge base.
"""
        
        contents = [
            genai_types.Content(
                role='user',
                parts=[genai_types.Part.from_text(text=prompt)]
            )
        ]
        
        response = client.models.generate_content(
            model=GEMINI_MODEL_NAME,
            contents=contents,
            config=genai_types.GenerateContentConfig(response_mime_type='application/json')
        )
        
        # Parse response
        raw_text = ''
        if hasattr(response, 'text') and response.text:
            raw_text = response.text
        elif getattr(response, 'candidates', None):
            for candidate in response.candidates:
                candidate_parts = getattr(getattr(candidate, 'content', None), 'parts', [])
                for part in candidate_parts:
                    raw_text += getattr(part, 'text', '')
        
        if raw_text:
            pricing_data = json.loads(raw_text)
            
            # Handle both dict and list responses
            if isinstance(pricing_data, list):
                # If it's a list, try to find the pricing object
                pricing_data = pricing_data[0] if pricing_data else {}
            
            if not isinstance(pricing_data, dict):
                print(f"[WARNING] Unexpected pricing data format: {type(pricing_data)}")
                return None, None
            
            cost_current = pricing_data.get('current_cost_usd_per_hour')
            cost_predicted = pricing_data.get('predicted_cost_usd_per_hour')
            
            # Validate costs are numeric
            try:
                if cost_current is not None:
                    cost_current = float(cost_current)
                if cost_predicted is not None:
                    cost_predicted = float(cost_predicted)
            except (TypeError, ValueError) as e:
                print(f"[WARNING] Invalid cost values: {e}")
                return None, None
            
            # Cache the result
            result = (cost_current, cost_predicted)
            _pricing_cache[cache_key] = result
            return result
        
        return None, None
        
    except json.JSONDecodeError as e:
        print(f"[WARNING] Failed to parse Gemini JSON response: {e}")
        print(f"[DEBUG] Raw response: {raw_text[:200]}...")
        return None, None
    except Exception as e:
        print(f"[WARNING] Failed to get costs from Gemini: {e}")
        import traceback
        traceback.print_exc()
        return None, None

# Global variables
model = None
scaler = None
label_encoder = None
feature_names = None
dataset = None
active_month_label = None

# Model expects 6 features (Hourly_Cost_USD removed from training)
MODEL_FEATURE_NAMES = [
    'Network_In_Mbps',
    'Network_Out_Mbps',
    'Response_Time_ms',
    'CPU_Utilization_Percent',
    'Memory_Utilization_Percent',
    'Disk_Usage_Percent'
]

def load_models():
    """Load pre-trained models from the models directory"""
    global model, scaler, label_encoder, feature_names
    
    try:
        models_dir = 'models'
        
        # Load LSTM model
        if os.path.exists(f'{models_dir}/lstm_model.h5'):
            print("Loading LSTM model...")
            model = keras.models.load_model(f'{models_dir}/lstm_model.h5')
            print("✅ LSTM model loaded successfully!")
        elif os.path.exists(f'{models_dir}/lstm_model.h5'):
            print("Loading LSTM model...")
            model = keras.models.load_model(f'{models_dir}/lstm_model.h5')
            print("✅ LSTM model loaded successfully!")
        else:
            print("❌ LSTM model file not found!")
            return False
        
        # Load scaler (CRITICAL - use the trained scaler)
        if os.path.exists(f'{models_dir}/scaler.pkl'):
            print("Loading scaler...")
            scaler = joblib.load(f'{models_dir}/scaler.pkl')
            print(f"✅ Scaler loaded successfully! Expected features: {scaler.n_features_in_}")
            print(f"✅ Scaler feature names: {list(scaler.feature_names_in_)}")
        else:
            print("❌ Scaler file not found!")
            return False
        
        # Load label encoder (optional)
        if os.path.exists(f'{models_dir}/label_encoder.pkl'):
            print("Loading label encoder...")
            label_encoder = joblib.load(f'{models_dir}/label_encoder.pkl')
            print("✅ Label encoder loaded successfully!")
        
        # Load feature names (optional)
        if os.path.exists(f'{models_dir}/feature_names.pkl'):
            print("Loading feature names...")
            feature_names = joblib.load(f'{models_dir}/feature_names.pkl')
            print(f"Loaded feature names: {feature_names}")
        else:
            feature_names = MODEL_FEATURE_NAMES.copy()
            print(f"Using default feature names: {feature_names}")
        
        # Verify model input shape matches our expectations (6 features)
        print(f"\nModel verification:")
        print(f"  Input shape: {model.input_shape}")
        print(f"  Output shape: {model.output_shape}")
        print(f"  Expected features: {len(MODEL_FEATURE_NAMES)} (6 features)")
        print(f"  Scaler features: {scaler.n_features_in_}")
        
        if scaler.n_features_in_ != len(MODEL_FEATURE_NAMES):
            print(f"⚠️  WARNING: Feature count mismatch!")
            print(f"   Model expects: {scaler.n_features_in_}")
            print(f"   Code provides: {len(MODEL_FEATURE_NAMES)}")
            print(f"   Please retrain the model with the correct 6 features")
            return False
        
        return True
        
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        return False

def load_dataset():
    """Load the monthly dataset from S3 using boto3"""
    global dataset, active_month_label
    try:
        # S3 bucket and file configuration
        s3_bucket = 'cloud-opt-poc'
        s3_key = 'monthly_cloud.csv'
        s3_region = 'eu-north-1'
        
        try:
            print(f"Loading dataset from S3: s3://{s3_bucket}/{s3_key}")
            
            # Load AWS credentials from CSV file
            credentials_file = 'demo-user_accessKeys.csv'
            if os.path.exists(credentials_file):
                creds_df = pd.read_csv(credentials_file)
                aws_access_key = creds_df['Access key ID'].iloc[0]
                aws_secret_key = creds_df['Secret access key'].iloc[0]
                
                # Initialize S3 client with credentials from CSV
                s3_client = boto3.client(
                    's3',
                    region_name=s3_region,
                    aws_access_key_id=aws_access_key,
                    aws_secret_access_key=aws_secret_key
                )
            else:
                # Fall back to default credentials (environment or ~/.aws/credentials)
                print("⚠️ Credentials file not found, using default AWS credentials")
                s3_client = boto3.client('s3', region_name=s3_region)
            
            # Get the file from S3
            response = s3_client.get_object(Bucket=s3_bucket, Key=s3_key)
            csv_content = response['Body'].read().decode('utf-8')
            
            # Load into pandas DataFrame
            dataset = pd.read_csv(StringIO(csv_content))
            print(f"✅ Dataset loaded from S3: {len(dataset)} records")
            
        except Exception as s3_error:
            print(f"❌ Failed to load from S3: {s3_error}")
            print(f"❌ Dataset could not be loaded. S3 is the only data source.")
            return False
        
        # Process timestamps and filter to previous month
        timestamp_col = None
        if 'Timestamp' in dataset.columns:
            dataset['Timestamp'] = pd.to_datetime(dataset['Timestamp'], errors='coerce')
            timestamp_col = 'Timestamp'
        elif 'Date' in dataset.columns:
            dataset['Date'] = pd.to_datetime(dataset['Date'], errors='coerce')
            dataset['Timestamp'] = dataset['Date']
            timestamp_col = 'Timestamp'

        now = pd.Timestamp.now().tz_localize(None)
        current_month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        previous_month_start = current_month_start - pd.DateOffset(months=1)
        
        # Get the previous month name (e.g., "October" if current is November)
        previous_month_name = previous_month_start.strftime('%B')
        active_month_label = previous_month_start.strftime('%B %Y')

        # First priority: check if there's a Month column
        if 'Month' in dataset.columns:
            month_strings = dataset['Month'].astype(str).str.strip()
            # Match just the month name (case-insensitive), ignoring year
            filtered = dataset.loc[month_strings.str.lower() == previous_month_name.lower()].copy()
            
            if filtered.empty:
                print(
                    f"⚠️ No records found in Month column for '{previous_month_name}'. Dataset will load with 0 rows."
                )
                dataset = filtered
            else:
                dataset = filtered.reset_index(drop=True)
                print(f"✅ Filtered dataset to {previous_month_name}: {len(dataset)} records")
        
        # Second priority: try Timestamp column if Month column not available or didn't work
        elif timestamp_col:
            valid_mask = dataset['Timestamp'].notna()
            if valid_mask.any():
                # Extract month name from Timestamp and match
                dataset['_temp_month'] = dataset['Timestamp'].dt.strftime('%B')
                filtered = dataset.loc[dataset['_temp_month'] == previous_month_name].copy()
                dataset.drop('_temp_month', axis=1, inplace=True)
                
                if filtered.empty:
                    print(
                        f"⚠️ No records found for month '{previous_month_name}' in dataset."
                    )
                    dataset = filtered
                else:
                    dataset = filtered.reset_index(drop=True)
                    print(
                        f"✅ Filtered dataset to {previous_month_name}: {len(dataset)} records"
                    )
            else:
                print("⚠️ Timestamp column exists but all values are null.")
        else:
            print("⚠️ Dataset missing timestamp/month information; unable to filter to previous month.")

        if 'Month' not in dataset.columns:
            dataset['Month'] = active_month_label
            print("✅ Month column created")
        else:
            # Overwrite to ensure consistent label for downstream filtering
            dataset['Month'] = active_month_label

        return True
        
    except Exception as e:
        print(f"Error loading dataset: {str(e)}") 
        return False

def get_monthly_data(month_name):
    """Get data for a specific month"""
    if dataset is None:
        return None
    
    try:
        month_data = dataset[dataset['Month'] == month_name]
        return month_data
    except Exception as e:
        print(f"Error getting monthly data: {str(e)}")
        return None

def calculate_instance_averages(month_data):
    """Calculate average metrics for each instance using the correct 7 features"""
    if month_data is None or len(month_data) == 0:
        return {}
    
    try:
        print(f"[DEBUG] Dataset columns: {list(month_data.columns)}")
        
        # Check which of our required features are available
        available_columns = [col for col in MODEL_FEATURE_NAMES if col in month_data.columns]
        missing_columns = [col for col in MODEL_FEATURE_NAMES if col not in month_data.columns]
        
        print(f"[DEBUG] Available features: {available_columns}")
        print(f"[DEBUG] Missing features: {missing_columns}")
        
        # Calculate averages for available columns
        instance_averages = month_data.groupby(['Instance_ID', 'Instance_Type'])[available_columns].mean().reset_index()
        
        # Add missing features with default values
        for missing_col in missing_columns:
            # Set to 0 for missing features; actual costs will come from Gemini
            instance_averages[missing_col] = 0
        
        print(f"[DEBUG] Final features in averages: {[col for col in instance_averages.columns if col in MODEL_FEATURE_NAMES]}")
        
        return instance_averages
    except Exception as e:
        print(f"Error calculating averages: {str(e)}")
        return pd.DataFrame()

def predict_optimal_instance(metrics):
    """Predict optimal instance type using the trained LSTM model and scaler - EXACT COPY FROM WORKING TEST"""
    if model is None or scaler is None:
        print("Model or scaler not loaded")
        return None
    
    try:
        # Extract features in the exact order expected by the scaler (6 features)
        features = []
        for feature_name in MODEL_FEATURE_NAMES:
            if feature_name in metrics:
                features.append(metrics[feature_name])
            else:
                features.append(0)  # Default for missing features
        
        print(f"[DEBUG] Using {len(features)} features")
        print(f"[DEBUG] Features: {dict(zip(MODEL_FEATURE_NAMES, features))}")
        
        # Prepare features array (6 features)
        features_array = np.array(features).reshape(1, -1)
        print(f"[DEBUG] Features array shape: {features_array.shape}")
        
        # Apply the trained scaler
        features_scaled = scaler.transform(features_array)
        print(f"[DEBUG] Scaled features: {features_scaled[0]}")
        
        # Reshape for LSTM: (batch_size=1, timesteps=5, features=6)
        timesteps = 5
        features_sequence = np.repeat(features_scaled, timesteps, axis=0)
        features_reshaped = features_sequence.reshape(1, timesteps, len(features))
        
        print(f"[DEBUG] Reshaped for LSTM: {features_reshaped.shape}")
        
        # Make prediction
        predictions = model.predict(features_reshaped, verbose=0)[0]
        predicted_class = np.argmax(predictions)
        confidence = np.max(predictions)
        
        predicted_instance = INSTANCE_TYPE_MAPPING.get(predicted_class, f'unknown_{predicted_class}')
        
        print(f"[DEBUG] LSTM Prediction: {predicted_class} -> {predicted_instance} (confidence: {confidence:.3f})")
        
        return {
            'instance_type': predicted_instance, 
            'confidence': float(confidence),  # Convert numpy float32 to Python float
            'prediction_code': int(predicted_class),  # Convert numpy int to Python int
            'probabilities': [float(p) for p in predictions],  # Convert all probabilities to Python floats
            'actual_features_used': len(features),
            'feature_names_used': MODEL_FEATURE_NAMES
        }
        
    except Exception as e:
        print(f"Error in LSTM prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def compare_metrics(current_metrics, predicted_metrics=None):
    """Compare current instance metrics with predicted instance metrics (if available)"""
    if predicted_metrics is None:
        predicted_metrics = {}
    
    comparison = {}
    for metric in MODEL_FEATURE_NAMES:
        current_value = current_metrics.get(metric, 0)
        predicted_value = predicted_metrics.get(metric, 0)
        
        if predicted_value > 0:
            difference_percent = ((current_value - predicted_value) / predicted_value) * 100
        else:
            difference_percent = 0
            
        comparison[metric] = {
            'current': round(current_value, 2),
            'predicted': round(predicted_value, 2),
            'difference': round(current_value - predicted_value, 2),
            'difference_percent': round(difference_percent, 2),
            'status': 'higher' if current_value > predicted_value else 'lower' if current_value < predicted_value else 'equal'
        }
    
    return comparison

def analyze_provisioning(current_type, lstm_type, gemini_type, current_metrics, cost_current=None, cost_lstm=None, cost_gemini=None):
    """Analyze provisioning by comparing current instance with both LSTM and Gemini recommendations
    
    Args:
        current_type: Current instance type name
        lstm_type: LSTM model recommended instance type
        gemini_type: Gemini AI recommended instance type
        current_metrics: Dictionary of current instance metrics
        cost_current: Hourly cost for current instance
        cost_lstm: Hourly cost for LSTM recommended instance
        cost_gemini: Hourly cost for Gemini recommended instance
    """
    metrics_comparison = compare_metrics(current_metrics)
    
    cpu_util = current_metrics.get('CPU_Utilization_Percent', 0)
    memory_util = current_metrics.get('Memory_Utilization_Percent', 0)
    
    # Use Gemini-provided costs only (no fallbacks)
    if cost_current is None or cost_current == 0:
        print(f"[WARNING] No cost available for {current_type}, skipping savings calculation")
        cost_current = 0
    if cost_lstm is None or cost_lstm == 0:
        print(f"[WARNING] No cost available for {lstm_type}, skipping savings calculation")
        cost_lstm = 0
    if cost_gemini is None or cost_gemini == 0:
        print(f"[WARNING] No cost available for {gemini_type}, skipping savings calculation")
        cost_gemini = 0
    
    print(f"[DEBUG] Gemini Costs - Current: ${cost_current:.4f}, LSTM: ${cost_lstm:.4f}, Gemini: ${cost_gemini:.4f}")
    
    # Determine if both models agree
    models_agree = lstm_type == gemini_type
    recommended_type = gemini_type  # Prefer Gemini's recommendation for final decision
    recommended_cost = cost_gemini
    
    # Calculate actual savings based on the recommendation
    # Positive = savings (current is more expensive), Negative = cost increase (upgrade needed)
    monthly_savings = (cost_current - recommended_cost) * 730
    
    # Determine provisioning status
    if current_type == lstm_type == gemini_type:
        status = 'optimal'
        recommendation = 'Both LSTM and Gemini agree: Current instance type is optimal'
        monthly_savings = 0  # No change = no savings
    elif models_agree and current_type != recommended_type:
        # Both models recommend the same change
        if monthly_savings > 0:
            status = 'over-provisioned'
            recommendation = f'Both models recommend {recommended_type} - potential savings ${monthly_savings/730:.4f}/hour'
        else:
            status = 'under-provisioned'
            recommendation = f'Both models recommend upgrading to {recommended_type} for better performance (additional ${abs(monthly_savings/730):.4f}/hour)'
    else:
        # Models disagree - analyze based on utilization and costs
        if cpu_util < 40 and memory_util < 40:
            status = 'over-provisioned'
            recommendation = f'Low utilization detected. Gemini recommends {gemini_type}, LSTM suggests {lstm_type}'
        elif cpu_util > 75 or memory_util > 75:
            status = 'under-provisioned'
            recommendation = f'High utilization detected. Gemini recommends {gemini_type}, LSTM suggests {lstm_type}'
        else:
            status = 'optimal'
            recommendation = f'Current instance adequate. Gemini suggests {gemini_type}, LSTM suggests {lstm_type}'
    
    return {
        'status': status,
        'recommendation': recommendation,
        'current_type': current_type,
        'lstm_type': lstm_type,
        'gemini_type': gemini_type,
        'models_agree': models_agree,
        'current_cost': cost_current,
        'lstm_cost': cost_lstm,
        'gemini_cost': cost_gemini,
        'potential_savings_lstm': round(cost_current - cost_lstm, 4),
        'potential_savings_gemini': round(cost_current - cost_gemini, 4),
        'potential_monthly_savings': round(monthly_savings, 2),
        'metrics_comparison': metrics_comparison
    }

@app.route('/')
def index():
    """Serve welcome page from the Tailwind frontend."""
    return app.send_static_file('index.html')


@app.route('/predict')
def predict_page():
    """Serve the guided prediction page."""
    return app.send_static_file('predict.html')


@app.route('/real-time')
def realtime_page():
    """Serve the real-time command center page."""
    return app.send_static_file('real-time.html')


@app.route('/js/<path:asset_path>')
def static_js(asset_path):
    """Serve frontend JavaScript assets."""
    return send_from_directory(os.path.join(app.static_folder, 'js'), asset_path)

@app.route('/analysis')
def analysis():
    """Legacy route that now points to the real-time command center."""
    return app.send_static_file('real-time.html')

@app.route('/recommendations')
def recommendations():
    """Legacy route that now points to the prediction lab."""
    return app.send_static_file('predict.html')


def build_gemini_prompt(instance_id, current_type, predicted_type, metrics_map):
    """Craft a structured prompt asking Gemini for an enriched metric table with full instance specs."""
    metrics_payload = {
        metric: {
            'current': values.get('current'),
            'predicted': values.get('predicted', values.get('standard')),
            'difference': values.get('difference'),
            'difference_percent': values.get('difference_percent'),
        }
        for metric, values in (metrics_map or {}).items()
    }

    prompt = f"""
You are an AWS EC2 instance optimization expert. Analyze the current instance and provide a detailed comparison with the recommended instance type, including complete specifications from AWS catalog.

Return ONLY valid JSON with this schema:
{{
    "caption": "string",
    "disclaimer": "string",
    "rows": [
        {{
            "metric": "string",
            "current": "string",
            "recommended": "string",
            "unit": "string"
        }}
    ],
    "recommendations": ["string", ...]
}}

Mandatory rows to include:
1. "vCPU" - Number of virtual CPUs for each instance type
2. "Instance Memory" - RAM in GiB for each instance type
3. "Network Performance" - Network bandwidth description
4. "Hourly Cost (On demand)" - AWS on-demand pricing in USD/hr
5. "Category" - Instance category (e.g., General Purpose, Compute Optimized, Memory Optimized, Burstable)
6. "CPU Utilization" - Current utilization percentage from telemetry
7. "Memory Utilization" - Current utilization percentage from telemetry

You may add up to 3 additional relevant rows (storage IOPS, network throughput, burst credits, etc.).

For the "recommendations" array, provide 3-5 actionable bullet points explaining:
- Cost savings or additional cost with exact dollar amounts
- Performance implications of the change
- Sizing risks or benefits
- Migration considerations

Instance identifier: {instance_id}
Current instance type: {current_type}
Recommended instance type: {predicted_type}

Workload telemetry (actual usage metrics):
{json.dumps(metrics_payload, indent=2)}

IMPORTANT: Look up the actual AWS EC2 specifications for both {current_type} and {predicted_type} from your knowledge base. Provide accurate vCPU count, memory, network performance, cost per hour, and category for each instance type.
    """
    return prompt


@app.route('/api/gemini_metrics', methods=['POST'])
def gemini_metrics():
    """Retrieve enriched instance comparison metrics via Gemini."""
    payload = request.json or {}
    instance_id = payload.get('instance_id', 'unknown')
    current_type = payload.get('current_type')
    predicted_type = payload.get('predicted_type')
    metrics_map = payload.get('metrics') or {}

    fallback = build_spec_fallback(current_type, predicted_type, metrics_map)

    if not current_type or not predicted_type:
        fallback['disclaimer'] = 'Missing instance types in request. Serving static reference data.'
        return jsonify(fallback)

    try:
        client = get_gemini_client()
        prompt = build_gemini_prompt(instance_id, current_type, predicted_type, metrics_map)
        contents = [
            genai_types.Content(
                role='user',
                parts=[genai_types.Part.from_text(text=prompt)]
            )
        ]
        response = client.models.generate_content(
            model=GEMINI_MODEL_NAME,
            contents=contents,
            config=genai_types.GenerateContentConfig(response_mime_type='application/json')
        )

        raw_text = ''
        if hasattr(response, 'text') and response.text:
            raw_text = response.text
        elif getattr(response, 'candidates', None):
            for candidate in response.candidates:
                candidate_parts = getattr(getattr(candidate, 'content', None), 'parts', [])
                for part in candidate_parts:
                    raw_text += getattr(part, 'text', '')
        if not raw_text:
            raw_text = ''.join(getattr(chunk, 'text', '') for chunk in getattr(response, 'parts', []))

        enriched = json.loads(raw_text)
        enriched.setdefault('caption', 'Gemini-assisted comparison')
        enriched.setdefault('disclaimer', 'Generated with Gemini using workload context.')
        if not isinstance(enriched.get('rows'), list) or not enriched['rows']:
            enriched['rows'] = fallback['rows']
        recs = enriched.get('recommendations')
        if not isinstance(recs, list) or not recs:
            enriched['recommendations'] = fallback['recommendations']
        enriched['generated'] = True
        return jsonify(enriched)
    except Exception as exc:
        fallback['disclaimer'] = f"Gemini unavailable ({exc}). Displaying static catalog specs instead."
        return jsonify(fallback)

@app.route('/api/analyze_month', methods=['POST'])
def analyze_month():
    """Analyze instances for a specific month with detailed predictions and comparisons"""
    global active_month_label
    try:
        data = request.json or {}
        month_name = data.get('month') or active_month_label

        if not month_name:
            return jsonify({'error': 'Month is required'}), 400
        
        # Get monthly data
        month_data = get_monthly_data(month_name)
        if month_data is None or len(month_data) == 0:
            return jsonify({'error': f'No data found for {month_name}'}), 404
        
        # Calculate instance averages
        instance_averages = calculate_instance_averages(month_data)
        if len(instance_averages) == 0:
            return jsonify({'error': 'Unable to calculate averages'}), 500
        
        results = []
        status_counts = {'optimal': 0, 'over-provisioned': 0, 'under-provisioned': 0}
        total_savings = 0
        
        # STEP 1: Collect all instances and their LSTM predictions
        instances_for_gemini = []
        lstm_predictions_map = {}
        
        for _, row in instance_averages.iterrows():
            instance_id = row['Instance_ID']
            current_type = row['Instance_Type']
            
            # Extract metrics
            metrics = {}
            for feature_name in MODEL_FEATURE_NAMES:
                if feature_name in row:
                    metrics[feature_name] = row[feature_name]
                else:
                    metrics[feature_name] = 0
            
            # Get LSTM prediction
            lstm_prediction = predict_optimal_instance(metrics)
            if not lstm_prediction:
                continue
            
            lstm_predictions_map[instance_id] = lstm_prediction
            
            # Prepare data for batch Gemini request
            instances_for_gemini.append({
                'instance_id': instance_id,
                'current_type': current_type,
                'metrics': metrics
            })
        
        # STEP 2: SINGLE BATCH API CALL to Gemini for ALL recommendations and costs
        print(f"[BATCH GEMINI] Fetching recommendations and costs for {len(instances_for_gemini)} instances in ONE request")
        gemini_batch_results = get_batch_gemini_recommendations_and_costs(instances_for_gemini)
        print(f"[BATCH GEMINI] Retrieved {len(gemini_batch_results)} recommendations")
        
        # STEP 3: Process all instances with batched results
        for data in instances_for_gemini:
            instance_id = row['Instance_ID']
            current_type = row['Instance_Type']
            
            # Extract the correct 7 features for prediction
            metrics = {}
            for feature_name in MODEL_FEATURE_NAMES:
                if feature_name in row:
                    metrics[feature_name] = row[feature_name]
                else:
                    metrics[feature_name] = 0
                    
            instance_id = data['instance_id']
            current_type = data['current_type']
            metrics = data['metrics']
            
            # Get LSTM prediction from map
            lstm_prediction = lstm_predictions_map.get(instance_id)
            if not lstm_prediction:
                continue
            
            lstm_type = lstm_prediction['instance_type']
            lstm_confidence = lstm_prediction['confidence']
            
            # Get Gemini results from batch response (costs from Gemini only)
            gemini_result = gemini_batch_results.get(instance_id)
            if gemini_result:
                gemini_type = gemini_result['recommended_type']
                gemini_reasoning = gemini_result['reasoning']
                gemini_confidence = gemini_result['confidence']
                cost_current = gemini_result.get('current_cost', 0)
                cost_gemini = gemini_result.get('recommended_cost', 0)
                
                # Get LSTM cost using reference pricing (already validated in Gemini batch)
                if lstm_type == gemini_type:
                    cost_lstm = cost_gemini
                elif lstm_type == current_type:
                    cost_lstm = cost_current
                else:
                    # Use reference pricing for LSTM prediction
                    cost_lstm = get_instance_price(lstm_type)
                    print(f"[PRICING] Using reference price for LSTM {lstm_type}: ${cost_lstm:.4f}/hr")
            else:
                # Fallback if Gemini batch failed
                gemini_type = lstm_type
                gemini_reasoning = "Gemini unavailable - using LSTM prediction"
                gemini_confidence = "unknown"
                cost_current = 0
                cost_lstm = 0
                cost_gemini = 0
            
            print(f"[DEBUG] {instance_id}: Current={current_type}(${cost_current:.4f}), LSTM={lstm_type}(${cost_lstm:.4f}), Gemini={gemini_type}(${cost_gemini:.4f})")
            
            # Compare all three (current vs LSTM vs Gemini)
            analysis = analyze_provisioning(
                current_type, 
                lstm_type,
                gemini_type,
                metrics,
                cost_current=cost_current,
                cost_lstm=cost_lstm,
                cost_gemini=cost_gemini
            )
            
            # Update counters
            status_counts[analysis['status']] += 1
            if analysis.get('potential_savings_gemini', 0) > 0:
                total_savings += analysis['potential_savings_gemini']
            
            results.append({
                'instance_id': instance_id,
                'current_type': current_type,
                'lstm_prediction': {
                    'instance_type': lstm_type,
                    'confidence': round(lstm_confidence * 100, 2),
                    'probabilities': lstm_prediction.get('probabilities', [])
                },
                'gemini_prediction': {
                    'instance_type': gemini_type,
                    'reasoning': gemini_reasoning,
                    'confidence': gemini_confidence
                },
                'models_agree': lstm_type == gemini_type,
                'metrics': {k: round(v, 2) if isinstance(v, (int, float)) else v for k, v in metrics.items()},
                'analysis': analysis
            })
        
        # Check if Gemini was available during analysis
        gemini_available = any(r.get('gemini_prediction', {}).get('confidence') != 'unknown' 
                              for r in results if r.get('gemini_prediction'))
        
        return jsonify({
            'month': month_name,
            'active_month': active_month_label,
            'total_instances': len(results),
            'status_summary': status_counts,
            'total_potential_savings': round(total_savings, 4),
            'gemini_available': gemini_available,
            'api_status': 'Gemini API available' if gemini_available else 'Using LSTM predictions with default costs',
            'results': results
        })
        
    except Exception as e:
        print(f"[ERROR] Error in analyze_month: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/debug_model', methods=['POST'])
def debug_model():
    """Debug the LSTM model behavior with different inputs"""
    try:
        # Test with different synthetic inputs (7 features only)
        test_cases = [
            # Low usage case - should predict small instance
            {'CPU_Utilization_Percent': 10, 'Memory_Utilization_Percent': 15, 'Network_In_Mbps': 1, 'Network_Out_Mbps': 1, 'Disk_Usage_Percent': 20, 'Response_Time_ms': 200, 'Hourly_Cost_USD': 0.01},
            # Medium usage case - should predict medium instance
            {'CPU_Utilization_Percent': 50, 'Memory_Utilization_Percent': 60, 'Network_In_Mbps': 10, 'Network_Out_Mbps': 10, 'Disk_Usage_Percent': 50, 'Response_Time_ms': 100, 'Hourly_Cost_USD': 0.1},
            # High usage case - should predict large instance
            {'CPU_Utilization_Percent': 90, 'Memory_Utilization_Percent': 85, 'Network_In_Mbps': 50, 'Network_Out_Mbps': 50, 'Disk_Usage_Percent': 80, 'Response_Time_ms': 50, 'Hourly_Cost_USD': 0.3},
            # Memory heavy case - should predict memory optimized
            {'CPU_Utilization_Percent': 30, 'Memory_Utilization_Percent': 95, 'Network_In_Mbps': 5, 'Network_Out_Mbps': 5, 'Disk_Usage_Percent': 60, 'Response_Time_ms': 80, 'Hourly_Cost_USD': 0.25},
            # Compute heavy case - should predict compute optimized
            {'CPU_Utilization_Percent': 95, 'Memory_Utilization_Percent': 40, 'Network_In_Mbps': 15, 'Network_Out_Mbps': 15, 'Disk_Usage_Percent': 45, 'Response_Time_ms': 60, 'Hourly_Cost_USD': 0.17},
            # Very low usage - should predict nano/micro
            {'CPU_Utilization_Percent': 5, 'Memory_Utilization_Percent': 10, 'Network_In_Mbps': 0.5, 'Network_Out_Mbps': 0.5, 'Disk_Usage_Percent': 15, 'Response_Time_ms': 300, 'Hourly_Cost_USD': 0.005}
        ]
        
        results = []
        for i, test_case in enumerate(test_cases):
            print(f"\n[DEBUG TEST {i+1}] Input: {test_case}")
            
            prediction = predict_optimal_instance(test_case)
            if prediction:
                results.append({
                    'test_case': i+1,
                    'input': test_case,
                    'prediction': prediction['instance_type'],
                    'confidence': prediction['confidence'],
                    'probabilities': prediction.get('probabilities', []),
                    'prediction_code': prediction.get('prediction_code', -1)
                })
        
        # Check if model is stuck
        unique_predictions = set([r['prediction'] for r in results])
        unique_codes = set([r['prediction_code'] for r in results])
        is_stuck = len(unique_predictions) == 1
        
        return jsonify({
            'model_stuck': is_stuck,
            'unique_predictions': len(unique_predictions),
            'unique_prediction_codes': len(unique_codes),
            'predictions': list(unique_predictions),
            'prediction_codes': list(unique_codes),
            'detailed_results': results,
            'model_summary': {
                'input_shape': str(model.input_shape) if model else 'No model',
                'output_shape': str(model.output_shape) if model else 'No model',
                'num_classes': len(INSTANCE_TYPE_MAPPING)
            },
            'analysis': {
                'total_tests': len(test_cases),
                'successful_predictions': len(results),
                'failed_predictions': len(test_cases) - len(results),
                'diversity_score': len(unique_predictions) / len(results) if results else 0
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/quick_test')
def quick_test():
    """Quick test with extreme cases to check model responsiveness"""
    try:
        # Test with extreme cases (7 features only)
        low_case = {
            'CPU_Utilization_Percent': 5, 
            'Memory_Utilization_Percent': 10, 
            'Network_In_Mbps': 0.5, 
            'Network_Out_Mbps': 0.5, 
            'Disk_Usage_Percent': 15, 
            'Response_Time_ms': 300, 
            'Hourly_Cost_USD': 0.005
        }
        
        high_case = {
            'CPU_Utilization_Percent': 95, 
            'Memory_Utilization_Percent': 90, 
            'Network_In_Mbps': 100, 
            'Network_Out_Mbps': 100, 
            'Disk_Usage_Percent': 90, 
            'Response_Time_ms': 30, 
            'Hourly_Cost_USD': 0.5
        }
        
        medium_case = {
            'CPU_Utilization_Percent': 50, 
            'Memory_Utilization_Percent': 50, 
            'Network_In_Mbps': 10, 
            'Network_Out_Mbps': 10, 
            'Disk_Usage_Percent': 50, 
            'Response_Time_ms': 100, 
            'Hourly_Cost_USD': 0.1
        }
        
        low_pred = predict_optimal_instance(low_case)
        high_pred = predict_optimal_instance(high_case)
        medium_pred = predict_optimal_instance(medium_case)
        
        predictions = [low_pred, high_pred, medium_pred]
        valid_predictions = [p for p in predictions if p is not None]
        
        if not valid_predictions:
            return jsonify({'error': 'All predictions failed'}), 500
            
        unique_predictions = set([p['instance_type'] for p in valid_predictions])
        
        return jsonify({
            'low_usage': low_pred,
            'high_usage': high_pred,
            'medium_usage': medium_pred,
            'same_prediction': len(unique_predictions) == 1,
            'unique_predictions': len(unique_predictions),
            'prediction_types': list(unique_predictions),
            'model_responsive': len(unique_predictions) > 1,
            'test_summary': {
                'total_tests': 3,
                'successful_tests': len(valid_predictions),
                'failed_tests': 3 - len(valid_predictions),
                'diversity': len(unique_predictions) / len(valid_predictions) if valid_predictions else 0
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/model_info')
def get_model_info():
    """Get detailed model information"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 404
            
        info = {
            'model_loaded': model is not None,
            'scaler_loaded': scaler is not None,
            'model_type': str(type(model)),
            'input_shape': str(model.input_shape),
            'output_shape': str(model.output_shape),
            'num_parameters': model.count_params(),
            'expected_features': MODEL_FEATURE_NAMES,
            'num_features': len(MODEL_FEATURE_NAMES),
            'instance_types': list(INSTANCE_TYPE_MAPPING.values()),
            'num_classes': len(INSTANCE_TYPE_MAPPING),
            'scaler_info': {
                'n_features_in': scaler.n_features_in_ if scaler else 'N/A',
                'feature_names': list(scaler.feature_names_in_) if scaler and hasattr(scaler, 'feature_names_in_') else 'N/A'
            }
        }
        
        # Add layer information
        layers = []
        for i, layer in enumerate(model.layers):
            layer_info = {
                'index': i,
                'name': layer.name,
                'type': layer.__class__.__name__
            }
            
            if hasattr(layer, 'units'):
                layer_info['units'] = layer.units
            if hasattr(layer, 'activation'):
                layer_info['activation'] = str(layer.activation)
            if hasattr(layer, 'input_shape'):
                layer_info['input_shape'] = str(layer.input_shape)
            if hasattr(layer, 'output_shape'):
                layer_info['output_shape'] = str(layer.output_shape)
                
            layers.append(layer_info)
        
        info['layers'] = layers
        
        return jsonify(info)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/test_single_prediction', methods=['POST'])
def test_single_prediction():
    """Test a single prediction with custom input"""
    try:
        data = request.json
        
        # Validate input has all required features
        missing_features = []
        for feature in MODEL_FEATURE_NAMES:
            if feature not in data:
                missing_features.append(feature)
        
        if missing_features:
            return jsonify({
                'error': 'Missing required features',
                'missing_features': missing_features,
                'required_features': MODEL_FEATURE_NAMES
            }), 400
        
        # Extract metrics
        metrics = {feature: data[feature] for feature in MODEL_FEATURE_NAMES}
        
        # Make LSTM prediction
        lstm_prediction = predict_optimal_instance(metrics)
        
        if lstm_prediction is None:
            return jsonify({'error': 'LSTM prediction failed'}), 500
        
        # Get Gemini recommendation
        gemini_recommendation = get_gemini_instance_recommendation(metrics)
        
        return jsonify({
            'input_metrics': metrics,
            'prediction': lstm_prediction,
            'predicted_instance_type': lstm_prediction['instance_type'],
            'gemini_recommendation': gemini_recommendation,
            'success': True
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/get_months')
def get_months():
    """Get available months"""
    global active_month_label
    if dataset is not None:
        months = dataset['Month'].dropna().unique().tolist()
        if active_month_label and active_month_label not in months:
            months.insert(0, active_month_label)
        elif active_month_label:
            months = [active_month_label] + [m for m in months if m != active_month_label]
        return jsonify({'months': months, 'active_month': active_month_label})
    return jsonify({'months': [], 'active_month': active_month_label})

@app.route('/api/instance_types')
def get_instance_types():
    """Get instance type mapping (model class to instance name)"""
    return jsonify({
        'mapping': INSTANCE_TYPE_MAPPING,
        'note': 'Detailed instance specifications are provided by Gemini AI'
    })



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
            'instance_types': sorted(dataset['Instance_Type'].unique().tolist()) if 'Instance_Type' in dataset.columns else [],
            'num_instances': len(dataset['Instance_ID'].unique()) if 'Instance_ID' in dataset.columns else 0,
            'date_range': {
                'min': str(dataset['Date'].min()) if 'Date' in dataset.columns else 'N/A',
                'max': str(dataset['Date'].max()) if 'Date' in dataset.columns else 'N/A'
            } if 'Date' in dataset.columns else 'N/A'
        }
        
        # Add feature statistics
        feature_stats = {}
        for feature in MODEL_FEATURE_NAMES:
            if feature in dataset.columns:
                feature_stats[feature] = {
                    'min': float(dataset[feature].min()),
                    'max': float(dataset[feature].max()),
                    'mean': float(dataset[feature].mean()),
                    'std': float(dataset[feature].std())
                }
        
        info['feature_statistics'] = feature_stats
        
        return jsonify(info)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Cloud Instance Optimizer...")
    print("="*60)
    
    models_loaded = load_models()
    dataset_loaded = load_dataset()
    
    if not models_loaded:
        print("❌ WARNING: Models not loaded. Predictions will not work.")
    else:
        print("✅ Models loaded successfully")
    
    if not dataset_loaded:
        print("❌ WARNING: Dataset not loaded. Some features may not work.")
    else:
        print("✅ Dataset loaded successfully")
    
    print("="*60)
    print("Available endpoints:")
    print("  GET  /                     - Main dashboard")
    print("  POST /api/analyze_month    - Analyze monthly data")
    print("  POST /api/debug_model      - Debug model behavior")
    print("  GET  /api/quick_test       - Quick model test")
    print("  GET  /api/model_info       - Get model information")
    print("  POST /api/test_single_prediction - Test single prediction")
    print("  GET  /api/dataset_info     - Get dataset information")
    print("  GET  /api/get_months       - Get available months")
    print("  GET  /api/instance_types   - Get instance type info")
    print("="*60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)