# Cloud Instance Optimization System

AI-powered cloud instance optimization using LSTM and Gemini AI with MVC architecture.

## 📁 Project Structure

```
cloud/
├── app_mvc.py                 # Main Flask application (MVC pattern)
├── app.py                     # Legacy monolithic app (backup)
│
├── config/                    # Configuration files
│   ├── app_config.json       # Application settings
│   ├── aws_pricing.json      # AWS EC2 pricing reference
│   ├── instance_type_mapping.json  # LSTM model class mapping
│   ├── model_config.json     # Model configuration
│   └── demo-user_accessKeys.csv    # AWS credentials
│
├── controllers/               # Business logic layer
│   ├── __init__.py
│   └── analysis_controller.py  # Instance analysis logic
│
├── services/                  # External service integrations
│   ├── __init__.py
│   ├── aws_service.py        # S3 and AWS operations
│   ├── gemini_service.py     # Gemini AI integration
│   └── lstm_service.py       # LSTM model predictions
│
├── utils/                     # Utilities
│   ├── __init__.py
│   └── config_loader.py      # Configuration loader
│
├── models/                    # Trained models
│   ├── cloud_instance_lstm_model.h5
│   ├── scaler.pkl
│   └── *.json
│
├── frontend/                  # Static frontend files
│   ├── index.html
│   ├── predict.html
│   ├── real-time.html
│   └── js/
│       ├── common.js
│       ├── predict.js
│       └── realtime.js
│
├── dataset/                   # Training datasets
│   └── *.csv
│
├── data/                      # Runtime data
│   └── (generated files)
│
└── scripts/                   # Utility scripts
    ├── collect_ec2_metrics.py    # Collect metrics from CloudWatch
    └── metrics_server.py         # Metrics exposure server
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Application
Edit `config/app_config.json` with your settings:
- AWS region and S3 bucket
- Gemini API key
- Application host and port

### 3. Run Application
```bash
python app_mvc.py
```

Access at: `http://localhost:5000`

## 📝 Configuration Files

### app_config.json
Main application configuration including AWS, Gemini, and Flask settings.

### aws_pricing.json
AWS EC2 on-demand pricing reference (us-east-1). Update periodically.

### instance_type_mapping.json
Maps LSTM model output classes (0-14) to AWS instance types.

### model_config.json
Defines the 6 features used by the LSTM model:
- Network_In_Mbps
- Network_Out_Mbps  
- Response_Time_ms
- CPU_Utilization_Percent
- Memory_Utilization_Percent
- Disk_Usage_Percent

## 🏗️ MVC Architecture

### Models (`services/lstm_service.py`)
- Loads and manages LSTM model
- Handles predictions with proper feature scaling
- Returns instance type recommendations

### Views (`frontend/`)
- HTML templates with Tailwind CSS
- JavaScript for dynamic interactions
- Real-time dashboard and prediction lab

### Controllers (`controllers/`)
- `analysis_controller.py`: Core analysis logic
  - Analyzes provisioning status
  - Compares LSTM vs Gemini predictions
  - Calculates cost savings

### Services (`services/`)
- `aws_service.py`: S3 operations and data loading
- `gemini_service.py`: Gemini AI integration
- `lstm_service.py`: LSTM model operations

## 📊 API Endpoints

### Frontend Routes
- `GET /` - Welcome page
- `GET /predict` - Prediction lab
- `GET /real-time` - Real-time dashboard

### API Routes
- `POST /api/analyze_month` - Analyze monthly instance data
- `POST /api/test_single_prediction` - Test single prediction
- `GET /api/get_months` - Get available months
- `GET /api/dataset_info` - Get dataset information
- `GET /api/model_info` - Get model configuration
- `GET /api/instance_types` - Get instance type mapping

## 🛠️ Utility Scripts

### Collect EC2 Metrics
```bash
python scripts/collect_ec2_metrics.py
```
Collects metrics from all running EC2 instances using CloudWatch and optionally saves to S3.

### Metrics Server
```bash
python scripts/metrics_server.py
```
Lightweight Flask server to expose instance metrics (deploy on EC2 instances).

## 🔧 Development

### Adding New Instance Types
1. Update `config/aws_pricing.json` with new pricing
2. Update `config/instance_type_mapping.json` if adding LSTM classes
3. Retrain model if needed

### Updating Configuration
All configurations are in JSON files - no code changes needed for:
- AWS settings
- Pricing updates
- Feature modifications
- Model paths

### Running Old Monolithic App
```bash
python app.py
```

## 📈 Features

- ✅ **Dual AI Predictions**: LSTM + Gemini AI
- ✅ **Real-time Analysis**: Analyze all instances in one click
- ✅ **Cost Optimization**: Accurate monthly savings calculations
- ✅ **MVC Architecture**: Clean separation of concerns
- ✅ **JSON Configuration**: No hardcoded values
- ✅ **S3 Integration**: Load datasets from cloud
- ✅ **Batch Processing**: Efficient Gemini API usage

## 🔒 Security Notes

- Keep `config/demo-user_accessKeys.csv` secure
- Don't commit API keys to version control
- Use environment variables in production
- Rotate AWS credentials regularly

## 📄 License

Internal Use Only
