"""
Configuration Loader Utility
Loads all JSON configuration files
"""
import json
import os

class Config:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._load_configs()
        return cls._instance
    
    def _load_configs(self):
        """Load all configuration files"""
        # Get absolute path to config directory
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_dir = os.path.join(base_dir, 'config')
        
        print(f"[CONFIG] Loading from: {config_dir}")
        
        # Load app configuration
        with open(os.path.join(config_dir, 'app_config.json'), 'r') as f:
            self.app_config = json.load(f)
        
        # Load AWS pricing
        with open(os.path.join(config_dir, 'aws_pricing.json'), 'r') as f:
            pricing_data = json.load(f)
            self.aws_pricing = pricing_data['pricing']
        
        # Load instance type mapping
        with open(os.path.join(config_dir, 'instance_type_mapping.json'), 'r') as f:
            mapping_data = json.load(f)
            # Convert string keys to integers
            self.instance_type_mapping = {int(k): v for k, v in mapping_data['mapping'].items()}
        
        # Load model configuration
        with open(os.path.join(config_dir, 'model_config.json'), 'r') as f:
            self.model_config = json.load(f)
    
    def get_app_config(self):
        return self.app_config
    
    def get_aws_config(self):
        return self.app_config['aws']
    
    def get_gemini_config(self):
        return self.app_config['gemini']
    
    def get_pricing(self):
        return self.aws_pricing
    
    def get_instance_price(self, instance_type):
        return self.aws_pricing.get(instance_type, 0)
    
    def get_instance_mapping(self):
        return self.instance_type_mapping
    
    def get_model_features(self):
        return self.model_config['features']
    
    def get_lstm_config(self):
        return self.model_config['lstm_config']

# Singleton instance
config = Config()
