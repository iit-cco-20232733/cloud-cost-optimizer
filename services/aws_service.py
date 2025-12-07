"""
AWS Service
Handles S3 and other AWS operations
"""
import boto3
import pandas as pd
from io import StringIO
import csv
from utils.config_loader import config

class AWSService:
    def __init__(self):
        self.aws_config = config.get_aws_config()
        self.region = self.aws_config['region']
        self.s3_bucket = self.aws_config['s3_bucket']
        self.s3_dataset_key = self.aws_config['s3_dataset_key']
        self.credentials_file = self.aws_config['credentials_file']
        self._s3_client = None
    
    def load_credentials(self):
        """Load AWS credentials from CSV"""
        try:
            df = pd.read_csv(self.credentials_file)
            if len(df) > 0:
                return df.iloc[0]['Access key ID'], df.iloc[0]['Secret access key']
            return None, None
        except Exception as e:
            print(f"Error loading credentials: {e}")
            return None, None
    
    def get_s3_client(self):
        """Get or create S3 client"""
        if self._s3_client is None:
            access_key, secret_key = self.load_credentials()
            if not access_key or not secret_key:
                raise RuntimeError("Failed to load AWS credentials")
            
            self._s3_client = boto3.client(
                's3',
                region_name=self.region,
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key
            )
        return self._s3_client
    
    def load_dataset_from_s3(self):
        """Load dataset from S3"""
        try:
            print(f"[S3] Loading dataset from s3://{self.s3_bucket}/{self.s3_dataset_key}")
            s3_client = self.get_s3_client()
            
            response = s3_client.get_object(Bucket=self.s3_bucket, Key=self.s3_dataset_key)
            csv_content = response['Body'].read().decode('utf-8')
            df = pd.read_csv(StringIO(csv_content))
            
            print(f"[S3] Loaded {len(df)} records from S3")
            return df
            
        except Exception as e:
            print(f"[ERROR] Failed to load from S3: {e}")
            return None
    
    def save_to_s3(self, df, s3_key):
        """Save DataFrame to S3"""
        try:
            s3_client = self.get_s3_client()
            csv_buffer = df.to_csv(index=False)
            
            s3_client.put_object(
                Bucket=self.s3_bucket,
                Key=s3_key,
                Body=csv_buffer,
                ContentType='text/csv'
            )
            
            print(f"[S3] Saved to s3://{self.s3_bucket}/{s3_key}")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to save to S3: {e}")
            return False

# Singleton instance
aws_service = AWSService()
