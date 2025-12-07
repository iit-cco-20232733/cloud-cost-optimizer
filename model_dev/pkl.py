import pandas as pd
import pickle
import numpy as np

def load_pkl_content(file_path):
    """
    Load and display PKL file content
    """
    try:
        print(f"Loading: {file_path}")
        
        # Load the pickle file
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"✅ Loaded successfully!")
        print(f"Data type: {type(data)}")
        
        # Show content based on type
        if isinstance(data, pd.DataFrame):
            print(f"Shape: {data.shape}")
            print(f"Columns: {list(data.columns)}")
            print("\nFirst 10 rows:")
            print(data.head(10))
            
            # Show data types
            print(f"\nData types:")
            print(data.dtypes)
            
        elif isinstance(data, dict):
            print(f"Dictionary with {len(data)} keys")
            print("Keys:", list(data.keys())[:20])  # Show first 20 keys
            
            # Show some values
            print("\nSample content:")
            count = 0
            for k, v in data.items():
                if count < 10:  # Show first 10 items
                    print(f"  {k}: {v}")
                    count += 1
                else:
                    break
                    
        elif isinstance(data, list):
            print(f"List with {len(data)} items")
            print("First 10 items:")
            for i, item in enumerate(data[:10]):
                print(f"  [{i}]: {item}")
                
        elif isinstance(data, np.ndarray):
            print(f"NumPy array")
            print(f"Shape: {data.shape}")
            print(f"Data type: {data.dtype}")
            print("Content preview:")
            print(data)
            
        else:
            print("Raw content:")
            print(data)
            
        return data
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def create_mapping(data):
    """
    Create Label_Encoded -> Instance_Type mapping
    """
    try:
        # Convert to DataFrame if needed
        if isinstance(data, pd.DataFrame):
            df = data
        elif isinstance(data, dict):
            df = pd.DataFrame(data)
        else:
            print("Cannot create mapping from this data type")
            return None
            
        # Check for required columns
        if 'Label_Encoded' not in df.columns or 'Instance_Type' not in df.columns:
            print("Missing required columns")
            print("Available columns:", list(df.columns))
            return None
            
        # Create mapping
        mapping = {}
        unique_pairs = df[['Label_Encoded', 'Instance_Type']].drop_duplicates()
        
        for _, row in unique_pairs.iterrows():
            mapping[row['Label_Encoded']] = row['Instance_Type']
            
        print(f"\nMapping created with {len(mapping)} entries:")
        print(mapping)
        
        return mapping
        
    except Exception as e:
        print(f"Error creating mapping: {e}")
        return None

# Usage
if __name__ == "__main__":
    # Replace with your actual file path
    pkl_file = "models/scaler.pkl"  # Change this to your PKL file path
    
    # Load and show content
    data = load_pkl_content(pkl_file)
    
    if data is not None:
        # Try to create the mapping
        mapping = create_mapping(data)
        
        if mapping:
            print(f"\n✅ Successfully created mapping dictionary!")