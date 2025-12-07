import pandas as pd
import pickle
import numpy as np
from collections import defaultdict

def load_and_analyze_data(file_path, file_type='csv'):
    """
    Load data from CSV or PKL file and analyze it
    
    Args:
        file_path (str): Path to the file
        file_type (str): 'csv' or 'pkl'
    
    Returns:
        pandas.DataFrame: Loaded data
    """
    try:
        if file_type.lower() == 'csv':
            # Load CSV file
            df = pd.read_csv(file_path)
            print(f"✅ Successfully loaded CSV file: {file_path}")
        elif file_type.lower() == 'pkl':
            # Load PKL file
            df = pd.read_pickle(file_path)
            print(f"✅ Successfully loaded PKL file: {file_path}")
        else:
            raise ValueError("file_type must be 'csv' or 'pkl'")
        
        return df
    
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return None

def explore_data(df):
    """
    Explore the dataset and show basic information
    """
    print("\n" + "="*60)
    print("📊 DATASET OVERVIEW")
    print("="*60)
    
    print(f"Shape: {df.shape}")
    print(f"Columns: {len(df.columns)}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print("\n📋 Column Names:")
    for i, col in enumerate(df.columns, 1):
        print(f"{i:2d}. {col}")
    
    print("\n🔍 Data Types:")
    print(df.dtypes)
    
    print("\n📈 Basic Statistics:")
    print(df.describe())
    
    print("\n❓ Missing Values:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("No missing values found!")
    
    return df

def create_label_instance_mapping(df):
    """
    Create a dictionary mapping Label_Encoded to Instance_Type
    
    Args:
        df (pandas.DataFrame): The dataset
    
    Returns:
        dict: Dictionary mapping {Label_Encoded: Instance_Type}
    """
    try:
        # Check if required columns exist
        required_cols = ['Label_Encoded', 'Instance_Type']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"❌ Missing columns: {missing_cols}")
            print("Available columns:", list(df.columns))
            return None
        
        # Create the mapping dictionary
        label_to_instance = {}
        
        # Get unique combinations
        unique_combinations = df[['Label_Encoded', 'Instance_Type']].drop_duplicates()
        
        print("\n" + "="*60)
        print("🗺️  LABEL TO INSTANCE TYPE MAPPING")
        print("="*60)
        
        for _, row in unique_combinations.iterrows():
            label_encoded = row['Label_Encoded']
            instance_type = row['Instance_Type']
            label_to_instance[label_encoded] = instance_type
        
        # Display the mapping
        print(f"Found {len(label_to_instance)} unique mappings:")
        print("\nMapping Dictionary:")
        print("{")
        for label, instance in sorted(label_to_instance.items()):
            print(f"    {label}: '{instance}',")
        print("}")
        
        # Show statistics
        print(f"\n📊 Mapping Statistics:")
        print(f"   • Unique Label_Encoded values: {len(label_to_instance)}")
        print(f"   • Unique Instance_Type values: {len(set(label_to_instance.values()))}")
        
        # Check for any inconsistencies
        inconsistencies = []
        for label in df['Label_Encoded'].unique():
            instances_for_label = df[df['Label_Encoded'] == label]['Instance_Type'].unique()
            if len(instances_for_label) > 1:
                inconsistencies.append((label, instances_for_label))
        
        if inconsistencies:
            print(f"\n⚠️  Warning: Found {len(inconsistencies)} labels with multiple instance types:")
            for label, instances in inconsistencies:
                print(f"   Label {label}: {list(instances)}")
        else:
            print("\n✅ No inconsistencies found - each label maps to exactly one instance type")
        
        return label_to_instance
        
    except Exception as e:
        print(f"❌ Error creating mapping: {e}")
        return None

def save_mapping_to_file(mapping_dict, output_file='label_instance_mapping.pkl'):
    """
    Save the mapping dictionary to a pickle file
    """
    try:
        with open(output_file, 'wb') as f:
            pickle.dump(mapping_dict, f)
        print(f"\n💾 Mapping dictionary saved to: {output_file}")
    except Exception as e:
        print(f"❌ Error saving mapping: {e}")

def analyze_instance_distribution(df):
    """
    Analyze the distribution of instance types and labels
    """
    print("\n" + "="*60)
    print("📈 DISTRIBUTION ANALYSIS")
    print("="*60)
    
    if 'Instance_Type' in df.columns:
        print("\n🖥️  Instance Type Distribution:")
        instance_counts = df['Instance_Type'].value_counts()
        for instance, count in instance_counts.items():
            percentage = (count / len(df)) * 100
            print(f"   {instance}: {count:,} ({percentage:.1f}%)")
    
    if 'Label_Encoded' in df.columns:
        print("\n🏷️  Label Distribution:")
        label_counts = df['Label_Encoded'].value_counts().sort_index()
        for label, count in label_counts.items():
            percentage = (count / len(df)) * 100
            print(f"   Label {label}: {count:,} ({percentage:.1f}%)")

# Main execution
if __name__ == "__main__":
    # File path - update this to your actual file path
    csv_file_path = "cloud_instance_dataset_enhanced.csv"
    
    print("🚀 Starting CSV Analysis...")
    
    # Load the CSV file
    df = load_and_analyze_data(csv_file_path, 'csv')
    
    if df is not None:
        # Explore the data
        df = explore_data(df)
        
        # Show first few rows
        print("\n" + "="*60)
        print("👀 FIRST 5 ROWS")
        print("="*60)
        print(df.head())
        
        # Analyze distributions
        analyze_instance_distribution(df)
        
        # Create the mapping dictionary
        mapping_dict = create_label_instance_mapping(df)
        
        if mapping_dict:
            # Save the mapping to a file
            save_mapping_to_file(mapping_dict)
            
            # You can also access the mapping dictionary directly
            print(f"\n🎯 You can now use the mapping_dict variable:")
            print("   mapping_dict =", dict(list(mapping_dict.items())[:3]), "...")
            
            # Example usage
            print(f"\n💡 Example usage:")
            if mapping_dict:
                first_label = next(iter(mapping_dict))
                print(f"   mapping_dict[{first_label}] = '{mapping_dict[first_label]}'")
    
    else:
        print("❌ Failed to load the dataset. Please check the file path and format.")

# Additional utility functions
def load_pkl_mapping(pkl_file_path):
    """
    Load a previously saved mapping dictionary from pickle file
    """
    try:
        with open(pkl_file_path, 'rb') as f:
            mapping = pickle.load(f)
        print(f"✅ Loaded mapping from {pkl_file_path}")
        return mapping
    except Exception as e:
        print(f"❌ Error loading mapping: {e}")
        return None

def reverse_mapping(mapping_dict):
    """
    Create reverse mapping: Instance_Type -> Label_Encoded
    """
    reverse_map = defaultdict(list)
    for label, instance in mapping_dict.items():
        reverse_map[instance].append(label)
    
    # Convert to regular dict and handle single values
    reverse_map = {k: v[0] if len(v) == 1 else v for k, v in reverse_map.items()}
    return dict(reverse_map)

# Example of how to use with different file types:
"""
# For CSV file:
df = load_and_analyze_data("cloud_instance_dataset_enhanced.csv", "csv")

# For PKL file:
df = load_and_analyze_data("your_file.pkl", "pkl")

# Create mapping:
mapping = create_label_instance_mapping(df)

# Use the mapping:
print("Label 0 corresponds to:", mapping.get(0, "Unknown"))
"""