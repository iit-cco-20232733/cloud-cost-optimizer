#!/usr/bin/env python3
"""
Find specific instance ID in monthly_cloud_instances_2024.csv
"""

import pandas as pd

def find_instance_rows(instance_id, file_path="monthly_cloud_instances_2024.csv"):
    """
    Find all rows containing the specific instance ID
    """
    try:
        # Load the CSV
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} total rows from {file_path}")
        
        # Find instance ID column (could be named different things)
        id_cols = [col for col in df.columns if 'id' in col.lower() or 'instance' in col.lower()]
        
        if not id_cols:
            print("No instance ID column found!")
            print(f"Available columns: {list(df.columns)}")
            return None
        
        # Use the first ID column found
        id_col = id_cols[0]
        print(f"Using column: '{id_col}'")
        
        # Filter rows with the specific instance ID
        matching_rows = df[df[id_col] == instance_id]
        
        if len(matching_rows) == 0:
            print(f"Instance ID '{instance_id}' not found in any rows")
            return None
        
        print(f"\nFound instance '{instance_id}' in {len(matching_rows)} rows:")
        print("Row numbers:", list(matching_rows.index + 1))  # +1 for human-readable row numbers
        
        # Display the matching rows
        for i, (idx, row) in enumerate(matching_rows.iterrows()):
            print(f"\n--- Row {idx + 1} ---")
            for col in df.columns:
                print(f"{col}: {row[col]}")
        
        return matching_rows
        
    except FileNotFoundError:
        print(f"File '{file_path}' not found")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def main():
    # The instance ID from your sample data
    instance_id = "i-5c1dd6e6f41d33eea"
    
    print(f"Searching for instance ID: {instance_id}")
    print("=" * 50)
    
    result = find_instance_rows(instance_id)
    
    if result is not None and len(result) > 0:
        print(f"\nSUMMARY: Found {len(result)} rows with instance '{instance_id}'")
    else:
        print(f"\nSUMMARY: Instance '{instance_id}' not found")

if __name__ == "__main__":
    main()