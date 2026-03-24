import pandas as pd
import numpy as np
import os

def generate_temporal_deltas(csv_path, window_size=10):
    """
    Calculates the rate of change (velocity) for BVP, HR, and EDA 
    over a specified window size (number of preceding windows).
    
    Formula: (val_t - val_{t-window_size}) / window_size
    """
    print(f"Loading dataset: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Ensure sorting by subject and sequence (assuming window_id or index implies timing within subject)
    # The request implies temporal continuity.
    df = df.sort_values(by=['subject_id', 'window_id']).reset_index(drop=True)
    
    features_to_delta = ['bvp_mean', 'hr_mean', 'eda_mean']
    
    for feat in features_to_delta:
        if feat not in df.columns:
            print(f"Warning: {feat} not found in columns. Skipping.")
            continue
            
        new_col = f"{feat}_delta_{window_size}"
        print(f"Calculating {new_col}...")
        
        # Calculate diff over window_size within each subject group
        # Using shift(window_size) to get the value 10 windows ago
        df[new_col] = df.groupby('subject_id')[feat].transform(
            lambda x: (x - x.shift(window_size)) / window_size
        )
        
        # Fill NaNs (start of each subject sequence) with 0.0 or forward fill logic
        # 0.0 is safer for "no change" assumption
        df[new_col] = df[new_col].fillna(0.0)
        
    return df

if __name__ == "__main__":
    input_file = r"c:\Users\91704\AI-Integration-in-Wearables-for-Clot-Monitoring\processed_data\integrated_features_balanced_v2.csv"
    output_file = r"c:\Users\91704\AI-Integration-in-Wearables-for-Clot-Monitoring\processed_data\integrated_features_v4_TEMPORAL.csv"
    
    if os.path.exists(input_file):
        processed_df = generate_temporal_deltas(input_file)
        processed_df.to_csv(output_file, index=False)
        print(f"Successfully saved temporal features to: {output_file}")
        
        # Verification snippet
        print("\nVerification (First 5 rows with deltas):")
        delta_cols = [c for c in processed_df.columns if 'delta' in c]
        print(processed_df[['subject_id', 'window_id'] + delta_cols].head(15))
    else:
        print(f"Error: Could not find {input_file}")
