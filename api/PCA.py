import pandas as pd
import numpy as np
import os
import argparse
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path

def process_embeddings(input_file):
    print(f"Processing file: {input_file}")
    
    # Create output directory if it doesn't exist
    output_dir = Path("6d")
    output_dir.mkdir(exist_ok=True)
    
    # Determine output filename (keeping the original name but in the 6d folder)
    output_filename = output_dir / Path(input_file).name
    
    # Load the embeddings
    try:
        data = pd.read_csv(input_file)
        print(f"Loaded data with shape: {data.shape}")
    except Exception as e:
        print(f"Error loading file: {e}")
        return
    
    # Check if we have numerical data
    numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
    if len(numeric_cols) == 0:
        print("No numeric columns found in the dataset")
        return
    
    # Use all numeric columns for PCA
    X = data[numeric_cols]
    print(f"Using {len(numeric_cols)} numeric columns for PCA")
    
    # Apply PCA to reduce to 6 dimensions
    pca = PCA(n_components=6)
    try:
        principal_components = pca.fit_transform(X)
        print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    except Exception as e:
        print(f"Error during PCA: {e}")
        return
    
    # Create a DataFrame with the principal components
    pca_df = pd.DataFrame(
        data=principal_components,
        columns=[f'PC{i+1}' for i in range(6)]
    )
    
    # Normalize the PCA vectors (L2 normalization)
    normalized_pca = pca_df.apply(lambda x: x / np.sqrt(np.sum(x**2)), axis=1)
    
    # Save the normalized PCA embeddings
    normalized_pca.to_csv(output_filename, index=False)
    print(f"Saved normalized PCA embeddings to: {output_filename}")
    
    # Print stats about the output
    max_values = normalized_pca.max()
    min_values = normalized_pca.min()
    
    print("\nMax values for each embedding:")
    print(max_values)
    
    print("\nMin values for each embedding:")
    print(min_values)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Reduce dimensionality of embeddings to 6D using PCA')
    parser.add_argument('input_file', type=str, help='Path to the input CSV file with embeddings')
    args = parser.parse_args()
    
    process_embeddings(args.input_file)