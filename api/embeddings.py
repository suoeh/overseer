import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import re
import hdbscan
from sklearn.neighbors import NearestNeighbors
import os

def clean_text(text):
    """Basic cleaning function for resume text"""
    if isinstance(text, str):
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    return ""

def generate_embeddings(df, resume_column='Resume_str', model_name="all-MiniLM-L6-v2"):
    """Generate embeddings for resume texts"""
    # Apply basic cleaning to Resume column
    df['cleaned_text'] = df[resume_column].apply(clean_text)
    
    # Load a pretrained Sentence Transformer model
    model = SentenceTransformer(model_name)
    
    # Calculate embeddings for all resumes
    resume_embeddings = model.encode(df['cleaned_text'].tolist(), show_progress_bar=True)
    
    print(f"Shape of embeddings array: {resume_embeddings.shape}")
    print(f"Sample embedding vector: {resume_embeddings[0][:10]}...")  # Show first 10 values
    
    return resume_embeddings

def find_dense_clusters(embeddings, min_cluster_size=5, min_samples=5, n_clusters=10):
    """Find the N densest clusters using HDBSCAN"""
    print(f"Clustering {embeddings.shape[0]} embeddings...")
    
    # Apply HDBSCAN clustering
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean',
        cluster_selection_method='eom'
    )
    cluster_labels = clusterer.fit_predict(embeddings)
    
    # Get number of unique clusters (excluding noise points labeled as -1)
    unique_clusters = np.unique(cluster_labels)
    unique_clusters = unique_clusters[unique_clusters != -1]
    num_clusters = len(unique_clusters)
    
    print(f"Found {num_clusters} clusters (excluding noise)")
    
    if num_clusters == 0:
        print("No clusters found. Try adjusting parameters.")
        return [], []
    
    # Calculate density for each cluster
    cluster_densities = {}
    cluster_indices = {}
    
    for label in unique_clusters:
        # Get points in this cluster
        cluster_points = embeddings[cluster_labels == label]
        cluster_point_indices = np.where(cluster_labels == label)[0]
        
        # Calculate average distance to k nearest neighbors as a proxy for density
        k = min(5, len(cluster_points) - 1)  # Use 5 neighbors or all if fewer points
        if k > 0:
            nbrs = NearestNeighbors(n_neighbors=k+1, metric='euclidean').fit(cluster_points)
            distances, _ = nbrs.kneighbors(cluster_points)
            # Exclude self-distance (first column)
            avg_dist = np.mean(distances[:, 1:])
            # Higher density = lower average distance
            density = 1.0 / (avg_dist + 1e-6)  # Add small value to avoid division by zero
        else:
            density = 0
        
        cluster_densities[label] = density
        cluster_indices[label] = cluster_point_indices.tolist()
    
    # Sort clusters by density (descending)
    sorted_clusters = sorted(cluster_densities.items(), key=lambda x: x[1], reverse=True)
    
    # Select the n densest clusters
    n_densest = min(n_clusters, len(sorted_clusters))
    densest_cluster_labels = [label for label, _ in sorted_clusters[:n_densest]]
    densest_cluster_indices = [cluster_indices[label] for label in densest_cluster_labels]
    
    # Print information about the densest clusters
    for i, (label, density) in enumerate(zip(densest_cluster_labels, 
                                           [cluster_densities[label] for label in densest_cluster_labels])):
        print(f"Cluster {label}: {len(cluster_indices[label])} points, density: {density:.4f}")
    
    # Calculate percentage of data assigned to clusters vs. marked as noise
    noise_count = np.sum(cluster_labels == -1)
    total_count = len(cluster_labels)
    noise_percentage = (noise_count / total_count) * 100
    print(f"Noise points: {noise_count} ({noise_percentage:.2f}% of data)")
    
    return densest_cluster_labels, densest_cluster_indices

def save_clusters_to_csv(df, cluster_indices, output_dir="clusters"):
    """Save each cluster as a separate CSV file"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    for i, indices in enumerate(cluster_indices):
        # Get the data for this cluster
        cluster_df = df.iloc[indices].copy()
        
        # Save to CSV
        output_file = os.path.join(output_dir, f"cluster_{i+1}.csv")
        cluster_df.to_csv(output_file, index=False)
        print(f"Saved cluster {i+1} with {len(indices)} resumes to {output_file}")
    
    # Also save a combined file with a cluster column
    combined_df = df.copy()
    combined_df['cluster'] = -1  # Initialize all as noise
    
    for i, indices in enumerate(cluster_indices):
        combined_df.loc[indices, 'cluster'] = i+1
    
    combined_file = os.path.join(output_dir, "all_clusters.csv")
    combined_df.to_csv(combined_file, index=False)
    print(f"Saved combined file with all clusters to {combined_file}")

def main(input_file=None):
    # Load the resume dataset
    print("Loading resume dataset...")
    try:
        if input_file and os.path.exists(input_file):
            print(f"Loading custom dataset from {input_file}")
            df = pd.read_csv(input_file)
        else:
            # No fallback to default dataset - just error out
            print("Error: No input file provided or file does not exist")
            return None, None
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None, None
    
    # Print dataset information
    print("DataFrame shape:", df.shape)
    print("\nDataFrame columns:", df.columns.tolist())
    
    # Check if embeddings already exist, load them if they do
    if os.path.exists('resume_embeddings.npy'):
        print("\nLoading existing embeddings from resume_embeddings.npy...")
        resume_embeddings = np.load('resume_embeddings.npy')
        
        # If embeddings exist but cleaned_text doesn't, regenerate cleaned_text
        if 'cleaned_text' not in df.columns:
            print("Regenerating cleaned text...")
            df['cleaned_text'] = df['Resume_str'].apply(clean_text)
    else:
        # Generate embeddings
        print("\nGenerating embeddings...")
        resume_embeddings = generate_embeddings(df)
        
        # Save embeddings for later use
        np.save('resume_embeddings.npy', resume_embeddings)
        print("Saved embeddings to resume_embeddings.npy")
    
    # Save cleaned dataframe for later use if not already saved
    if not os.path.exists('cleaned_resumes.csv'):
        df.to_csv('cleaned_resumes.csv', index=False)
        print("Saved cleaned dataframe to cleaned_resumes.csv")
    
    # Find the N densest clusters
    n_clusters = 6  # Number of clusters to save
    _, densest_cluster_indices = find_dense_clusters(
        resume_embeddings, 
        min_cluster_size=10,  # Adjust as needed
        min_samples=5,        # Adjust as needed
        n_clusters=n_clusters
    )
    
    # Save clusters to CSV files
    if densest_cluster_indices:
        save_clusters_to_csv(df, densest_cluster_indices)
    
    return df, resume_embeddings

if __name__ == "__main__":
    df, embeddings = main()
 
# # Optional: Calculate similarity matrix between resumes
# # Warning: This can be memory-intensive for large datasets
# # similarities = model.similarity(resume_embeddings, resume_embeddings)
# # print(f"Similarity matrix shape: {similarities.shape}")
# 
# # Save embeddings for later use
# np.save('resume_embeddings.npy', resume_embeddings)
# 
# # If you want to add embeddings back to the dataframe as a new column
# # This converts each embedding array to a list and stores it in the DataFrame
# df['embedding'] = list(resume_embeddings)
# 
# # Now you can use these embeddings for downstream tasks like clustering or classification 