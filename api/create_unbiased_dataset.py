import pandas as pd
import numpy as np
from pathlib import Path
import os
import random
import json
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def create_unbiased_dataset():
    """
    Creates an unbiased version of the original dataset by removing 
    50% of the datapoints from selected clusters and outputs only 6D
    PCA-reduced versions as NPY files.
    """
    print("Creating unbiased dataset using embedding-based approach...")
    
    # Check if cleaned_resumes.csv exists
    cleaned_file = Path("cleaned_resumes.csv")
    if not cleaned_file.exists():
        print("Error: cleaned_resumes.csv not found. Run embeddings.py first.")
        return
    
    # Check if embeddings file exists
    embeddings_file = Path("resume_embeddings.npy")
    if not embeddings_file.exists():
        print("Error: resume_embeddings.npy not found. Run embeddings.py first.")
        return
    
    # Load all embeddings
    all_embeddings = np.load(embeddings_file)
    print(f"Loaded {len(all_embeddings)} embeddings from resume_embeddings.npy")
    
    # Load the full cleaned dataset
    df_full = pd.read_csv(cleaned_file)
    print(f"Loaded cleaned_resumes.csv with {len(df_full)} rows")
    
    # Check if cluster directories exist
    clusters_dir = Path("clusters")
    if not clusters_dir.exists():
        print("Error: clusters directory not found. Run embeddings.py first.")
        return
    
    # Get the lists of indices that belong to each cluster
    top_clusters = [1, 2, 3]
    cluster_embeddings_map = {}
    
    # Load cluster embeddings from JSON files
    for cluster_num in top_clusters:
        embedding_file = clusters_dir / f"cluster_{cluster_num}_embeddings.json"
        if not embedding_file.exists():
            print(f"Warning: {embedding_file} not found, skipping cluster {cluster_num}")
            continue
        
        try:
            with open(embedding_file, 'r') as f:
                embeddings_data = json.load(f)
            
            print(f"Loaded embeddings for cluster {cluster_num}: {len(embeddings_data['embeddings'])} entries")
            cluster_embeddings_map[cluster_num] = embeddings_data
        except Exception as e:
            print(f"Error loading embeddings for cluster {cluster_num}: {e}")
            continue
    
    # Track indices to remove based on embeddings
    indices_to_remove = []
    removed_counts = {}
    
    # Process each cluster
    for cluster_num, embeddings_data in cluster_embeddings_map.items():
        cluster_embeddings = embeddings_data['embeddings']
        resume_ids = [emb['resume_id'] for emb in cluster_embeddings]
        
        # Get unique resume_ids (in case there are duplicates)
        unique_resume_ids = list(set(resume_ids))
        print(f"Cluster {cluster_num} has {len(unique_resume_ids)} unique resume IDs")
        
        # Randomly select 50% to remove
        random.seed(42)  # For reproducibility
        num_to_remove = len(unique_resume_ids) // 2
        resume_ids_to_remove = random.sample(unique_resume_ids, k=num_to_remove)
        
        # Add to the overall list of indices to remove
        indices_to_remove.extend(resume_ids_to_remove)
        removed_counts[cluster_num] = num_to_remove
        
        print(f"Selected {num_to_remove} entries from cluster {cluster_num} for removal")
    
    # Remove duplicates in case some resumes belong to multiple clusters
    unique_indices_to_remove = list(set(indices_to_remove))
    print(f"Total unique indices to remove: {len(unique_indices_to_remove)}")
    
    # Create a copy of the full dataset
    df_unbiased = df_full.copy()
    
    # Remove the selected indices
    df_unbiased = df_unbiased.drop(unique_indices_to_remove)
    
    # Calculate remaining entries per cluster
    for cluster_num in removed_counts:
        remaining = len(cluster_embeddings_map[cluster_num]['embeddings']) - removed_counts[cluster_num]
        print(f"Removed {removed_counts[cluster_num]} entries from cluster {cluster_num}, {remaining} remain")
    
    # Create output directory if it doesn't exist
    output_dir = Path("unbiased_dataset")
    output_dir.mkdir(exist_ok=True)
    
    # Save the unbiased dataset
    output_file = output_dir / "unbiased_resumes.csv"
    df_unbiased.to_csv(output_file, index=False)
    print(f"Unbiased dataset saved to {output_file}")
    
    # Save the removed entries as a separate file for reference
    removed_df = df_full.loc[unique_indices_to_remove]
    removed_file = output_dir / "removed_entries.csv"
    removed_df.to_csv(removed_file, index=False)
    print(f"Removed entries saved to {removed_file}")
    
    # Create a mask for indices to keep
    keep_mask = np.ones(len(all_embeddings), dtype=bool)
    for idx in unique_indices_to_remove:
        if idx < len(keep_mask):
            keep_mask[idx] = False
    
    # Split the embeddings
    unbiased_embeddings = all_embeddings[keep_mask]
    removed_embeddings = all_embeddings[~keep_mask]
    
    print(f"Split embeddings: {len(unbiased_embeddings)} kept, {len(removed_embeddings)} removed")
    
    # Save the original 384D embeddings
    print("\nSaving original 384D embeddings for the unbiased dataset...")
    unbiased_384d_file = output_dir / "unbiased_embeddings_384d.npy"
    removed_384d_file = output_dir / "removed_embeddings_384d.npy"
    
    np.save(unbiased_384d_file, unbiased_embeddings)
    np.save(removed_384d_file, removed_embeddings)
    
    print(f"Saved 384D embeddings as NPY files:")
    print(f"  - Unbiased embeddings (384D): {unbiased_384d_file}")
    print(f"  - Removed embeddings (384D): {removed_384d_file}")
    
    # Apply PCA directly to the embeddings to get 6D versions
    print("\nApplying PCA to reduce embeddings to 6D...")
    pca = PCA(n_components=6)
    
    # Fit PCA on all embeddings for consistent transformation
    pca.fit(all_embeddings)
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    
    # Transform each set to 6D
    all_6d = pca.transform(all_embeddings)
    unbiased_6d = pca.transform(unbiased_embeddings)
    removed_6d = pca.transform(removed_embeddings)
    
    # Normalize the vectors (L2 normalization)
    def normalize_embeddings(emb):
        norm = np.sqrt(np.sum(emb**2, axis=1)).reshape(-1, 1)
        return emb / norm
    
    all_6d_normalized = normalize_embeddings(all_6d)
    unbiased_6d_normalized = normalize_embeddings(unbiased_6d)
    removed_6d_normalized = normalize_embeddings(removed_6d)
    
    # Save directly as NPY files
    all_6d_file = output_dir / "all_embeddings_6d.npy"
    unbiased_6d_file = output_dir / "unbiased_embeddings_6d.npy"
    removed_6d_file = output_dir / "removed_embeddings_6d.npy"
    
    np.save(all_6d_file, all_6d_normalized)
    np.save(unbiased_6d_file, unbiased_6d_normalized)
    np.save(removed_6d_file, removed_6d_normalized)
    
    print(f"Saved 6D embeddings as NPY files:")
    print(f"  - All embeddings (6D): {all_6d_file}")
    print(f"  - Unbiased embeddings (6D): {unbiased_6d_file}")
    print(f"  - Removed embeddings (6D): {removed_6d_file}")
    
    # Print file sizes
    original_size_mb = os.path.getsize(cleaned_file) / (1024 * 1024)
    unbiased_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    removed_size_mb = os.path.getsize(removed_file) / (1024 * 1024)
    original_emb_size_mb = os.path.getsize(embeddings_file) / (1024 * 1024)
    
    # Get 6D file sizes
    all_6d_size_mb = os.path.getsize(all_6d_file) / (1024 * 1024)
    unbiased_6d_size_mb = os.path.getsize(unbiased_6d_file) / (1024 * 1024)
    removed_6d_size_mb = os.path.getsize(removed_6d_file) / (1024 * 1024)
    
    # Get 384D file sizes
    unbiased_384d_size_mb = os.path.getsize(unbiased_384d_file) / (1024 * 1024)
    removed_384d_size_mb = os.path.getsize(removed_384d_file) / (1024 * 1024)
    
    print(f"\nFile sizes:")
    print(f"Original cleaned_resumes.csv: {original_size_mb:.2f} MB")
    print(f"Unbiased dataset: {unbiased_size_mb:.2f} MB")
    print(f"Removed entries: {removed_size_mb:.2f} MB")
    print(f"Original embeddings (384D): {original_emb_size_mb:.2f} MB")
    print(f"Unbiased embeddings (384D): {unbiased_384d_size_mb:.2f} MB")
    print(f"Removed embeddings (384D): {removed_384d_size_mb:.2f} MB")
    print(f"All embeddings (6D): {all_6d_size_mb:.2f} MB")
    print(f"Unbiased embeddings (6D): {unbiased_6d_size_mb:.2f} MB")
    print(f"Removed embeddings (6D): {removed_6d_size_mb:.2f} MB")
    
    # Save summary statistics
    summary = {
        "original_count": len(df_full),
        "unbiased_count": len(df_unbiased),
        "removed_count": len(unique_indices_to_remove),
        "removal_percentage": (len(unique_indices_to_remove) / len(df_full)) * 100,
        "cluster_removal_counts": removed_counts,
        "file_sizes": {
            "original_mb": original_size_mb,
            "unbiased_mb": unbiased_size_mb,
            "removed_mb": removed_size_mb,
            "original_embeddings_mb": original_emb_size_mb,
            "all_embeddings_6d_mb": all_6d_size_mb,
            "unbiased_embeddings_6d_mb": unbiased_6d_size_mb,
            "removed_embeddings_6d_mb": removed_6d_size_mb,
            "unbiased_embeddings_384d_mb": unbiased_384d_size_mb,
            "removed_embeddings_384d_mb": removed_384d_size_mb
        }
    }
    
    # Save summary as text file
    with open(output_dir / "unbiasing_summary.txt", "w") as f:
        f.write("UNBIASED DATASET SUMMARY (EMBEDDING-BASED)\n")
        f.write("========================================\n\n")
        f.write(f"Original dataset size: {summary['original_count']} entries ({original_size_mb:.2f} MB)\n")
        f.write(f"Unbiased dataset size: {summary['unbiased_count']} entries ({unbiased_size_mb:.2f} MB)\n")
        f.write(f"Removed entries: {summary['removed_count']} entries ({removed_size_mb:.2f} MB)\n")
        f.write(f"Overall removal percentage: {summary['removal_percentage']:.2f}%\n\n")
        
        f.write("Embeddings information:\n")
        f.write(f"  Original embeddings (384D): {len(all_embeddings)} vectors ({original_emb_size_mb:.2f} MB)\n")
        f.write(f"  Unbiased embeddings (384D): {len(unbiased_embeddings)} vectors ({unbiased_384d_size_mb:.2f} MB)\n")
        f.write(f"  Removed embeddings (384D): {len(removed_embeddings)} vectors ({removed_384d_size_mb:.2f} MB)\n")
        f.write(f"  All embeddings (6D): {len(all_6d_normalized)} vectors ({all_6d_size_mb:.2f} MB)\n")
        f.write(f"  Unbiased embeddings (6D): {len(unbiased_6d_normalized)} vectors ({unbiased_6d_size_mb:.2f} MB)\n")
        f.write(f"  Removed embeddings (6D): {len(removed_6d_normalized)} vectors ({removed_6d_size_mb:.2f} MB)\n\n")
        
        f.write("Removal by cluster (using embeddings):\n")
        for cluster_num in top_clusters:
            if cluster_num in removed_counts:
                count = removed_counts[cluster_num]
                original = len(cluster_embeddings_map[cluster_num]['embeddings'])
                remaining = original - count
                percentage = (count / original) * 100
                f.write(f"  Cluster {cluster_num}: Removed {count}/{original} entries ({percentage:.2f}%), {remaining} remain\n")
    
    print(f"Summary saved to {output_dir}/unbiasing_summary.txt")
    
    # Update flask_app.py download paths dynamically
    update_download_paths(output_dir)
    
    return df_unbiased, removed_df

def update_download_paths(output_dir):
    """Update the download paths in flask_app.py to include the new embedding files"""
    try:
        flask_app_path = Path("flask_app.py")
        if not flask_app_path.exists():
            print("Warning: flask_app.py not found, skipping download path updates")
            return
            
        # For simplicity, we'll just print a message suggesting manual update
        print("\nIMPORTANT: You may need to update flask_app.py to include new download paths:")
        print('Add "all_embeddings_6d": "unbiased_dataset/all_embeddings_6d.npy" to file_paths dictionary')
        print('Add "unbiased_embeddings_6d": "unbiased_dataset/unbiased_embeddings_6d.npy" to file_paths dictionary')
        print('Add "removed_embeddings_6d": "unbiased_dataset/removed_embeddings_6d.npy" to file_paths dictionary')
        print('Add "unbiased_embeddings_384d": "unbiased_dataset/unbiased_embeddings_384d.npy" to file_paths dictionary')
        print('Add "removed_embeddings_384d": "unbiased_dataset/removed_embeddings_384d.npy" to file_paths dictionary')
    except Exception as e:
        print(f"Error updating download paths: {e}")

if __name__ == "__main__":
    create_unbiased_dataset()