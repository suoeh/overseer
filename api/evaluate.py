import numpy as np
import argparse
from sklearn.cluster import KMeans
from collections import Counter
import matplotlib.pyplot as plt
from pathlib import Path
import time

def main():
    parser = argparse.ArgumentParser(description='Evaluate embeddings using K-means clustering.')
    # parser.add_argument('--embeddings_path', type=str, default='./unbiased_dataset/unbiased_embeddings_384d.npy',
                        # help='Path to the 384D embeddings numpy array')
    parser.add_argument('--embeddings_path', type=str, default='./resume_embeddings.npy',
                        help='Path to the 384D embeddings numpy array')
    parser.add_argument('--n_clusters', type=int, default=20,
                        help='Number of clusters to form')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed for K-means')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                        help='Directory to save evaluation results')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"Loading embeddings from {args.embeddings_path}...")
    embeddings = np.load(args.embeddings_path)
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Perform K-means clustering
    print(f"\nPerforming K-means clustering with {args.n_clusters} clusters...")
    start_time = time.time()
    kmeans = KMeans(n_clusters=args.n_clusters, random_state=args.random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    elapsed_time = time.time() - start_time
    print(f"K-means clustering completed in {elapsed_time:.2f} seconds")
    
    # Count the number of points in each cluster
    cluster_counts = Counter(cluster_labels)
    total_points = len(cluster_labels)
    
    # Find the 10 largest clusters
    largest_clusters = cluster_counts.most_common(10)
    
    print("\n=== 10 Largest Clusters ===")
    print("Rank | Cluster ID | Size | Percentage")
    print("-" * 40)
    for i, (cluster_id, count) in enumerate(largest_clusters, 1):
        percentage = (count / total_points) * 100
        print(f"{i:4d} | {cluster_id:10d} | {count:4d} | {percentage:6.2f}%")
    
    # Calculate and display statistics
    cluster_sizes = list(cluster_counts.values())
    min_size = min(cluster_sizes)
    max_size = max(cluster_sizes)
    avg_size = total_points / args.n_clusters
    median_size = np.median(cluster_sizes)
    
    print("\n=== Clustering Statistics ===")
    print(f"Total number of points: {total_points}")
    print(f"Number of clusters: {args.n_clusters}")
    print(f"Smallest cluster size: {min_size}")
    print(f"Largest cluster size: {max_size}")
    print(f"Average cluster size: {avg_size:.2f}")
    print(f"Median cluster size: {median_size:.2f}")
    
    # Calculate inertia (sum of squared distances to closest centroid)
    print(f"Inertia (sum of squared distances): {kmeans.inertia_:.2f}")
    
    # Plot cluster size distribution
    plt.figure(figsize=(10, 6))
    plt.hist(cluster_sizes, bins=20)
    plt.title('Distribution of Cluster Sizes')
    plt.xlabel('Cluster Size')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / 'cluster_size_distribution.png')
    print(f"\nSaved cluster size distribution plot to {output_dir / 'cluster_size_distribution.png'}")
    
    # Save cluster assignments
    cluster_assignment_path = output_dir / 'cluster_assignments.npy'
    np.save(cluster_assignment_path, cluster_labels)
    print(f"Saved cluster assignments to {cluster_assignment_path}")
    
    # Save cluster statistics as CSV
    with open(output_dir / 'cluster_statistics.csv', 'w') as f:
        f.write("cluster_id,size,percentage\n")
        for cluster_id, count in cluster_counts.items():
            percentage = (count / total_points) * 100
            f.write(f"{cluster_id},{count},{percentage:.2f}\n")
    print(f"Saved cluster statistics to {output_dir / 'cluster_statistics.csv'}")

if __name__ == "__main__":
    main()
