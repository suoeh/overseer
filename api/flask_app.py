from flask import Flask, jsonify, request, send_file
import pandas as pd
import os
import json
import subprocess
from pathlib import Path
from flask_cors import CORS  # Import CORS
import uuid
import shutil
import numpy as np

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})  # Enable CORS with more specific configuration

# Create uploads directory if it doesn't exist
UPLOAD_FOLDER = Path("uploads")
UPLOAD_FOLDER.mkdir(exist_ok=True)

@app.route('/')
def index():
    return "Bias Unbiasing API"

@app.route('/api/health', methods=['GET'])
def health_check():
    """Basic health check endpoint"""
    return jsonify({"status": "healthy", "message": "API is running"})

@app.route('/api/datasets/available', methods=['GET'])
def available_datasets():
    """Return information about which datasets are available"""
    available = {
        "cleaned_resumes": os.path.exists("cleaned_resumes.csv"),
        "unbiased_resumes": os.path.exists("unbiased_dataset/unbiased_resumes.csv"),
        "removed_entries": os.path.exists("unbiased_dataset/removed_entries.csv"),
        "all_clusters": os.path.exists("clusters/all_clusters.csv"),
        "cluster_analysis": os.path.exists("cluster_analysis"),
    }
    
    # Check for individual cluster files
    available["individual_clusters"] = []
    if os.path.exists("clusters"):
        cluster_files = list(Path("clusters").glob("cluster_*.csv"))
        available["individual_clusters"] = [f.name for f in cluster_files]
    
    return jsonify(available)

@app.route('/api/cleaned_resumes', methods=['GET'])
def get_cleaned_resumes():
    """Return the cleaned resumes dataset with pagination"""
    file_path = "cleaned_resumes.csv"
    if not os.path.exists(file_path):
        return jsonify({"error": "Cleaned resumes dataset not found"}), 404
    
    return get_paginated_dataset(file_path, "cleaned_resumes")

@app.route('/api/unbiased_resumes', methods=['GET'])
def get_unbiased_resumes():
    """Return the unbiased resumes dataset with pagination"""
    file_path = "unbiased_dataset/unbiased_resumes.csv"
    if not os.path.exists(file_path):
        return jsonify({"error": "Unbiased resumes dataset not found"}), 404
    
    return get_paginated_dataset(file_path, "unbiased_resumes")

@app.route('/api/removed_entries', methods=['GET'])
def get_removed_entries():
    """Return the removed entries dataset with pagination"""
    file_path = "unbiased_dataset/removed_entries.csv"
    if not os.path.exists(file_path):
        return jsonify({"error": "Removed entries dataset not found"}), 404
    
    return get_paginated_dataset(file_path, "removed_entries")

@app.route('/api/all_clusters', methods=['GET'])
def get_all_clusters():
    """Return the all clusters dataset with pagination"""
    file_path = "clusters/all_clusters.csv"
    if not os.path.exists(file_path):
        return jsonify({"error": "All clusters dataset not found"}), 404
    
    return get_paginated_dataset(file_path, "all_clusters")

@app.route('/api/unbiased_embeddings_data', methods=['GET'])
def get_unbiased_embeddings_data():
    """Return the unbiased embeddings as JSON"""
    file_path = "unbiased_dataset/unbiased_embeddings_6d.npy"
    if not os.path.exists(file_path):
        return jsonify({"error": "Unbiased embeddings not found"}), 404
    
    try:
        # Load the numpy array
        embeddings = np.load(file_path)
        
        # Convert to list for JSON serialization
        embeddings_list = embeddings.tolist()
        
        # Return the full embeddings data
        return jsonify({
            "success": True,
            "embeddings": embeddings_list,
            "shape": embeddings.shape,
            "dimensions": embeddings.shape[1] if len(embeddings.shape) > 1 else 0,
            "count": embeddings.shape[0]
        })
    except Exception as e:
        return jsonify({"error": f"Error processing embeddings: {str(e)}"}), 500

@app.route('/api/removed_embeddings_data', methods=['GET'])
def get_removed_embeddings_data():
    """Return the removed embeddings as JSON"""
    file_path = "unbiased_dataset/removed_embeddings_6d.npy"
    if not os.path.exists(file_path):
        return jsonify({"error": "Removed embeddings not found"}), 404
    
    try:
        # Load the numpy array
        embeddings = np.load(file_path)
        
        # Convert to list for JSON serialization
        embeddings_list = embeddings.tolist()
        
        # Return the full embeddings data
        return jsonify({
            "success": True,
            "embeddings": embeddings_list,
            "shape": embeddings.shape,
            "dimensions": embeddings.shape[1] if len(embeddings.shape) > 1 else 0,
            "count": embeddings.shape[0]
        })
    except Exception as e:
        return jsonify({"error": f"Error processing embeddings: {str(e)}"}), 500

def get_paginated_dataset(file_path, dataset_name):
    """Helper function to return a paginated dataset"""
    # Get pagination parameters
    page = request.args.get('page', default=1, type=int)
    page_size = request.args.get('page_size', default=100, type=int)
    
    # Limit page_size to prevent huge responses
    page_size = min(page_size, 1000)
    
    try:
        # Load the dataset
        df = pd.read_csv(file_path)
        
        # Get total records and pages
        total_records = len(df)
        total_pages = (total_records + page_size - 1) // page_size
        
        # Validate page number
        if page < 1:
            page = 1
        if page > total_pages and total_pages > 0:
            page = total_pages
        
        # Calculate start and end indices
        start_idx = (page - 1) * page_size
        end_idx = min(start_idx + page_size, total_records)
        
        # Get the page of data
        records = df.iloc[start_idx:end_idx].to_dict('records')
        
        # Return the paginated response
        return jsonify({
            "dataset": dataset_name,
            "total_records": total_records,
            "total_pages": total_pages,
            "current_page": page,
            "page_size": page_size,
            "records": records
        })
    
    except Exception as e:
        return jsonify({"error": f"Error reading dataset: {str(e)}"}), 500

@app.route('/api/clusters', methods=['GET'])
def get_clusters():
    """Return embeddings for all clusters"""
    # First check if clusters directory exists
    clusters_dir = Path("clusters")
    if not clusters_dir.exists():
        return jsonify({"error": "Clusters directory not found"}), 404
    
    # Look for 6D embedding files instead of CSV files
    embedding_files = list(clusters_dir.glob("cluster_*_embeddings_6d.npy"))
    
    # If no embedding files found, try looking in unbiased_dataset directory
    if not embedding_files:
        embedding_files = list(Path("unbiased_dataset").glob("cluster_*_embeddings_6d.npy"))
    
    if not embedding_files:
        return jsonify({"error": "No cluster embedding files found"}), 404
    
    try:
        all_clusters = {}
        
        for embedding_file in embedding_files:
            # Extract cluster number from filename (cluster_1_embeddings_6d.npy -> 1)
            file_name = embedding_file.stem
            parts = file_name.split("_")
            if len(parts) >= 2:
                cluster_id = int(parts[1])
            else:
                continue  # Skip if filename format is unexpected
                
            # Load the embeddings
            try:
                embeddings = np.load(embedding_file)
                
                # Convert to list for JSON serialization
                embeddings_list = embeddings.tolist()
                
                # Store metadata about the embeddings
                all_clusters[f"cluster_{cluster_id}"] = {
                    "count": len(embeddings),
                    "dimensions": embeddings.shape[1] if len(embeddings.shape) > 1 else 0,
                    "embeddings": embeddings_list
                }
            except Exception as e:
                print(f"Error loading embeddings for cluster {cluster_id}: {str(e)}")
                continue
        
        if not all_clusters:
            return jsonify({"error": "Failed to load any cluster embeddings"}), 500
            
        return jsonify({
            "total_clusters": len(all_clusters),
            "clusters": all_clusters
        })
    
    except Exception as e:
        return jsonify({"error": f"Error reading cluster embeddings: {str(e)}"}), 500

@app.route('/api/clusters/<cluster_id>', methods=['GET'])
def get_cluster(cluster_id):
    """Return a specific cluster by ID"""
    try:
        cluster_id = int(cluster_id)
        file_path = f"clusters/cluster_{cluster_id}.csv"
        
        if not os.path.exists(file_path):
            return jsonify({"error": f"Cluster {cluster_id} not found"}), 404
        
        return get_paginated_dataset(file_path, f"cluster_{cluster_id}")
    
    except ValueError:
        return jsonify({"error": "Cluster ID must be a number"}), 400
    except Exception as e:
        return jsonify({"error": f"Error reading cluster: {str(e)}"}), 500

@app.route('/api/analysis/clusters', methods=['GET'])
def get_all_cluster_analyses():
    """Return all cluster analyses"""
    file_path = "cluster_analysis/all_clusters_analysis.json"
    if not os.path.exists(file_path):
        return jsonify({"error": "Cluster analyses not found"}), 404
    
    try:
        with open(file_path, 'r') as f:
            analyses = json.load(f)
        
        return jsonify(analyses)
    
    except Exception as e:
        return jsonify({"error": f"Error reading analyses: {str(e)}"}), 500

@app.route('/api/analysis/clusters/<cluster_id>', methods=['GET'])
def get_cluster_analysis(cluster_id):
    """Return the analysis for a specific cluster"""
    try:
        cluster_id = int(cluster_id)
        file_path = f"cluster_analysis/cluster_{cluster_id}_analysis.txt"
        
        if not os.path.exists(file_path):
            return jsonify({"error": f"Analysis for cluster {cluster_id} not found"}), 404
        
        with open(file_path, 'r') as f:
            analysis = f.read()
        
        return jsonify({
            "cluster_id": cluster_id,
            "analysis": analysis
        })
    
    except ValueError:
        return jsonify({"error": "Cluster ID must be a number"}), 400
    except Exception as e:
        return jsonify({"error": f"Error reading analysis: {str(e)}"}), 500

@app.route('/api/summary', methods=['GET'])
def get_unbiasing_summary():
    """Return the unbiasing summary if it exists"""
    file_path = "unbiased_dataset/unbiasing_summary.txt"
    
    if not os.path.exists(file_path):
        return jsonify({"error": "Unbiasing summary not found"}), 404
    
    try:
        with open(file_path, 'r') as f:
            summary = f.read()
        
        return jsonify({
            "summary": summary
        })
    
    except Exception as e:
        return jsonify({"error": f"Error reading summary: {str(e)}"}), 500

@app.route('/api/download/<file_type>', methods=['GET'])
def download_file(file_type):
    """Direct file download endpoint for various outputs"""
    # Define file paths for different file types
    file_paths = {
        "cleaned_resumes": "cleaned_resumes.csv",
        "unbiased_resumes": "unbiased_dataset/unbiased_resumes.csv",
        "removed_entries": "unbiased_dataset/removed_entries.csv",
        "all_clusters": "clusters/all_clusters.csv",
        "embeddings": "resume_embeddings.npy",
        "summary": "unbiased_dataset/unbiasing_summary.txt"
    }
    
    # Check if the requested file type exists
    if file_type not in file_paths:
        return jsonify({"error": f"Unknown file type: {file_type}"}), 404
    
    file_path = file_paths[file_type]
    if not os.path.exists(file_path):
        return jsonify({"error": f"File {file_type} not found"}), 404
    
    try:
        return send_file(file_path, as_attachment=True)
    except Exception as e:
        return jsonify({"error": f"Error downloading file: {str(e)}"}), 500

# Add file upload endpoint
@app.route('/api/upload', methods=['POST'])
def upload_file():
    """
    Upload a CSV file to use as the dataset for the analysis pipeline
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
        
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
        
    if not file.filename.endswith('.csv'):
        return jsonify({"error": "Only CSV files are supported"}), 400
    
    # Create a unique ID for this upload
    upload_id = str(uuid.uuid4())
    upload_dir = UPLOAD_FOLDER / upload_id
    upload_dir.mkdir(exist_ok=True)
    
    # Save the uploaded file
    file_path = upload_dir / "Resume.csv"
    file.save(file_path)
    
    try:
        # Validate the CSV file
        df = pd.read_csv(file_path)
        if 'Resume_str' not in df.columns:
            return jsonify({
                "error": "Invalid CSV format. The file must contain a 'Resume_str' column."
            }), 400
            
        rows_count = len(df)
        
        # Run the pipeline asynchronously
        # This will be a non-blocking call
        subprocess.Popen(
            ["python", "./main.py", "--input", str(file_path), "--job_id", upload_id],
            # Redirect output to a log file
            stdout=open(upload_dir / "pipeline.log", "w"),
            stderr=subprocess.STDOUT
        )
        
        return jsonify({
            "message": "File uploaded successfully. Processing started.",
            "job_id": upload_id,
            "rows_count": rows_count,
            "status": "processing"
        })
        
    except Exception as e:
        # Clean up on error
        shutil.rmtree(upload_dir, ignore_errors=True)
        return jsonify({"error": f"Error processing file: {str(e)}"}), 500

@app.route('/api/jobs/<job_id>/status', methods=['GET'])
def get_job_status(job_id):
    """Check the status of a processing job"""
    job_dir = UPLOAD_FOLDER / job_id
    
    if not job_dir.exists():
        return jsonify({"error": "Job not found"}), 404
    
    # Check for completion indicators
    completed = (job_dir / "completed").exists()
    failed = (job_dir / "failed").exists()
    
    # Get log contents if available
    log_path = job_dir / "pipeline.log"
    log_content = ""
    if log_path.exists():
        with open(log_path, "r") as f:
            log_content = f.read()
    
    if completed:
        return jsonify({
            "job_id": job_id,
            "status": "completed",
            "log": log_content
        })
    elif failed:
        return jsonify({
            "job_id": job_id,
            "status": "failed",
            "log": log_content
        })
    else:
        return jsonify({
            "job_id": job_id,
            "status": "processing",
            "log": log_content
        })

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=3002) 