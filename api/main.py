import os
import logging
import pandas as pd
from pathlib import Path
import time
import sys
import argparse
import shutil

# Import functions from our other files
from embeddings import main as embeddings_main
from cluster_analysis import main as cluster_analysis_main
from create_unbiased_dataset import create_unbiased_dataset

# Setup argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="Unbiasing pipeline")
    parser.add_argument("--input", type=str, help="Path to input CSV file")
    parser.add_argument("--job_id", type=str, help="Job ID for this processing task")
    return parser.parse_args()

# Setup logging
def setup_logging(job_id=None):
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    if job_id:
        # Use job_id for the log filename
        log_file = log_dir / f"unbiasing_pipeline_{job_id}.log"
        # Also create a symbolic link to the job directory
        job_dir = Path("uploads") / job_id
        job_log_file = job_dir / "pipeline.log"
    else:
        log_file = log_dir / f"unbiasing_pipeline_{timestamp}.log"
    
    # Configure logging to write to both file and console
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logging.info(f"Starting unbiasing pipeline. Logs will be saved to {log_file}")
    return log_file

def count_files_in_dir(directory):
    """Count files in a directory and report their sizes"""
    if not os.path.exists(directory):
        return "Directory does not exist"
    
    result = []
    total_size_mb = 0
    
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        if os.path.isfile(file_path):
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            total_size_mb += size_mb
            result.append(f"{file} ({size_mb:.2f} MB)")
    
    return f"{len(result)} files, total size: {total_size_mb:.2f} MB\n" + "\n".join(result)

def report_file_info(file_path):
    """Report information about a file"""
    if not os.path.exists(file_path):
        return f"{file_path} does not exist"
    
    size_mb = os.path.getsize(file_path) / (1024 * 1024)
    
    if file_path.endswith('.csv'):
        try:
            df = pd.read_csv(file_path)
            return f"{file_path}: {len(df)} rows, {len(df.columns)} columns, {size_mb:.2f} MB"
        except Exception as e:
            return f"{file_path}: {size_mb:.2f} MB (Error reading CSV: {e})"
    else:
        return f"{file_path}: {size_mb:.2f} MB"

def main():
    """Main function to execute the entire unbiasing pipeline"""
    # Parse command-line arguments
    args = parse_args()
    
    # Setup logging
    log_file = setup_logging(args.job_id)
    
    try:
        # Set the input file path
        input_file = args.input
        job_id = args.job_id
        
        if input_file:
            logging.info(f"Using custom input file: {input_file}")
        else:
            logging.info("No input file specified, will use default dataset")
        
        # Create job completion marker if job_id is provided
        job_dir = None
        if job_id:
            job_dir = Path("uploads") / job_id
            
        # Step 1: Generate embeddings and find clusters
        logging.info("=" * 80)
        logging.info("STEP 1: GENERATING EMBEDDINGS AND FINDING CLUSTERS")
        logging.info("=" * 80)
        
        df, embeddings = embeddings_main(input_file)
        
        if df is None or embeddings is None:
            logging.error("Failed to generate embeddings. Exiting.")
            if job_dir:
                (job_dir / "failed").touch()
            return
        
        logging.info(f"Embeddings shape: {embeddings.shape}")
        logging.info(f"Generated cleaned_resumes.csv: {report_file_info('cleaned_resumes.csv')}")
        logging.info(f"Generated resume_embeddings.npy: {report_file_info('resume_embeddings.npy')}")
        
        # Report on clusters
        clusters_dir = Path("clusters")
        if clusters_dir.exists():
            logging.info(f"Clusters directory contents:\n{count_files_in_dir('clusters')}")
        
        # Step 2: Analyze clusters using Cohere
        logging.info("\n" + "=" * 80)
        logging.info("STEP 2: ANALYZING CLUSTERS USING COHERE")
        logging.info("=" * 80)
        
        cluster_analysis_main()
        
        # Report on analysis results
        analysis_dir = Path("cluster_analysis")
        if analysis_dir.exists():
            logging.info(f"Cluster analysis directory contents:\n{count_files_in_dir('cluster_analysis')}")
        
        # Step 3: Create unbiased dataset
        logging.info("\n" + "=" * 80)
        logging.info("STEP 3: CREATING UNBIASED DATASET")
        logging.info("=" * 80)
        
        unbiased_df, removed_df = create_unbiased_dataset()
        
        # Report on unbiased dataset
        unbiased_dir = Path("unbiased_dataset")
        if unbiased_dir.exists():
            logging.info(f"Unbiased dataset directory contents:\n{count_files_in_dir('unbiased_dataset')}")
        
        # Final summary
        logging.info("\n" + "=" * 80)
        logging.info("PIPELINE COMPLETE - SUMMARY")
        logging.info("=" * 80)
        
        logging.info(f"Original dataset: {report_file_info('cleaned_resumes.csv')}")
        logging.info(f"Unbiased dataset: {report_file_info('unbiased_dataset/unbiased_resumes.csv')}")
        logging.info(f"Removed entries: {report_file_info('unbiased_dataset/removed_entries.csv')}")
        
        if os.path.exists('unbiased_dataset/unbiasing_summary.txt'):
            with open('unbiased_dataset/unbiasing_summary.txt', 'r') as f:
                summary = f.read()
                logging.info(f"Unbiasing summary:\n{summary}")
        
        logging.info("Unbiasing pipeline completed successfully!")
        
        # Create completion marker if job_id is provided
        if job_dir:
            (job_dir / "completed").touch()
        
    except Exception as e:
        logging.error(f"Error in pipeline: {str(e)}", exc_info=True)
        # Create failure marker if job_id is provided
        if job_id and job_dir:
            (job_dir / "failed").touch()
    
    logging.info(f"Complete log available at: {log_file}")

if __name__ == "__main__":
    main() 