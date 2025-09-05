#!/usr/bin/env python3
"""Test script to download the dataset and create DuckDB."""

import os
import sys
from pathlib import Path

# Add server module to path
sys.path.insert(0, str(Path(__file__).parent))

from server.data_access import DataAccess
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    print("Testing data download and DuckDB creation...")
    
    # Get configuration from environment
    cache_dir = os.getenv("DATA_CACHE_DIR", "./cache")
    dataset_repo = os.getenv("DATASET_REPO", "ClimatePolicyRadar/all-document-text-data")
    dataset_revision = os.getenv("DATASET_REVISION", "main")
    hf_token = os.getenv("HF_TOKEN")
    
    print(f"Cache directory: {cache_dir}")
    print(f"Dataset repo: {dataset_repo}")
    print(f"Dataset revision: {dataset_revision}")
    print(f"HF Token present: {'Yes' if hf_token else 'No'}")
    
    # Initialize DataAccess which will trigger download
    print("\nInitializing DataAccess (this will download the dataset)...")
    try:
        data_access = DataAccess(
            cache_dir=cache_dir,
            dataset_repo=dataset_repo,
            dataset_revision=dataset_revision,
            hf_token=hf_token
        )
        print("DataAccess initialized successfully!")
        
        # Test a simple query
        print("\nTesting database query...")
        result = data_access.execute_query("SELECT COUNT(*) as count FROM open_data LIMIT 1")
        print(f"Total rows in dataset: {result[0]['count']}")
        
        # Close the connection
        data_access.close()
        print("\nDatabase connection closed.")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())