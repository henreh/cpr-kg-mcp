#!/usr/bin/env python3
"""Test the search functionality."""

import sys
from pathlib import Path

# Add server module to path
sys.path.insert(0, str(Path(__file__).parent))

from server.data_access import DataAccess
from server.concepts import ConceptStore  
from server.search import SearchEngine
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

def main():
    print("Testing search functionality...")
    
    # Initialize components
    data_access = DataAccess(
        cache_dir=os.getenv("DATA_CACHE_DIR", "./cache"),
        dataset_repo=os.getenv("DATASET_REPO"),
        dataset_revision=os.getenv("DATASET_REVISION"),
        hf_token=os.getenv("HF_TOKEN")
    )
    
    concept_store = ConceptStore(
        sparql_endpoint=os.getenv("SPARQL_ENDPOINT"),
        cache_dir=os.getenv("DATA_CACHE_DIR", "./cache")
    )
    
    search_engine = SearchEngine(data_access, concept_store)
    
    # Test search
    print("\nSearching for 'climate change mitigation'...")
    results, total = search_engine.search_passages(
        query="climate change mitigation",
        limit=5
    )
    
    print(f"Found {total} total results, showing top 5:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result.title}")
        print(f"   Score: {result.score:.2f}")
        print(f"   Snippet: {result.snippet[:100]}...")
    
    # Close connection
    data_access.close()
    print("\nTest complete!")

if __name__ == "__main__":
    main()