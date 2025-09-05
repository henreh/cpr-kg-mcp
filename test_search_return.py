#!/usr/bin/env python3
"""Test that search returns proper results."""

import sys
from pathlib import Path
import json

# Add server module to path
sys.path.insert(0, str(Path(__file__).parent))

from server.data_access import DataAccess
from server.concepts import ConceptStore  
from server.search import SearchEngine
from server.schemas import SearchFilters
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

def main():
    print("Testing search return values...")
    
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
    
    # Test search with the same function signature as the tool
    print("\nTesting search with 'climate change mitigation'...")
    results, total = search_engine.search_passages(
        query="climate change mitigation",
        filters=None,
        limit=3,
        offset=0
    )
    
    print(f"Found {total} total results, returning {len(results)} results")
    
    # Test converting to dict (like the tool does)
    print("\nConverting results to dicts:")
    dict_results = [result.model_dump() for result in results]
    
    # Print as JSON to see exact structure
    print("\nJSON representation:")
    print(json.dumps(dict_results, indent=2))
    
    # Close connection
    data_access.close()
    print("\nTest complete!")

if __name__ == "__main__":
    main()