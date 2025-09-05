#!/usr/bin/env python3
"""Test that search handles null values properly."""

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
    print("Testing null value handling in search...")
    
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
    
    # Test search with a query that might return documents with null titles
    print("\nSearching for 'climate change mitigation policies'...")
    try:
        results, total = search_engine.search_passages(
            query="climate change mitigation policies",
            filters=None,
            limit=5,
            offset=0
        )
        
        print(f"✓ Found {total} total results, returning {len(results)} results")
        
        # Check each result for proper title handling
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Title: {result.title}")
            print(f"   ID: {result.id}")
            print(f"   Score: {result.score:.2f}")
            
            # Verify title is not None
            assert result.title is not None, "Title should never be None"
            assert isinstance(result.title, str), "Title should be a string"
            
        print("\n✓ All results have valid titles (no None values)")
        
    except Exception as e:
        print(f"✗ Error during search: {e}")
        return 1
    
    # Close connection
    data_access.close()
    print("\nTest complete - null handling working correctly!")
    return 0

if __name__ == "__main__":
    sys.exit(main())