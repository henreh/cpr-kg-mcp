"""
Data access layer for Climate Policy Radar MCP Server.
Handles DuckDB connections, HuggingFace dataset downloads, and data views.
"""

import os
import logging
from typing import Optional, List, Dict, Any
from pathlib import Path
import json
from datetime import datetime, timedelta

import duckdb
from huggingface_hub import snapshot_download
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class DataAccess:
    """Manages DuckDB connections and HuggingFace dataset access."""
    
    def __init__(
        self,
        cache_dir: str = "./cache",
        dataset_repo: str = "ClimatePolicyRadar/all-document-text-data",
        dataset_revision: str = "main",
        hf_token: Optional[str] = None
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.dataset_repo = dataset_repo
        self.dataset_revision = dataset_revision
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        
        self.conn: Optional[duckdb.DuckDBPyConnection] = None
        self._download_timestamp: Optional[datetime] = None
        
        self._init_database()
    
    def _init_database(self):
        """Initialize DuckDB connection and download dataset if needed."""
        try:
            # Download dataset if not already cached
            self._ensure_dataset_downloaded()
            
            # Initialize DuckDB connection
            self.conn = duckdb.connect(":memory:")
            
            # Create HuggingFace secret if token is available
            if self.hf_token:
                self.conn.execute(
                    f"CREATE SECRET hf_token (TYPE HUGGINGFACE, TOKEN '{self.hf_token}')"
                )
            
            # Create views for the dataset
            parquet_path = self.cache_dir / "*.parquet"
            self.conn.execute(
                f"CREATE VIEW open_data AS SELECT * FROM read_parquet('{parquet_path}')"
            )
            
            # Create English-only view for faster queries
            self.conn.execute("""
                CREATE VIEW open_data_english AS 
                SELECT * FROM open_data 
                WHERE "text_block.language" = 'en'
            """)
            
            # Log dataset statistics
            stats = self.conn.execute(
                "SELECT COUNT(*) as total_rows, COUNT(DISTINCT document_id) as total_docs FROM open_data"
            ).fetchone()
            logger.info(f"Dataset loaded: {stats[0]} rows, {stats[1]} documents")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def _ensure_dataset_downloaded(self):
        """Download HuggingFace dataset if not already cached."""
        marker_file = self.cache_dir / ".download_info.json"
        
        # Check if dataset is already downloaded
        if marker_file.exists():
            with open(marker_file, 'r') as f:
                info = json.load(f)
                if info.get("revision") == self.dataset_revision:
                    self._download_timestamp = datetime.fromisoformat(info["timestamp"])
                    logger.info(f"Using cached dataset from {self._download_timestamp}")
                    return
        
        logger.info(f"Downloading dataset {self.dataset_repo} revision {self.dataset_revision}")
        
        # Download dataset
        snapshot_download(
            repo_id=self.dataset_repo,
            repo_type="dataset",
            local_dir=str(self.cache_dir),
            revision=self.dataset_revision,
            allow_patterns=["*.parquet"],
            token=self.hf_token
        )
        
        # Save download info
        self._download_timestamp = datetime.now()
        with open(marker_file, 'w') as f:
            json.dump({
                "revision": self.dataset_revision,
                "timestamp": self._download_timestamp.isoformat(),
                "repo": self.dataset_repo
            }, f)
        
        logger.info("Dataset download complete")
    
    def execute_query(self, query: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
        """Execute a DuckDB query and return results as list of dicts."""
        if not self.conn:
            raise RuntimeError("Database connection not initialized")
        
        try:
            if params:
                result = self.conn.execute(query, params)
            else:
                result = self.conn.execute(query)
            
            # Get column names
            columns = [desc[0] for desc in result.description]
            
            # Convert to list of dicts
            rows = []
            for row in result.fetchall():
                rows.append(dict(zip(columns, row)))
            
            return rows
        
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise
    
    def get_document_by_slug(self, slug: str) -> Optional[Dict[str, Any]]:
        """Get document metadata by slug."""
        query = """
            SELECT DISTINCT
                document_id,
                "document_metadata.family_title" as family_title,
                "document_metadata.document_title" as document_title,
                "document_metadata.geographies" as geographies,
                "document_metadata.languages" as languages,
                "document_metadata.corpus_type_name" as corpus_type_name,
                "document_metadata.slug" as slug
            FROM open_data
            WHERE "document_metadata.slug" = ?
            LIMIT 1
        """
        
        results = self.execute_query(query, (slug,))
        return results[0] if results else None
    
    def get_passages_by_slug(self, slug: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get ordered passages for a document."""
        query = """
            SELECT 
                "text_block.index" as index,
                "text_block.page_number" as page_number,
                "text_block.type" as type,
                "text_block.text" as text,
                "text_block.language" as language
            FROM open_data
            WHERE "document_metadata.slug" = ?
            ORDER BY "text_block.index" ASC
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        return self.execute_query(query, (slug,))
    
    def search_documents(
        self,
        text_query: Optional[str] = None,
        iso3_filter: Optional[List[str]] = None,
        language_filter: Optional[List[str]] = None,
        corpus_type_filter: Optional[List[str]] = None,
        family_slug_filter: Optional[List[str]] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Search documents with filters."""
        
        # Build base query
        conditions = []
        params = []
        
        # Use English view if language filter is English only
        if language_filter == ["en"]:
            base_table = "open_data_english"
        else:
            base_table = "open_data"
            if language_filter:
                lang_conditions = " OR ".join(["\"text_block.language\" = ?" for _ in language_filter])
                conditions.append(f"({lang_conditions})")
                params.extend(language_filter)
        
        # Add text search
        if text_query:
            conditions.append("\"text_block.text\" ILIKE ?")
            params.append(f"%{text_query}%")
        
        # Add ISO3 filter
        if iso3_filter:
            iso_conditions = []
            for iso in iso3_filter:
                iso_conditions.append("? = ANY(\"document_metadata.geographies\")")
                params.append(iso)
            conditions.append(f"({' OR '.join(iso_conditions)})")
        
        # Add corpus type filter
        if corpus_type_filter:
            corpus_conditions = " OR ".join(["\"document_metadata.corpus_type_name\" = ?" for _ in corpus_type_filter])
            conditions.append(f"({corpus_conditions})")
            params.extend(corpus_type_filter)
        
        # Add family slug filter
        if family_slug_filter:
            family_conditions = " OR ".join(["\"document_metadata.slug\" LIKE ?" for _ in family_slug_filter])
            conditions.append(f"({family_conditions})")
            params.extend([f"{slug}%" for slug in family_slug_filter])
        
        # Build final query
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        query = f"""
            SELECT 
                "document_metadata.slug" as slug,
                document_id as document_id,
                "document_metadata.document_title" as document_title,
                "document_metadata.family_title" as family_title,
                "document_metadata.geographies" as geographies,
                "document_metadata.languages" as languages,
                "document_metadata.corpus_type_name" as corpus_type_name,
                "text_block.index" as text_index,
                "text_block.page_number" as page_number,
                "text_block.type" as text_type,
                "text_block.text" as text,
                "text_block.language" as text_language
            FROM {base_table}
            WHERE {where_clause}
            ORDER BY "document_metadata.family_title", "document_metadata.document_title", "text_block.index"
            LIMIT ? OFFSET ?
        """
        
        params.extend([limit, offset])
        return self.execute_query(query, tuple(params))
    
    def get_jurisdiction_stats(self, iso3: str) -> Dict[str, Any]:
        """Get statistics for a jurisdiction."""
        
        # Documents by corpus type
        corpus_query = """
            SELECT 
                "document_metadata.corpus_type_name" as corpus_type,
                COUNT(DISTINCT document_id) as count
            FROM open_data
            WHERE ? = ANY("document_metadata.geographies")
            GROUP BY "document_metadata.corpus_type_name"
            ORDER BY count DESC
        """
        
        corpus_stats = self.execute_query(corpus_query, (iso3,))
        
        # Top languages
        lang_query = """
            SELECT 
                "text_block.language" as language,
                COUNT(*) as count
            FROM open_data
            WHERE ? = ANY("document_metadata.geographies")
            GROUP BY "text_block.language"
            ORDER BY count DESC
            LIMIT 10
        """
        
        lang_stats = self.execute_query(lang_query, (iso3,))
        
        return {
            "docs_by_corpus_type": {row["corpus_type"]: row["count"] for row in corpus_stats if row["corpus_type"]},
            "top_languages": {row["language"]: row["count"] for row in lang_stats if row["language"]}
        }
    
    def get_families_overview(
        self, 
        iso3_filter: Optional[List[str]] = None,
        language_filter: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Get overview of document families."""
        
        conditions = []
        params = []
        
        if iso3_filter:
            iso_conditions = []
            for iso in iso3_filter:
                iso_conditions.append("? = ANY(\"document_metadata.geographies\")")
                params.append(iso)
            conditions.append(f"({' OR '.join(iso_conditions)})")
        
        if language_filter:
            lang_conditions = " OR ".join(["? = ANY(\"document_metadata.languages\")" for _ in language_filter])
            conditions.append(f"({lang_conditions})")
            params.extend(language_filter)
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        query = f"""
            SELECT 
                "document_metadata.family_title" as family_title,
                COUNT(DISTINCT document_id) as doc_count,
                MIN("document_metadata.slug") as sample_slug,
                MIN("document_metadata.document_title") as sample_title
            FROM open_data
            WHERE {where_clause}
            GROUP BY "document_metadata.family_title"
            ORDER BY doc_count DESC, family_title
        """
        
        return self.execute_query(query, tuple(params))
    
    def get_cache_age(self) -> int:
        """Get age of cached data in minutes."""
        if not self._download_timestamp:
            return -1
        
        age = datetime.now() - self._download_timestamp
        return int(age.total_seconds() / 60)
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None