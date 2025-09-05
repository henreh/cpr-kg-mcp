"""
Basic tests for Climate Policy Radar MCP Server.
Tests core functionality and Deep-Research compatibility.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from server.schemas import (
    SearchInput, SearchResult, FetchInput, Evidence,
    ConceptInfo, ConceptFindInput, SearchFilters
)
from server.search import SearchEngine
from server.concepts import ConceptStore


class TestSchemas:
    """Test Pydantic schemas."""
    
    def test_search_input_validation(self):
        """Test SearchInput validation."""
        # Valid input
        search = SearchInput(query="climate change", limit=10)
        assert search.query == "climate change"
        assert search.limit == 10
        
        # With filters
        filters = SearchFilters(iso3=["BRA", "ZAF"], language=["en"])
        search_with_filters = SearchInput(query="solar", filters=filters)
        assert search_with_filters.filters.iso3 == ["BRA", "ZAF"]
    
    def test_evidence_structure(self):
        """Test Evidence object structure."""
        evidence = Evidence(
            id="cpr://doc/test-slug#p=123",
            document={
                "slug": "test-slug",
                "document_id": "doc-123",
                "title": "Test Document",
                "family_title": "Test Family",
                "geographies": ["BRA"],
                "languages": ["en"],
                "url": "https://app.climatepolicyradar.org/documents/test-slug"
            },
            passage={
                "index": 123,
                "page_number": 5,
                "type": "body",
                "text": "This is test passage text."
            },
            concept_matches=[],
            score=0.5
        )
        
        assert evidence.id == "cpr://doc/test-slug#p=123"
        assert evidence.document.slug == "test-slug"
        assert evidence.passage.index == 123
    
    def test_concept_info_structure(self):
        """Test ConceptInfo structure."""
        concept = ConceptInfo(
            qid="Q1167",
            preferred_label="Solar energy",
            aliases=["Solar power", "Photovoltaic"],
            description="Energy from the sun"
        )
        
        assert concept.qid == "Q1167"
        assert len(concept.aliases) == 2


class TestSearchEngine:
    """Test search engine functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_data_access = MagicMock()
        self.mock_concept_store = MagicMock()
        self.search_engine = SearchEngine(self.mock_data_access, self.mock_concept_store)
    
    def test_tokenize(self):
        """Test text tokenization."""
        text = "Climate Change Policy 2024"
        tokens = self.search_engine.tokenize(text)
        assert tokens == ["climate", "change", "policy", "2024"]
    
    def test_bm25_scoring(self):
        """Test BM25 scoring calculation."""
        query_terms = ["climate", "policy"]
        document_text = "This document discusses climate policy and climate change mitigation."
        
        score = self.search_engine.calculate_bm25_score(
            query_terms,
            document_text,
            avg_doc_length=100,
            total_docs=1000
        )
        
        assert score > 0
        assert isinstance(score, float)
    
    def test_fetch_evidence_parsing(self):
        """Test resource ID parsing in fetch_evidence."""
        # Mock data access responses
        self.mock_data_access.get_document_by_slug.return_value = {
            "document_id": "doc-123",
            "document_title": "Test Doc",
            "family_title": "Test Family",
            "geographies": ["BRA"],
            "languages": ["en"],
            "slug": "test-slug"
        }
        
        self.mock_data_access.get_passages_by_slug.return_value = [
            {
                "index": 42,
                "page_number": 3,
                "type": "body",
                "text": "Test passage text",
                "language": "en"
            }
        ]
        
        # Test fetching evidence
        evidence_list = self.search_engine.fetch_evidence(["cpr://doc/test-slug#p=42"])
        
        assert len(evidence_list) == 1
        assert evidence_list[0].id == "cpr://doc/test-slug#p=42"
        assert evidence_list[0].passage.index == 42


class TestConceptStore:
    """Test concept store functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.concept_store = ConceptStore(cache_dir="/tmp/test_cache")
    
    def test_expand_concept_to_terms(self):
        """Test concept expansion to terms."""
        # Mock the alias map
        self.concept_store.alias_map = {
            "Q1167": {
                "label": "Solar energy",
                "aliases": ["Solar power", "Photovoltaic"]
            }
        }
        
        terms = self.concept_store.expand_concept_to_terms("Q1167")
        assert "Solar energy" in terms
        assert "Solar power" in terms
        assert "Photovoltaic" in terms
        assert len(terms) == 3
    
    def test_cache_key_generation(self):
        """Test cache key generation for SPARQL queries."""
        query = "SELECT * WHERE { ?s ?p ?o } LIMIT 10"
        cache_key = f"sparql_{hash(query)}"
        
        # Set cached value
        self.concept_store._set_cached(cache_key, {"test": "data"})
        
        # Retrieve cached value
        cached = self.concept_store._get_cached(cache_key)
        assert cached == {"test": "data"}


class TestDeepResearchCompatibility:
    """Test Deep-Research compatibility requirements."""
    
    def test_search_returns_ids_and_snippets(self):
        """Test that search returns lightweight IDs with snippets."""
        result = SearchResult(
            id="cpr://doc/test-slug#p=123",
            title="Test Document",
            snippet="This is a test snippet...",
            score=0.75
        )
        
        # Check required fields for Deep-Research
        assert hasattr(result, 'id')
        assert hasattr(result, 'snippet')
        assert result.id.startswith("cpr://")
    
    def test_fetch_hydrates_full_evidence(self):
        """Test that fetch returns full Evidence objects."""
        evidence = Evidence(
            id="cpr://doc/test-slug#p=123",
            document={
                "slug": "test-slug",
                "document_id": "doc-123",
                "title": "Test Document",
                "family_title": "Test Family",
                "geographies": ["BRA"],
                "languages": ["en"],
                "url": "https://app.climatepolicyradar.org/documents/test-slug"
            },
            passage={
                "index": 123,
                "type": "body",
                "text": "Full passage text here..."
            }
        )
        
        # Check all required fields are present
        assert evidence.document.url.startswith("https://")
        assert evidence.passage.text
        assert evidence.id == "cpr://doc/test-slug#p=123"


class TestURIFormat:
    """Test canonical URI formats."""
    
    def test_document_passage_uri(self):
        """Test document passage URI format."""
        slug = "brazil-climate-law-2024"
        index = 42
        uri = f"cpr://doc/{slug}#p={index}"
        
        assert uri == "cpr://doc/brazil-climate-law-2024#p=42"
        assert uri.startswith("cpr://doc/")
        assert "#p=" in uri
    
    def test_family_uri(self):
        """Test family URI format."""
        family_slug = "brazil-climate-framework"
        uri = f"cpr://family/{family_slug}"
        
        assert uri == "cpr://family/brazil-climate-framework"
    
    def test_concept_uri(self):
        """Test concept URI format."""
        qid = "Q1167"
        uri = f"cpr://concept/{qid}"
        
        assert uri == "cpr://concept/Q1167"


@pytest.mark.asyncio
class TestServerStartup:
    """Test server startup and initialization."""
    
    @patch('server.server.DataAccess')
    @patch('server.server.ConceptStore')
    @patch('server.server.SearchEngine')
    async def test_startup_initialization(self, mock_search, mock_concepts, mock_data):
        """Test that startup initializes all components."""
        from server.server import startup
        
        # Mock environment variables
        with patch.dict('os.environ', {
            'DATA_CACHE_DIR': '/tmp/cache',
            'DATASET_REPO': 'test/repo',
            'DATASET_REVISION': 'main',
            'SPARQL_ENDPOINT': 'http://test.sparql'
        }):
            await startup()
            
            # Verify components were initialized
            mock_data.assert_called_once()
            mock_concepts.assert_called_once()
            mock_search.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])