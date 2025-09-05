#!/usr/bin/env python3
"""
Climate Policy Radar MCP Server
Exposes climate law & policy knowledge graph to agentic clients via FastMCP.
"""

import os
import sys
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager

from fastmcp import FastMCP
from pydantic import Field
from dotenv import load_dotenv

# Import our modules
from .schemas import (
    SearchInput, SearchResult, FetchInput, Evidence,
    ConceptFindInput, ConceptInfo, ConceptNeighborsInput, ConceptEdge,
    ConceptPropertiesInput, PropertyInfo, ConceptMentionsInput,
    FamilyOverviewInput, FamilyInfo, JurisdictionOverviewInput, JurisdictionStats,
    CompareJurisdictionsInput, ComparisonResult, PolicyLineageInput, PolicyDocument,
    HealthStatus, SearchFilters,
    SearchStatsResult, FallbackSearchResult, ExpandedSearchResult, FacetedSearchResult,
    SimilarDocument, TermAssociation, CoverageAnalysis, CommonPhrase,
    PaginatedSearchResult, QuerySuggestion, BatchSearchQuery, BatchSearchResult,
    CrossReferenceResult, QueryExplanation, QueryComparison, QueryProfile, FilterValidation
)
from .data_access import DataAccess
from .concepts import ConceptStore
from .search import SearchEngine
from .enhanced_search import EnhancedSearchEngine
from .analytics import AnalyticsEngine
from .batch_debug import BatchDebugEngine

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize data components
data_access: Optional[DataAccess] = None
concept_store: Optional[ConceptStore] = None
search_engine: Optional[SearchEngine] = None
enhanced_search: Optional[EnhancedSearchEngine] = None
analytics_engine: Optional[AnalyticsEngine] = None
batch_debug: Optional[BatchDebugEngine] = None


def initialize_components():
    """Initialize data components immediately on module load."""
    global data_access, concept_store, search_engine, enhanced_search, analytics_engine, batch_debug
    
    logger.info("Initializing Climate Policy Radar data components...")
    
    # Get configuration from environment
    cache_dir = os.getenv("DATA_CACHE_DIR", "./cache")
    dataset_repo = os.getenv("DATASET_REPO", "ClimatePolicyRadar/all-document-text-data")
    dataset_revision = os.getenv("DATASET_REVISION", "main")
    sparql_endpoint = os.getenv("SPARQL_ENDPOINT", "https://climatepolicyradar.wikibase.cloud/query/sparql")
    hf_token = os.getenv("HF_TOKEN")
    
    # Initialize components
    try:
        logger.info("Initializing DataAccess (downloading dataset if needed)...")
        data_access = DataAccess(
            cache_dir=cache_dir,
            dataset_repo=dataset_repo,
            dataset_revision=dataset_revision,
            hf_token=hf_token
        )
        
        logger.info("Initializing ConceptStore...")
        concept_store = ConceptStore(
            sparql_endpoint=sparql_endpoint,
            cache_dir=cache_dir
        )
        
        logger.info("Initializing SearchEngine...")
        search_engine = SearchEngine(data_access, concept_store)
        
        logger.info("Initializing EnhancedSearchEngine...")
        enhanced_search = EnhancedSearchEngine(data_access, concept_store)
        
        logger.info("Initializing AnalyticsEngine...")
        analytics_engine = AnalyticsEngine(data_access, concept_store)
        
        logger.info("Initializing BatchDebugEngine...")
        batch_debug = BatchDebugEngine(data_access, enhanced_search, analytics_engine)
        
        logger.info("All components initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        raise


# Initialize components immediately when module loads
initialize_components()


@asynccontextmanager
async def lifespan(app):
    """Manage server lifecycle."""
    # Components are already initialized
    logger.info("Server lifecycle started")
    
    try:
        yield  # Server runs here
    finally:
        # Shutdown - close database connection
        global data_access
        if data_access:
            data_access.close()
            logger.info("Database connection closed")


# Initialize FastMCP server with lifespan
mcp = FastMCP(
    name="Climate Policy Radar KG",
    lifespan=lifespan
)


# ============================================================================
# RESPONSE WRAPPER
# ============================================================================

CRITICAL_INSTRUCTIONS = """CRITICAL REQUIREMENTS:

1. Provenance: Every factual assertion must be backed by a direct quote from source material
2. Citations: Use format (Document Title, p.XXX) for all quotes
3. Direct Quotes: Include relevant verbatim quotes, properly marked with quotation marks"""


def wrap_response(response: Any) -> Dict[str, Any]:
    """Wrap any response with critical instructions."""
    if isinstance(response, dict):
        return {
            "critical_instructions": CRITICAL_INSTRUCTIONS,
            **response
        }
    elif isinstance(response, list):
        return {
            "critical_instructions": CRITICAL_INSTRUCTIONS,
            "results": response
        }
    elif hasattr(response, 'model_dump'):
        return {
            "critical_instructions": CRITICAL_INSTRUCTIONS,
            **response.model_dump()
        }
    else:
        return {
            "critical_instructions": CRITICAL_INSTRUCTIONS,
            "data": response
        }


# ============================================================================
# DEEP-RESEARCH COMPATIBLE TOOLS
# ============================================================================

@mcp.tool
def search(
    query: str = Field(description="Search query text"),
    filters: Optional[SearchFilters] = Field(default=None, description="Optional filters"),
    limit: int = Field(default=50, ge=1, le=500, description="Maximum results to return"),
    offset: int = Field(default=0, ge=0, description="Offset for pagination")
) -> Dict[str, Any]:
    """
    Generic retrieval over passages. Deep-Research compatible.
    Returns lightweight resource IDs with snippets for efficient scanning.
    """
    if not search_engine:
        raise RuntimeError("Search engine not initialized")
    
    results, _ = search_engine.search_passages(
        query=query,
        filters=filters.model_dump() if filters else None,
        limit=limit,
        offset=offset
    )
    
    # Convert Pydantic models to dicts and wrap with critical instructions
    return wrap_response([result.model_dump() for result in results])


@mcp.tool
def fetch(ids: List[str] = Field(description="List of resource IDs to fetch")) -> Dict[str, Any]:
    """
    Hydrate resource IDs to full Evidence objects. Deep-Research compatible.
    Takes IDs from search results and returns complete passage data with provenance.
    """
    if not search_engine:
        raise RuntimeError("Search engine not initialized")
    
    evidence_list = search_engine.fetch_evidence(ids)
    # Convert Pydantic models to dicts and wrap with critical instructions
    return wrap_response([evidence.model_dump() for evidence in evidence_list])


# ============================================================================
# CONCEPT STORE TOOLS
# ============================================================================

@mcp.tool
def concept_find(
    q: Optional[str] = Field(default=None, description="Search term for concept label/alias"),
    qid: Optional[str] = Field(default=None, description="Specific QID to look up"),
    limit: int = Field(default=20, ge=1, le=100)
) -> Dict[str, Any]:
    """
    Search concepts by label, alias, or QID.
    Returns concept information including labels, aliases, and descriptions.
    """
    if not concept_store:
        raise RuntimeError("Concept store not initialized")
    
    if qid:
        # Lookup specific QID
        concept = concept_store.get_concept_by_qid(qid)
        if concept:
            return wrap_response([ConceptInfo(**concept).model_dump()])
        return wrap_response([])
    
    elif q:
        # Search by term
        concepts = concept_store.find_concepts(q, limit)
        return wrap_response([ConceptInfo(**c).model_dump() for c in concepts])
    
    else:
        raise ValueError("Either 'q' or 'qid' must be provided")


@mcp.tool
def concept_neighbors(
    qid: str = Field(description="QID of the concept"),
    direction: str = Field(default="both", description="Direction: both, out, or in"),
    predicates: Optional[List[str]] = Field(default=None, description="Filter by specific predicates"),
    hops: int = Field(default=1, ge=1, le=2, description="Number of hops"),
    limit: int = Field(default=100, ge=1, le=500)
) -> Dict[str, Any]:
    """
    Get 1-hop or 2-hop neighborhood of a concept.
    Returns edges with subject, predicate, object, and labels.
    """
    if not concept_store:
        raise RuntimeError("Concept store not initialized")
    
    edges = concept_store.get_concept_neighbors(
        qid=qid,
        direction=direction,
        predicates=predicates,
        hops=hops,
        limit=limit
    )
    
    return wrap_response(edges)


@mcp.tool
def concept_properties(
    qid: Optional[str] = Field(default=None, description="QID for concept-specific properties")
) -> Dict[str, Any]:
    """
    Get frequency of properties used globally or for a specific concept.
    Returns properties sorted by usage frequency.
    """
    if not concept_store:
        raise RuntimeError("Concept store not initialized")
    
    properties = concept_store.get_property_frequencies(qid)
    return wrap_response([PropertyInfo(**p).model_dump() for p in properties])


@mcp.tool
def concept_mentions(
    qids: List[str] = Field(description="List of concept QIDs to find"),
    iso3: Optional[List[str]] = Field(default=None, description="Filter by ISO3 country codes"),
    limit: int = Field(default=100, ge=1, le=500)
) -> Dict[str, Any]:
    """
    Find passages likely mentioning specific concepts.
    Expands concepts to labels/aliases and searches in text.
    Returns Evidence objects with concept match offsets.
    """
    if not search_engine:
        raise RuntimeError("Search engine not initialized")
    
    mentions = search_engine.find_concept_mentions(
        concept_qids=qids,
        iso3_filter=iso3,
        limit=limit
    )
    return wrap_response([m.model_dump() for m in mentions])


# ============================================================================
# DOCUMENT & JURISDICTION TOOLS
# ============================================================================

@mcp.tool
def family_overview(
    iso3: Optional[List[str]] = Field(default=None, description="Filter by ISO3 codes"),
    language: Optional[List[str]] = Field(default=None, description="Filter by languages")
) -> Dict[str, Any]:
    """
    Group documents by policy family.
    Returns family titles with document counts and sample documents.
    """
    if not data_access:
        raise RuntimeError("Data access not initialized")
    
    families = data_access.get_families_overview(
        iso3_filter=iso3,
        language_filter=language
    )
    
    result = []
    for family in families:
        result.append(FamilyInfo(
            family_title=family["family_title"],
            doc_count=family["doc_count"],
            sample_doc={
                "slug": family["sample_slug"],
                "title": family["sample_title"]
            }
        ))
    
    return wrap_response(result)


@mcp.tool
def jurisdiction_overview(iso3: str = Field(description="ISO3 country code")) -> Dict[str, Any]:
    """
    Get distribution summaries for a jurisdiction.
    Returns documents by corpus type, top languages, and optionally top concepts.
    """
    if not data_access:
        raise RuntimeError("Data access not initialized")
    
    stats = data_access.get_jurisdiction_stats(iso3)
    
    result = JurisdictionStats(
        docs_by_corpus_type=stats["docs_by_corpus_type"],
        top_languages=stats["top_languages"]
    )
    return wrap_response(result.model_dump())


@mcp.tool
def compare_jurisdictions(
    iso3_a: str = Field(description="First ISO3 country code"),
    iso3_b: str = Field(description="Second ISO3 country code"),
    by: str = Field(default="corpus_type", description="Compare by: corpus_type or language")
) -> Dict[str, Any]:
    """
    Side-by-side comparison of two jurisdictions.
    Returns bins with counts for each jurisdiction.
    """
    if not data_access:
        raise RuntimeError("Data access not initialized")
    
    # Get stats for both jurisdictions
    stats_a = data_access.get_jurisdiction_stats(iso3_a)
    stats_b = data_access.get_jurisdiction_stats(iso3_b)
    
    if by == "corpus_type":
        # Get all corpus types
        all_types = set(stats_a["docs_by_corpus_type"].keys()) | set(stats_b["docs_by_corpus_type"].keys())
        bins = sorted(list(all_types))
        
        counts_a = [stats_a["docs_by_corpus_type"].get(t, 0) for t in bins]
        counts_b = [stats_b["docs_by_corpus_type"].get(t, 0) for t in bins]
    
    else:  # by == "language"
        # Get top languages
        all_langs = set(list(stats_a["top_languages"].keys())[:10]) | set(list(stats_b["top_languages"].keys())[:10])
        bins = sorted(list(all_langs))
        
        counts_a = [stats_a["top_languages"].get(l, 0) for l in bins]
        counts_b = [stats_b["top_languages"].get(l, 0) for l in bins]
    
    result = ComparisonResult(
        bins=bins,
        a=counts_a,
        b=counts_b
    )
    return wrap_response(result.model_dump())


@mcp.tool
def policy_lineage(
    family_slug: str = Field(description="Family slug to trace"),
    include_docs: bool = Field(default=True, description="Include document details")
) -> Dict[str, Any]:
    """
    Follow policy family history.
    Returns ordered list of documents in the same family.
    """
    if not data_access:
        raise RuntimeError("Data access not initialized")
    
    # Get all documents with family slug prefix
    results = data_access.search_documents(
        family_slug_filter=[family_slug],
        limit=100
    )
    
    # Group by document and sort
    docs_seen = set()
    policy_docs = []
    
    for i, result in enumerate(results):
        doc_id = result["document_id"]
        if doc_id not in docs_seen:
            docs_seen.add(doc_id)
            policy_docs.append(PolicyDocument(
                slug=result["slug"],
                title=result["document_title"],
                document_id=doc_id,
                order=i
            ))
    
    return wrap_response([doc.model_dump() for doc in policy_docs])


# ============================================================================
# ADMIN & HEALTH TOOLS
# ============================================================================

@mcp.tool
def list_resources() -> Dict[str, Any]:
    """
    Enumerate available views and cache status.
    Returns information about data resources and their states.
    """
    if not data_access:
        raise RuntimeError("Data access not initialized")
    
    # Get row counts
    open_data_count = data_access.execute_query("SELECT COUNT(*) as cnt FROM open_data")[0]["cnt"]
    english_count = data_access.execute_query("SELECT COUNT(*) as cnt FROM open_data_english")[0]["cnt"]
    
    # Get unique documents
    unique_docs = data_access.execute_query("SELECT COUNT(DISTINCT document_id) as cnt FROM open_data")[0]["cnt"]
    
    return {
        "views": {
            "open_data": {
                "rows": open_data_count,
                "description": "All document passages"
            },
            "open_data_english": {
                "rows": english_count,
                "description": "English-only passages"
            }
        },
        "statistics": {
            "total_passages": open_data_count,
            "english_passages": english_count,
            "unique_documents": unique_docs
        },
        "cache_age_minutes": data_access.get_cache_age()
    }


@mcp.tool
def health() -> HealthStatus:
    """
    Check system health and connectivity.
    Returns dataset revision, SPARQL status, and cache ages.
    """
    if not data_access or not concept_store:
        return HealthStatus(
            status="unhealthy",
            dataset_revision="unknown",
            sparql_reachable=False,
            cache_ages={}
        )
    
    # Check SPARQL connectivity
    sparql_ok = concept_store.is_reachable()
    
    # Get DuckDB row count
    try:
        row_count = data_access.execute_query("SELECT COUNT(*) as cnt FROM open_data")[0]["cnt"]
    except:
        row_count = None
    
    return HealthStatus(
        status="healthy" if sparql_ok else "degraded",
        dataset_revision=data_access.dataset_revision,
        sparql_reachable=sparql_ok,
        cache_ages={
            "dataset_minutes": data_access.get_cache_age()
        },
        duckdb_rows=row_count
    )


# ============================================================================
# RESOURCES
# ============================================================================

@mcp.resource("cpr://status")
def get_status() -> Dict[str, Any]:
    """
    Current server status and configuration.
    """
    result = {
        "server": "Climate Policy Radar MCP",
        "version": "0.1.0",
        "status": "operational",
        "timestamp": datetime.now().isoformat()
    }
    return wrap_response(result)


@mcp.resource("cpr://dataset/revision")
def get_dataset_revision() -> Dict[str, Any]:
    """
    Dataset revision information.
    """
    if not data_access:
        return wrap_response({"revision": "unknown"})
    
    result = {
        "repo": data_access.dataset_repo,
        "revision": data_access.dataset_revision,
        "cache_dir": str(data_access.cache_dir)
    }
    return wrap_response(result)


# ============================================================================
# ENHANCED SEARCH TOOLS
# ============================================================================

@mcp.tool
def search_with_stats(
    query: str = Field(description="Search query text"),
    filters: Optional[SearchFilters] = Field(default=None, description="Optional filters"),
    return_stats: bool = Field(default=True, description="Return search statistics"),
    limit: int = Field(default=50, ge=1, le=500)
) -> SearchStatsResult:
    """
    Full-text search with detailed statistics about results.
    Returns counts of documents with each term and combinations.
    """
    if not enhanced_search:
        raise RuntimeError("Enhanced search engine not initialized")
    
    result = enhanced_search.search_with_stats(
        query=query,
        filters=filters.model_dump() if filters else None,
        return_stats=return_stats,
        limit=limit
    )
    
    return SearchStatsResult(
        results=result["results"],
        stats=result.get("stats")
    )


@mcp.tool
def search_with_fallback(
    query: str = Field(description="Search query text"),
    fallback_mode: str = Field(default="progressive", description="Fallback mode: progressive or any_token"),
    min_results: int = Field(default=5, description="Minimum results before fallback"),
    filters: Optional[SearchFilters] = Field(default=None),
    limit: int = Field(default=50)
) -> FallbackSearchResult:
    """
    Token-based fallback search that progressively relaxes criteria.
    Tries AND logic first, then falls back to OR if needed.
    """
    if not enhanced_search:
        raise RuntimeError("Enhanced search engine not initialized")
    
    result = enhanced_search.search_with_fallback(
        query=query,
        fallback_mode=fallback_mode,
        min_results=min_results,
        filters=filters.model_dump() if filters else None,
        limit=limit
    )
    
    return FallbackSearchResult(**result)


@mcp.tool
def search_expanded(
    query: str = Field(description="Search query text"),
    use_synonyms: bool = Field(default=True, description="Expand query with synonyms"),
    filters: Optional[SearchFilters] = Field(default=None),
    limit: int = Field(default=50)
) -> ExpandedSearchResult:
    """
    Search with pre-computed synonym expansion.
    Expands terms like 'livestock' to include 'cattle', 'cows', etc.
    """
    if not enhanced_search:
        raise RuntimeError("Enhanced search engine not initialized")
    
    result = enhanced_search.search_expanded(
        query=query,
        use_synonyms=use_synonyms,
        filters=filters.model_dump() if filters else None,
        limit=limit
    )
    
    return ExpandedSearchResult(**result)


@mcp.tool
def filter_search(
    text_contains: Optional[List[str]] = Field(default=None, description="Text must contain these terms"),
    text_mode: str = Field(default="ALL", description="ALL or ANY for text terms"),
    countries: Optional[List[str]] = Field(default=None, description="ISO3 country codes"),
    year_range: Optional[List[int]] = Field(default=None, description="[start_year, end_year]"),
    document_types: Optional[List[str]] = Field(default=None),
    min_score: Optional[float] = Field(default=None),
    limit: int = Field(default=50)
) -> List[SearchResult]:
    """
    Multi-field filter search using pure SQL.
    Combines text, geography, time, and document type filters.
    """
    if not enhanced_search:
        raise RuntimeError("Enhanced search engine not initialized")
    
    year_tuple = tuple(year_range) if year_range and len(year_range) == 2 else None
    
    results = enhanced_search.filter_search(
        text_contains=text_contains,
        text_mode=text_mode,
        countries=countries,
        year_range=year_tuple,
        document_types=document_types,
        min_score=min_score,
        limit=limit
    )
    return wrap_response([r.model_dump() for r in results])


@mcp.tool
def get_facets(
    base_query: Optional[str] = Field(default=None, description="Base search query"),
    limit_per_facet: int = Field(default=10, description="Max items per facet")
) -> Dict[str, Any]:
    """
    Get faceted search counts for filtering.
    Returns distribution by country, year, document type, and co-occurring terms.
    """
    if not enhanced_search:
        raise RuntimeError("Enhanced search engine not initialized")
    
    facets = enhanced_search.get_facets(
        base_query=base_query,
        limit_per_facet=limit_per_facet
    )
    
    facets_result = FacetedSearchResult(**facets)
    return wrap_response(facets_result.model_dump())


@mcp.tool
def find_similar(
    document_id: Optional[str] = Field(default=None),
    slug: Optional[str] = Field(default=None),
    similarity_metric: str = Field(default="shared_keywords", description="Metric: shared_keywords, jaccard, or dice"),
    limit: int = Field(default=10)
) -> Dict[str, Any]:
    """
    Find similar documents based on shared terms.
    Uses SQL set operations without ML.
    """
    if not enhanced_search:
        raise RuntimeError("Enhanced search engine not initialized")
    
    if not document_id and not slug:
        raise ValueError("Either document_id or slug must be provided")
    
    similar_docs = enhanced_search.find_similar(
        document_id=document_id,
        slug=slug,
        similarity_metric=similarity_metric,
        limit=limit
    )
    
    return wrap_response([SimilarDocument(**doc).model_dump() for doc in similar_docs])


# ============================================================================
# ANALYTICS TOOLS
# ============================================================================

@mcp.tool
def get_term_associations(
    term: str = Field(description="Term to analyze"),
    window_size: int = Field(default=50, description="Context window size"),
    min_frequency: int = Field(default=5),
    limit: int = Field(default=20)
) -> Dict[str, Any]:
    """
    Get terms that frequently co-occur with the given term.
    Returns frequency and average distance metrics.
    """
    if not analytics_engine:
        raise RuntimeError("Analytics engine not initialized")
    
    result = analytics_engine.get_term_associations(
        term=term,
        window_size=window_size,
        min_frequency=min_frequency,
        limit=limit
    )
    return wrap_response(result)


@mcp.tool
def analyze_coverage(
    topics: List[str] = Field(description="Topics to analyze"),
    by_country: bool = Field(default=True, description="Include country breakdown")
) -> Dict[str, Any]:
    """
    Analyze document coverage for given topics.
    Shows documents with all, any, or specific combinations.
    """
    if not analytics_engine:
        raise RuntimeError("Analytics engine not initialized")
    
    result = analytics_engine.analyze_coverage(
        topics=topics,
        by_country=by_country
    )
    
    coverage_result = CoverageAnalysis(**result)
    return wrap_response(coverage_result.model_dump())


@mcp.tool
def get_common_phrases(
    containing: Optional[str] = Field(default=None, description="Term that must appear in phrase"),
    ngram_size: int = Field(default=3, description="N-gram size"),
    min_frequency: int = Field(default=10),
    limit: int = Field(default=20)
) -> Dict[str, Any]:
    """
    Extract common n-gram phrases from the corpus.
    """
    if not analytics_engine:
        raise RuntimeError("Analytics engine not initialized")
    
    phrases = analytics_engine.get_common_phrases(
        containing=containing,
        ngram_size=ngram_size,
        min_frequency=min_frequency,
        limit=limit
    )
    
    return wrap_response([CommonPhrase(**p).model_dump() for p in phrases])


@mcp.tool
def get_concept_tree(
    root: Optional[str] = Field(default=None, description="Root concept to start from")
) -> Dict[str, Any]:
    """
    Get concept hierarchy navigation tree.
    Returns pre-defined taxonomy structure.
    """
    if not analytics_engine:
        raise RuntimeError("Analytics engine not initialized")
    
    result = analytics_engine.get_concept_tree(root=root)
    return wrap_response(result)


@mcp.tool
def search_paginated(
    query: str = Field(description="Search query"),
    page: int = Field(default=1, ge=1),
    per_page: int = Field(default=20, ge=1, le=100),
    filters: Optional[Dict[str, Any]] = Field(default=None),
    return_context: bool = Field(default=True)
) -> Dict[str, Any]:
    """
    Smart pagination with refinement suggestions.
    Returns results with pagination info and suggested refinements.
    """
    if not analytics_engine:
        raise RuntimeError("Analytics engine not initialized")
    
    result = analytics_engine.search_paginated(
        query=query,
        page=page,
        per_page=per_page,
        filters=filters,
        return_context=return_context
    )
    
    paginated_result = PaginatedSearchResult(**result)
    return wrap_response(paginated_result.model_dump())


@mcp.tool
def suggest_query_terms(
    partial: str = Field(description="Partial term to complete"),
    limit: int = Field(default=10)
) -> Dict[str, Any]:
    """
    Suggest query terms based on partial input.
    Returns terms with document counts.
    """
    if not analytics_engine:
        raise RuntimeError("Analytics engine not initialized")
    
    suggestions = analytics_engine.suggest_query_terms(
        partial=partial,
        limit=limit
    )
    
    return wrap_response([QuerySuggestion(**s).model_dump() for s in suggestions])


@mcp.tool
def get_term_stats(
    limit: int = Field(default=100, description="Number of top terms to return")
) -> Dict[str, Any]:
    """
    Get overall term frequency statistics for the corpus.
    Returns total terms, unique terms, and most common.
    """
    if not analytics_engine:
        raise RuntimeError("Analytics engine not initialized")
    
    result = analytics_engine.get_term_stats(limit=limit)
    return wrap_response(result)


# ============================================================================
# BATCH OPERATIONS
# ============================================================================

@mcp.tool
def batch_search(
    queries: List[BatchSearchQuery] = Field(description="List of queries to execute"),
    deduplicate: bool = Field(default=True, description="Remove duplicate results"),
    max_total_results: int = Field(default=500)
) -> Dict[str, Any]:
    """
    Execute multiple search queries in batch.
    Combines results with optional deduplication.
    """
    if not batch_debug:
        raise RuntimeError("Batch debug engine not initialized")
    
    # Convert Pydantic models to dicts
    query_dicts = [q.model_dump() for q in queries]
    
    result = batch_debug.batch_search(
        queries=query_dicts,
        deduplicate=deduplicate,
        max_total_results=max_total_results
    )
    
    batch_result = BatchSearchResult(**result)
    return wrap_response(batch_result.model_dump())


@mcp.tool
def find_documents_mentioning_all(
    terms: List[str] = Field(description="Terms that must all appear"),
    proximity: Optional[int] = Field(default=None, description="Max distance between terms"),
    filters: Optional[Dict[str, Any]] = Field(default=None),
    limit: int = Field(default=50)
) -> Dict[str, Any]:
    """
    Find documents mentioning all specified terms.
    Optionally enforce proximity constraints.
    """
    if not batch_debug:
        raise RuntimeError("Batch debug engine not initialized")
    
    results = batch_debug.find_documents_mentioning_all(
        terms=terms,
        proximity=proximity,
        filters=filters,
        limit=limit
    )
    
    return wrap_response([CrossReferenceResult(**r).model_dump() for r in results])


# ============================================================================
# DEBUG & TRANSPARENCY TOOLS
# ============================================================================

@mcp.tool
def explain_search(
    query: str = Field(description="Query to explain"),
    filters: Optional[SearchFilters] = Field(default=None)
) -> Dict[str, Any]:
    """
    Explain how a search query will be executed.
    Shows parsed tokens, SQL, and execution plan.
    """
    if not batch_debug:
        raise RuntimeError("Batch debug engine not initialized")
    
    explanation = batch_debug.explain_search(
        query=query,
        filters=filters.model_dump() if filters else None
    )
    
    query_explanation = QueryExplanation(**explanation)
    return wrap_response(query_explanation.model_dump())


@mcp.tool
def compare_queries(
    query_a: str = Field(description="First query"),
    query_b: str = Field(description="Second query"),
    limit: int = Field(default=20)
) -> Dict[str, Any]:
    """
    Compare results from two different queries.
    Shows unique and overlapping results with statistics.
    """
    if not batch_debug:
        raise RuntimeError("Batch debug engine not initialized")
    
    comparison = batch_debug.compare_queries(
        query_a=query_a,
        query_b=query_b,
        limit=limit
    )
    
    query_comparison = QueryComparison(**comparison)
    return wrap_response(query_comparison.model_dump())


@mcp.tool
def profile_query(
    query: str = Field(description="Query to profile"),
    filters: Optional[SearchFilters] = Field(default=None),
    iterations: int = Field(default=3, ge=1, le=10)
) -> Dict[str, Any]:
    """
    Profile query performance with detailed timing.
    Runs query multiple times and reports statistics.
    """
    if not batch_debug:
        raise RuntimeError("Batch debug engine not initialized")
    
    profile = batch_debug.profile_query(
        query=query,
        filters=filters.model_dump() if filters else None,
        iterations=iterations
    )
    
    query_profile = QueryProfile(**profile)
    return wrap_response(query_profile.model_dump())


@mcp.tool
def validate_filters(
    filters: Dict[str, Any] = Field(description="Filters to validate")
) -> Dict[str, Any]:
    """
    Validate filter values against actual corpus data.
    Returns valid/invalid values and suggestions.
    """
    if not batch_debug:
        raise RuntimeError("Batch debug engine not initialized")
    
    validation = batch_debug.validate_filters(filters)
    
    filter_validation = FilterValidation(**validation)
    return wrap_response(filter_validation.model_dump())


@mcp.tool
def get_query_history(
    limit: int = Field(default=20, ge=1, le=100)
) -> Dict[str, Any]:
    """
    Get recent query execution history.
    Shows queries, filters, and execution times.
    """
    if not batch_debug:
        raise RuntimeError("Batch debug engine not initialized")
    
    history = batch_debug.get_query_history(limit=limit)
    return wrap_response(history)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point for the server."""
    
    # Get port from environment or use default
    port = int(os.getenv("PORT", 8000))
    transport = os.getenv("TRANSPORT", "http")
    
    logger.info(f"Starting server on port {port} with transport {transport}")
    
    # Run the server
    if transport == "http":
        mcp.run(port=port, transport="http")
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()