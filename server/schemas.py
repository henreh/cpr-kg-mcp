"""
Pydantic models for Climate Policy Radar MCP Server.
Defines the schemas for Evidence objects, tool inputs/outputs, and API responses.
"""

from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """Document metadata structure."""
    slug: str
    document_id: str
    title: str
    family_title: str
    geographies: List[str]
    languages: List[str]
    url: str
    corpus_type_name: Optional[str] = None


class Passage(BaseModel):
    """Text passage within a document."""
    index: int
    page_number: Optional[int] = None
    type: str = Field(description="body|title|table|...")
    text: str
    language: Optional[str] = None


class ConceptMatch(BaseModel):
    """Concept match within a passage."""
    qid: str
    label: str
    match_type: Literal["label", "alias", "string-search"]
    offsets: List[List[int]] = Field(default_factory=list, description="[[start, end], ...]")


class Evidence(BaseModel):
    """Evidence object returned by most tools."""
    id: str = Field(description="cpr://doc/{slug}#p={index}")
    document: DocumentMetadata
    passage: Passage
    concept_matches: List[ConceptMatch] = Field(default_factory=list)
    score: float = 0.0


class SearchFilters(BaseModel):
    """Filters for search operations."""
    iso3: Optional[List[str]] = None
    language: Optional[List[str]] = None
    corpus_type: Optional[List[str]] = None
    family_slug: Optional[List[str]] = None
    concept_qids: Optional[List[str]] = None
    max_passages_per_doc: int = 3


class SearchInput(BaseModel):
    """Input schema for search tool."""
    query: str
    filters: Optional[SearchFilters] = None
    limit: int = Field(default=50, ge=1, le=500)
    offset: int = Field(default=0, ge=0)


class SearchResult(BaseModel):
    """Lightweight search result."""
    id: str
    title: str
    snippet: str
    score: float = 0.0


class FetchInput(BaseModel):
    """Input schema for fetch tool."""
    ids: List[str]


class ConceptInfo(BaseModel):
    """Concept information from Wikibase."""
    qid: str
    preferred_label: str
    aliases: List[str] = Field(default_factory=list)
    description: Optional[str] = None
    instance_of: Optional[str] = None
    broader: Optional[List[str]] = None
    narrower: Optional[List[str]] = None


class ConceptFindInput(BaseModel):
    """Input for concept_find tool."""
    q: Optional[str] = None
    qid: Optional[str] = None
    limit: int = Field(default=20, ge=1, le=100)


class ConceptNeighborsInput(BaseModel):
    """Input for concept_neighbors tool."""
    qid: str
    direction: Literal["both", "out", "in"] = "both"
    predicates: Optional[List[str]] = None
    hops: int = Field(default=1, ge=1, le=2)
    limit: int = Field(default=100, ge=1, le=500)


class ConceptEdge(BaseModel):
    """Edge in concept graph."""
    subject_qid: str
    predicate_pid: str
    object_qid: str
    labels: Dict[str, str] = Field(default_factory=dict)
    counts: Optional[int] = None


class PropertyInfo(BaseModel):
    """Property frequency information."""
    pid: str
    propLabel: str
    count: int


class ConceptPropertiesInput(BaseModel):
    """Input for concept_properties tool."""
    qid: Optional[str] = None


class ConceptMentionsInput(BaseModel):
    """Input for concept_mentions tool."""
    qids: List[str]
    iso3: Optional[List[str]] = None
    limit: int = Field(default=100, ge=1, le=500)


class FamilyInfo(BaseModel):
    """Policy family information."""
    family_title: str
    doc_count: int
    sample_doc: Dict[str, str]


class FamilyOverviewInput(BaseModel):
    """Input for family_overview tool."""
    iso3: Optional[List[str]] = None
    language: Optional[List[str]] = None


class JurisdictionStats(BaseModel):
    """Jurisdiction statistics."""
    docs_by_corpus_type: Dict[str, int]
    top_concepts: Optional[List[Dict[str, Any]]] = None
    top_languages: Dict[str, int]


class JurisdictionOverviewInput(BaseModel):
    """Input for jurisdiction_overview tool."""
    iso3: str


class CompareJurisdictionsInput(BaseModel):
    """Input for compare_jurisdictions tool."""
    iso3_a: str
    iso3_b: str
    by: Literal["corpus_type", "language"] = "corpus_type"


class ComparisonResult(BaseModel):
    """Result of jurisdiction comparison."""
    bins: List[str]
    a: List[int]
    b: List[int]


class PolicyLineageInput(BaseModel):
    """Input for policy_lineage tool."""
    family_slug: str
    include_docs: bool = True


class PolicyDocument(BaseModel):
    """Document in policy lineage."""
    slug: str
    title: str
    document_id: str
    order: int


class HealthStatus(BaseModel):
    """Health check response."""
    status: str
    dataset_revision: str
    sparql_reachable: bool
    cache_ages: Dict[str, int]
    duckdb_rows: Optional[int] = None


class SearchStatsResult(BaseModel):
    """Search results with statistics."""
    results: List[SearchResult]
    stats: Optional[Dict[str, Any]] = None


class FallbackSearchResult(BaseModel):
    """Fallback search result."""
    results: List[SearchResult]
    search_mode: str
    terms_used: List[str]
    terms_dropped: Optional[List[str]] = None


class ExpandedSearchResult(BaseModel):
    """Search with synonym expansion result."""
    results: List[SearchResult]
    original_terms: List[str]
    expanded_terms: List[str]
    synonyms_used: bool


class FacetedSearchResult(BaseModel):
    """Faceted search results."""
    countries: Dict[str, int]
    years: Dict[str, int]
    document_types: Dict[str, int]
    co_occurring_terms: Optional[Dict[str, int]] = None


class SimilarDocument(BaseModel):
    """Similar document result."""
    document_id: str
    slug: str
    title: str
    similarity_score: float
    shared_terms: List[str]


class TermAssociation(BaseModel):
    """Term co-occurrence result."""
    term: str
    frequency: int
    distance_avg: float


class CoverageAnalysis(BaseModel):
    """Document coverage analysis result."""
    total_documents: int
    documents_with_any: int
    documents_with_all: int
    topics_analyzed: List[str]
    coverage_by_country: Optional[Dict[str, Dict[str, int]]] = None
    missing_combinations: Optional[List[Dict[str, Any]]] = None


class CommonPhrase(BaseModel):
    """Common n-gram phrase."""
    phrase: str
    frequency: int


class PaginatedSearchResult(BaseModel):
    """Paginated search with context."""
    results: List[Dict[str, Any]]
    pagination: Dict[str, Any]
    refinement_options: Optional[Dict[str, List[str]]] = None


class QuerySuggestion(BaseModel):
    """Query term suggestion."""
    term: str
    document_count: int
    frequency: int


class BatchSearchQuery(BaseModel):
    """Individual query in batch search."""
    query: str
    limit: int = 10
    filters: Optional[Dict[str, Any]] = None
    tag: Optional[str] = None


class BatchSearchResult(BaseModel):
    """Batch search results."""
    results: List[Dict[str, Any]]
    total_results: int
    deduplicated: bool
    query_metadata: List[Dict[str, Any]]
    queries_executed: int


class CrossReferenceResult(BaseModel):
    """Cross-reference search result."""
    document_id: str
    slug: str
    title: str
    text_snippet: str
    terms_found: Optional[List[str]] = None
    term_positions: Optional[Dict[str, int]] = None
    proximity_span: Optional[int] = None


class QueryExplanation(BaseModel):
    """Query execution explanation."""
    original_query: str
    parsed_tokens: List[str]
    expanded_terms: Optional[List[str]] = None
    sql_generated: str
    filters_applied: Dict[str, Any]
    execution_plan: Dict[str, Any]
    optimization_hints: List[str]


class QueryComparison(BaseModel):
    """Query comparison result."""
    query_a: str
    query_b: str
    statistics: Dict[str, Any]
    sample_results: Dict[str, List[Dict[str, Any]]]


class QueryProfile(BaseModel):
    """Query performance profile."""
    query: str
    filters: Dict[str, Any]
    iterations: int
    timings: List[Dict[str, Any]]
    summary: Dict[str, float]
    performance_rating: str


class FilterValidation(BaseModel):
    """Filter validation result."""
    iso3: Optional[Dict[str, Any]] = None
    document_types: Optional[Dict[str, Any]] = None
    year_range: Optional[Dict[str, Any]] = None