"""
Enhanced search functionality for Climate Policy Radar MCP Server.
Implements DuckDB-computable search features with statistics and fallback modes.
"""

import re
import math
import json
from typing import List, Dict, Any, Optional, Set, Tuple, Union
from collections import Counter, defaultdict
from datetime import datetime
import logging

from .schemas import Evidence, SearchResult, DocumentMetadata, Passage, ConceptMatch

logger = logging.getLogger(__name__)


class EnhancedSearchEngine:
    """Enhanced search with statistics, fallback, and analytics."""
    
    def __init__(self, data_access, concept_store):
        self.data_access = data_access
        self.concept_store = concept_store
        
        # BM25 parameters
        self.k1 = 1.2
        self.b = 0.75
        
        # Synonym mapping (pre-computed)
        self.synonyms = {
            "livestock": ["cattle", "ruminants", "cows", "sheep", "goats", "swine", "pigs"],
            "methane": ["CH4", "natural gas"],
            "emissions": ["pollution", "GHG", "greenhouse gas", "greenhouse gases"],
            "agriculture": ["farming", "agricultural", "farms", "cultivation"],
            "climate": ["global warming", "climate change"],
            "carbon": ["CO2", "carbon dioxide"],
            "energy": ["power", "electricity", "renewable energy"],
            "forest": ["deforestation", "afforestation", "reforestation", "woodland"],
            "water": ["hydro", "aquatic", "marine", "ocean"],
            "adaptation": ["resilience", "adaptive capacity"],
            "mitigation": ["reduction", "abatement", "sequestration"]
        }
        
        # Build reverse synonym map
        self.reverse_synonyms = {}
        for primary, syns in self.synonyms.items():
            for syn in syns:
                self.reverse_synonyms[syn.lower()] = primary
        
        # Cache for term statistics
        self._term_stats_cache = {}
        self._co_occurrence_cache = {}
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        if not text:
            return []
        return re.findall(r'\b\w+\b', text.lower())
    
    def expand_with_synonyms(self, terms: List[str]) -> Set[str]:
        """Expand terms with their synonyms."""
        expanded = set(terms)
        for term in terms:
            term_lower = term.lower()
            # Add direct synonyms
            if term_lower in self.synonyms:
                expanded.update(self.synonyms[term_lower])
            # Check if it's a synonym of something else
            if term_lower in self.reverse_synonyms:
                primary = self.reverse_synonyms[term_lower]
                expanded.add(primary)
                expanded.update(self.synonyms.get(primary, []))
        return expanded
    
    def search_with_stats(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        return_stats: bool = True,
        limit: int = 50,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Full-text search with statistics about the search results.
        Returns both results and statistics about document distribution.
        """
        
        # Tokenize query
        query_terms = self.tokenize(query)
        
        # Build SQL query parts
        where_clauses = []
        params = []
        
        # Add text search conditions
        for term in query_terms:
            where_clauses.append("LOWER(\"text_block.text\") LIKE ?")
            params.append(f"%{term}%")
        
        # Add filters
        if filters:
            if filters.get("iso3"):
                iso3_list = filters["iso3"]
                iso_conditions = []
                for iso in iso3_list:
                    iso_conditions.append("? = ANY(\"document_metadata.geographies\")")
                    params.append(iso)
                where_clauses.append(f"({' OR '.join(iso_conditions)})")
            
            if filters.get("year_range"):
                year_start, year_end = filters["year_range"]
                where_clauses.append("CAST(SUBSTR(\"document_metadata.document_title\", -4) AS INTEGER) BETWEEN ? AND ?")
                params.extend([year_start, year_end])
            
            if filters.get("document_types"):
                doc_types = filters["document_types"]
                placeholders = ",".join(["?"] * len(doc_types))
                where_clauses.append(f"\"document_metadata.corpus_type_name\" IN ({placeholders})")
                params.extend(doc_types)
        
        # Base query for results
        base_where = " AND ".join(where_clauses) if where_clauses else "1=1"
        
        # Get results
        results_query = f"""
            SELECT *, 
                   ROW_NUMBER() OVER (PARTITION BY document_id ORDER BY "text_block.index") as passage_rank,
                   "text_block.text" as text,
                   "document_metadata.slug" as slug,
                   "document_metadata.document_title" as document_title,
                   "document_metadata.family_title" as family_title,
                   "text_block.index" as text_index
            FROM open_data
            WHERE {base_where}
            ORDER BY "document_metadata.family_title", "document_metadata.document_title", "text_block.index"
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])
        
        results = self.data_access.execute_query(results_query, params)
        
        # Convert to SearchResult objects
        search_results = []
        for row in results:
            passage_id = f"cpr://doc/{row['slug']}#p={row['text_index']}"
            snippet = row['text']  # Return full passage text
            
            search_results.append(SearchResult(
                id=passage_id,
                title=row.get('document_title') or row.get('family_title') or "Untitled",
                snippet=snippet,
                score=row.get('passage_rank', 0)
            ))
        
        # Calculate statistics if requested
        stats = {}
        if return_stats:
            # Total documents searched
            total_query = f"""
                SELECT COUNT(DISTINCT document_id) as total_documents,
                       COUNT(*) as total_passages
                FROM open_data
            """
            total_stats = self.data_access.execute_query(total_query)[0]
            
            # Documents with each term
            term_stats = {}
            for term in query_terms:
                term_query = """
                    SELECT COUNT(DISTINCT document_id) as doc_count,
                           COUNT(*) as passage_count
                    FROM open_data
                    WHERE LOWER(\"text_block.text\") LIKE ?
                """
                term_result = self.data_access.execute_query(term_query, [f"%{term}%"])[0]
                term_stats[term] = {
                    "documents": term_result["doc_count"],
                    "passages": term_result["passage_count"]
                }
            
            # Documents with all terms (AND)
            all_terms_query = f"""
                SELECT COUNT(DISTINCT document_id) as doc_count,
                       COUNT(*) as passage_count
                FROM open_data
                WHERE {base_where}
            """
            all_terms_result = self.data_access.execute_query(
                all_terms_query, 
                params[:len(query_terms)]  # Only term params, not limit/offset
            )[0]
            
            # Documents with any term (OR)
            any_where = " OR ".join(["LOWER(\"text_block.text\") LIKE ?" for _ in query_terms])
            any_terms_query = f"""
                SELECT COUNT(DISTINCT document_id) as doc_count,
                       COUNT(*) as passage_count
                FROM open_data
                WHERE {any_where}
            """
            any_params = [f"%{term}%" for term in query_terms]
            any_terms_result = self.data_access.execute_query(any_terms_query, any_params)[0]
            
            stats = {
                "total_documents_searched": total_stats["total_documents"],
                "total_passages_searched": total_stats["total_passages"],
                "term_statistics": term_stats,
                "documents_with_all_terms": all_terms_result["doc_count"],
                "passages_with_all_terms": all_terms_result["passage_count"],
                "documents_with_any_term": any_terms_result["doc_count"],
                "passages_with_any_term": any_terms_result["passage_count"]
            }
        
        return {
            "results": search_results,
            "stats": stats if return_stats else None
        }
    
    def search_with_fallback(
        self,
        query: str,
        fallback_mode: str = "progressive",
        min_results: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 50
    ) -> Dict[str, Any]:
        """
        Token-based fallback search.
        
        Modes:
        - progressive: Try all terms (AND), then progressively remove terms
        - any_token: Fall back to OR if AND doesn't return enough results
        """
        
        query_terms = self.tokenize(query)
        if not query_terms:
            return {"results": [], "search_mode": "empty_query"}
        
        # Try with all terms (AND)
        all_results = self._search_with_terms(query_terms, "AND", filters, limit)
        
        if len(all_results) >= min_results:
            return {
                "results": all_results,
                "search_mode": "all_terms",
                "terms_used": query_terms
            }
        
        # Fallback strategies
        if fallback_mode == "progressive":
            # Try removing terms one by one (least important first)
            for i in range(1, len(query_terms)):
                subset_terms = query_terms[:len(query_terms)-i]
                subset_results = self._search_with_terms(subset_terms, "AND", filters, limit)
                
                if len(subset_results) >= min_results:
                    return {
                        "results": subset_results,
                        "search_mode": "partial_terms",
                        "terms_used": subset_terms,
                        "terms_dropped": query_terms[len(query_terms)-i:]
                    }
        
        # Final fallback: OR query
        or_results = self._search_with_terms(query_terms, "OR", filters, limit)
        return {
            "results": or_results,
            "search_mode": "any_term",
            "terms_used": query_terms
        }
    
    def _search_with_terms(
        self,
        terms: List[str],
        operator: str,
        filters: Optional[Dict[str, Any]],
        limit: int
    ) -> List[SearchResult]:
        """Helper to search with specific terms and operator."""
        
        if operator == "AND":
            where_parts = ['LOWER("text_block.text") LIKE ?' for _ in terms]
            where_clause = " AND ".join(where_parts)
        else:  # OR
            where_parts = ['LOWER("text_block.text") LIKE ?' for _ in terms]
            where_clause = " OR ".join(where_parts)
        
        params = [f"%{term}%" for term in terms]
        
        # Add filters
        filter_clause = ""
        if filters:
            filter_parts = []
            if filters.get("iso3"):
                iso3_list = filters["iso3"]
                for iso in iso3_list:
                    filter_parts.append("? = ANY(\"document_metadata.geographies\")")
                    params.append(iso)
            
            if filters.get("document_types"):
                doc_types = filters["document_types"]
                placeholders = ",".join(["?"] * len(doc_types))
                filter_parts.append(f"\"document_metadata.corpus_type_name\" IN ({placeholders})")
                params.extend(doc_types)
            
            if filter_parts:
                filter_clause = " AND " + " AND ".join(filter_parts)
        
        query = f"""
            SELECT *,
                   "text_block.text" as text,
                   "document_metadata.slug" as slug,
                   "document_metadata.document_title" as document_title,
                   "document_metadata.family_title" as family_title,
                   "text_block.index" as text_index
            FROM open_data
            WHERE ({where_clause}){filter_clause}
            ORDER BY "document_metadata.family_title", "document_metadata.document_title", "text_block.index"
            LIMIT ?
        """
        params.append(limit)
        
        results = self.data_access.execute_query(query, params)
        
        # Convert to SearchResult
        search_results = []
        for row in results:
            passage_id = f"cpr://doc/{row['slug']}#p={row['text_index']}"
            snippet = row['text']  # Return full passage text
            
            search_results.append(SearchResult(
                id=passage_id,
                title=row.get('document_title') or row.get('family_title') or "Untitled",
                snippet=snippet,
                score=1.0
            ))
        
        return search_results
    
    def search_expanded(
        self,
        query: str,
        use_synonyms: bool = True,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 50
    ) -> Dict[str, Any]:
        """
        Search with pre-computed synonym expansion.
        """
        
        query_terms = self.tokenize(query)
        
        # Expand with synonyms if requested
        if use_synonyms:
            expanded_terms = self.expand_with_synonyms(query_terms)
        else:
            expanded_terms = set(query_terms)
        
        # Build SQL with all expanded terms
        where_parts = []
        params = []
        for term in expanded_terms:
            where_parts.append('LOWER("text_block.text") LIKE ?')
            params.append(f"%{term.lower()}%")
        
        where_clause = " OR ".join(where_parts)  # OR for synonyms
        
        # Add filters
        filter_clause = ""
        if filters:
            filter_parts = []
            if filters.get("iso3"):
                iso3_list = filters["iso3"]
                for iso in iso3_list:
                    filter_parts.append("? = ANY(\"document_metadata.geographies\")")
                    params.append(iso)
            
            if filter_clause:
                filter_clause = " AND " + " AND ".join(filter_parts)
        
        query = f"""
            SELECT *,
                   "text_block.text" as text,
                   "document_metadata.slug" as slug,
                   "document_metadata.document_title" as document_title,
                   "document_metadata.family_title" as family_title,
                   "text_block.index" as text_index
            FROM open_data
            WHERE ({where_clause}){filter_clause}
            ORDER BY "document_metadata.family_title", "document_metadata.document_title", "text_block.index"
            LIMIT ?
        """
        params.append(limit)
        
        results = self.data_access.execute_query(query, params)
        
        # Convert and return
        search_results = []
        for row in results:
            passage_id = f"cpr://doc/{row['slug']}#p={row['text_index']}"
            snippet = row['text']  # Return full passage text
            
            search_results.append(SearchResult(
                id=passage_id,
                title=row.get('document_title') or row.get('family_title') or "Untitled",
                snippet=snippet,
                score=1.0
            ))
        
        return {
            "results": search_results,
            "original_terms": query_terms,
            "expanded_terms": list(expanded_terms),
            "synonyms_used": use_synonyms
        }
    
    def filter_search(
        self,
        text_contains: Optional[List[str]] = None,
        text_mode: str = "ALL",  # ALL or ANY
        countries: Optional[List[str]] = None,
        year_range: Optional[Tuple[int, int]] = None,
        document_types: Optional[List[str]] = None,
        min_score: Optional[float] = None,
        limit: int = 50
    ) -> List[SearchResult]:
        """
        Multi-field filter search with pure SQL WHERE clauses.
        """
        
        where_parts = []
        params = []
        
        # Text search
        if text_contains:
            text_conditions = ['LOWER("text_block.text") LIKE ?' for _ in text_contains]
            text_params = [f"%{term}%" for term in text_contains]
            
            if text_mode == "ALL":
                where_parts.append("(" + " AND ".join(text_conditions) + ")")
            else:  # ANY
                where_parts.append("(" + " OR ".join(text_conditions) + ")")
            params.extend(text_params)
        
        # Country filter
        if countries:
            country_conditions = []
            for country in countries:
                country_conditions.append("? = ANY(\"document_metadata.geographies\")")
                params.append(country)
            where_parts.append(f"({' OR '.join(country_conditions)})")
        
        # Year range filter
        if year_range:
            where_parts.append("""
                CAST(
                    CASE 
                        WHEN "document_metadata.document_title" SIMILAR TO '%[0-9]{4}%' 
                        THEN regexp_extract("document_metadata.document_title", '[0-9]{4}')
                        ELSE NULL
                    END AS INTEGER
                ) BETWEEN ? AND ?
            """)
            params.extend(year_range)
        
        # Document type filter
        if document_types:
            placeholders = ",".join(["?"] * len(document_types))
            where_parts.append(f"corpus_type_name IN ({placeholders})")
            params.extend(document_types)
        
        # Build final query
        where_clause = " AND ".join(where_parts) if where_parts else "1=1"
        
        query = f"""
            SELECT *,
                   LENGTH("text_block.text") as text_length,
                   ARRAY_LENGTH("document_metadata.geographies") as geo_count,
                   "text_block.text" as text,
                   "document_metadata.slug" as slug,
                   "document_metadata.document_title" as document_title,
                   "document_metadata.family_title" as family_title,
                   "text_block.index" as text_index
            FROM open_data
            WHERE {where_clause}
            ORDER BY "document_metadata.family_title", "document_metadata.document_title", "text_block.index"
            LIMIT ?
        """
        params.append(limit)
        
        results = self.data_access.execute_query(query, params)
        
        # Convert to SearchResult with optional scoring
        search_results = []
        for row in results:
            passage_id = f"cpr://doc/{row['slug']}#p={row['text_index']}"
            snippet = row['text']  # Return full passage text
            
            # Calculate simple score based on text length and matches
            score = 1.0
            if text_contains:
                text_lower = row['text'].lower()
                match_count = sum(1 for term in text_contains if term.lower() in text_lower)
                score = match_count / len(text_contains)
            
            if not min_score or score >= min_score:
                search_results.append(SearchResult(
                    id=passage_id,
                    title=row.get('document_title') or row.get('family_title') or "Untitled",
                    snippet=snippet,
                    score=score
                ))
        
        return search_results
    
    def get_facets(self, base_query: str = None, limit_per_facet: int = 10) -> Dict[str, Any]:
        """
        Get faceted search counts for filtering.
        Returns counts by country, year, document type, and co-occurring terms.
        """
        
        # Base condition
        base_where = "1=1"
        params = []
        
        if base_query:
            query_terms = self.tokenize(base_query)
            text_conditions = ['LOWER("text_block.text") LIKE ?' for _ in query_terms]
            base_where = " AND ".join(text_conditions)
            params = [f"%{term}%" for term in query_terms]
        
        facets = {}
        
        # Country facets
        country_query = f"""
            SELECT "document_metadata.geographies"[1] as country, 
                   COUNT(DISTINCT document_id) as doc_count
            FROM open_data
            WHERE {base_where}
              AND "document_metadata.geographies"[1] IS NOT NULL
            GROUP BY "document_metadata.geographies"[1]
            ORDER BY doc_count DESC
            LIMIT ?
        """
        country_params = params + [limit_per_facet]
        country_results = self.data_access.execute_query(country_query, country_params)
        facets["countries"] = {row["country"]: row["doc_count"] for row in country_results}
        
        # Year facets (extract from document title)
        year_query = f"""
            SELECT CAST(regexp_extract("document_metadata.document_title", '[0-9]{{4}}') AS INTEGER) as year,
                   COUNT(DISTINCT document_id) as doc_count
            FROM open_data
            WHERE {base_where}
              AND "document_metadata.document_title" SIMILAR TO '%[0-9]{{4}}%'
            GROUP BY year
            HAVING year IS NOT NULL
            ORDER BY year DESC
            LIMIT ?
        """
        year_params = params + [limit_per_facet]
        year_results = self.data_access.execute_query(year_query, year_params)
        facets["years"] = {str(row["year"]): row["doc_count"] for row in year_results}
        
        # Document type facets
        type_query = f"""
            SELECT "document_metadata.corpus_type_name" as corpus_type_name,
                   COUNT(DISTINCT document_id) as doc_count
            FROM open_data
            WHERE {base_where}
              AND "document_metadata.corpus_type_name" IS NOT NULL
            GROUP BY "document_metadata.corpus_type_name"
            ORDER BY doc_count DESC
            LIMIT ?
        """
        type_params = params + [limit_per_facet]
        type_results = self.data_access.execute_query(type_query, type_params)
        facets["document_types"] = {row["corpus_type_name"]: row["doc_count"] for row in type_results}
        
        # Co-occurring terms (if base query provided)
        if base_query:
            # Get sample of matching documents to find co-occurring terms
            sample_query = f"""
                SELECT "text_block.text" as text
                FROM open_data
                WHERE {base_where}
                LIMIT 100
            """
            sample_results = self.data_access.execute_query(sample_query, params)
            
            # Count term frequencies
            term_counts = Counter()
            base_terms = set(self.tokenize(base_query))
            
            for row in sample_results:
                tokens = self.tokenize(row["text"])
                # Count terms that aren't in the original query
                for token in tokens:
                    if token not in base_terms and len(token) > 3:
                        term_counts[token] += 1
            
            # Get top co-occurring terms
            facets["co_occurring_terms"] = dict(term_counts.most_common(limit_per_facet))
        
        return facets
    
    def find_similar(
        self,
        document_id: str = None,
        slug: str = None,
        similarity_metric: str = "shared_keywords",
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Find similar documents based on shared terms.
        Uses SQL set operations, no ML required.
        """
        
        # Get the reference document
        if slug:
            ref_query = """
                SELECT document_id, 
                       "text_block.text" as text, 
                       "document_metadata.family_title" as family_title, 
                       "document_metadata.document_title" as document_title,
                       "document_metadata.slug" as slug
                FROM open_data
                WHERE "document_metadata.slug" = ?
                LIMIT 1
            """
            ref_result = self.data_access.execute_query(ref_query, [slug])
        elif document_id:
            ref_query = """
                SELECT document_id, 
                       "text_block.text" as text, 
                       "document_metadata.family_title" as family_title, 
                       "document_metadata.document_title" as document_title,
                       "document_metadata.slug" as slug
                FROM open_data
                WHERE document_id = ?
                LIMIT 1
            """
            ref_result = self.data_access.execute_query(ref_query, [document_id])
        else:
            raise ValueError("Either document_id or slug must be provided")
        
        if not ref_result:
            return []
        
        ref_doc = ref_result[0]
        ref_text = ref_doc["text"]
        ref_tokens = set(self.tokenize(ref_text))
        
        # Remove very common words (simple stopword filtering)
        stopwords = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "from"}
        ref_tokens = {t for t in ref_tokens if t not in stopwords and len(t) > 2}
        
        # Find documents with overlapping terms
        # Build query to find documents with any of the reference terms
        term_conditions = ['LOWER("text_block.text") LIKE ?' for _ in ref_tokens]
        where_clause = " OR ".join(term_conditions)
        params = [f"%{term}%" for term in ref_tokens]
        
        similar_query = f"""
            SELECT DISTINCT document_id, 
                   "document_metadata.family_title" as family_title,
                   "document_metadata.document_title" as document_title,
                   "document_metadata.slug" as slug,
                   "text_block.text" as text
            FROM open_data
            WHERE ({where_clause})
              AND document_id != ?
            LIMIT ?
        """
        params.extend([ref_doc["document_id"], limit * 5])  # Get more for scoring
        
        similar_results = self.data_access.execute_query(similar_query, params)
        
        # Score each document by similarity
        scored_docs = []
        for doc in similar_results:
            doc_tokens = set(self.tokenize(doc["text"]))
            doc_tokens = {t for t in doc_tokens if t not in stopwords and len(t) > 2}
            
            # Calculate similarity based on metric
            if similarity_metric == "jaccard":
                # Jaccard similarity: |A ∩ B| / |A ∪ B|
                intersection = len(ref_tokens & doc_tokens)
                union = len(ref_tokens | doc_tokens)
                score = intersection / union if union > 0 else 0
            
            elif similarity_metric == "dice":
                # Dice coefficient: 2 * |A ∩ B| / (|A| + |B|)
                intersection = len(ref_tokens & doc_tokens)
                total = len(ref_tokens) + len(doc_tokens)
                score = (2 * intersection) / total if total > 0 else 0
            
            else:  # shared_keywords
                # Simple shared keyword count
                score = len(ref_tokens & doc_tokens)
            
            scored_docs.append({
                "document_id": doc["document_id"],
                "slug": doc["slug"],
                "title": doc.get("document_title") or doc.get("family_title"),
                "similarity_score": score,
                "shared_terms": list(ref_tokens & doc_tokens)[:10]  # Top 10 shared terms
            })
        
        # Sort by similarity score
        scored_docs.sort(key=lambda x: -x["similarity_score"])
        
        return scored_docs[:limit]