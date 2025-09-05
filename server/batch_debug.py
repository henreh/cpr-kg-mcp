"""
Batch operations and debugging features for Climate Policy Radar MCP Server.
Implements multi-query execution, cross-reference search, and query explanation.
"""

import time
import json
from typing import List, Dict, Any, Optional, Set
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class BatchDebugEngine:
    """Batch operations and debugging/transparency features."""
    
    def __init__(self, data_access, enhanced_search, analytics):
        self.data_access = data_access
        self.enhanced_search = enhanced_search
        self.analytics = analytics
        
        # Query execution history for debugging
        self.query_history = []
    
    def batch_search(
        self,
        queries: List[Dict[str, Any]],
        deduplicate: bool = True,
        max_total_results: int = 500
    ) -> Dict[str, Any]:
        """
        Execute multiple search queries in batch.
        
        Each query dict should contain:
        - query: search text
        - limit: max results for this query
        - filters: optional filters
        - tag: optional tag to identify results
        """
        
        all_results = []
        seen_ids = set() if deduplicate else None
        query_metadata = []
        
        for query_spec in queries:
            query_text = query_spec.get("query", "")
            limit = query_spec.get("limit", 10)
            filters = query_spec.get("filters")
            tag = query_spec.get("tag", query_text[:20])
            
            # Execute search
            start_time = time.time()
            
            search_results = self.enhanced_search.search_with_stats(
                query=query_text,
                filters=filters,
                return_stats=False,
                limit=limit
            )
            
            execution_time = time.time() - start_time
            
            # Process results
            query_result_count = 0
            for result in search_results["results"]:
                if deduplicate:
                    if result.id in seen_ids:
                        continue
                    seen_ids.add(result.id)
                
                # Add source query tag
                result_dict = result.model_dump() if hasattr(result, 'model_dump') else result
                result_dict["source_query"] = tag
                all_results.append(result_dict)
                query_result_count += 1
                
                if len(all_results) >= max_total_results:
                    break
            
            # Track metadata
            query_metadata.append({
                "query": query_text,
                "tag": tag,
                "results_returned": query_result_count,
                "execution_time_ms": round(execution_time * 1000, 2)
            })
            
            if len(all_results) >= max_total_results:
                break
        
        return {
            "results": all_results[:max_total_results],
            "total_results": len(all_results),
            "deduplicated": deduplicate,
            "query_metadata": query_metadata,
            "queries_executed": len(query_metadata)
        }
    
    def find_documents_mentioning_all(
        self,
        terms: List[str],
        proximity: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Find documents mentioning all specified terms.
        Optionally enforce proximity constraints.
        """
        
        # Build base query for all terms
        where_conditions = []
        params = []
        
        for term in terms:
            where_conditions.append("LOWER(\"text_block.text\") LIKE ?")
            params.append(f"%{term.lower()}%")
        
        # Add filters
        if filters:
            if filters.get("iso3"):
                iso3_list = filters["iso3"]
                for iso in iso3_list:
                    where_conditions.append("? = ANY(\"document_metadata.geographies\")")
                    params.append(iso)
            
            if filters.get("document_types"):
                doc_types = filters["document_types"]
                placeholders = ",".join(["?"] * len(doc_types))
                where_conditions.append(f"\"document_metadata.corpus_type_name\" IN ({placeholders})")
                params.extend(doc_types)
        
        where_clause = " AND ".join(where_conditions)
        
        # Query for documents
        query = f"""
            SELECT *,
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
        params.append(limit * 2)  # Get extra for proximity filtering
        
        results = self.data_access.execute_query(query, params)
        
        # Filter by proximity if specified
        filtered_results = []
        
        for row in results:
            text_lower = row["text"].lower()
            
            if proximity:
                # Check if all terms appear within proximity window
                positions = {}
                for term in terms:
                    term_lower = term.lower()
                    pos = text_lower.find(term_lower)
                    if pos == -1:
                        break
                    positions[term] = pos
                
                if len(positions) == len(terms):
                    # Check proximity
                    all_positions = list(positions.values())
                    min_pos = min(all_positions)
                    max_pos = max(all_positions)
                    
                    if max_pos - min_pos <= proximity:
                        # Terms are within proximity
                        filtered_results.append({
                            "document_id": row["document_id"],
                            "slug": row["slug"],
                            "title": row.get("document_title") or row.get("family_title"),
                            "text_snippet": row["text"],  # Return full passage text
                            "term_positions": positions,
                            "proximity_span": max_pos - min_pos
                        })
            else:
                # No proximity requirement
                filtered_results.append({
                    "document_id": row["document_id"],
                    "slug": row["slug"],
                    "title": row.get("document_title") or row.get("family_title"),
                    "text_snippet": row["text"],  # Return full passage text
                    "terms_found": terms
                })
            
            if len(filtered_results) >= limit:
                break
        
        return filtered_results
    
    def explain_search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Explain how a search query will be executed.
        Shows parsed tokens, SQL generated, and execution plan.
        """
        
        # Parse query
        tokens = self.enhanced_search.tokenize(query)
        
        # Check for synonym expansion
        expanded_terms = self.enhanced_search.expand_with_synonyms(tokens)
        
        # Build SQL query
        where_conditions = []
        params = []
        
        for token in tokens:
            where_conditions.append("LOWER(\"text_block.text\") LIKE ?")
            params.append(f"%{token}%")
        
        # Add filters
        filter_sql = ""
        if filters:
            filter_parts = []
            if filters.get("iso3"):
                filter_parts.append(f"\"document_metadata.geographies\"[1] IN ({','.join(filters['iso3'])})")
            if filters.get("document_types"):
                filter_parts.append(f"\"document_metadata.corpus_type_name\" IN ({','.join(filters['document_types'])})")
            if filter_parts:
                filter_sql = " AND " + " AND ".join(filter_parts)
        
        sql_query = f"""
            SELECT * FROM open_data
            WHERE {' AND '.join(where_conditions)}{filter_sql}
            ORDER BY "document_metadata.family_title", "document_metadata.document_title", "text_block.index"
            LIMIT 50
        """
        
        # Estimate documents to scan
        estimate_query = """
            SELECT COUNT(*) as total_docs,
                   COUNT(DISTINCT document_id) as unique_docs
            FROM open_data
        """
        estimates = self.data_access.execute_query(estimate_query)[0]
        
        # Simulate execution timing
        start_time = time.time()
        test_query = """
            SELECT COUNT(*) as cnt
            FROM open_data
            WHERE LOWER("text_block.text") LIKE ?
            LIMIT 1
        """
        self.data_access.execute_query(test_query, [f"%{tokens[0]}%" if tokens else "%"])
        sample_time = time.time() - start_time
        
        explanation = {
            "original_query": query,
            "parsed_tokens": tokens,
            "expanded_terms": list(expanded_terms) if len(expanded_terms) > len(tokens) else None,
            "sql_generated": sql_query,
            "filters_applied": filters or {},
            "execution_plan": {
                "index_used": "text_fts_index (if available)",
                "scan_type": "Full table scan with LIKE filters",
                "documents_in_corpus": estimates["unique_docs"],
                "total_passages": estimates["total_docs"],
                "estimated_time_ms": round(sample_time * 1000 * len(tokens), 2)
            },
            "optimization_hints": []
        }
        
        # Add optimization hints
        if len(tokens) > 5:
            explanation["optimization_hints"].append(
                "Consider using fewer search terms for faster results"
            )
        
        if not filters:
            explanation["optimization_hints"].append(
                "Adding filters (country, document type) can significantly speed up search"
            )
        
        if len(expanded_terms) > len(tokens) * 2:
            explanation["optimization_hints"].append(
                "Synonym expansion added many terms; consider disabling for faster search"
            )
        
        return explanation
    
    def compare_queries(
        self,
        query_a: str,
        query_b: str,
        limit: int = 20
    ) -> Dict[str, Any]:
        """
        Compare results from two different queries.
        Shows unique and overlapping results.
        """
        
        # Execute both queries
        results_a = self.enhanced_search.search_with_stats(
            query=query_a,
            return_stats=False,
            limit=limit * 2
        )
        
        results_b = self.enhanced_search.search_with_stats(
            query=query_b,
            return_stats=False,
            limit=limit * 2
        )
        
        # Extract IDs
        ids_a = {r.id for r in results_a["results"]}
        ids_b = {r.id for r in results_b["results"]}
        
        # Find overlaps and differences
        overlap = ids_a & ids_b
        unique_a = ids_a - ids_b
        unique_b = ids_b - ids_a
        
        # Get sample results from each category
        overlap_results = [
            r for r in results_a["results"] 
            if r.id in overlap
        ][:limit//3]
        
        unique_a_results = [
            r for r in results_a["results"]
            if r.id in unique_a
        ][:limit//3]
        
        unique_b_results = [
            r for r in results_b["results"]
            if r.id in unique_b
        ][:limit//3]
        
        return {
            "query_a": query_a,
            "query_b": query_b,
            "statistics": {
                "results_a": len(ids_a),
                "results_b": len(ids_b),
                "overlap_count": len(overlap),
                "unique_to_a": len(unique_a),
                "unique_to_b": len(unique_b),
                "jaccard_similarity": len(overlap) / len(ids_a | ids_b) if (ids_a | ids_b) else 0
            },
            "sample_results": {
                "overlapping": [r.model_dump() if hasattr(r, 'model_dump') else r for r in overlap_results],
                "unique_to_query_a": [r.model_dump() if hasattr(r, 'model_dump') else r for r in unique_a_results],
                "unique_to_query_b": [r.model_dump() if hasattr(r, 'model_dump') else r for r in unique_b_results]
            }
        }
    
    def profile_query(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        iterations: int = 3
    ) -> Dict[str, Any]:
        """
        Profile query performance with detailed timing.
        """
        
        timings = []
        
        for i in range(iterations):
            start_time = time.time()
            
            # Execute search
            results = self.enhanced_search.search_with_stats(
                query=query,
                filters=filters,
                return_stats=True,
                limit=50
            )
            
            end_time = time.time()
            execution_time = (end_time - start_time) * 1000  # Convert to ms
            
            timings.append({
                "iteration": i + 1,
                "execution_time_ms": round(execution_time, 2),
                "results_returned": len(results["results"]),
                "stats_calculated": results["stats"] is not None
            })
        
        # Calculate statistics
        avg_time = sum(t["execution_time_ms"] for t in timings) / len(timings)
        min_time = min(t["execution_time_ms"] for t in timings)
        max_time = max(t["execution_time_ms"] for t in timings)
        
        # Add to query history
        self.query_history.append({
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "filters": filters,
            "avg_execution_time_ms": round(avg_time, 2)
        })
        
        # Keep only last 100 queries in history
        if len(self.query_history) > 100:
            self.query_history = self.query_history[-100:]
        
        return {
            "query": query,
            "filters": filters or {},
            "iterations": iterations,
            "timings": timings,
            "summary": {
                "average_time_ms": round(avg_time, 2),
                "min_time_ms": round(min_time, 2),
                "max_time_ms": round(max_time, 2),
                "variance_ms": round(max_time - min_time, 2)
            },
            "performance_rating": self._rate_performance(avg_time)
        }
    
    def _rate_performance(self, avg_time_ms: float) -> str:
        """Rate query performance based on execution time."""
        if avg_time_ms < 50:
            return "excellent"
        elif avg_time_ms < 200:
            return "good"
        elif avg_time_ms < 500:
            return "acceptable"
        elif avg_time_ms < 1000:
            return "slow"
        else:
            return "very slow"
    
    def get_query_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get recent query execution history.
        """
        return self.query_history[-limit:]
    
    def validate_filters(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate filter values against actual data in the corpus.
        """
        
        validation_results = {}
        
        # Validate ISO3 codes
        if "iso3" in filters:
            valid_iso3_query = """
                SELECT DISTINCT "document_metadata.geographies"[1] as iso3
                FROM open_data
                WHERE "document_metadata.geographies"[1] IS NOT NULL
            """
            valid_iso3_results = self.data_access.execute_query(valid_iso3_query)
            valid_iso3 = {row["iso3"] for row in valid_iso3_results}
            
            provided_iso3 = set(filters["iso3"]) if isinstance(filters["iso3"], list) else {filters["iso3"]}
            invalid_iso3 = provided_iso3 - valid_iso3
            
            validation_results["iso3"] = {
                "provided": list(provided_iso3),
                "valid": list(provided_iso3 & valid_iso3),
                "invalid": list(invalid_iso3),
                "all_valid": len(invalid_iso3) == 0
            }
        
        # Validate document types
        if "document_types" in filters:
            valid_types_query = """
                SELECT DISTINCT "document_metadata.corpus_type_name" as corpus_type_name
                FROM open_data
                WHERE "document_metadata.corpus_type_name" IS NOT NULL
            """
            valid_types_results = self.data_access.execute_query(valid_types_query)
            valid_types = {row["corpus_type_name"] for row in valid_types_results}
            
            provided_types = set(filters["document_types"]) if isinstance(filters["document_types"], list) else {filters["document_types"]}
            invalid_types = provided_types - valid_types
            
            validation_results["document_types"] = {
                "provided": list(provided_types),
                "valid": list(provided_types & valid_types),
                "invalid": list(invalid_types),
                "all_valid": len(invalid_types) == 0,
                "suggestions": list(valid_types)[:10] if invalid_types else []
            }
        
        # Validate year range
        if "year_range" in filters:
            year_start, year_end = filters["year_range"]
            
            # Get actual year range in data
            year_query = """
                SELECT 
                    MIN(CAST(regexp_extract("document_metadata.document_title", '[0-9]{4}') AS INTEGER)) as min_year,
                    MAX(CAST(regexp_extract("document_metadata.document_title", '[0-9]{4}') AS INTEGER)) as max_year
                FROM open_data
                WHERE "document_metadata.document_title" SIMILAR TO '%[0-9]{4}%'
            """
            year_result = self.data_access.execute_query(year_query)[0]
            
            validation_results["year_range"] = {
                "provided": [year_start, year_end],
                "data_range": [year_result["min_year"], year_result["max_year"]],
                "valid": year_start <= year_end and year_start >= year_result["min_year"] and year_end <= year_result["max_year"],
                "warnings": []
            }
            
            if year_start > year_end:
                validation_results["year_range"]["warnings"].append("Start year is after end year")
            if year_start < year_result["min_year"]:
                validation_results["year_range"]["warnings"].append(f"Start year is before earliest data ({year_result['min_year']})")
            if year_end > year_result["max_year"]:
                validation_results["year_range"]["warnings"].append(f"End year is after latest data ({year_result['max_year']})")
        
        return validation_results