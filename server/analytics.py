"""
Pre-computed analytics and navigation features for Climate Policy Radar MCP Server.
Implements term co-occurrence, coverage analysis, and concept navigation.
"""

import re
import json
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import Counter, defaultdict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class AnalyticsEngine:
    """Pre-computed analytics and navigation features."""
    
    def __init__(self, data_access, concept_store):
        self.data_access = data_access
        self.concept_store = concept_store
        
        # Concept hierarchy (pre-defined taxonomy)
        self.concept_hierarchy = {
            "agriculture": {
                "livestock": ["cattle", "sheep", "goats", "swine", "poultry", "dairy"],
                "crops": ["rice", "wheat", "corn", "soybeans", "cotton", "sugarcane"],
                "practices": ["fertilization", "irrigation", "tillage", "crop rotation", "organic farming"],
                "land use": ["deforestation", "afforestation", "grassland", "pasture"]
            },
            "energy": {
                "fossil fuels": ["coal", "oil", "natural gas", "petroleum"],
                "renewable": ["solar", "wind", "hydro", "geothermal", "biomass"],
                "efficiency": ["insulation", "LED", "smart grid", "energy storage"],
                "sectors": ["electricity", "heating", "transport", "industry"]
            },
            "emissions": {
                "greenhouse gases": ["CO2", "methane", "N2O", "fluorinated gases"],
                "sources": ["combustion", "industrial processes", "agriculture", "waste"],
                "sinks": ["forests", "oceans", "soil", "wetlands"],
                "reduction": ["capture", "storage", "sequestration", "offset"]
            },
            "adaptation": {
                "impacts": ["sea level rise", "drought", "flooding", "heat waves"],
                "sectors": ["water", "agriculture", "health", "infrastructure"],
                "measures": ["early warning", "resilience", "disaster risk", "climate services"]
            },
            "policy": {
                "instruments": ["carbon tax", "emissions trading", "subsidies", "regulations"],
                "targets": ["NDC", "net zero", "carbon neutral", "renewable targets"],
                "governance": ["monitoring", "reporting", "verification", "transparency"],
                "finance": ["climate finance", "green bonds", "adaptation fund", "loss and damage"]
            }
        }
        
        # Pre-compute some statistics on initialization
        self._term_freq_cache = {}
        self._co_occurrence_cache = {}
        self._ngram_cache = {}
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        if not text:
            return []
        return re.findall(r'\b\w+\b', text.lower())
    
    def get_term_associations(
        self,
        term: str,
        window_size: int = 50,
        min_frequency: int = 5,
        limit: int = 20
    ) -> Dict[str, Any]:
        """
        Get terms that frequently co-occur with the given term.
        Uses sliding window analysis on the corpus.
        """
        
        # Query for documents containing the term
        query = """
            SELECT "text_block.text" as text
            FROM open_data
            WHERE LOWER("text_block.text") LIKE ?
            LIMIT 1000
        """
        results = self.data_access.execute_query(query, [f"%{term.lower()}%"])
        
        # Analyze co-occurrences
        co_occurrences = defaultdict(lambda: {"frequency": 0, "distances": []})
        term_lower = term.lower()
        
        for row in results:
            tokens = self.tokenize(row["text"])
            
            # Find positions of target term
            term_positions = [i for i, t in enumerate(tokens) if t == term_lower]
            
            for pos in term_positions:
                # Look within window
                start = max(0, pos - window_size)
                end = min(len(tokens), pos + window_size + 1)
                
                for i in range(start, end):
                    if i == pos:
                        continue
                    
                    other_term = tokens[i]
                    if len(other_term) > 2:  # Skip short words
                        distance = abs(i - pos)
                        co_occurrences[other_term]["frequency"] += 1
                        co_occurrences[other_term]["distances"].append(distance)
        
        # Filter by minimum frequency and calculate average distances
        filtered = {}
        for other_term, data in co_occurrences.items():
            if data["frequency"] >= min_frequency:
                avg_distance = sum(data["distances"]) / len(data["distances"])
                filtered[other_term] = {
                    "frequency": data["frequency"],
                    "distance_avg": round(avg_distance, 2)
                }
        
        # Sort by frequency and return top results
        sorted_terms = sorted(filtered.items(), key=lambda x: -x[1]["frequency"])
        
        return dict(sorted_terms[:limit])
    
    def analyze_coverage(
        self,
        topics: List[str],
        by_country: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze document coverage for given topics.
        Shows which documents contain all, any, or specific combinations.
        """
        
        # Build queries for each topic
        topic_conditions = []
        params = []
        for topic in topics:
            topic_conditions.append("LOWER(\"text_block.text\") LIKE ?")
            params.append(f"%{topic.lower()}%")
        
        # Documents with ANY topic
        any_query = f"""
            SELECT COUNT(DISTINCT document_id) as doc_count
            FROM open_data
            WHERE {' OR '.join(topic_conditions)}
        """
        any_result = self.data_access.execute_query(any_query, params)[0]
        
        # Documents with ALL topics
        all_query = f"""
            SELECT COUNT(DISTINCT document_id) as doc_count
            FROM open_data
            WHERE {' AND '.join(topic_conditions)}
        """
        all_result = self.data_access.execute_query(all_query, params)[0]
        
        # Get total document count
        total_query = "SELECT COUNT(DISTINCT document_id) as total FROM open_data"
        total_result = self.data_access.execute_query(total_query)[0]
        
        result = {
            "total_documents": total_result["total"],
            "documents_with_any": any_result["doc_count"],
            "documents_with_all": all_result["doc_count"],
            "topics_analyzed": topics
        }
        
        # Coverage by country if requested
        if by_country:
            country_any_query = f"""
                SELECT "document_metadata.geographies"[1] as country,
                       COUNT(DISTINCT document_id) as any_count
                FROM open_data
                WHERE ({' OR '.join(topic_conditions)})
                  AND "document_metadata.geographies"[1] IS NOT NULL
                GROUP BY "document_metadata.geographies"[1]
                ORDER BY any_count DESC
                LIMIT 20
            """
            country_any = self.data_access.execute_query(country_any_query, params)
            
            country_all_query = f"""
                SELECT "document_metadata.geographies"[1] as country,
                       COUNT(DISTINCT document_id) as all_count
                FROM open_data
                WHERE ({' AND '.join(topic_conditions)})
                  AND "document_metadata.geographies"[1] IS NOT NULL
                GROUP BY "document_metadata.geographies"[1]
                ORDER BY all_count DESC
                LIMIT 20
            """
            country_all = self.data_access.execute_query(country_all_query, params)
            
            # Merge results
            coverage_by_country = {}
            for row in country_any:
                coverage_by_country[row["country"]] = {"any": row["any_count"], "all": 0}
            for row in country_all:
                if row["country"] in coverage_by_country:
                    coverage_by_country[row["country"]]["all"] = row["all_count"]
                else:
                    coverage_by_country[row["country"]] = {"any": 0, "all": row["all_count"]}
            
            result["coverage_by_country"] = coverage_by_country
        
        # Find missing combinations (pairs with no documents)
        if len(topics) > 1:
            missing_combinations = []
            for i in range(len(topics)):
                for j in range(i + 1, len(topics)):
                    pair_query = """
                        SELECT COUNT(DISTINCT document_id) as cnt
                        FROM open_data
                        WHERE LOWER("text_block.text") LIKE ? AND LOWER("text_block.text") LIKE ?
                    """
                    pair_result = self.data_access.execute_query(
                        pair_query, 
                        [f"%{topics[i].lower()}%", f"%{topics[j].lower()}%"]
                    )[0]
                    
                    if pair_result["cnt"] < 5:  # Threshold for "missing"
                        missing_combinations.append({
                            "terms": [topics[i], topics[j]],
                            "document_count": pair_result["cnt"]
                        })
            
            result["missing_combinations"] = missing_combinations
        
        return result
    
    def get_common_phrases(
        self,
        containing: Optional[str] = None,
        ngram_size: int = 3,
        min_frequency: int = 10,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Extract common n-gram phrases from the corpus.
        """
        
        # Query for text samples
        if containing:
            query = """
                SELECT "text_block.text" as text
                FROM open_data
                WHERE LOWER("text_block.text") LIKE ?
                LIMIT 500
            """
            params = [f"%{containing.lower()}%"]
        else:
            query = """
                SELECT "text_block.text" as text
                FROM open_data
                TABLESAMPLE BERNOULLI(1)
                LIMIT 500
            """
            params = []
        
        results = self.data_access.execute_query(query, params)
        
        # Extract n-grams
        ngram_counts = Counter()
        containing_lower = containing.lower() if containing else None
        
        for row in results:
            tokens = self.tokenize(row["text"])
            
            # Generate n-grams
            for i in range(len(tokens) - ngram_size + 1):
                ngram = tokens[i:i + ngram_size]
                
                # Check if it contains the target word
                if containing_lower:
                    if not any(containing_lower in token for token in ngram):
                        continue
                
                # Skip if contains very short words
                if all(len(token) > 2 for token in ngram):
                    phrase = " ".join(ngram)
                    ngram_counts[phrase] += 1
        
        # Filter by minimum frequency
        filtered_phrases = [
            {"phrase": phrase, "frequency": count}
            for phrase, count in ngram_counts.items()
            if count >= min_frequency
        ]
        
        # Sort by frequency
        filtered_phrases.sort(key=lambda x: -x["frequency"])
        
        return filtered_phrases[:limit]
    
    def get_concept_tree(self, root: str = None) -> Dict[str, Any]:
        """
        Get concept hierarchy navigation tree.
        Returns pre-defined taxonomy structure.
        """
        
        if root and root in self.concept_hierarchy:
            return {root: self.concept_hierarchy[root]}
        elif root:
            # Search within hierarchy
            for category, subcategories in self.concept_hierarchy.items():
                if root in subcategories:
                    return {root: subcategories[root]}
                for subcat, terms in subcategories.items():
                    if root == subcat:
                        return {root: terms}
            return {}
        else:
            # Return full hierarchy
            return self.concept_hierarchy
    
    def get_term_stats(self, limit: int = 100) -> Dict[str, Any]:
        """
        Get overall term frequency statistics for the corpus.
        """
        
        # Sample documents for term analysis
        sample_query = """
            SELECT "text_block.text" as text
            FROM open_data
            TABLESAMPLE BERNOULLI(0.5)
            LIMIT 1000
        """
        results = self.data_access.execute_query(sample_query)
        
        # Count all terms
        all_terms = Counter()
        for row in results:
            tokens = self.tokenize(row["text"])
            all_terms.update(tokens)
        
        # Calculate statistics
        total_terms = sum(all_terms.values())
        unique_terms = len(all_terms)
        hapax_legomena = sum(1 for count in all_terms.values() if count == 1)
        
        # Get most common terms
        most_common = [
            {"term": term, "frequency": count}
            for term, count in all_terms.most_common(limit)
        ]
        
        return {
            "total_terms": total_terms,
            "unique_terms": unique_terms,
            "most_common": most_common,
            "hapax_legomena": hapax_legomena,
            "average_term_frequency": round(total_terms / unique_terms, 2) if unique_terms > 0 else 0
        }
    
    def search_paginated(
        self,
        query: str,
        page: int = 1,
        per_page: int = 20,
        filters: Optional[Dict[str, Any]] = None,
        return_context: bool = True
    ) -> Dict[str, Any]:
        """
        Smart pagination with refinement options.
        """
        
        offset = (page - 1) * per_page
        
        # Get main results
        main_query = """
            SELECT *,
                   "text_block.text" as text,
                   "document_metadata.slug" as slug,
                   "document_metadata.document_title" as document_title,
                   "document_metadata.family_title" as family_title,
                   "text_block.index" as text_index
            FROM open_data
            WHERE LOWER("text_block.text") LIKE ?
        """
        params = [f"%{query.lower()}%"]
        
        # Add filters
        if filters:
            if filters.get("iso3"):
                main_query += " AND ? = ANY(\"document_metadata.geographies\")"
                params.append(filters["iso3"])
            if filters.get("year"):
                main_query += " AND \"document_metadata.document_title\" LIKE ?"
                params.append(f"%{filters['year']}%")
        
        main_query += " ORDER BY \"document_metadata.family_title\", \"document_metadata.document_title\", \"text_block.index\" LIMIT ? OFFSET ?"
        params.extend([per_page, offset])
        
        results = self.data_access.execute_query(main_query, params)
        
        # Get total count
        count_query = """
            SELECT COUNT(*) as total
            FROM open_data
            WHERE LOWER("text_block.text") LIKE ?
        """
        count_params = [f"%{query.lower()}%"]
        if filters and filters.get("iso3"):
            count_query += " AND ? = ANY(\"document_metadata.geographies\")"
            count_params.append(filters["iso3"])
        
        total_count = self.data_access.execute_query(count_query, count_params)[0]["total"]
        
        # Calculate pagination info
        total_pages = (total_count + per_page - 1) // per_page
        
        response = {
            "results": results,
            "pagination": {
                "current_page": page,
                "total_pages": total_pages,
                "total_results": total_count,
                "results_per_page": per_page,
                "has_next": page < total_pages,
                "has_previous": page > 1
            }
        }
        
        # Add refinement options if requested
        if return_context:
            # Get top terms to add
            add_terms_query = """
                SELECT "text_block.text" as text
                FROM open_data
                WHERE LOWER("text_block.text") LIKE ?
                LIMIT 100
            """
            add_terms_results = self.data_access.execute_query(add_terms_query, [f"%{query.lower()}%"])
            
            # Extract frequent terms not in query
            term_counts = Counter()
            query_terms = set(self.tokenize(query))
            
            for row in add_terms_results:
                tokens = self.tokenize(row["text"])
                for token in tokens:
                    if token not in query_terms and len(token) > 3:
                        term_counts[token] += 1
            
            # Get top countries for filtering
            country_query = """
                SELECT "document_metadata.geographies"[1] as country, COUNT(*) as cnt
                FROM open_data
                WHERE LOWER("text_block.text") LIKE ?
                  AND "document_metadata.geographies"[1] IS NOT NULL
                GROUP BY "document_metadata.geographies"[1]
                ORDER BY cnt DESC
                LIMIT 5
            """
            country_results = self.data_access.execute_query(country_query, [f"%{query.lower()}%"])
            
            response["refinement_options"] = {
                "add_term": [
                    f"{term} ({count})" 
                    for term, count in term_counts.most_common(5)
                ],
                "filter_country": [
                    f"{row['country']} ({row['cnt']})"
                    for row in country_results
                ]
            }
        
        return response
    
    def suggest_query_terms(
        self,
        partial: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Suggest query terms based on partial input.
        Uses term frequency in the corpus.
        """
        
        # Query for documents containing the partial term
        query = """
            SELECT "text_block.text" as text
            FROM open_data
            WHERE LOWER("text_block.text") LIKE ?
            LIMIT 500
        """
        results = self.data_access.execute_query(query, [f"%{partial.lower()}%"])
        
        # Extract and count terms starting with partial
        term_counts = Counter()
        partial_lower = partial.lower()
        
        for row in results:
            tokens = self.tokenize(row["text"])
            for token in tokens:
                if token.startswith(partial_lower) and len(token) > len(partial):
                    term_counts[token] += 1
        
        # Get document counts for top terms
        suggestions = []
        for term, freq in term_counts.most_common(limit):
            # Get document count
            doc_count_query = """
                SELECT COUNT(DISTINCT document_id) as cnt
                FROM open_data
                WHERE LOWER("text_block.text") LIKE ?
            """
            doc_count = self.data_access.execute_query(doc_count_query, [f"%{term}%"])[0]["cnt"]
            
            suggestions.append({
                "term": term,
                "document_count": doc_count,
                "frequency": freq
            })
        
        return suggestions