"""
Search and ranking module for Climate Policy Radar MCP Server.
Implements BM25-like scoring and result ranking.
"""

import re
import math
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import Counter
import logging

from .schemas import Evidence, SearchResult, DocumentMetadata, Passage, ConceptMatch

logger = logging.getLogger(__name__)


class SearchEngine:
    """Handles search ranking and result processing."""
    
    def __init__(self, data_access, concept_store):
        self.data_access = data_access
        self.concept_store = concept_store
        
        # BM25 parameters
        self.k1 = 1.2  # Term frequency saturation parameter
        self.b = 0.75   # Length normalization parameter
        
        # Cache for IDF values
        self._idf_cache = {}
    
    def tokenize(self, text: str) -> List[str]:
        """Simple tokenization - split on non-alphanumeric and lowercase."""
        if not text:
            return []
        
        # Convert to lowercase and split on non-alphanumeric
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens
    
    def calculate_bm25_score(
        self,
        query_terms: List[str],
        document_text: str,
        avg_doc_length: float = 500.0,
        total_docs: int = 10000
    ) -> float:
        """Calculate BM25-like score for a document."""
        
        doc_tokens = self.tokenize(document_text)
        doc_length = len(doc_tokens)
        
        if doc_length == 0:
            return 0.0
        
        # Count term frequencies
        term_freq = Counter(doc_tokens)
        
        score = 0.0
        for term in query_terms:
            if term not in term_freq:
                continue
            
            # Term frequency in document
            tf = term_freq[term]
            
            # IDF calculation (simplified - would need corpus statistics in production)
            # Using cached or estimated IDF
            if term not in self._idf_cache:
                # Estimate based on term rarity (simplified)
                # In production, this would be calculated from actual corpus statistics
                estimated_df = max(1, total_docs * 0.1)  # Assume 10% of docs contain term
                self._idf_cache[term] = math.log((total_docs - estimated_df + 0.5) / (estimated_df + 0.5))
            
            idf = self._idf_cache[term]
            
            # BM25 formula
            norm_factor = 1 - self.b + self.b * (doc_length / avg_doc_length)
            tf_component = (tf * (self.k1 + 1)) / (tf + self.k1 * norm_factor)
            
            score += idf * tf_component
        
        return score
    
    def search_passages(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 50,
        offset: int = 0
    ) -> Tuple[List[SearchResult], int]:
        """Search for passages matching query with filters."""
        
        # Expand concepts if provided
        expanded_terms = set()
        if filters and filters.get("concept_qids"):
            expanded_terms = self.concept_store.expand_concepts_to_terms(filters["concept_qids"])
        
        # Combine query with expanded concept terms
        search_text = query
        if expanded_terms:
            # Add concept terms to search
            search_text = f"{query} {' '.join(expanded_terms)}"
        
        # Execute database search
        results = self.data_access.search_documents(
            text_query=search_text if search_text.strip() else None,
            iso3_filter=filters.get("iso3") if filters else None,
            language_filter=filters.get("language") if filters else None,
            corpus_type_filter=filters.get("corpus_type") if filters else None,
            family_slug_filter=filters.get("family_slug") if filters else None,
            limit=limit * 3,  # Get more results for ranking
            offset=offset
        )
        
        # Score and rank results
        query_terms = self.tokenize(query)
        scored_results = []
        
        for row in results:
            # Calculate BM25 score
            text = row.get("text", "")
            score = self.calculate_bm25_score(query_terms, text)
            
            # Boost score for concept matches
            if expanded_terms:
                text_lower = text.lower()
                for term in expanded_terms:
                    if term.lower() in text_lower:
                        score += 0.5  # Concept match bonus
            
            # Create search result
            passage_id = f"cpr://doc/{row['slug']}#p={row['text_index']}"
            
            # Return full passage text
            snippet = text  # Return full passage text
            
            # Handle potential None values
            title = row.get("document_title") or row.get("family_title") or "Untitled Document"
            
            result = SearchResult(
                id=passage_id,
                title=title,
                snippet=snippet,
                score=score
            )
            
            scored_results.append((score, result, row))
        
        # Sort by score (descending), then by document order
        scored_results.sort(key=lambda x: (-x[0], x[2]["family_title"], x[2]["document_title"], x[2]["text_index"]))
        
        # Apply max_passages_per_doc filter if specified
        if filters and filters.get("max_passages_per_doc"):
            max_per_doc = filters["max_passages_per_doc"]
            doc_counts = {}
            filtered_results = []
            
            for score, result, row in scored_results:
                doc_id = row["document_id"]
                if doc_counts.get(doc_id, 0) < max_per_doc:
                    filtered_results.append((score, result, row))
                    doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1
            
            scored_results = filtered_results
        
        # Apply final limit
        final_results = [result for _, result, _ in scored_results[:limit]]
        total_count = len(scored_results)
        
        return final_results, total_count
    
    def fetch_evidence(self, ids: List[str]) -> List[Evidence]:
        """Fetch full evidence objects for given IDs."""
        
        evidence_list = []
        
        for resource_id in ids:
            # Parse ID format: cpr://doc/{slug}#p={index}
            if not resource_id.startswith("cpr://doc/"):
                logger.warning(f"Invalid resource ID format: {resource_id}")
                continue
            
            try:
                # Extract slug and passage index
                parts = resource_id.replace("cpr://doc/", "").split("#p=")
                if len(parts) != 2:
                    logger.warning(f"Invalid resource ID format: {resource_id}")
                    continue
                
                slug = parts[0]
                passage_index = int(parts[1])
                
                # Get document metadata
                doc_meta = self.data_access.get_document_by_slug(slug)
                if not doc_meta:
                    logger.warning(f"Document not found: {slug}")
                    continue
                
                # Get passages
                passages = self.data_access.get_passages_by_slug(slug)
                
                # Find specific passage
                target_passage = None
                for passage in passages:
                    if passage["index"] == passage_index:
                        target_passage = passage
                        break
                
                if not target_passage:
                    logger.warning(f"Passage not found: {slug}#p={passage_index}")
                    continue
                
                # Create Evidence object
                document = DocumentMetadata(
                    slug=slug,
                    document_id=doc_meta["document_id"],
                    title=doc_meta.get("document_title") or doc_meta.get("family_title") or "Untitled Document",
                    family_title=doc_meta.get("family_title") or "Unknown Family",
                    geographies=doc_meta.get("geographies", []),
                    languages=doc_meta.get("languages", []),
                    url=f"https://app.climatepolicyradar.org/documents/{slug}",
                    corpus_type_name=doc_meta.get("corpus_type_name")
                )
                
                passage = Passage(
                    index=passage_index,
                    page_number=target_passage.get("page_number"),
                    type=target_passage.get("type", "body"),
                    text=target_passage.get("text", ""),
                    language=target_passage.get("language")
                )
                
                evidence = Evidence(
                    id=resource_id,
                    document=document,
                    passage=passage,
                    concept_matches=[],  # Will be populated when searching with concepts
                    score=0.0
                )
                
                evidence_list.append(evidence)
            
            except Exception as e:
                logger.error(f"Error fetching evidence for {resource_id}: {e}")
                continue
        
        return evidence_list
    
    def find_concept_mentions(
        self,
        concept_qids: List[str],
        iso3_filter: Optional[List[str]] = None,
        limit: int = 100
    ) -> List[Evidence]:
        """Find passages mentioning specific concepts."""
        
        # Expand concepts to terms
        concept_terms = {}
        for qid in concept_qids:
            terms = self.concept_store.expand_concept_to_terms(qid)
            concept_terms[qid] = terms
        
        # Build search query from all terms
        all_terms = set()
        for terms in concept_terms.values():
            all_terms.update(terms)
        
        search_query = " ".join(all_terms)
        
        # Search for documents
        results = self.data_access.search_documents(
            text_query=search_query,
            iso3_filter=iso3_filter,
            limit=limit * 2  # Get more for filtering
        )
        
        evidence_list = []
        
        for row in results[:limit]:
            text = row.get("text", "")
            text_lower = text.lower()
            
            # Find concept matches in text
            concept_matches = []
            for qid, terms in concept_terms.items():
                for term in terms:
                    term_lower = term.lower()
                    if term_lower in text_lower:
                        # Find all occurrences
                        offsets = []
                        start = 0
                        while True:
                            pos = text_lower.find(term_lower, start)
                            if pos == -1:
                                break
                            offsets.append([pos, pos + len(term)])
                            start = pos + 1
                        
                        if offsets:
                            # Determine match type
                            concept_info = self.concept_store.alias_map.get(qid, {})
                            if term == concept_info.get("label"):
                                match_type = "label"
                            elif term in concept_info.get("aliases", []):
                                match_type = "alias"
                            else:
                                match_type = "string-search"
                            
                            concept_matches.append(ConceptMatch(
                                qid=qid,
                                label=term,
                                match_type=match_type,
                                offsets=offsets
                            ))
            
            # Skip if no concept matches found
            if not concept_matches:
                continue
            
            # Create Evidence object
            passage_id = f"cpr://doc/{row['slug']}#p={row['text_index']}"
            
            document = DocumentMetadata(
                slug=row["slug"],
                document_id=row["document_id"],
                title=row.get("document_title") or row.get("family_title") or "Untitled Document",
                family_title=row.get("family_title") or "Unknown Family",
                geographies=row.get("geographies", []),
                languages=row.get("languages", []),
                url=f"https://app.climatepolicyradar.org/documents/{row['slug']}",
                corpus_type_name=row.get("corpus_type_name")
            )
            
            passage = Passage(
                index=row["text_index"],
                page_number=row.get("page_number"),
                type=row.get("text_type", "body"),
                text=text,
                language=row.get("text_language")
            )
            
            evidence = Evidence(
                id=passage_id,
                document=document,
                passage=passage,
                concept_matches=concept_matches,
                score=len(concept_matches)  # Simple score based on number of matches
            )
            
            evidence_list.append(evidence)
        
        # Sort by score (number of concept matches)
        evidence_list.sort(key=lambda x: -x.score)
        
        return evidence_list