"""
SPARQL client for CPR Wikibase concept store.
Handles concept queries, label expansion, and graph traversal.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Set
from datetime import datetime, timedelta
from pathlib import Path
import httpx
from functools import lru_cache

logger = logging.getLogger(__name__)


class ConceptStore:
    """Client for CPR Wikibase SPARQL endpoint."""
    
    def __init__(
        self,
        sparql_endpoint: str = "https://climatepolicyradar.wikibase.cloud/query/sparql",
        cache_dir: str = "./cache",
        cache_ttl_minutes: int = 5
    ):
        self.sparql_endpoint = sparql_endpoint
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.cache_ttl = timedelta(minutes=cache_ttl_minutes)
        self._cache: Dict[str, Any] = {}
        self._cache_timestamps: Dict[str, datetime] = {}
        
        # Load or create alias map cache
        self.alias_map_file = self.cache_dir / "alias_map.json"
        self.alias_map = self._load_alias_map()
    
    def _load_alias_map(self) -> Dict[str, Dict[str, Any]]:
        """Load cached alias map or create empty one."""
        if self.alias_map_file.exists():
            try:
                with open(self.alias_map_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load alias map: {e}")
        return {}
    
    def _save_alias_map(self):
        """Save alias map to cache."""
        try:
            with open(self.alias_map_file, 'w') as f:
                json.dump(self.alias_map, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save alias map: {e}")
    
    def _get_cached(self, key: str) -> Optional[Any]:
        """Get cached result if still valid."""
        if key in self._cache:
            timestamp = self._cache_timestamps.get(key)
            if timestamp and (datetime.now() - timestamp) < self.cache_ttl:
                return self._cache[key]
        return None
    
    def _set_cached(self, key: str, value: Any):
        """Set cached result."""
        self._cache[key] = value
        self._cache_timestamps[key] = datetime.now()
    
    async def execute_sparql(self, query: str) -> Optional[Dict[str, Any]]:
        """Execute SPARQL query with retry and caching."""
        
        # Check cache
        cache_key = f"sparql_{hash(query)}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached
        
        # Execute query
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    self.sparql_endpoint,
                    data={"query": query},
                    headers={"Accept": "application/sparql-results+json"}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    self._set_cached(cache_key, result)
                    return result
                else:
                    logger.error(f"SPARQL query failed: {response.status_code}")
                    return None
        
        except Exception as e:
            logger.error(f"SPARQL query error: {e}")
            return None
    
    def execute_sparql_sync(self, query: str) -> Optional[Dict[str, Any]]:
        """Synchronous version of execute_sparql."""
        cache_key = f"sparql_{hash(query)}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached
        
        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.post(
                    self.sparql_endpoint,
                    data={"query": query},
                    headers={"Accept": "application/sparql-results+json"}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    self._set_cached(cache_key, result)
                    return result
                else:
                    logger.error(f"SPARQL query failed: {response.status_code}")
                    return None
        
        except Exception as e:
            logger.error(f"SPARQL query error: {e}")
            return None
    
    def find_concepts(self, search_term: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Search for concepts by label or alias."""
        
        query = f"""
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
        PREFIX wikibase: <http://wikiba.se/ontology#>
        PREFIX bd: <http://www.bigdata.com/rdf#>
        
        SELECT DISTINCT ?item ?itemLabel ?itemDescription ?altLabel WHERE {{
          {{
            ?item rdfs:label ?label .
            FILTER(CONTAINS(LCASE(?label), LCASE("{search_term}")))
          }} UNION {{
            ?item skos:altLabel ?altLabel .
            FILTER(CONTAINS(LCASE(?altLabel), LCASE("{search_term}")))
          }}
          
          SERVICE wikibase:label {{ 
            bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". 
            ?item rdfs:label ?itemLabel .
            ?item schema:description ?itemDescription .
          }}
        }}
        LIMIT {limit}
        """
        
        result = self.execute_sparql_sync(query)
        if not result:
            return []
        
        concepts = []
        for binding in result.get("results", {}).get("bindings", []):
            qid = binding["item"]["value"].split("/")[-1]
            
            # Update alias map
            if qid not in self.alias_map:
                self.alias_map[qid] = {
                    "label": binding.get("itemLabel", {}).get("value", ""),
                    "aliases": []
                }
            
            if "altLabel" in binding:
                alias = binding["altLabel"]["value"]
                if alias not in self.alias_map[qid]["aliases"]:
                    self.alias_map[qid]["aliases"].append(alias)
            
            concepts.append({
                "qid": qid,
                "preferred_label": binding.get("itemLabel", {}).get("value", ""),
                "description": binding.get("itemDescription", {}).get("value"),
                "aliases": self.alias_map.get(qid, {}).get("aliases", [])
            })
        
        self._save_alias_map()
        return concepts
    
    def get_concept_by_qid(self, qid: str) -> Optional[Dict[str, Any]]:
        """Get concept details by QID."""
        
        query = f"""
        PREFIX ent: <https://climatepolicyradar.wikibase.cloud/entity/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
        PREFIX wikibase: <http://wikiba.se/ontology#>
        PREFIX bd: <http://www.bigdata.com/rdf#>
        PREFIX wdt: <https://climatepolicyradar.wikibase.cloud/prop/direct/>
        
        SELECT ?itemLabel ?itemDescription ?altLabel ?instanceOf ?instanceOfLabel ?broader ?broaderLabel WHERE {{
          BIND(ent:{qid} AS ?item)
          
          OPTIONAL {{ ?item skos:altLabel ?altLabel }}
          OPTIONAL {{ ?item wdt:P6 ?instanceOf }}
          OPTIONAL {{ ?item wdt:P9 ?broader }}
          
          SERVICE wikibase:label {{ 
            bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". 
            ?item rdfs:label ?itemLabel .
            ?item schema:description ?itemDescription .
            ?instanceOf rdfs:label ?instanceOfLabel .
            ?broader rdfs:label ?broaderLabel .
          }}
        }}
        """
        
        result = self.execute_sparql_sync(query)
        if not result or not result.get("results", {}).get("bindings"):
            return None
        
        bindings = result["results"]["bindings"]
        if not bindings:
            return None
        
        # Collect all aliases
        aliases = []
        for binding in bindings:
            if "altLabel" in binding:
                alias = binding["altLabel"]["value"]
                if alias not in aliases:
                    aliases.append(alias)
        
        # Get first binding for main info
        first = bindings[0]
        
        concept = {
            "qid": qid,
            "preferred_label": first.get("itemLabel", {}).get("value", ""),
            "description": first.get("itemDescription", {}).get("value"),
            "aliases": aliases
        }
        
        if "instanceOf" in first:
            concept["instance_of"] = first["instanceOfLabel"]["value"]
        
        if "broader" in first:
            concept["broader"] = [first["broaderLabel"]["value"]]
        
        # Update alias map
        self.alias_map[qid] = {
            "label": concept["preferred_label"],
            "aliases": aliases
        }
        self._save_alias_map()
        
        return concept
    
    def get_concept_neighbors(
        self,
        qid: str,
        direction: str = "both",
        predicates: Optional[List[str]] = None,
        hops: int = 1,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get neighbors of a concept in the graph."""
        
        # Build predicate filter
        pred_filter = ""
        if predicates:
            pred_list = ", ".join([f"wdt:{p}" for p in predicates])
            pred_filter = f"FILTER(?p IN ({pred_list}))"
        else:
            pred_filter = 'FILTER(STRSTARTS(STR(?p), "https://climatepolicyradar.wikibase.cloud/prop/direct/"))'
        
        # Build query based on direction
        if direction == "out":
            pattern = f"ent:{qid} ?p ?object ."
        elif direction == "in":
            pattern = f"?subject ?p ent:{qid} ."
        else:  # both
            pattern = f"""
            {{ ent:{qid} ?p ?object . }}
            UNION
            {{ ?subject ?p ent:{qid} . }}
            """
        
        query = f"""
        PREFIX ent: <https://climatepolicyradar.wikibase.cloud/entity/>
        PREFIX wdt: <https://climatepolicyradar.wikibase.cloud/prop/direct/>
        PREFIX wikibase: <http://wikiba.se/ontology#>
        PREFIX bd: <http://www.bigdata.com/rdf#>
        
        SELECT ?subject ?subjectLabel ?predicate ?predicateLabel ?object ?objectLabel WHERE {{
          {pattern}
          {pred_filter}
          
          ?predicate wikibase:directClaim ?p .
          
          SERVICE wikibase:label {{ 
            bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". 
          }}
        }}
        LIMIT {limit}
        """
        
        result = self.execute_sparql_sync(query)
        if not result:
            return []
        
        edges = []
        for binding in result.get("results", {}).get("bindings", []):
            edge = {}
            
            if "subject" in binding:
                edge["subject_qid"] = binding["subject"]["value"].split("/")[-1]
                edge["subject_label"] = binding.get("subjectLabel", {}).get("value", "")
            else:
                edge["subject_qid"] = qid
            
            if "object" in binding:
                edge["object_qid"] = binding["object"]["value"].split("/")[-1]
                edge["object_label"] = binding.get("objectLabel", {}).get("value", "")
            else:
                edge["object_qid"] = qid
            
            edge["predicate_pid"] = binding["predicate"]["value"].split("/")[-1]
            edge["predicate_label"] = binding.get("predicateLabel", {}).get("value", "")
            
            edges.append(edge)
        
        return edges
    
    def get_property_frequencies(self, qid: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get frequency of properties used (globally or for a concept)."""
        
        if qid:
            # Properties for specific concept
            query = f"""
            PREFIX ent: <https://climatepolicyradar.wikibase.cloud/entity/>
            PREFIX wikibase: <http://wikiba.se/ontology#>
            PREFIX bd: <http://www.bigdata.com/rdf#>
            
            SELECT ?prop ?propLabel (COUNT(?o) as ?count) WHERE {{
              ent:{qid} ?p ?o .
              ?prop wikibase:directClaim ?p .
              FILTER(STRSTARTS(STR(?p), "https://climatepolicyradar.wikibase.cloud/prop/direct/"))
              
              SERVICE wikibase:label {{ 
                bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". 
              }}
            }}
            GROUP BY ?prop ?propLabel
            ORDER BY DESC(?count)
            """
        else:
            # Global property frequencies
            query = """
            PREFIX wikibase: <http://wikiba.se/ontology#>
            PREFIX bd: <http://www.bigdata.com/rdf#>
            
            SELECT ?prop ?propLabel (COUNT(?s) as ?count) WHERE {
              ?s ?p ?o .
              ?prop wikibase:directClaim ?p .
              FILTER(STRSTARTS(STR(?p), "https://climatepolicyradar.wikibase.cloud/prop/direct/"))
              
              SERVICE wikibase:label { 
                bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". 
              }
            }
            GROUP BY ?prop ?propLabel
            ORDER BY DESC(?count)
            LIMIT 50
            """
        
        result = self.execute_sparql_sync(query)
        if not result:
            return []
        
        properties = []
        for binding in result.get("results", {}).get("bindings", []):
            properties.append({
                "pid": binding["prop"]["value"].split("/")[-1],
                "propLabel": binding.get("propLabel", {}).get("value", ""),
                "count": int(binding["count"]["value"])
            })
        
        return properties
    
    def expand_concept_to_terms(self, qid: str) -> Set[str]:
        """Expand a concept QID to its label and all aliases."""
        
        # Check cache first
        if qid in self.alias_map:
            terms = {self.alias_map[qid]["label"]}
            terms.update(self.alias_map[qid]["aliases"])
            return terms
        
        # Fetch from SPARQL
        concept = self.get_concept_by_qid(qid)
        if not concept:
            return set()
        
        terms = {concept["preferred_label"]}
        terms.update(concept.get("aliases", []))
        return terms
    
    def expand_concepts_to_terms(self, qids: List[str]) -> Set[str]:
        """Expand multiple concept QIDs to their labels and aliases."""
        terms = set()
        for qid in qids:
            terms.update(self.expand_concept_to_terms(qid))
        return terms
    
    def is_reachable(self) -> bool:
        """Check if SPARQL endpoint is reachable."""
        try:
            query = "SELECT ?s WHERE { ?s ?p ?o } LIMIT 1"
            result = self.execute_sparql_sync(query)
            return result is not None
        except:
            return False