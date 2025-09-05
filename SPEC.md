Doc — here’s a tight, build-ready spec for a **Dockerised FastMCP server** that **builds and serves** your climate law & policy knowledge graph to agentic clients. It’s opinionated, evidence-centric, and junior-engineer friendly.

# 1) Purpose & scope

Expose **primitive, composable KG actions** so an agent can: (a) find relevant law/policy passages with provenance, (b) traverse climate concepts (Wikibase), (c) summarise/compare across jurisdictions, and (d) fetch evidence bundles for citations.

# 2) Tech stack (concrete)

* **MCP framework:** FastMCP v2 (HTTP or SSE). Tools + Resources; optional Prompts.&#x20;
* **Deep-Research compatibility:** implement **exactly** `search` and `fetch` tools alongside domain tools.&#x20;
* **Storage:** DuckDB over a cached HF dataset; in-memory graph views (NetworkX optional).
* **Concept store:** CPR Wikibase SPARQL endpoint (read-only).
* **Optional auth:** Eunomia or Permit.io middleware for policy-based authorization.

# 3) Data inputs (as used in the tutorial notebook)

* **Documents dataset** (Parquet lake): `ClimatePolicyRadar/all-document-text-data` (via `huggingface_hub.snapshot_download`).
  Key columns used in the tutorial:
  `document_id`, `document_metadata.corpus_type_name`, `document_metadata.document_title`, `document_metadata.family_title`, `document_metadata.geographies` (ISO-3 list), `document_metadata.languages`, `document_metadata.slug`, `text_block.index`, `text_block.language`, `text_block.page_number`, `text_block.type`, `text_block.text`.
* **Concept store:** CPR **Wikibase** (labels/aliases/relationships) via SPARQL at `https://climatepolicyradar.wikibase.cloud/query/sparql`.
* **Assumption:** no writes; server is a read layer with local caches.

# 4) Canonical IDs & evidence schema

Use **stable URIs** so agents can stitch results deterministically.

* **Document passage ID**:
  `cpr://doc/{slug}#p={text_block.index}`
* **Document family ID**:
  `cpr://family/{slug}` (or family slug)
* **Concept ID**:
  `cpr://concept/{QID}` (e.g., `Q1167`)
* **Evidence object (returned by most tools)**:

```json
{
  "id": "cpr://doc/{slug}#p={index}",
  "document": {
    "slug": "string",
    "document_id": "string",
    "title": "string",
    "family_title": "string",
    "geographies": ["ISO3", "..."],
    "languages": ["en", "..."],
    "url": "https://app.climatepolicyradar.org/documents/{slug}"
  },
  "passage": {
    "index": 123,
    "page_number": 4,
    "type": "body|title|table|…",
    "text": "…exact text…"
  },
  "concept_matches": [
    {"qid": "Q1167", "label": "Solar energy", "match_type": "label|alias|string-search", "offsets": [[start, end], "..."]}
  ],
  "score": 0.0
}
```

# 5) MCP tools (APIs) — primitives an agent can chain

## A. Required Deep-Research pair

1. `search` — **generic retrieval over passages**
   **Input**:

```json
{
  "query": "string",
  "filters": {
    "iso3": ["BRA", "ZAF"],
    "language": ["en"],
    "corpus_type": ["National Law", "Executive"],
    "family_slug": ["optional"],
    "concept_qids": ["Q####", "..."],  // optional: expand to labels/aliases
    "max_passages_per_doc": 3
  },
  "limit": 50,
  "offset": 0
}
```

**Behaviour:**

* If `concept_qids` present: expand to `(preferred_label ∪ aliases)` and OR into full-text `ILIKE` search over `text_block.text`.
* Apply filters against DuckDB views (`open_data`, `open_data_english`).
* Rank: BM25-ish simple scoring (fallback: term hits + rare term weight); break ties by recency if a date is available, else by lower `text_block.index`.
  **Output:** array of **resource IDs** (`cpr://doc/{slug}#p={index}`) with lightweight metadata (title, snippet).

2. `fetch` — **hydrate IDs to full evidence**
   **Input:** `{ "ids": ["cpr://doc/{slug}#p={index}", "..."] }`
   **Output:** array of the **Evidence objects** defined above.
   *These two make your server plug-and-play with ChatGPT Deep Research.*&#x20;

## B. Concept store (Wikibase) traversal

3. `concept_find` — **search concepts by label/alias/QID**
   **Input:** `{ "q": "solar", "limit": 20 }` or `{ "qid": "Q1167" }`
   **Output:** concepts with `{qid, preferred_label, aliases[], description?, instance_of?, broader?, narrower?}`.

4. `concept_neighbors` — **1-hop/2-hop neighborhood**
   **Input:**

```json
{"qid":"Q1167","direction":"both|out|in","predicates":["P###","..."],"hops":1,"limit":100}
```

**Output:** edges `{subject_qid, predicate_pid, object_qid, labels, counts?}`.
*(Backed by SPARQL; cache results; include direct properties only via `wikibase:directClaim` filter.)*

5. `concept_properties` — **frequency of used properties** (global or per concept)
   **Input:** `{ "qid": "optional" }`
   **Output:** `[{pid, propLabel, count}]`.
   *(This mirrors the “all properties used” SPARQL the notebook sketches.)*

## C. KG-aware document lookups

6. `concept_mentions` — **find passages likely mentioning concept(s)**
   **Input:** `{ "qids": ["Q1167","Q758"], "iso3": ["BRA"], "limit": 100 }`
   **Behaviour:** expand qids → `(label ∪ aliases)`; string-match in `text_block.text`; return evidence with offsets.
   **Output:** array of Evidence; `concept_matches` populated.

7. `family_overview` — **group documents by policy family**
   **Input:** `{ "iso3": ["BRA"], "language":["en"] }`
   **Output:** `[{family_title, doc_count, sample_doc: {slug,title}}]`.

8. `jurisdiction_overview` — **distribution summaries**
   **Input:** `{ "iso3": "BRA" }`
   **Output:**

* docs by `corpus_type_name`
* top concepts by co-occurrence (if `concept_mentions` cache exists)
* top languages
  (All as small typed arrays so agents can graph.)

9. `compare_jurisdictions` — **side-by-side counts**
   **Input:** `{ "iso3_a":"BRA", "iso3_b":"ZAF", "by":"corpus_type|language" }`
   **Output:** `{ bins: ["National Law","…"], a: [..], b: [..] }`.

10. `policy_lineage` — **follow family history**
    **Input:** `{ "family_slug":"...", "include_docs": true }`
    **Output:** ordered list of documents (titles, slugs) that belong to the same family; if you later map legal relations (amends/repeals) in Wikibase, include edges.

## D. Graph introspection & admin

11. `list_resources` — enumerate available views / cache status.
12. `health` — dataset revision, SPARQL reachability, cache ages.

> All tools return **pure JSON** with explicit, predictable shapes. Keep inputs primitive (strings, arrays, numbers); keep outputs tiny but composable.

# 6) SQL/SPARQL sketches (for implementers)

**DuckDB setup**

* Cache HF dataset (pin `REVISION`); build views:

```sql
CREATE VIEW open_data        AS SELECT * FROM read_parquet('./cache/*.parquet');
CREATE VIEW open_data_english AS SELECT * FROM open_data WHERE "text_block.language" = 'en';
```

* Basic filtered doc list by ISO-3:

```sql
SELECT DISTINCT document_id,
       "document_metadata.family_title",
       "document_metadata.document_title",
       "document_metadata.slug"
FROM open_data_english
WHERE 'BRA' = ANY("document_metadata.geographies")
ORDER BY "document_metadata.family_title";
```

* Fetch ordered passages for a slug:

```sql
SELECT "text_block.index","text_block.page_number","text_block.type","text_block.text"
FROM open_data_english
WHERE "document_metadata.slug" = ?
ORDER BY "text_block.index" ASC;
```

**SPARQL endpoint** (CPR Wikibase)

* **Properties by frequency** (direct claims only):

```sparql
SELECT ?prop ?propLabel (COUNT(?s) as ?count) WHERE {
  ?s ?p ?o .
  ?prop wikibase:directClaim ?p.
  FILTER(STRSTARTS(STR(?p), "https://climatepolicyradar.wikibase.cloud/prop/direct/"))
  SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
}
GROUP BY ?prop ?propLabel
ORDER BY DESC(?count)
```

* **Neighbors for a QID** (sketch):

```sparql
PREFIX ent: <https://climatepolicyradar.wikibase.cloud/entity/>
SELECT ?subject ?subjectLabel ?predicate ?predicateLabel ?object ?objectLabel WHERE {
  { BIND(ent:Q1167 AS ?subject) ?subject ?p ?object .
    FILTER(STRSTARTS(STR(?p), "https://climatepolicyradar.wikibase.cloud/prop/direct/")) }
  ?predicate wikibase:directClaim ?p .
  SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
}
LIMIT 200
```

# 7) Server contract (FastMCP)

**General**

* All tools decorated `@mcp.tool` with docstrings explaining inputs/outputs.
* `search` returns only IDs + small snippets; `fetch` hydrates to Evidence.&#x20;
* Register **Resources** for static metadata (e.g., `cpr://status`, `cpr://dataset/revision`).&#x20;

**Auth (optional, strongly recommended)**

* Add **Eunomia** or **Permit.io** middleware to gate `tools/list`, `tools/call`, etc.; configure via env.
* Example one-liner add-middleware & policy file (`mcp_policies.json`).&#x20;
* Permit.io supports **RBAC/ABAC**, including conditions on **tool arguments**.

# 8) Dockerisation

**Dockerfile (multi-stage, reproducible)**

```dockerfile
FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y curl build-essential && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip uv

WORKDIR /app
COPY server/pyproject.toml server/uv.lock* /app/
RUN uv sync --frozen

COPY server /app
# cache mount for HF data
VOLUME ["/data/cache"]
ENV DATA_CACHE_DIR=/data/cache \
    HF_HOME=/data/hf \
    SPARQL_ENDPOINT=https://climatepolicyradar.wikibase.cloud/query/sparql \
    DATASET_REPO=ClimatePolicyRadar/all-document-text-data \
    DATASET_REVISION=main \
    PORT=8000

EXPOSE 8000
CMD ["uv", "run", "server.py", "--port", "8000", "--transport", "http"]
```

**docker-compose.yml**

```yaml
services:
  kg-mcp:
    build: .
    ports: ["8000:8000"]
    environment:
      - DATASET_REPO=ClimatePolicyRadar/all-document-text-data
      - DATASET_REVISION=main
      - SPARQL_ENDPOINT=https://climatepolicyradar.wikibase.cloud/query/sparql
      # Optional auth:
      - FASTMCP_SERVER_AUTH=AZURE   # or configure Eunomia/Permit
    volumes:
      - ./var/cache:/data/cache
```

* If you layer **auth**: follow FastMCP provider/middleware envs (Azure OAuth, Eunomia/Permit).

# 9) Implementation notes (junior-friendly)

* **Project layout**

```
server/
  server.py           # FastMCP app + tools
  data_access.py      # duckdb session, HF snapshot, helpers
  concepts.py         # SPARQL client, label/alias expansion, neighbor queries
  search.py           # ranking & filtering
  schemas.py          # Pydantic models for inputs/outputs (Evidence, etc.)
  cache/              # optional json/parquet caches (gitignored)
```

* **DuckDB session**: open once on startup; build `open_data` / `open_data_english` views.
* **HF snapshot**: pin `DATASET_REVISION`; create DuckDB **SECRET** for HF auth if needed (same pattern as the notebook).
* **SPARQL**: simple `POST` with `Accept: application/sparql-results+json`; 10s timeout; on failure → cached results or 502.
* **Scoring**: start simple (token overlap); expose `score` transparently so agents can reason about uncertainty.
* **Pagination**: every list-returning tool takes `limit`/`offset`.
* **Determinism**: sort secondarily by `(family_title, document_title, text_block.index)`.

# 10) Testing

* **Unit**:

  * `concept_find("solar")` returns at least one `QID`, labels non-empty.
  * `search("solar", filters={iso3:["BRA"]})` returns only `BRA` documents; IDs well-formed.
  * `fetch([id])` returns exact passage text and valid provenance URL.
* **E2E**:

  * Run server, call `tools/list` and ensured `search`/`fetch` exposed (Deep-Research).&#x20;
  * If auth enabled, unauthorized calls to `tools/call` are blocked per policy.

# 11) Performance & ops

* **Caching**: on first use, materialise:

  * `alias_map.parquet` for concept label/alias lookups;
  * `mentions_index.parquet` if you precompute string matches for top N concepts.
* **Rate limits**: guard SPARQL with 1–2 qps + 5-min TTL cache.
* **Observability**: log tool name, args (redacted), result counts, latency; add auth audit logs if using middleware.&#x20;

# 12) Security (optional but recommended)

* **Eunomia** quick add (JSON policy file + middleware).&#x20;
* **Permit.io** for RBAC/ABAC/REBAC with argument-level conditions; supports JWT/header identities.

---

If you want, I’ll turn this into **ready-to-run code** (`server.py`, `Dockerfile`, `docker-compose.yml`) with Pydantic schemas and stubbed SQL/SPARQL calls wired up, then we can iterate on ranking and a tiny mentions cache.
