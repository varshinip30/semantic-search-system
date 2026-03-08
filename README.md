Semantic Search System with Fuzzy Clustering and Semantic Cache
Overview

A lightweight semantic search system for the 20 Newsgroups dataset (~20,000 news posts, 20 categories).

Core Components:

Fuzzy Clustering → soft document membership across clusters

Semantic Cache → avoids redundant computation for similar queries

FastAPI Service → live API with query handling and cache management

Project Structure
semantic-search-system
├── main.py
├── requirements.txt
├── Dockerfile
├── templates/index.html
├── cache/semantic_cache.py
└── clustering/fuzzy_cluster.py
Installation & Running

Clone the repository:

git clone <repository-url>
cd semantic-search-system

Install dependencies:

pip install -r requirements.txt

Start the FastAPI server:

uvicorn main:app --reload

Open in browser: http://127.0.0.1:8000

Optional Docker (Bonus)

docker build -t semantic-search .
docker run -p 8000:8000 semantic-search
API Endpoints
POST /query

Checks semantic cache and returns results for a query.

Example request:

{
  "query": "machine learning techniques"
}

Example response:

{
  "query": "machine learning techniques",
  "cache_hit": true,
  "matched_query": "machine learning",
  "similarity_score": 0.772,
  "result": "Document index: 6121",
  "dominant_cluster": 1
}
GET /cache/stats

Returns cache metrics.

{
  "total_entries": 42,
  "hit_count": 17,
  "miss_count": 25,
  "hit_rate": 0.405
}
DELETE /cache

Flushes cache and resets stats.

Features

Semantic search using sentence embeddings

Vector similarity search with FAISS

Fuzzy clustering (soft cluster assignments)

Semantic caching for repeated or similar queries

FastAPI backend + simple web UI

Optional Docker containerization

Notes

Dataset preprocessing and embedding choices are handled in main.py.

Cache similarity threshold is tunable; helps balance hit rate vs. accuracy.

Fuzzy clusters capture overlapping semantic topics in the dataset