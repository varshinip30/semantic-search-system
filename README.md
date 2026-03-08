# Semantic Search System with Fuzzy Clustering and Semantic Cache

## Overview

A lightweight **semantic search system** for the **20 Newsgroups dataset** with:

- **Fuzzy Clustering** – soft document clusters  
- **Semantic Cache** – avoids redundant computations  
- **FastAPI Service** – live API for queries and cache stats

---

## Project Structure
```text
semantic-search-system
├── main.py
├── requirements.txt
├── Dockerfile
├── templates/index.html
├── cache/semantic_cache.py
└── clustering/fuzzy_cluster.py