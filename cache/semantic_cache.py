import numpy as np
from typing import Optional

class SemanticCache:
    def __init__(self, similarity_threshold: float = 0.75):
        self.cache = []  # list of dicts
        self.similarity_threshold = similarity_threshold
        self.hit_count = 0
        self.miss_count = 0

    def lookup(self, query_embedding: np.ndarray) -> Optional[dict]:
        """
        Check if a query is semantically close enough to a cached query.
        Returns the cache entry if found, else None.
        """
        for entry in self.cache:
            cached_embedding = entry["embedding"]
            similarity = self.cosine_similarity(query_embedding, cached_embedding)
            if similarity >= self.similarity_threshold:
                self.hit_count += 1
                entry["similarity_score"] = similarity
                return entry

        self.miss_count += 1
        return None

    def add(self, query: str, embedding: np.ndarray, result_text: str, cluster: int):
        """
        Add a new cache entry with query, embedding, cluster, and actual document text.
        """
        cache_entry = {
            "query": query,
            "embedding": embedding,
            "matched_query": query,
            "result_text": result_text,  # actual document content
            "cluster": cluster,
            "similarity_score": None
        }
        self.cache.append(cache_entry)

    def clear(self):
        self.cache = []
        self.hit_count = 0
        self.miss_count = 0

    def stats(self):
        total_entries = len(self.cache)
        hit_rate = self.hit_count / (self.hit_count + self.miss_count) if (self.hit_count + self.miss_count) > 0 else 0.0
        return {
            "total_entries": total_entries,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": hit_rate
        }

    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))