from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request, FastAPI
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from cache.semantic_cache import SemanticCache
from data.load_dataset import documents, original_documents  # cleaned and original posts

app = FastAPI()
templates = Jinja2Templates(directory="templates")

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Loading FAISS index...")
index = faiss.read_index("vector_store/faiss_index.bin")

print("Loading embeddings...")
doc_embeddings = np.load("vector_store/document_embeddings.npy")

print("Loading cluster memberships...")
membership = np.load("vector_store/membership_matrix.npy")

cache = SemanticCache()

SIMILARITY_THRESHOLD = 0.25  

@app.get("/", response_class=HTMLResponse)
def homepage(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/query")
def query_search(body: dict):
    query = body["query"]
    query_embedding = model.encode([query])[0]

    # --- Check cache first ---
    cache_result = cache.lookup(query_embedding)
    if cache_result:
        return {
            "query": query,
            "cache_hit": True,
            "matched_query": cache_result["matched_query"],
            "similarity_score": cache_result["similarity_score"],
            "result": cache_result["result_text"],  # full original document
            "dominant_cluster": cache_result["cluster"]
        }

    # --- Cache miss: search FAISS ---
    query_vector = np.array([query_embedding]).astype("float32")
    distances, indices = index.search(query_vector, 1)
    best_idx = int(indices[0][0])

    # Cosine similarity to verify match
    similarity_score = np.dot(query_embedding, doc_embeddings[best_idx]) / (
        np.linalg.norm(query_embedding) * np.linalg.norm(doc_embeddings[best_idx])
    )

    if similarity_score < SIMILARITY_THRESHOLD:
        result_text = "No matching content found in dataset."
    else:
        result_text = original_documents[best_idx]

    dominant_cluster = int(np.argmax(membership[:, best_idx]))

    # Add to cache
    cache.add(
        query=query,
        embedding=query_embedding,
        result_text=result_text,
        cluster=dominant_cluster
    )

    return {
        "query": query,
        "cache_hit": False,
        "matched_query": None,
        "similarity_score": float(similarity_score),
        "result": result_text,
        "dominant_cluster": dominant_cluster
    }

@app.get("/cache/stats")
def cache_stats():
    return cache.stats()

@app.delete("/cache")
def clear_cache():
    cache.clear()
    return {"message": "Cache cleared"}