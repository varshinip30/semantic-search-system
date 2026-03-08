import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

print("Loading embeddings...")
doc_embeddings = np.load("vector_store/document_embeddings.npy")

print("Loading model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

while True:
    query = input("\nEnter search query (or type exit): ")

    if query.lower() == "exit":
        break

    query_embedding = model.encode([query])

    similarities = cosine_similarity(query_embedding, doc_embeddings)[0]

    top_indices = np.argsort(similarities)[-5:][::-1]

    print("\nTop results:")
    for i, idx in enumerate(top_indices):
        print(f"{i+1}. Document index: {idx} | Score: {similarities[idx]:.4f}")