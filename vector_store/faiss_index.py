import numpy as np
import faiss

print("Loading document embeddings...")

embeddings = np.load("vector_store/document_embeddings.npy")

# FAISS requires float32
embeddings = embeddings.astype("float32")

dimension = embeddings.shape[1]

print("Creating FAISS index...")

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

print("Number of vectors in index:", index.ntotal)

# save index
faiss.write_index(index, "vector_store/faiss_index.bin")

print("FAISS index saved successfully!")