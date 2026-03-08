import numpy as np
import skfuzzy as fuzz

print("Loading embeddings...")

embeddings = np.load("vector_store/document_embeddings.npy")

print("Embedding shape:", embeddings.shape)

data = embeddings.T

n_clusters = 20

print("Running fuzzy clustering...")

cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
    data,
    c=n_clusters,
    m=2,
    error=0.005,
    maxiter=1000
)

print("Clustering finished")

np.save("vector_store/membership_matrix.npy", u)

print("Membership matrix saved!")