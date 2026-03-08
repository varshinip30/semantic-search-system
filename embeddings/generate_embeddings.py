from sklearn.datasets import fetch_20newsgroups
from sentence_transformers import SentenceTransformer
from utils.preprocessing import clean_text
import numpy as np
import os

print("Loading dataset...")

data = fetch_20newsgroups(
    subset='all',
    remove=('headers', 'footers', 'quotes')
)

documents = [clean_text(doc) for doc in data.data]

print("Number of documents:", len(documents))


print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")


print("Generating embeddings (this may take a few minutes)...")

embeddings = model.encode(
    documents,
    show_progress_bar=True,
    batch_size=32
)


print("Embedding shape:", embeddings.shape)


os.makedirs("vector_store", exist_ok=True)

np.save("vector_store/document_embeddings.npy", embeddings)

print("Embeddings saved successfully!")