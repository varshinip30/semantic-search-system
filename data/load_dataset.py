import zipfile
import os
import pickle
from sklearn.datasets import fetch_20newsgroups
from utils.preprocessing import clean_text

# Paths
BASE_DIR = os.path.dirname(__file__)
ZIP_PATH = os.path.join(BASE_DIR, "twenty+newsgroups.zip")
EXTRACT_FOLDER = os.path.join(BASE_DIR, "twenty_newsgroups_data")

DOCS_PATH = os.path.join(EXTRACT_FOLDER, "documents.pkl")
ORIG_DOCS_PATH = os.path.join(EXTRACT_FOLDER, "original_documents.pkl")

# Extract zip if not already done
if not os.path.exists(EXTRACT_FOLDER):
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_FOLDER)
    print(f"Dataset extracted to {EXTRACT_FOLDER}")
else:
    print(f"Dataset already extracted at {EXTRACT_FOLDER}")

# If processed dataset already exists, load it
if os.path.exists(DOCS_PATH) and os.path.exists(ORIG_DOCS_PATH):

    print("Loading cached documents...")

    with open(DOCS_PATH, "rb") as f:
        documents = pickle.load(f)

    with open(ORIG_DOCS_PATH, "rb") as f:
        original_documents = pickle.load(f)

else:
    print("Processing dataset...")

    data = fetch_20newsgroups(
        subset='all',
        remove=('headers', 'footers', 'quotes'),
        data_home=EXTRACT_FOLDER
    )

    # Raw documents with punctuation
    original_documents = data.data

    # Cleaned documents for embeddings/search
    documents = [clean_text(doc) for doc in original_documents]

    # Save processed data using pickle (safe for text)
    with open(DOCS_PATH, "wb") as f:
        pickle.dump(documents, f)

    with open(ORIG_DOCS_PATH, "wb") as f:
        pickle.dump(original_documents, f)

print("Number of documents:", len(documents))
print("\nCleaned Example:\n")
print(documents[0][:500])