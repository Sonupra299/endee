import requests
import numpy as np

print("NumPy version:", np.__version__)
from sentence_transformers import SentenceTransformer
import numpy as np

# Sample documents
documents = [
    "AI is transforming healthcare",
    "Machine learning is a subset of AI",
    "Python is popular for data science",
    "Vector databases enable semantic search"
]

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Convert documents to vectors
doc_embeddings = model.encode(documents)

# Search function
def semantic_search(query):
    query_embedding = model.encode([query])[0]
    scores = np.dot(doc_embeddings, query_embedding)
    top_k = 3
    top_indices = np.argsort(scores)[-top_k:][::-1]
    return [documents[i] for i in top_indices]

# Test
query = input("Enter your search query: ")
print("Query:", query)
results = semantic_search(query)

print("Top results:")
for r in results:
    print("-", r)

