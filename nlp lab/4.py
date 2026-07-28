import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

# Store documents
docs = []

# Get number of documents
n = int(input("Enter number of documents: "))

# Read documents
for i in range(n):
    docs.append(input("Enter document: "))

# Get search query
query = input("\nEnter search query: ")

# Convert documents to TF-IDF vectors
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(docs)

# Convert query to TF-IDF vector
query_vec = vectorizer.transform([query])

# Calculate TF-IDF cosine similarity
scores = cosine_similarity(query_vec, X)

print("\nTF-IDF Similarity Scores:")
for i, s in enumerate(scores[0]):
    print("Document", i + 1, ":", round(s, 3))

# Apply LSA using Truncated SVD
svd = TruncatedSVD(n_components=2)
X_lsa = svd.fit_transform(X)
query_lsa = svd.transform(query_vec)

# Calculate LSA cosine similarity
lsa_scores = cosine_similarity(query_lsa, X_lsa)

print("\nLSA Similarity Scores:")
for i, s in enumerate(lsa_scores[0]):
    print("Document", i + 1, ":", round(s, 3))

# Find the most relevant document
best = np.argmax(lsa_scores)

print("\nMost Relevant Document:")
print(docs[best])