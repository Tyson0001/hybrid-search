from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from app.graph import add_profile

# 🔹 Load Model
model = SentenceTransformer('all-MiniLM-L6-v2')

# 🔹 Load Dataset
df = pd.read_csv("data/profiles.csv")
df = df.fillna("")

# 🔹 Convert to docs (structured + text field)
docs = df.to_dict(orient="records")

for doc in docs:
    doc["text"] = " ".join(str(v) for v in doc.values())

# 🔹 Text list for models
texts = [doc["text"] for doc in docs]

# 🔹 Dense Embeddings (FAISS)
embeddings = model.encode(texts)

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

# 🔹 TF-IDF (BM25 alternative)
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(texts)

# 🔹 Knowledge Graph (limit for speed)
for doc in docs[:50]:
    add_profile(doc["text"])


# 🔹 Dense Search
def dense_search(query):
    q_emb = model.encode([query])
    D, I = index.search(q_emb, 5)
    return list(I[0])


# 🔹 Lexical Search (TF-IDF)
def bm25_search(query):
    q_vec = vectorizer.transform([query])
    scores = (tfidf_matrix @ q_vec.T).toarray().ravel()
    return list(np.argsort(scores)[::-1][:5])


# 🔹 Hybrid Fusion (RRF)
def rrf(bm25, dense, k=60):
    scores = {}

    for rank, doc in enumerate(bm25):
        scores[doc] = scores.get(doc, 0) + 1/(k + rank)

    for rank, doc in enumerate(dense):
        scores[doc] = scores.get(doc, 0) + 1/(k + rank)

    return sorted(scores, key=scores.get, reverse=True)
