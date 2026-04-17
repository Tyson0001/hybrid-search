import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer

# -------------------------------
# 🔹 Load Dataset (dynamic schema)
# -------------------------------
df = pd.read_csv("data/profiles.csv")
df = df.fillna("")

# Combine all columns into one text field
df["text"] = df.astype(str).agg(" ".join, axis=1)

docs = df.to_dict(orient="records")
texts = df["text"].tolist()

# -------------------------------
# 🔹 TF-IDF (Lexical Search)
# -------------------------------
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    stop_words="english"
)

tfidf_matrix = vectorizer.fit_transform(texts)

# -------------------------------
# 🔹 Sentence-BERT (Semantic Search)
# -------------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(texts, show_progress_bar=True)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# -------------------------------
# 🔹 Query Expansion (Intent Understanding)
# -------------------------------
def expand_query(q):
    q = q.lower()

    if "data analyst" in q:
        return [q, "data analysis", "data science", "sql analytics"]

    if "python developer" in q:
        return [q, "backend developer", "software engineer python"]

    if "machine learning" in q:
        return [q, "ml engineer", "deep learning", "ai engineer"]

    return [q]

# -------------------------------
# 🔹 BM25 / TF-IDF Search
# -------------------------------
def bm25_search(query):
    queries = expand_query(query)

    scores = np.zeros(len(texts))

    for q in queries:
        q_vec = vectorizer.transform([q])
        scores += (tfidf_matrix @ q_vec.T).toarray().flatten()

    return np.argsort(scores)[::-1]

# -------------------------------
# 🔹 Dense Search (FAISS)
# -------------------------------
def dense_search(query):
    queries = expand_query(query)

    scores = np.zeros(len(texts))

    for q in queries:
        q_vec = model.encode([q])
        D, I = index.search(q_vec, len(texts))
        scores[I[0]] += D[0]

    return np.argsort(scores)

# -------------------------------
# 🔹 RRF Fusion
# -------------------------------
def rrf(bm25, dense, k=60):
    scores = {}

    for rank, doc_id in enumerate(bm25):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)

    for rank, doc_id in enumerate(dense):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)

    return sorted(scores, key=scores.get, reverse=True)
