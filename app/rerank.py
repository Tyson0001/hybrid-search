from sentence_transformers import SentenceTransformer
import numpy as np

from app.retrieval import docs

model = SentenceTransformer('all-MiniLM-L6-v2')


def rerank(query, doc_ids):
    q_emb = model.encode([query])[0]

    scores = []

    for i in doc_ids:
        text = docs[i].get("text", "")   # ✅ safe access
        d_emb = model.encode([text])[0]

        score = np.dot(q_emb, d_emb)
        scores.append((i, score))

    # sort by score
    scores.sort(key=lambda x: x[1], reverse=True)

    return scores
