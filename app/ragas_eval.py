import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from app.retrieval import dense_search, bm25_search, rrf, docs
from app.rerank import rerank

# -------------------------------
# 🔹 Load embedding model
# -------------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------------
# 🔹 Sample Queries
# -------------------------------
queries = [
    "data analyst",
    "python developer",
    "machine learning"
]

# -------------------------------
# 🔹 Helper Functions
# -------------------------------
def tokenize(text):
    return set(text.lower().split())

# -------------------------------
# 🔹 Context Precision
# -------------------------------
def context_precision(query, contexts):
    q_tokens = tokenize(query)

    relevant = 0
    for ctx in contexts:
        if len(q_tokens & tokenize(ctx)) > 0:
            relevant += 1

    return relevant / len(contexts) if contexts else 0

# -------------------------------
# 🔹 Context Recall
# -------------------------------
def context_recall(query, contexts):
    q_tokens = tokenize(query)

    covered = set()
    for ctx in contexts:
        covered |= (q_tokens & tokenize(ctx))

    return len(covered) / len(q_tokens) if q_tokens else 0

# -------------------------------
# 🔹 Answer Relevance (Embedding)
# -------------------------------
def answer_relevance(query, answer):
    q_vec = model.encode([query])
    a_vec = model.encode([answer])

    return cosine_similarity(q_vec, a_vec)[0][0]

# -------------------------------
# 🔹 Faithfulness (Embedding)
# -------------------------------
def faithfulness(answer, contexts):
    ctx_text = " ".join(contexts)

    a_vec = model.encode([answer])
    c_vec = model.encode([ctx_text])

    return cosine_similarity(a_vec, c_vec)[0][0]

# -------------------------------
# 🔹 Run Evaluation
# -------------------------------
def run_ragas():

    for q in queries:
        print("\n==============================")
        print(f"Query: {q}")

        # Retrieval
        bm25 = bm25_search(q)
        dense = dense_search(q)
        fused = rrf(bm25, dense)
        ranked = rerank(q, fused[:30])

        # Top contexts
        contexts = []
        for i, _ in ranked[:5]:
            contexts.append(docs[int(i)]["text"])

        # Fake answer (top result)
        answer = contexts[0] if contexts else ""

        # Metrics
        cp = context_precision(q, contexts)
        cr = context_recall(q, contexts)
        ar = answer_relevance(q, answer)
        fa = faithfulness(answer, contexts)

        # Print results
        print("Top Results:")
        for i, ctx in enumerate(contexts):
            print(f"{i+1}. {ctx[:100]}...")

        print("\n📊 RAGAS (LLM-FREE APPROX)")
        print(f"Context Precision: {round(cp,3)}")
        print(f"Context Recall: {round(cr,3)}")
        print(f"Answer Relevance: {round(ar,3)}")
        print(f"Faithfulness: {round(fa,3)}")


# -------------------------------
# 🔹 Run
# -------------------------------
if __name__ == "__main__":
    run_ragas()
