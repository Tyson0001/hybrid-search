from app.retrieval import dense_search, bm25_search, rrf, docs
from app.rerank import rerank

# 🔹 STEP 1 — DEFINE GROUND TRUTH (YOU MUST EDIT THIS AFTER SEEING OUTPUT)
# These IDs should be taken from your actual results
ground_truth = {
    "data analyst": [316, 656, 576],
    "python developer": [244, 259],
    "machine learning": [376, 364]
}


# 🔹 Precision@K
def precision_at_k(relevant, retrieved, k=5):
    retrieved_k = retrieved[:k]
    return len(set(relevant) & set(retrieved_k)) / k


# 🔹 Recall@K
def recall_at_k(relevant, retrieved, k=5):
    retrieved_k = retrieved[:k]
    return len(set(relevant) & set(retrieved_k)) / len(relevant)


# 🔹 MAP (Mean Average Precision)
def average_precision(relevant, retrieved):
    score = 0
    hits = 0

    for i, doc in enumerate(retrieved):
        if doc in relevant:
            hits += 1
            score += hits / (i + 1)

    return score / len(relevant) if relevant else 0


# 🔹 nDCG (Normalized Discounted Cumulative Gain)
def ndcg_at_k(relevant, retrieved, k=5):
    dcg = 0

    for i, doc in enumerate(retrieved[:k]):
        if doc in relevant:
            dcg += 1 / (i + 1)

    idcg = sum(1 / (i + 1) for i in range(min(len(relevant), k)))

    return dcg / idcg if idcg > 0 else 0


# 🔹 MAIN EVALUATION FUNCTION
def evaluate():
    for query, relevant_docs in ground_truth.items():
        print("\n==============================")
        print(f"Query: {query}")

        # 🔹 Run hybrid search
        bm25 = bm25_search(query)
        dense = dense_search(query)

        fused = rrf(bm25, dense)
        ranked = rerank(query, fused[:10])

        # Convert to list of IDs
        retrieved_ids = [int(i) for i, _ in ranked]

        print("\nTop Retrieved IDs:", retrieved_ids[:10])

        # 🔹 Show actual text (VERY IMPORTANT for debugging)
        print("\nTop Results Text:")
        for i in retrieved_ids[:5]:
            print(f"{i} → {docs[i]['text'][:120]}")

        # 🔹 Metrics
        p = precision_at_k(relevant_docs, retrieved_ids)
        r = recall_at_k(relevant_docs, retrieved_ids)
        ap = average_precision(relevant_docs, retrieved_ids)
        ndcg = ndcg_at_k(relevant_docs, retrieved_ids)

        print("\n📊 Metrics:")
        print("Precision@5:", round(p, 3))
        print("Recall@5:", round(r, 3))
        print("MAP:", round(ap, 3))
        print("nDCG:", round(ndcg, 3))


# 🔹 RUN FILE
if __name__ == "__main__":
    evaluate()
