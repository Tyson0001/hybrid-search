def explain_result(query, doc, score):
    return {
        "keyword_match": query.lower() in doc.lower(),
        "semantic_score": round(score, 3),
        "reasoning": [
            "BM25 keyword match" if query.lower() in doc.lower() else "No keyword match",
            "Semantic similarity via SBERT",
            "Graph-based expansion used"
        ]
    }
