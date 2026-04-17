from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from app.retrieval import dense_search, bm25_search, rrf, docs
from app.rerank import rerank
from app.utils import expand_query

app = FastAPI()


# 🔹 Home Page
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
    <body>
        <h1>🔍 Hybrid Search System (Profiles)</h1>
        <form action="/search">
            <input type="text" name="q" placeholder="Enter query (e.g. python developer)">
            <button type="submit">Search</button>
        </form>
    </body>
    </html>
    """


# 🔹 Search Page
@app.get("/search", response_class=HTMLResponse)
def search(q: str):

    # 🔹 Query Expansion
    queries = expand_query(q)

    bm25_all = []
    dense_all = []

    for query in queries:
        bm25_all += bm25_search(query)
        dense_all += dense_search(query)

    # 🔹 Hybrid Fusion
    fused = rrf(bm25_all, dense_all)

    # 🔹 Re-ranking
    ranked = rerank(q, fused[:10])

    results_html = ""

    for i, score in ranked[:5]:
        i = int(i)
        doc = docs[i]

        text = doc.get("text", "")
        preview = text[:150] + "..." if len(text) > 200 else text

        # 🔹 Explanation
        explanation = []

        if q.lower() in text.lower():
            explanation.append("✔ Keyword match")

        if score > 0.5:
            explanation.append("✔ Strong semantic similarity")

        explanation.append("✔ Hybrid ranking (BM25 + Semantic + RRF)")

        results_html += f"""
        <div style='margin:15px;padding:15px;border:1px solid black;border-radius:8px'>
            <p><b>Profile:</b> {preview}</p>
            <p><b>Score:</b> {round(score,3)}</p>
            <p><b>Explanation:</b> {' | '.join(explanation)}</p>
            <p>🔗 Connected in Knowledge Graph</p>
        </div>
        """

    return f"""
    <html>
    <body>
        <h2>Results for: {q}</h2>
        <p>Hybrid Search (BM25 + Semantic + RRF)</p>

        {results_html}

        <br><a href="/">⬅ Back</a>
    </body>
    </html>
    """
