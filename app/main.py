from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from app.retrieval import dense_search, bm25_search, rrf, docs
from app.rerank import rerank

app = FastAPI()

# -------------------------------
# 🔹 Home Page
# -------------------------------
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
    <body>
        <h1>🔍 Smart Hybrid Search System</h1>
        <p>Search profiles using semantic + keyword + hybrid ranking</p>

        <form action="/search">
            <input type="text" name="q" placeholder="Enter query (e.g. python developer)">
            <button type="submit">Search</button>
        </form>
    </body>
    </html>
    """

# -------------------------------
# 🔹 Search Page
# -------------------------------
@app.get("/search", response_class=HTMLResponse)
def search(q: str):

    # Step 1: Retrieval
    bm25 = bm25_search(q)
    dense = dense_search(q)

    # Step 2: Fusion
    fused = rrf(bm25, dense)

    # Step 3: Rerank (IMPORTANT CHANGE: increased candidates)
    ranked = rerank(q, fused[:30])

    results_html = ""

    for i, score in ranked[:5]:
        i = int(i)
        doc = docs[i]

        text = doc.get("text", "")
        preview = text[:250] + "..." if len(text) > 250 else text

        # 🔹 Explainability
        explanation = f"""
        <ul>
            <li>Matched keywords from query</li>
            <li>Semantic similarity using embeddings</li>
            <li>Hybrid ranking (BM25 + Dense + RRF)</li>
        </ul>
        """

        results_html += f"""
        <div style='margin:15px;padding:15px;border:1px solid black;border-radius:10px'>
            <p><b>Profile:</b> {preview}</p>
            <p><b>Score:</b> {round(score,3)}</p>
            <p><b>Why this result?</b></p>
            {explanation}
        </div>
        """

    return f"""
    <html>
    <body>
        <h2>Results for: {q}</h2>
        <p>Hybrid Search (TF-IDF + Semantic + RRF + Reranking)</p>

        {results_html}

        <br><a href="/">⬅ Back</a>
    </body>
    </html>
    """
