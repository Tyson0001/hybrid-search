from langgraph.graph import StateGraph
from typing import TypedDict, List

from langchain_openai import ChatOpenAI

from app.retrieval import dense_search, bm25_search, rrf, docs
from app.rerank import rerank


# 🔹 LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)


# 🔹 State
class State(TypedDict):
    query: str
    contexts: List[str]
    answer: str


# 🔹 STEP 1: Retrieval
def retrieve(state: State):
    q = state["query"]

    bm25 = bm25_search(q)
    dense = dense_search(q)
    fused = rrf(bm25, dense)
    ranked = rerank(q, fused[:5])

    contexts = [docs[int(i)]["text"] for i, _ in ranked]

    return {"contexts": contexts}


# 🔹 STEP 2: Generate Answer (LLM)
def generate(state: State):
    query = state["query"]
    contexts = state["contexts"]

    prompt = f"""
    Answer the query using the context below.

    Query: {query}

    Context:
    {contexts}

    Answer:
    """

    response = llm.invoke(prompt)

    return {"answer": response.content}


# 🔹 STEP 3: Evaluate (Answer Relevance + Faithfulness)
def evaluate_node(state: State):
    query = state["query"]
    contexts = state["contexts"]
    answer = state["answer"]

    # 🔹 Answer Relevance
    relevance_prompt = f"""
    Query: {query}
    Answer: {answer}

    Rate how relevant the answer is to the query from 0 to 1.
    Only return a number.
    """

    relevance = llm.invoke(relevance_prompt).content

    # 🔹 Faithfulness
    faithfulness_prompt = f"""
    Context: {contexts}
    Answer: {answer}

    Is the answer fully supported by the context?
    Return a score between 0 and 1.
    """

    faithfulness = llm.invoke(faithfulness_prompt).content

    print("\n========================")
    print(f"Query: {query}")
    print("Answer:", answer[:150])
    print("Answer Relevance:", relevance)
    print("Faithfulness:", faithfulness)


# 🔹 BUILD GRAPH
builder = StateGraph(State)

builder.add_node("retrieve", retrieve)
builder.add_node("generate", generate)
builder.add_node("evaluate", evaluate_node)

builder.set_entry_point("retrieve")

builder.add_edge("retrieve", "generate")
builder.add_edge("generate", "evaluate")

graph = builder.compile()


# 🔹 RUN
queries = [
    "data analyst",
    "python developer",
    "machine learning"
]

if __name__ == "__main__":
    for q in queries:
        graph.invoke({"query": q})
