"""
Service 2: Semantic Search Service
Uses ChromaDB with file persistence and sentence-transformers embeddings
over a curated dataset of research abstracts.
"""

import os
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from langchain_core.tools import tool
from assignment_chat.tools_api import _search_pubmed_ids, _fetch_pubmed_data

# ── ChromaDB setup ────────────────────────────────────────────────────────────
DB_PATH        = os.path.join(os.path.dirname(__file__), "../data/chroma_db")
# DB_PATH        =  "../data/chroma_db"
COLLECTION_NAME = "research_abstracts"

def get_collection():
    """Return (or create) the ChromaDB collection."""
    client = chromadb.PersistentClient(path=DB_PATH)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"}
    )
    return collection



def populate_if_empty():
    """Seed the collection with curated research abstracts if empty."""
    collection = get_collection()
    if collection.count() > 0:
        return  # already populated
    
    # df = pd.read_csv("../citations.csv")
    # papers = build_abstracts_from_citation(df)

    # query = _fetch_pubmed_data(_search_pubmed_ids("Creighton SE[au] OR 'vision AND psychophysics'",30))
    sec_papers = _fetch_pubmed_data(_search_pubmed_ids("Creighton SE[au]",4))
    for idx, paper in enumerate(sec_papers):
        sec_papers[idx]['id'] = f"Creighton_{idx}"

    # pmids = _search_pubmed_ids("face perception AND 2026[pd]",26)
    # time.sleep(0.5)
    # other_papers = _fetch_pubmed_data(pmids)
    # for idx, paper in enumerate(other_papers):
    #     other_papers[idx]['id'] = f"Other_{idx}"
    papers = sec_papers #+ other_papers

    ids = [p["id"] for p in papers]
    metadata = [{"title": p['title']} for p in papers]
    docs = [p["abstract"] for p in papers]

    collection.add(documents=docs, ids=ids, metadatas=metadata)
    print(f"[semantic_service] Seeded {len(papers)} abstracts into ChromaDB.")


@tool
def semantic_search(query: str, n_results: int = 3) -> str:
    """
    Perform semantic search over the research abstracts collection.
    Returns a formatted string of the top matching abstracts.

    Use this tool when the user asks how similar a paper is to the
    papers contained in the collection "research_abstracts"
    """
    try:
        populate_if_empty()
        collection = get_collection()

        results = collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )

        docs      = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        if not docs:
            return "No relevant research abstracts found."

        output = []
        for i, (doc, meta, dist) in enumerate(zip(docs, metadatas, distances)):
            similarity = round((1 - dist) * 100, 1)
            output.append(
                f"[Result {i+1}] {meta.get('title', 'Untitled')} "
                f"(Relevance: {similarity}%)\n{doc[:300]}..."
            )

        return "\n\n".join(output)

    except Exception as e:
        return f"Semantic search encountered an error: {str(e)}"