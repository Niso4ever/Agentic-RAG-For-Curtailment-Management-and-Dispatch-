import os
import json
from typing import List, Dict, Any

import faiss
import numpy as np
from pypdf import PdfReader

from openai import OpenAI

from app.config import OPENAI_API_KEY, EMBED_MODEL

# --- Config ---
client = OpenAI(api_key=OPENAI_API_KEY)

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PDF_DIR = os.path.join(BASE_DIR, "data", "pdfs")
INDEX_PATH = os.path.join(BASE_DIR, "data", "rag_index.faiss")
META_PATH = os.path.join(BASE_DIR, "data", "rag_meta.json")


# ---------- Embeddings ----------
def embed_text(text: str) -> List[float]:
    resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=text
    )
    return resp.data[0].embedding


# ---------- PDF Chunking ----------
def _load_pdfs() -> List[Dict[str, Any]]:
    docs = []
    for fname in os.listdir(PDF_DIR):
        if not fname.lower().endswith(".pdf"):
            continue
        path = os.path.join(PDF_DIR, fname)
        reader = PdfReader(path)
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            text = text.strip()
            if not text:
                continue
            docs.append({
                "source": fname,
                "page": i + 1,
                "text": text
            })
    return docs


# ---------- Build / Load Index ----------
def build_index() -> None:
    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)

    docs = _load_pdfs()
    if not docs:
        print("No PDFs found in", PDF_DIR)
        return

    vectors = []
    for d in docs:
        vec = embed_text(d["text"])
        vectors.append(vec)

    dim = len(vectors[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(vectors, dtype="float32"))

    faiss.write_index(index, INDEX_PATH)

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)

    print(f"Built RAG index with {len(docs)} chunks.")


def _load_index():
    if not (os.path.exists(INDEX_PATH) and os.path.exists(META_PATH)):
        print("RAG index not found. Building now...")
        build_index()

    if not (os.path.exists(INDEX_PATH) and os.path.exists(META_PATH)):
        raise RuntimeError("Failed to build RAG index; ensure PDFs exist in data/pdfs.")

    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return index, meta


# ---------- Retrieval API ----------
def retrieve_grounded_knowledge(query: str, k: int = 3) -> Dict[str, Any]:
    """
    Main function used by your agent tools.
    Returns top-k chunks with metadata.
    """
    index, meta = _load_index()

    q_vec = np.array([embed_text(query)], dtype="float32")
    distances, indices = index.search(q_vec, k)

    hits = []
    for rank, idx in enumerate(indices[0]):
        if idx < 0 or idx >= len(meta):
            continue
        doc = meta[idx]
        hits.append({
            "rank": rank + 1,
            "distance": float(distances[0][rank]),
            "source": doc["source"],
            "page": doc["page"],
            "text": doc["text"][:800]  # truncate long pages
        })

    return {
        "query": query,
        "results": hits
    }
if __name__ == "__main__":
    build_index()
