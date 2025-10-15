import json
import numpy as np
from openai import OpenAI

client = OpenAI()

VECTOR_STORE_FILE = "rag/vector_store.json"

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def load_vector_store():
    with open(VECTOR_STORE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def retrieve_context(query, top_k=4):
    vector_store = load_vector_store()
    query_emb = client.embeddings.create(
        model="text-embedding-3-large",
        input=query
    ).data[0].embedding

    scored = [(cosine_similarity(np.array(query_emb), np.array(v["embedding"])), v) for v in vector_store]
    top = sorted(scored, key=lambda x: x[0], reverse=True)[:top_k]
    return "\n\n".join([t[1]["text"] for t in top])