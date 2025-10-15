import json, numpy as np
from openai import OpenAI

client = OpenAI()

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def retrieve_context(query, k=4):
    with open("rag/vector_store.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    query_emb = client.embeddings.create(model="text-embedding-3-large", input=query).data[0].embedding
    scored = [(cosine_similarity(np.array(query_emb), np.array(d["embedding"])), d) for d in data]
    top = sorted(scored, key=lambda x: x[0], reverse=True)[:k]
    return "\n\n".join([t[1]["text"] for t in top])