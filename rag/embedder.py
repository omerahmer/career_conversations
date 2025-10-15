import os, json
from openai import OpenAI
from pypdf import PdfReader

client = OpenAI()

VECTOR_STORE_FILE = "rag/vector_store.json"

def embed_text_chunks(text_chunks):
    embeddings = []
    for t in text_chunks:
        emb = client.embeddings.create(
            model="text-embedding-3-large",
            input=t
        ).data[0].embedding
        embeddings.append({"text": t, "embedding": emb})
    return embeddings

def embed_sources(base_folder):
    all_text = ""
    for fname in os.listdir(base_folder):
        path = os.path.join(base_folder, fname)
        if fname.endswith(".pdf"):
            reader = PdfReader(path)
            for page in reader.pages:
                if t := page.extract_text():
                    all_text += t + "\n\n"
        elif fname.endswith(".txt"):
            with open(path, "r", encoding="utf-8") as f:
                all_text += f.read() + "\n\n"

    chunks = [c.strip() for c in all_text.split("\n\n") if c.strip()]
    vector_store = embed_text_chunks(chunks)

    os.makedirs("rag", exist_ok=True)
    with open(VECTOR_STORE_FILE, "w", encoding="utf-8") as f:
        json.dump(vector_store, f, ensure_ascii=False, indent=2)

    print(f"Embedded {len(chunks)} chunks and saved to {VECTOR_STORE_FILE}")
