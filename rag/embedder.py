import os, json
from openai import OpenAI
from pypdf import PdfReader

client = OpenAI()

def load_sources():
    sources = {}

    pdf_path = os.path.join("me", "linkedin.pdf")
    if os.path.exists(pdf_path):
        reader = PdfReader(pdf_path)
        linkedin_text = ""
        for page in reader.pages:
            if text := page.extract_text():
                linkedin_text += text + "\n"
        sources["linkedin.pdf"] = linkedin_text

    for fname in ["summary.txt", "projects.txt"]:
        path = os.path.join("me", fname)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                sources[fname] = f.read()
    return sources

def embed_sources():
    data = load_sources()
    chunks = []

    for name, text in data.items():
        for part in text.split("\n\n"):
            if part.strip():
                chunks.append({"source": name, "text": part.strip()})

    print(f"Embedding {len(chunks)} text chunks")
    embeddings = []
    for c in chunks:
        emb = client.embeddings.create(model="text-embedding-3-large", input=c["text"]).data[0].embedding
        embeddings.append({"source": c["source"], "text": c["text"], "embedding": emb})

    os.makedirs("rag", exist_ok=True)
    with open("rag/vector_store.json", "w", encoding="utf-8") as f:
        json.dump(embeddings, f, indent=2)
    print("Saved updated embeddings to rag/vector_store.json")

if __name__ == "__main__":
    embed_sources()
