from openai import OpenAI
from pypdf import PdfReader
import numpy as np
import json
import os

class RAG:
    def __init__(self, data_dir="me/data", embed_file="me/data/embeddings.json"):
        self.openai = OpenAI()
        self.data_dir = data_dir
        self.embed_file = embed_file
        self.docs, self.embeds = self._load_or_build_embeddings()

    def _embed_texts(self, texts):
        response = self.openai.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )
        return [d.embedding for d in response.data]

    def _load_or_build_embeddings(self):
        if os.path.exists(self.embed_file):
            with open(self.embed_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data["docs"], np.array(data["embeds"])

        texts = []
        for file in os.listdir(self.data_dir):
            path = os.path.join(self.data_dir, file)
            if file.endswith(".txt"):
                with open(path, "r", encoding="utf-8") as f:
                    texts.append(f.read())
            elif file.endswith(".pdf"):
                pdf_text = ""
                for page in PdfReader(path).pages:
                    pdf_text += page.extract_text() or ""
                texts.append(pdf_text)

        embeds = self._embed_texts(texts)
        with open(self.embed_file, "w", encoding="utf-8") as f:
            json.dump({"docs": texts, "embeds": embeds}, f)
        return texts, np.array(embeds)

    def retrieve(self, query, top_k=3):
        query_embed = self._embed_texts([query])[0]
        sims = np.dot(self.embeds, query_embed) / (np.linalg.norm(self.embeds, axis=1) * np.linalg.norm(query_embed))
        top_indices = np.argsort(sims)[-top_k:][::-1]
        return [self.docs[i] for i in top_indices]
