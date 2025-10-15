from openai import OpenAI
import gradio as gr
from pypdf import PdfReader
import numpy as np
from evaluator import Evaluator

client = OpenAI()

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def embed_text_chunks(text_chunks):
    embeddings = []
    for t in text_chunks:
        emb = client.embeddings.create(
            model="text-embedding-3-large",
            input=t
        ).data[0].embedding
        embeddings.append({"text": t, "embedding": emb})
    return embeddings

def retrieve_context(query, vector_store, top_k=4):
    query_emb = client.embeddings.create(
        model="text-embedding-3-large",
        input=query
    ).data[0].embedding
    scored = [(cosine_similarity(np.array(query_emb), np.array(v["embedding"])), v) for v in vector_store]
    top = sorted(scored, key=lambda x: x[0], reverse=True)[:top_k]
    return "\n\n".join([t[1]["text"] for t in top])

# ---------- Bot Creation ----------

def create_bot(resume_file, summary_file=None, projects_file=None):
    resume_text = ""
    if resume_file:
        reader = PdfReader(resume_file.name)
        for page in reader.pages:
            if t := page.extract_text():
                resume_text += t + "\n"

    summary_text = ""
    if summary_file:
        summary_text = summary_file.read().decode("utf-8")

    projects_text = ""
    if projects_file:
        projects_text = projects_file.read().decode("utf-8")

    # Chunk all texts
    all_text = resume_text + "\n\n" + summary_text + "\n\n" + projects_text
    chunks = [c.strip() for c in all_text.split("\n\n") if c.strip()]
    vector_store = embed_text_chunks(chunks)

    evaluator = Evaluator("User Bot", summary_text, resume_text)

    # The chat function to be used in Gradio
    def chat_fn(message, history):
        context = retrieve_context(message, vector_store)
        system_prompt = f"""You are acting as this user's career bot. Use the uploaded resume, summary, and projects to answer questions professionally and accurately.
        
## Retrieved Context:
{context}
"""
        messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": message}]
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        reply = response.choices[0].message.content

        # Evaluate response
        evaluation = evaluator.evaluate(reply, message, history)
        if not evaluation.is_acceptable:
            # Rerun with feedback
            updated_system = system_prompt + f"\n\nPrevious reply failed evaluation:\n{evaluation.feedback}"
            messages = [{"role": "system", "content": updated_system}] + history + [{"role": "user", "content": message}]
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages
            )
            reply = response.choices[0].message.content

        return reply

    return chat_fn

# ---------- Gradio UI ----------

with gr.Blocks() as demo:
    gr.Markdown("### Upload your resume to create a personalized career chatbot")

    resume_input = gr.File(label="Upload PDF Resume")
    summary_input = gr.File(label="Optional: Upload summary.txt", file_types=[".txt"])
    projects_input = gr.File(label="Optional: Upload projects.txt", file_types=[".txt"])
    start_btn = gr.Button("Create Bot")
    chatbot = gr.Chatbot()
    
    def start_bot(resume, summary, projects):
        chat_fn = create_bot(resume, summary, projects)
        return chat_fn

    start_btn.click(start_bot, inputs=[resume_input, summary_input, projects_input], outputs=chatbot)

demo.launch()
