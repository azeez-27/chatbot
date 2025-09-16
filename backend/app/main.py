# backend/app/main.py
import os
import shutil
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Optional, List

from dotenv import load_dotenv
load_dotenv()

from haystack import Document, Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.builders import ChatPromptBuilder
from haystack.dataclasses import ChatMessage

# Optional OpenAI generator
from haystack.components.generators.chat import OpenAIChatGenerator

from .ingest_utils import pdf_to_text, chunk_text, docs_from_text_chunks

app = FastAPI(title="RAG Chatbot API")

# Global containers
DOCUMENT_STORE = None
DOC_EMBEDDER = None

# Separate embedders/retrievers for two pipelines (avoid sharing instances)
TEXT_EMBEDDER_RET = None
TEXT_EMBEDDER_RAG = None
RETRIEVER_RET = None
RETRIEVER_RAG = None

RAG_PIPELINE = None
RETRIEVAL_PIPELINE = None
CHAT_GENERATOR = None

from openai import OpenAI

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo-0125")  # change as needed

@app.on_event("startup")
def startup_event():
    global DOCUMENT_STORE, DOC_EMBEDDER
    global TEXT_EMBEDDER_RET, TEXT_EMBEDDER_RAG
    global RETRIEVER_RET, RETRIEVER_RAG
    global RAG_PIPELINE, RETRIEVAL_PIPELINE, CHAT_GENERATOR

    # 1) Document store (single store is fine)
    DOCUMENT_STORE = InMemoryDocumentStore()

    # 2) Document embedder (used for indexing documents) — single instance OK
    DOC_EMBEDDER = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
    DOC_EMBEDDER.warm_up()

    # 3) Create separate text embedders for each pipeline
    TEXT_EMBEDDER_RET = SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
    TEXT_EMBEDDER_RET.warm_up()

    TEXT_EMBEDDER_RAG = SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
    TEXT_EMBEDDER_RAG.warm_up()

    # 4) Create two retrievers (they can share DOCUMENT_STORE)
    RETRIEVER_RET = InMemoryEmbeddingRetriever(document_store=DOCUMENT_STORE)
    RETRIEVER_RAG = InMemoryEmbeddingRetriever(document_store=DOCUMENT_STORE)

    # 5) Optional: Chat generator (requires OPENAI_API_KEY)
    if OPENAI_API_KEY:
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
        CHAT_GENERATOR = OpenAIChatGenerator(model=OPENAI_MODEL)
    else:
        CHAT_GENERATOR = None

    # 6) Prompt builder: small template — set required_variables
    template = [
        ChatMessage.from_user(
            """
            You are an AI assistant of "ucc.co.ug". Given the following context, answer the user's question as accurately and concisely as possible. when user greet or initiate the conversation, you have to greet that user properly. When you couldn't find the asked data, then you have to say "I'm sorry. I  couldn't find the relevant information." 

            Context:
            {% for document in documents %}
            {{ document.content }}
            {% endfor %}

            Question: {{ question }}
            Answer:
            """
        )
    ]
    prompt_builder = ChatPromptBuilder(template=template, required_variables=["question", "documents"])

    # 7) Build retrieval-only pipeline (text_embedder -> retriever)
    RETRIEVAL_PIPELINE = Pipeline()
    RETRIEVAL_PIPELINE.add_component("text_embedder", TEXT_EMBEDDER_RET)
    RETRIEVAL_PIPELINE.add_component("retriever", RETRIEVER_RET)
    RETRIEVAL_PIPELINE.connect("text_embedder.embedding", "retriever.query_embedding")

    # 8) Build RAG pipeline (separate instances)
    if CHAT_GENERATOR:
        RAG_PIPELINE = Pipeline()
        RAG_PIPELINE.add_component("text_embedder", TEXT_EMBEDDER_RAG)
        RAG_PIPELINE.add_component("retriever", RETRIEVER_RAG)
        RAG_PIPELINE.add_component("prompt_builder", prompt_builder)
        RAG_PIPELINE.add_component("llm", CHAT_GENERATOR)
        RAG_PIPELINE.connect("text_embedder.embedding", "retriever.query_embedding")
        RAG_PIPELINE.connect("retriever", "prompt_builder")
        RAG_PIPELINE.connect("prompt_builder.prompt", "llm.messages")
    else:
        RAG_PIPELINE = None

    print("Startup complete: pipelines ready.")

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    models = client.models.list()

    for m in models.data:
     print(m.id)


class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5

@app.post("/ingest")
async def ingest_file(file: UploadFile = File(...)):
    """Upload PDF or plain text. The endpoint will extract text -> chunk -> embed -> write to document store."""
    if file.content_type not in ("application/pdf", "text/plain"):
        raise HTTPException(status_code=400, detail="Only PDF and text files are supported in this demo.")

    tmp_dir = tempfile.mkdtemp()
    try:
        tmp_path = os.path.join(tmp_dir, file.filename)
        with open(tmp_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Extract text
        if file.content_type == "application/pdf":
            text = pdf_to_text(tmp_path)
        else:
            text = open(tmp_path, "r", encoding="utf-8").read()

        if not text.strip():
            return {"ok": False, "message": "No text extracted"}

        # chunk + make Documents
        chunks = chunk_text(text, chunk_size=1000, overlap=200)
        docs = docs_from_text_chunks(chunks, source=file.filename)

        # embed documents (doc embedder) and write to store
        docs_with_embeddings = DOC_EMBEDDER.run(docs)
        DOCUMENT_STORE.write_documents(docs_with_embeddings["documents"])

        return {"ok": True, "indexed_chunks": len(docs)}

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

@app.post("/query")
async def query(req: QueryRequest):
    """Query the system. If OpenAI key is present, returns the LLM answer (RAG). Otherwise returns top documents retrieved."""
    question = req.query
    top_k = req.top_k or 5

    if RAG_PIPELINE:
        print("OpenAI Key found")
        payload = {"text_embedder": {"text": question}, "prompt_builder": {"question": question}}
        resp = RAG_PIPELINE.run(payload)
        # The exact shape: resp["llm"]["replies"][0].text
        try:
            answer = resp["llm"]["replies"][0].text
        except Exception:
            answer = None
        retrieved = resp.get("retriever", {}).get("documents", [])
        return {"answer": answer, "retrieved": [{"content": d.content, "meta": d.meta} for d in retrieved]}
    else:
        # retrieval-only
        print("OpenAI Key Nottt found")
        resp = RETRIEVAL_PIPELINE.run({"text_embedder": {"text": question}})
        retrieved = resp.get("retriever", {}).get("documents", [])
        snippets = [{"content": d.content, "meta": d.meta} for d in retrieved[:top_k]]
        return {"answer": None, "retrieved": snippets}

@app.get("/health")
def health():
    return {"ok": True}

# If running locally via `python -m uvicorn app.main:app --reload`
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)