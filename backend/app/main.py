# backend/app/main.py
import os
import shutil
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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

# Pinecone imports
from pinecone import Pinecone, ServerlessSpec
import numpy as np

from .ingest_utils import pdf_to_text, chunk_text, docs_from_text_chunks

app = FastAPI(title="RAG Chatbot API")

origins = [
    "http://localhost",
    "http://localhost:5173",
    "http://127.0.0.1",
    "http://127.0.0.1:5173",
    "https://ucc-chatbot.netlify.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

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

# Pinecone globals
PINECONE_CLIENT = None
PINECONE_INDEX = None

# simple in-memory store: { session_id: [ {"role":"user","content":...}, {"role":"assistant","content":...}, ... ] }
CONVERSATIONS: dict = {}

from openai import OpenAI

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")  # change as needed

# Pinecone configuration
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "ucc-chatbot")
PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT", "us-east-1")


OPENAI_CLIENT: OpenAI | None = None

class PineconeRetriever:
    """Custom retriever that mimics Haystack's InMemoryEmbeddingRetriever interface"""
    
    def __init__(self, pinecone_index, top_k: int = 5):
        self.pinecone_index = pinecone_index
        self.top_k = top_k
    
    def retrieve(self, query_embedding: List[float], top_k: Optional[int] = None) -> List[Document]:
        """Run retrieval and return documents in Haystack format"""
        k = top_k or self.top_k
        
        try:
            # Query Pinecone
            results = self.pinecone_index.query(
                vector=query_embedding,
                top_k=k,
                include_metadata=True
            )
            
            # Convert to Haystack Document format
            documents = []
            for match in results.matches:
                # Reconstruct document from metadata
                doc = Document(
                    content=match.metadata.get('content', ''),
                    meta={
                        'source': match.metadata.get('source', ''),
                        'score': match.score,
                        'id': match.id
                    }
                )
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            print(f"Error retrieving from Pinecone: {e}")
            return {"documents": []}

def store_in_pinecone(documents: List[Document]):
    """Store documents in Pinecone with embeddings"""
    if not PINECONE_INDEX or not documents:
        return
    
    try:
        vectors = []
        for i, doc in enumerate(documents):
            if hasattr(doc, 'embedding') and doc.embedding:
                vector_data = {
                    'id': f"doc_{hash(doc.content)}_{i}",
                    'values': doc.embedding,
                    'metadata': {
                        'content': doc.content,
                        'source': doc.meta.get('source', '') if doc.meta else ''
                    }
                }
                vectors.append(vector_data)
        
        if vectors:
            # Batch upsert to Pinecone
            PINECONE_INDEX.upsert(vectors=vectors)
            print(f"Stored {len(vectors)} documents in Pinecone")
            
    except Exception as e:
        print(f"Error storing in Pinecone: {e}")

@app.on_event("startup")
def startup_event():
    global OPENAI_CLIENT
    global DOCUMENT_STORE, DOC_EMBEDDER
    global TEXT_EMBEDDER_RET, TEXT_EMBEDDER_RAG
    global RETRIEVER_RET, RETRIEVER_RAG
    global RAG_PIPELINE, RETRIEVAL_PIPELINE, CHAT_GENERATOR
    global PINECONE_INDEX, PINECONE_CLIENT

    if OPENAI_API_KEY:
        # instantiate the new SDK client; it accepts api_key param or will use env var
        OPENAI_CLIENT = OpenAI(api_key=OPENAI_API_KEY)
    else:
        OPENAI_CLIENT = None

    # Initialize Pinecone
    if PINECONE_API_KEY:
        try:
            PINECONE_CLIENT = Pinecone(api_key=PINECONE_API_KEY)
            
            # Get or create index
            existing_indexes = [idx.name for idx in PINECONE_CLIENT.list_indexes()]
            
            if PINECONE_INDEX_NAME not in existing_indexes:
                # Create index with 384 dimensions for sentence-transformers/all-MiniLM-L6-v2
                PINECONE_CLIENT.create_index(
                    name=PINECONE_INDEX_NAME,
                    dimension=384,  # all-MiniLM-L6-v2 embedding dimension
                    metric="cosine",
                    spec=ServerlessSpec(
                    cloud="aws",
                    region=PINECONE_ENVIRONMENT
       )
                )
                print(f"Created Pinecone index: {PINECONE_INDEX_NAME}")
            
            PINECONE_INDEX = PINECONE_CLIENT.Index(PINECONE_INDEX_NAME)
            print(f"Connected to Pinecone index: {PINECONE_INDEX_NAME}")
            
        except Exception as e:
            print(f"Error initializing Pinecone: {e}")
            PINECONE_CLIENT = None
            PINECONE_INDEX = None
    else:
        print("No Pinecone API key found, using in-memory storage only")  

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
    # RETRIEVER_RET = InMemoryEmbeddingRetriever(document_store=DOCUMENT_STORE)
    # RETRIEVER_RAG = InMemoryEmbeddingRetriever(document_store=DOCUMENT_STORE)

     # 4) Create retrievers - use Pinecone if available, otherwise in-memory
    if PINECONE_INDEX:
        RETRIEVER_RET = PineconeRetriever(PINECONE_INDEX)
        RETRIEVER_RAG = PineconeRetriever(PINECONE_INDEX)
        print("Using Pinecone retrievers")
    else:
        RETRIEVER_RET = InMemoryEmbeddingRetriever(document_store=DOCUMENT_STORE)
        RETRIEVER_RAG = InMemoryEmbeddingRetriever(document_store=DOCUMENT_STORE)
        print("Using in-memory retrievers")

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
            **Instructions:**
            1.  **Greeting:** If the user greets or initiates the conversation (e.g., "hello", "hi", "how are you?"), respond with a proper greeting. Do not search for information.
            2.  **Clarification:** If the user's question is vague, incomplete, or gibberish (e.g., "do", "cancel", "kjvbfvb", "what about"), you must ask for clarification. Do not attempt to answer or search for a response. A good response would be "Could you please clarify your question?" or "I'm sorry, I don't understand. Could you please provide more details?".
            3.  **Relevant Information:** For all other questions, answer as accurately and concisely as possible using ONLY the provided context.
            4.  **No Relevant Information:** If the question is clear but you cannot find the answer in the provided context, you must respond with: "I'm sorry. I couldn't find the relevant information."

            **Context**:
            {% for document in documents %}
            {{ document.content }}
            {% endfor %}

            **Conversation history (most recent last):**
            {% for msg in history %}
            {{ msg.role }}: {{ msg.content }}
            {% endfor %}

            **Question:** {{ question }}
            **Answer:**
            """
        )
    ]
    prompt_builder = ChatPromptBuilder(template=template, required_variables=["question", "documents", "history"])

    # 7) Build retrieval-only pipeline (text_embedder -> retriever)
    # RETRIEVAL_PIPELINE = Pipeline()
    # RETRIEVAL_PIPELINE.add_component("text_embedder", TEXT_EMBEDDER_RET)
    # RETRIEVAL_PIPELINE.add_component("retriever", RETRIEVER_RET)
    # RETRIEVAL_PIPELINE.connect("text_embedder.embedding", "retriever.query_embedding")

    # 7) Build retrieval-only pipeline
    RETRIEVAL_PIPELINE = Pipeline()
    RETRIEVAL_PIPELINE.add_component("text_embedder", TEXT_EMBEDDER_RET)
    
    if not PINECONE_INDEX:
    # Only add retriever to pipeline if using in-memory storage
        RETRIEVAL_PIPELINE.add_component("retriever", RETRIEVER_RET)
        RETRIEVAL_PIPELINE.connect("text_embedder.embedding", "retriever.query_embedding")

    # 8) Build RAG pipeline (separate instances)
    if CHAT_GENERATOR:
        RAG_PIPELINE = Pipeline()
        RAG_PIPELINE.add_component("text_embedder", TEXT_EMBEDDER_RAG)
        # RAG_PIPELINE.add_component("retriever", RETRIEVER_RAG)
        RAG_PIPELINE.add_component("prompt_builder", prompt_builder)
        RAG_PIPELINE.add_component("llm", CHAT_GENERATOR)
        # RAG_PIPELINE.connect("text_embedder.embedding", "retriever.query_embedding")
        # RAG_PIPELINE.connect("retriever", "prompt_builder")
        # RAG_PIPELINE.connect("prompt_builder.prompt", "llm.messages")

        if not PINECONE_INDEX:
            # Only add retriever to pipeline if using in-memory storage
            RAG_PIPELINE.add_component("retriever", RETRIEVER_RAG)
            RAG_PIPELINE.connect("text_embedder.embedding", "retriever.query_embedding")
            RAG_PIPELINE.connect("retriever", "prompt_builder")
        else:
            # For Pinecone, we'll handle retrieval manually and connect directly to prompt_builder
            pass
        
        RAG_PIPELINE.connect("prompt_builder.prompt", "llm.messages")
    else:
        RAG_PIPELINE = None

    print("Startup complete: pipelines ready.")

    # code to print available models for the key
    # client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # models = client.models.list()

    # for m in models.data:
    #  print(m.id)

# Add message to session history
def add_to_history(session_id: str, role: str, content: str):
    if not session_id:
        return
    CONVERSATIONS.setdefault(session_id, []).append({"role": role, "content": content})

# Get recent history in chat format for OpenAI, trimming tokens by count or length (naive)
def get_history_messages(session_id: str, max_messages: int = 10):
    history = CONVERSATIONS.get(session_id, [])
    # return last N messages (you can improve token counting using tiktoken)
    return history[-max_messages:]


class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5
    session_id: Optional[str] = None   # new: client-provided session id (or server creates one)

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
        # DOCUMENT_STORE.write_documents(docs_with_embeddings["documents"])
        embedded_docs = docs_with_embeddings["documents"]

        # Store in both Pinecone and in-memory (fallback)
        if PINECONE_INDEX:
            store_in_pinecone(embedded_docs)
        
        # Also store in in-memory for fallback
        DOCUMENT_STORE.write_documents(embedded_docs)

        return {"ok": True, "indexed_chunks": len(docs)}

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

@app.post("/query")
async def query(req: QueryRequest):
    """Query the system. If OpenAI key is present, returns the LLM answer (RAG). Otherwise returns top documents retrieved."""
    question = req.query
    top_k = req.top_k or 5
    session_id = req.session_id or 'abc123'

    if RAG_PIPELINE:
        print("OpenAI Key found")

        # get_history_messages returns list of {"role": "...", "content": "..."}
        history_msgs = get_history_messages(session_id, max_messages=12)

        if PINECONE_INDEX:
            # Custom pipeline execution for Pinecone
            # 1. Get query embedding
            embedding_result = TEXT_EMBEDDER_RAG.run(text=question)
            query_embedding = embedding_result["embedding"]
            
            
            # 2. Retrieve from Pinecone
            retrieved_docs = RETRIEVER_RAG.retrieve(query_embedding=query_embedding, top_k=top_k)
            
            # 3. Build prompt and get LLM response
            prompt_result = RAG_PIPELINE.get_component("prompt_builder").run(
                question=question, 
                documents=retrieved_docs, 
                history=history_msgs
            )
            
            llm_result = CHAT_GENERATOR.run(messages=prompt_result["prompt"])
            
            try:
                answer = llm_result["replies"][0].text
            except Exception as e:
                print("Exception: ", e)
                answer = None
                
        else:
            # Original pipeline execution for in-memory
            payload = {
                "text_embedder": {"text": question}, 
                "prompt_builder": {"question": question, "history": history_msgs}
            }
            
            resp = RAG_PIPELINE.run(payload)
            try:
                answer = resp["llm"]["replies"][0].text
            except Exception as e:
                print("Exception: ", e)
                answer = None
            retrieved_docs = resp.get("retriever", {}).get("documents", [])
        
        # Persist the current user and assistant messages into session history
        add_to_history(session_id, "user", question)
        if answer:
            add_to_history(session_id, "assistant", answer)

        return {"answer": answer, "retrieved": [{"content": d.content, "meta": d.meta} for d in retrieved_docs]}
    else:
        # retrieval-only
        print("OpenAI Key Not found")
        # resp = RETRIEVAL_PIPELINE.run({"text_embedder": {"text": question}})
        # retrieved = resp.get("retriever", {}).get("documents", [])
        # snippets = [{"content": d.content, "meta": d.meta} for d in retrieved[:top_k]]
        # return {"answer": None, "retrieved": snippets}

        if PINECONE_INDEX:
            # Custom retrieval for Pinecone
            embedding_result = TEXT_EMBEDDER_RET.run(text=question)
            query_embedding = embedding_result["embedding"]
            retrieved_docs = RETRIEVER_RET.retrieve(query_embedding=query_embedding, top_k=top_k)
        else:
            # Original pipeline execution
            resp = RETRIEVAL_PIPELINE.run({"text_embedder": {"text": question}})
            retrieved_docs = resp.get("retriever", {}).get("documents", [])
        
        snippets = [{"content": d.content, "meta": d.meta} for d in retrieved_docs[:top_k]]
        return {"answer": None, "retrieved": snippets}
    
@app.get("/pinecone/status")
def pinecone_status():
    """Check Pinecone connection status"""
    if PINECONE_INDEX:
        try:
            stats = PINECONE_INDEX.describe_index_stats()
            return {
                "connected": True,
                "index_name": PINECONE_INDEX_NAME,
                "total_vectors": stats.total_vector_count,
                "dimension": stats.dimension
            }
        except Exception as e:
            return {"connected": False, "error": str(e)}
    else:
        return {"connected": False, "message": "Pinecone not configured"}    

@app.get("/health")
def health():
    return {"ok": True}

# If running locally via `python -m uvicorn app.main:app --reload`
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)