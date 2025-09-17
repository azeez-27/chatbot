Run the following commands:

cd chatbot-rag/backend

python -m venv .venv

.venv\Scripts\activate

pip install -r requirements.txt


uvicorn app.main:app --reload --port 8000

curl -X POST "http://localhost:8000/ingest" -F "file=@/path/to/your.pdf;type=application/pdf"


curl -X POST "http://localhost:8000/query" -H "Content-Type: application/json" -d '{"query":"What is our return policy?"}'
