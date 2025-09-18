Run the following commands to run the bot:

cd chatbot/backend

python -m venv .venv

.venv\Scripts\activate

pip install -r requirements.txt

uvicorn app.main:app --reload --port 8000


# to provide files for the bot 
curl -X POST "http://localhost:8000/ingest" -F "file=@/path/to/your.pdf;type=application/pdf"

# to query the bot
curl -X POST "http://localhost:8000/query" -H "Content-Type: application/json" -d '{"query":"What is our return policy?"}'
