# ollama-local-rag-min
Minimal Ollama Local RAG Setup

## Setup
Setup environment
```bash
uv venv local --python 3.11
```
Install dependencies
```bash
uv pip install -r requirements.txt
```

### Start Vector Database
Pull Image
```bash
docker pull qdrant/qdrant
```
Start Qdrant Image
```bash
docker run -p 6333:6333 -p 6334:6334 \
    -v "$(pwd)/qdrant_storage:/qdrant/storage:z" \
    qdrant/qdrant
```