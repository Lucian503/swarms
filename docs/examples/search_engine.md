# Self-Hosted AI Search Engine

A fully self-hosted, privacy-preserving search engine built on the Swarms framework. Uses AI swarm agents to process queries, search local document indices, and synthesize results — all without paid APIs or external services.

## Features

- **Cost-Free Operation**: No paid APIs, tokens, or services required. Uses local models (e.g., Ollama) or runs in direct search mode with zero LLM dependency.
- **Privacy-First**: All data stays local. Documents are indexed and searched on your machine.
- **BM25 Ranking**: Industry-standard relevance ranking algorithm for accurate results.
- **AI-Enhanced Search** (Optional): Swarm agents refine queries and synthesize results when agent mode is enabled.
- **REST API**: Built-in HTTP server for easy integration with any application.
- **File Indexing**: Index local files and directories (supports `.txt`, `.md`, `.csv`, `.json`, `.py`, `.html`, `.xml`, `.yaml`, and more).
- **Persistent Index**: Save and load document indices to/from disk.
- **Thread-Safe**: Safe to use in multi-threaded applications.

## Quick Start

### Installation

The search engine is included in the Swarms package:

```bash
pip install swarms
```

### Basic Usage (No LLM Required)

```python
from swarms.tools.search_engine import DocumentStore

# Create a document store
store = DocumentStore(name="my_docs")

# Index documents
store.add_document("doc1", "Python Guide", "Python is a programming language for AI.")
store.add_document("doc2", "Java Basics", "Java is used in enterprise applications.")

# Index a directory of files
store.index_directory("/path/to/your/documents")

# Search
results = store.search("Python programming", top_k=5)
for r in results:
    print(f"{r.title} (score: {r.score:.4f})")
    print(f"  {r.content[:100]}")
```

### Using the Search Swarm (With AI Agents)

For enhanced search with query refinement and result synthesis:

```python
from swarms.structs.search_swarm import SearchSwarm

# Create a search swarm with a free local model
swarm = SearchSwarm(
    documents_directory="/path/to/your/documents",
    model_name="ollama/llama3",  # Free local model
    agent_mode=True,
)

# AI-enhanced search
result = swarm.run("What are the best practices for machine learning?")
print(result)
```

### Direct Search Mode (No LLM)

For environments without an LLM, use direct search mode:

```python
from swarms.structs.search_swarm import SearchSwarm

swarm = SearchSwarm(
    documents_directory="/path/to/docs",
    agent_mode=False,  # No LLM required
)

# Direct search returns formatted results
result = swarm.run("machine learning")
print(result)

# Or get structured results
results = swarm.search("machine learning", top_k=5)
for r in results:
    print(f"{r.title}: {r.score:.4f}")
```

## REST API Server

### Starting the Server

```bash
# Basic usage - index a directory and start serving
python -m swarms.cli.search_server --docs /path/to/documents --port 8000

# With persistent index
python -m swarms.cli.search_server --docs /path/to/documents --persist ./index.json

# With AI agent mode (requires local LLM like Ollama)
python -m swarms.cli.search_server --docs /path/to/documents --agent-mode --model ollama/llama3
```

### API Endpoints

#### Search Documents
```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "machine learning", "top_k": 5}'
```

Response:
```json
{
  "query": "machine learning",
  "result_count": 3,
  "results": [
    {
      "doc_id": "/path/to/ml_guide.txt",
      "title": "ml_guide.txt",
      "content": "Machine learning is a subset of AI...",
      "score": 2.4531,
      "metadata": {
        "filepath": "/path/to/ml_guide.txt",
        "extension": ".txt"
      }
    }
  ]
}
```

#### Index a File
```bash
curl -X POST http://localhost:8000/index/file \
  -H "Content-Type: application/json" \
  -d '{"filepath": "/path/to/new_document.txt"}'
```

#### Index a Directory
```bash
curl -X POST http://localhost:8000/index/directory \
  -H "Content-Type: application/json" \
  -d '{"directory": "/path/to/new_docs", "recursive": true}'
```

#### Get Index Statistics
```bash
curl http://localhost:8000/stats
```

#### Health Check
```bash
curl http://localhost:8000/health
```

## Configuration

### Supported File Types

The search engine indexes the following file types by default:

| Extension | Type |
|-----------|------|
| `.txt` | Plain text |
| `.md` | Markdown |
| `.csv` | CSV data |
| `.json` | JSON data |
| `.py` | Python source |
| `.js` | JavaScript source |
| `.html` | HTML pages |
| `.xml` | XML documents |
| `.yaml`/`.yml` | YAML config |
| `.rst` | reStructuredText |
| `.cfg`/`.ini`/`.toml` | Configuration |
| `.log` | Log files |

### BM25 Parameters

Fine-tune search relevance by adjusting BM25 parameters:

```python
store = DocumentStore(
    name="tuned_store",
    bm25_k1=1.2,  # Term frequency saturation (default: 1.5)
    bm25_b=0.75,  # Length normalization (default: 0.75)
)
```

### Using Free Local Models

The search swarm works with any model supported by LiteLLM. For free, local operation:

1. **Install Ollama**: [ollama.ai](https://ollama.ai)
2. **Pull a model**: `ollama pull llama3`
3. **Use in SearchSwarm**:
   ```python
   swarm = SearchSwarm(
       model_name="ollama/llama3",
       agent_mode=True,
   )
   ```

## Architecture

The search engine consists of three main components:

### 1. DocumentStore (`swarms/tools/search_engine.py`)
- Pure Python TF-IDF indexing with BM25 ranking
- Thread-safe document management
- File and directory indexing
- Index persistence (save/load to JSON)

### 2. SearchSwarm (`swarms/structs/search_swarm.py`)
- Orchestrates AI agents for enhanced search
- **Query Analyzer Agent**: Refines and expands search queries
- **Result Synthesizer Agent**: Synthesizes results into coherent answers
- Falls back to direct search if agents are unavailable

### 3. REST API (`swarms/cli/search_server.py`)
- Lightweight HTTP server (Python standard library)
- No additional web framework dependencies
- CORS support for browser-based clients

## Private Practice Use

For private practice deployment:

1. **Keep data local**: All indexing and search happens on your machine.
2. **Use direct search mode** (`agent_mode=False`) for zero external dependencies.
3. **Or use Ollama** for AI-enhanced search with local LLM inference.
4. **Persist your index** to avoid re-indexing on restart.
5. **Bind to localhost** for security: `--host 127.0.0.1`

```bash
# Secure local deployment
python -m swarms.cli.search_server \
  --host 127.0.0.1 \
  --port 8000 \
  --docs /path/to/practice/documents \
  --persist ./practice_index.json
```

## Maintenance

### Updating the Index

Add new documents at any time via the API or programmatically:

```python
swarm.index_file("/path/to/new_document.txt")
swarm.index_directory("/path/to/new_docs/")
swarm.save_index()  # Persist changes
```

### Index Statistics

Monitor your index health:

```python
stats = swarm.get_stats()
print(f"Documents: {stats['document_count']}")
print(f"Terms: {stats['unique_terms']}")
```

### Backup

The index is stored as a single JSON file. Back it up regularly:

```bash
cp ./practice_index.json ./backups/practice_index_$(date +%Y%m%d).json
```
