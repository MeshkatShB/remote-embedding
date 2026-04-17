# remote-embedding

`remote-embedding` packages two things together:

- A FastAPI server that exposes a `/embed` API backed by local Hugging Face models.
- A LangChain-compatible `RemoteEmbeddings` client that calls that server remotely.

This lets multiple applications share a single loaded embedding model instance instead of each process loading its own copy. On constrained GPUs, that reduces duplicated VRAM usage and makes it easier to serve embeddings from limited hardware.

## Install

```bash
pip install remote-embedding
```

## Package Layout

The import package is `remote_embedding`.

```python
from remote_embedding import RemoteEmbeddings
```

## Run The Server

Set the environment variables your model needs. You can copy values from `.env.example` into your own `.env` file, or set them directly in the shell.

PowerShell:

```powershell
$env:EMBEDDING_MODEL_NAME="BAAI/bge-base-en-v1.5"
$env:EMBEDDING_DIR="C:\\path\\to\\model-cache"
$env:DEVICE="cpu"
```

Bash:

```bash
export EMBEDDING_MODEL_NAME=BAAI/bge-base-en-v1.5
export EMBEDDING_DIR=/path/to/model-cache
export DEVICE=cpu
```

You can also configure the server with CLI flags:

```bash
remote-embedding-server \
  --host 0.0.0.0 \
  --port 5055 \
  --model-name BAAI/bge-base-en-v1.5 \
  --embedding-dir /path/to/model-cache \
  --device cuda
```

Start the API:

```bash
remote-embedding-server
```

Or:

```bash
python -m remote_embedding
```

Defaults:

- `HOST=0.0.0.0`
- `PORT=5055`

CLI flags override environment variables for the current process.

## Configuration

Server configuration:

- `HOST`: bind address for the FastAPI server
- `PORT`: bind port for the FastAPI server
- `EMBEDDING_MODEL_NAME`: default model to preload and use when a request does not pass `model_name`
- `EMBEDDING_DIR`: optional local cache/model directory for Hugging Face downloads or local files
- `DEVICE`: device passed to `HuggingFaceEmbeddings`, such as `cpu` or `cuda`

Client configuration through `RemoteEmbeddings(...)`:

- `base_url`: full server URL, such as `http://127.0.0.1:5055`
- `timeout`: request timeout in seconds
- `expected_dimensions`: optional validation for returned vector size
- `model_name`: optional per-client default model name sent with each request

If `EMBEDDING_MODEL_NAME` is configured on the server, the server can preload one shared embedding model instance and let multiple applications reuse it. That is what saves VRAM versus loading the same model separately in each application process.

## Use The Client

```python
from remote_embedding import RemoteEmbeddings

embeddings = RemoteEmbeddings(
    base_url="http://127.0.0.1:5055",
    timeout=120,
    expected_dimensions=768,
    model_name="BAAI/bge-base-en-v1.5",
)

docs = embeddings.embed_documents(["hello world", "remote embeddings"])
query = embeddings.embed_query("search text")
```
