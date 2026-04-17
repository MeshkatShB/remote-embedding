# remote-embedding

`remote-embedding` packages two things together:

- A FastAPI server that exposes a `/embed` API backed by local Hugging Face models.
- A LangChain-compatible `RemoteEmbeddings` client that calls that server remotely.

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

Set the environment variables your model needs.

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

## Use The Client

```python
from remote_embedding import RemoteEmbeddings

embeddings = RemoteEmbeddings(
    base_url="http://127.0.0.1:5055",
    model_name="BAAI/bge-base-en-v1.5",
)

docs = embeddings.embed_documents(["hello world", "remote embeddings"])
query = embeddings.embed_query("search text")
```

## Build For PyPI

Build distributions locally:

```bash
python -m pip install --upgrade build
python -m build
```

This creates:

- `dist/*.tar.gz`
- `dist/*.whl`

Upload with Twine:

```bash
python -m pip install --upgrade twine
python -m twine upload dist/*
```
