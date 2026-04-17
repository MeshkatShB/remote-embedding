"""FastAPI application serving remote embedding inference."""

import asyncio
import argparse
import os
from contextlib import asynccontextmanager
from typing import Literal, Optional, Union

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel, Field

load_dotenv()


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


HOST = os.getenv("HOST", "0.0.0.0")
PORT = _env_int("PORT", 5055)
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
EMBEDDING_DIR = os.getenv("EMBEDDING_DIR")
DEVICE = os.getenv("DEVICE")


class EmbeddingRequest(BaseModel):
    input: Union[str, list[str]] = Field(..., description="String or list of strings")
    mode: Literal["documents", "query"] = "documents"
    model_name: Optional[str] = None
    instruction: Optional[str] = None


class EmbeddingResponse(BaseModel):
    model: str
    dimensions: int
    count: int
    data: list[list[float]]


class HealthResponse(BaseModel):
    status: str
    model: str
    device: Optional[str]


class EmbeddingService:
    def __init__(self) -> None:
        self.embed_models: dict[str, HuggingFaceEmbeddings] = {}
        self.lock = asyncio.Lock()

    def _resolve_model_name(self, model_name: Optional[str] = None) -> str:
        resolved_model_name = (model_name or EMBEDDING_MODEL_NAME or "").strip()
        if not resolved_model_name:
            raise RuntimeError(
                "No embedding model specified. Set EMBEDDING_MODEL_NAME or pass model_name in the request."
            )
        return resolved_model_name

    def load(self, model_name: Optional[str] = None) -> HuggingFaceEmbeddings:
        resolved_model_name = self._resolve_model_name(model_name)
        if resolved_model_name in self.embed_models:
            return self.embed_models[resolved_model_name]

        embed_model = HuggingFaceEmbeddings(
            model_name=resolved_model_name,
            model_kwargs={
                "device": DEVICE,
                "local_files_only": True,
                "trust_remote_code": True,
            },
            cache_folder=EMBEDDING_DIR,
        )
        self.embed_models[resolved_model_name] = embed_model
        return embed_model

    async def embed_documents(
        self,
        texts: list[str],
        model_name: Optional[str] = None,
    ) -> list[list[float]]:
        embed_model = self.load(model_name)

        # Serialize GPU access to avoid VRAM spikes from concurrent requests.
        async with self.lock:
            return await asyncio.to_thread(embed_model.embed_documents, texts)

    async def embed_query(self, text: str, model_name: Optional[str] = None) -> list[float]:
        embed_model = self.load(model_name)

        async with self.lock:
            return await asyncio.to_thread(embed_model.embed_query, text)


svc = EmbeddingService()


@asynccontextmanager
async def lifespan(_: FastAPI):
    if EMBEDDING_MODEL_NAME:
        svc.load()
    yield


app = FastAPI(title="Shared Embedding Service", version="0.1.0", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    configured_model_name = (EMBEDDING_MODEL_NAME or "").strip()
    loaded_model_name = configured_model_name or next(iter(svc.embed_models), "")

    if not loaded_model_name:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return HealthResponse(
        status="ok",
        model=loaded_model_name,
        device=DEVICE,
    )


@app.post("/embed", response_model=EmbeddingResponse)
async def embed(req: EmbeddingRequest) -> EmbeddingResponse:
    texts = [req.input] if isinstance(req.input, str) else req.input

    if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
        raise HTTPException(status_code=400, detail="Input must contain non-empty strings")

    resolved_model_name = (req.model_name or EMBEDDING_MODEL_NAME or "").strip()
    if not resolved_model_name:
        raise HTTPException(
            status_code=400,
            detail="No embedding model specified. Set EMBEDDING_MODEL_NAME or pass model_name.",
        )

    try:
        if req.mode == "query":
            if len(texts) != 1:
                raise HTTPException(
                    status_code=400,
                    detail="mode='query' requires a single input string",
                )
            vectors = [await svc.embed_query(texts[0], model_name=resolved_model_name)]
        else:
            vectors = await svc.embed_documents(texts, model_name=resolved_model_name)

        dimensions = len(vectors[0]) if vectors else 0
        return EmbeddingResponse(
            model=resolved_model_name,
            dimensions=dimensions,
            count=len(vectors),
            data=vectors,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {exc}") from exc


def configure_runtime(
    *,
    host: str,
    port: int,
    embedding_model_name: Optional[str],
    embedding_dir: Optional[str],
    device: Optional[str],
) -> None:
    global HOST
    global PORT
    global EMBEDDING_MODEL_NAME
    global EMBEDDING_DIR
    global DEVICE

    HOST = host
    PORT = port
    EMBEDDING_MODEL_NAME = embedding_model_name
    EMBEDDING_DIR = embedding_dir
    DEVICE = device


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="remote-embedding-server",
        description=(
            "Run a shared embedding server so multiple applications can reuse one "
            "loaded embedding model instance."
        ),
    )
    parser.add_argument("--host", default=HOST, help="Bind host for the API server.")
    parser.add_argument("--port", type=int, default=PORT, help="Bind port for the API server.")
    parser.add_argument(
        "--model-name",
        default=EMBEDDING_MODEL_NAME,
        help="Default embedding model name to preload and use when requests omit model_name.",
    )
    parser.add_argument(
        "--embedding-dir",
        default=EMBEDDING_DIR,
        help="Optional Hugging Face cache/model directory.",
    )
    parser.add_argument(
        "--device",
        default=DEVICE,
        help="Torch device passed to HuggingFaceEmbeddings, for example cpu or cuda.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    configure_runtime(
        host=args.host,
        port=args.port,
        embedding_model_name=args.model_name,
        embedding_dir=args.embedding_dir,
        device=args.device,
    )
    uvicorn.run(app, host=HOST, port=PORT)
