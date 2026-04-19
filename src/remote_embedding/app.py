"""FastAPI application serving remote embedding inference."""

import asyncio
import argparse
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Literal, Optional, Union

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel, Field

load_dotenv()
logger = logging.getLogger("remote_embedding.server")


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def _parse_json_mapping(value: Optional[str], *, source: str) -> dict[str, Any]:
    if not value:
        return {}

    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{source} must be valid JSON.") from exc

    if not isinstance(parsed, dict):
        raise ValueError(f"{source} must be a JSON object.")

    return parsed


def _merge_mappings(*mappings: Optional[dict[str, Any]]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for mapping in mappings:
        if mapping:
            merged.update(mapping)
    return merged


HOST = os.getenv("HOST", "0.0.0.0")
PORT = _env_int("PORT", 5055)
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
EMBEDDING_DIR = os.getenv("EMBEDDING_DIR")
DEVICE = os.getenv("DEVICE")
MODEL_KWARGS = _parse_json_mapping(os.getenv("MODEL_KWARGS"), source="MODEL_KWARGS")
ENCODE_KWARGS = _parse_json_mapping(os.getenv("ENCODE_KWARGS"), source="ENCODE_KWARGS")


class EmbeddingRequest(BaseModel):
    input: Union[str, list[str]] = Field(..., description="String or list of strings")
    mode: Literal["documents", "query"] = "documents"
    model_name: Optional[str] = None
    instruction: Optional[str] = None
    embedding_dir: Optional[str] = None
    model_kwargs: Optional[dict[str, Any]] = None
    encode_kwargs: Optional[dict[str, Any]] = None


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

    def _cache_key(
        self,
        model_name: str,
        embedding_dir: Optional[str],
        model_kwargs: dict[str, Any],
        encode_kwargs: dict[str, Any],
    ) -> str:
        return json.dumps(
            {
                "model_name": model_name,
                "embedding_dir": embedding_dir,
                "model_kwargs": model_kwargs,
                "encode_kwargs": encode_kwargs,
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    def load(
        self,
        model_name: Optional[str] = None,
        embedding_dir: Optional[str] = None,
        model_kwargs: Optional[dict[str, Any]] = None,
        encode_kwargs: Optional[dict[str, Any]] = None,
    ) -> HuggingFaceEmbeddings:
        resolved_model_name = self._resolve_model_name(model_name)
        resolved_embedding_dir = embedding_dir if embedding_dir is not None else EMBEDDING_DIR
        resolved_model_kwargs = _merge_mappings(
            {
                "device": DEVICE,
                "local_files_only": True,
                "trust_remote_code": True,
            },
            MODEL_KWARGS,
            model_kwargs,
        )
        resolved_encode_kwargs = _merge_mappings(ENCODE_KWARGS, encode_kwargs)
        cache_key = self._cache_key(
            resolved_model_name,
            resolved_embedding_dir,
            resolved_model_kwargs,
            resolved_encode_kwargs,
        )
        if cache_key in self.embed_models:
            return self.embed_models[cache_key]

        logger.info(
            "Loading embedding model '%s' with cache dir '%s'.",
            resolved_model_name,
            resolved_embedding_dir or "<default>",
        )
        embed_model = HuggingFaceEmbeddings(
            model_name=resolved_model_name,
            model_kwargs=resolved_model_kwargs,
            encode_kwargs=resolved_encode_kwargs,
            cache_folder=resolved_embedding_dir,
        )
        self.embed_models[cache_key] = embed_model
        return embed_model

    async def embed_documents(
        self,
        texts: list[str],
        model_name: Optional[str] = None,
        embedding_dir: Optional[str] = None,
        model_kwargs: Optional[dict[str, Any]] = None,
        encode_kwargs: Optional[dict[str, Any]] = None,
    ) -> list[list[float]]:
        embed_model = self.load(
            model_name,
            embedding_dir=embedding_dir,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )

        # Serialize GPU access to avoid VRAM spikes from concurrent requests.
        async with self.lock:
            return await asyncio.to_thread(embed_model.embed_documents, texts)

    async def embed_query(
        self,
        text: str,
        model_name: Optional[str] = None,
        embedding_dir: Optional[str] = None,
        model_kwargs: Optional[dict[str, Any]] = None,
        encode_kwargs: Optional[dict[str, Any]] = None,
    ) -> list[float]:
        embed_model = self.load(
            model_name,
            embedding_dir=embedding_dir,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )

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
    loaded_model_name = configured_model_name or next(
        (
            model.model_name
            for model in svc.embed_models.values()
            if getattr(model, "model_name", None)
        ),
        "",
    )

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
            vectors = [
                await svc.embed_query(
                    texts[0],
                    model_name=resolved_model_name,
                    embedding_dir=req.embedding_dir,
                    model_kwargs=req.model_kwargs,
                    encode_kwargs=req.encode_kwargs,
                )
            ]
        else:
            vectors = await svc.embed_documents(
                texts,
                model_name=resolved_model_name,
                embedding_dir=req.embedding_dir,
                model_kwargs=req.model_kwargs,
                encode_kwargs=req.encode_kwargs,
            )

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
    model_kwargs: dict[str, Any],
    encode_kwargs: dict[str, Any],
) -> None:
    global HOST
    global PORT
    global EMBEDDING_MODEL_NAME
    global EMBEDDING_DIR
    global DEVICE
    global MODEL_KWARGS
    global ENCODE_KWARGS

    HOST = host
    PORT = port
    EMBEDDING_MODEL_NAME = embedding_model_name
    EMBEDDING_DIR = embedding_dir
    DEVICE = device
    MODEL_KWARGS = model_kwargs
    ENCODE_KWARGS = encode_kwargs


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
    parser.add_argument(
        "--model-kwargs",
        default=json.dumps(MODEL_KWARGS) if MODEL_KWARGS else None,
        help="JSON object merged into HuggingFaceEmbeddings model_kwargs.",
    )
    parser.add_argument(
        "--encode-kwargs",
        default=json.dumps(ENCODE_KWARGS) if ENCODE_KWARGS else None,
        help="JSON object passed to HuggingFaceEmbeddings encode_kwargs.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    model_kwargs = _parse_json_mapping(args.model_kwargs, source="--model-kwargs")
    encode_kwargs = _parse_json_mapping(args.encode_kwargs, source="--encode-kwargs")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    configure_runtime(
        host=args.host,
        port=args.port,
        embedding_model_name=args.model_name,
        embedding_dir=args.embedding_dir,
        device=args.device,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )
    logger.info(
        "Starting remote-embedding-server on %s:%s with default model '%s' and embedding dir '%s'.",
        HOST,
        PORT,
        EMBEDDING_MODEL_NAME or "<per-request>",
        EMBEDDING_DIR or "<default>",
    )
    uvicorn.run(app, host=HOST, port=PORT)
