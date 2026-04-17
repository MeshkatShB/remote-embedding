"""Client for the remote embedding FastAPI service."""

from typing import Optional

import requests
from langchain_core.embeddings import Embeddings
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class RemoteEmbeddings(Embeddings):
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:5055",
        timeout: int = 300,
        expected_dimensions: Optional[int] = None,
        model_name: Optional[str] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.expected_dimensions = expected_dimensions
        self.model_name = model_name

        self.session = requests.Session()
        retries = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST", "GET"],
        )
        adapter = HTTPAdapter(max_retries=retries)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def _check_dim(self, vectors: list[list[float]]) -> None:
        if not vectors or self.expected_dimensions is None:
            return
        if len(vectors[0]) != self.expected_dimensions:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.expected_dimensions}, got {len(vectors[0])}"
            )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        payload = {"input": texts, "mode": "documents"}
        if self.model_name:
            payload["model_name"] = self.model_name

        response = self.session.post(
            f"{self.base_url}/embed",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        vectors = response.json()["data"]
        self._check_dim(vectors)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        payload = {"input": text, "mode": "query"}
        if self.model_name:
            payload["model_name"] = self.model_name

        response = self.session.post(
            f"{self.base_url}/embed",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        vectors = response.json()["data"]
        self._check_dim(vectors)
        return vectors[0]
