"""RAG backend — LanceDB store + Ollama embeddings (nomic-embed-text).

Lazy: no Ollama HTTP call or LanceDB connection until first index/query.
Graceful: index/query catch backend errors and return a structured
unavailable result so memory + wiki keep working.
"""
from __future__ import annotations

import math
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from . import paths
from .config import Config
from .logging_util import log

EMBEDDING_DIM = 768
TABLE_NAME = "memories"


def embed_text(url: str, model: str, text: str, timeout: float = 30.0) -> list[float]:
    """Call Ollama /api/embeddings synchronously. Raises on transport error."""
    import httpx

    payload = {"model": model, "prompt": text}
    with httpx.Client(timeout=timeout) as client:
        r = client.post(url, json=payload)
        r.raise_for_status()
        data = r.json()
    emb = data.get("embedding")
    if not isinstance(emb, list) or not emb:
        raise RuntimeError(f"empty embedding response: {data!r}")
    return [float(x) for x in emb]


class LanceDBStore:
    """Open or create the per-instance LanceDB and ensure schema.

    Lazy: connect() is only called from index/query.
    """

    def __init__(self, instance: str, cfg: Config):
        self.instance = instance
        self.cfg = cfg
        self._db = None
        self._table = None

    def _connect(self) -> None:
        if self._table is not None:
            return
        import lancedb
        import pyarrow as pa

        db_dir = paths.rag_index_dir(self.instance)
        db_dir.mkdir(parents=True, exist_ok=True)
        self._db = lancedb.connect(str(db_dir))
        try:
            tables_resp = self._db.list_tables()
            tables = getattr(tables_resp, "tables", tables_resp)
            existing = set(tables)
        except (AttributeError, TypeError):
            existing = set(self._db.table_names())
        if TABLE_NAME in existing:
            self._table = self._db.open_table(TABLE_NAME)
        else:
            schema = pa.schema(
                [
                    pa.field("id", pa.string()),
                    pa.field("text", pa.string()),
                    pa.field("kind", pa.string()),
                    pa.field("source_path", pa.string()),
                    pa.field("category", pa.string()),
                    pa.field("ts", pa.string()),
                    pa.field(
                        "embedding",
                        pa.list_(pa.float32(), EMBEDDING_DIM),
                    ),
                ]
            )
            self._table = self._db.create_table(TABLE_NAME, schema=schema)

    def _embed(self, text: str) -> list[float]:
        return embed_text(
            self.cfg.rag_embed_url, self.cfg.rag_embed_model, text
        )

    def index(
        self,
        text: str,
        kind: str = "manual",
        source_path: str = "",
        category: str = "",
    ) -> dict[str, Any]:
        if not getattr(self.cfg, "rag_enabled", True):
            return {"ok": False, "error": "RAG backend disabled by config"}
        try:
            self._connect()
            emb = self._embed(text)
            row_id = uuid.uuid4().hex
            self._table.add(
                [
                    {
                        "id": row_id,
                        "text": text,
                        "kind": kind,
                        "source_path": source_path,
                        "category": category,
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "embedding": emb,
                    }
                ]
            )
            return {"ok": True, "id": row_id}
        except Exception as e:
            log("rag_index_error", error=str(e))
            return {"ok": False, "error": f"RAG backend unavailable: {e}"}

    def query(
        self, question: str, k: int = 5, kind: Optional[str] = None
    ) -> dict[str, Any]:
        if not getattr(self.cfg, "rag_enabled", True):
            return {"ok": False, "error": "RAG backend disabled by config"}
        try:
            self._connect()
            emb = self._embed(question)
            search = self._table.search(emb).limit(max(1, int(k)))
            if kind:
                search = search.where(f"kind = '{kind}'", prefilter=True)
            rows = search.to_list()
            results = []
            for r in rows:
                results.append(
                    {
                        "id": r.get("id"),
                        "text": r.get("text"),
                        "kind": r.get("kind"),
                        "source_path": r.get("source_path"),
                        "category": r.get("category"),
                        "ts": r.get("ts"),
                        "distance": float(r.get("_distance", 0.0)),
                    }
                )
            return {"ok": True, "results": results}
        except Exception as e:
            log("rag_query_error", error=str(e))
            return {"ok": False, "error": f"RAG backend unavailable: {e}"}


# Module-level lazy singleton keyed by instance name.
_STORES: dict[str, LanceDBStore] = {}


def store_for(instance: str, cfg: Config) -> LanceDBStore:
    s = _STORES.get(instance)
    if s is None:
        s = LanceDBStore(instance, cfg)
        _STORES[instance] = s
    return s


def reset_stores() -> None:
    """Test helper — clear singletons between tests."""
    _STORES.clear()
