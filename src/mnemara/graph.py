"""Graph backend — kuzu-backed property graph for relational recall.

Lazy: kuzu DB is not opened until the first graph tool call.
Graceful: if kuzu import fails or the DB is unavailable, every method
returns a structured {ok: False, error: ...} dict so memory + wiki +
RAG keep working.

Schema:
  Node(id STRING PRIMARY KEY, label STRING, properties STRING /* JSON */, created_at STRING)
  Edge(FROM Node TO Node, id STRING, relationship STRING, properties STRING /* JSON */, created_at STRING)
"""
from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from . import paths
from .config import Config
from .logging_util import log


_NOW = lambda: datetime.now(timezone.utc).isoformat()


def _safe_str(v: Any) -> str:
    if v is None:
        return ""
    return str(v)


class KuzuStore:
    """Open or create the per-instance Kuzu DB and ensure schema."""

    def __init__(self, instance: str, cfg: Config):
        self.instance = instance
        self.cfg = cfg
        self._db = None
        self._conn = None
        self._init_error: Optional[str] = None

    def _connect(self) -> Optional[str]:
        """Open the DB and ensure schema. Returns an error string on failure,
        None on success."""
        if self._conn is not None:
            return None
        if self._init_error is not None:
            return self._init_error
        try:
            import kuzu  # type: ignore
        except Exception as e:  # pragma: no cover
            msg = f"kuzu import failed: {e}"
            self._init_error = msg
            log("graph_import_error", error=str(e))
            return msg
        try:
            d = paths.graph_dir(self.instance)
            d.mkdir(parents=True, exist_ok=True)
            db_path = d / "kuzu_db"
            self._db = kuzu.Database(str(db_path))
            self._conn = kuzu.Connection(self._db)
            self._ensure_schema()
            return None
        except Exception as e:
            msg = f"kuzu connect failed: {e}"
            self._init_error = msg
            log("graph_connect_error", error=str(e))
            return msg

    def _ensure_schema(self) -> None:
        # Try create — ignore "already exists" errors.
        try:
            self._conn.execute(
                "CREATE NODE TABLE Node("
                "id STRING PRIMARY KEY, "
                "label STRING, "
                "properties STRING, "
                "created_at STRING)"
            )
        except Exception:
            pass
        try:
            self._conn.execute(
                "CREATE REL TABLE Edge("
                "FROM Node TO Node, "
                "id STRING, "
                "relationship STRING, "
                "properties STRING, "
                "created_at STRING)"
            )
        except Exception:
            pass

    # ------------------------------------------------------------------ checks

    def _check(self) -> Optional[dict[str, Any]]:
        if not getattr(self.cfg, "graph_enabled", True):
            return {"ok": False, "error": "Graph backend disabled by config"}
        err = self._connect()
        if err:
            return {"ok": False, "error": f"Graph backend unavailable: {err}"}
        return None

    # ------------------------------------------------------------------ CRUD

    def add_node(
        self,
        label: str,
        properties: Optional[dict[str, Any]] = None,
        node_id: Optional[str] = None,
    ) -> dict[str, Any]:
        chk = self._check()
        if chk is not None:
            return chk
        nid = node_id or uuid.uuid4().hex
        try:
            props_json = json.dumps(properties or {})
            self._conn.execute(
                "CREATE (n:Node {id: $id, label: $label, "
                "properties: $p, created_at: $t})",
                {"id": nid, "label": _safe_str(label), "p": props_json, "t": _NOW()},
            )
            return {"ok": True, "id": nid}
        except Exception as e:
            log("graph_add_node_error", error=str(e))
            return {"ok": False, "error": f"Graph backend unavailable: {e}"}

    def find_or_create_node(
        self, label: str, properties: dict[str, Any], match_key: str = "ref"
    ) -> dict[str, Any]:
        """Return existing node id matching label + properties[match_key],
        or create a new one. The match key is searched within properties JSON."""
        chk = self._check()
        if chk is not None:
            return chk
        match_val = properties.get(match_key) if properties else None
        if match_val is not None:
            # Cypher LIKE match against the JSON-encoded properties string.
            try:
                needle = f'"{match_key}": "{match_val}"'
                res = self._conn.execute(
                    "MATCH (n:Node) WHERE n.label = $label AND n.properties "
                    "CONTAINS $needle RETURN n.id AS id LIMIT 1",
                    {"label": _safe_str(label), "needle": needle},
                )
                if res.has_next():
                    row = res.get_next()
                    return {"ok": True, "id": row[0], "created": False}
            except Exception as e:
                log("graph_find_node_error", error=str(e))
        out = self.add_node(label, properties)
        if out.get("ok"):
            out["created"] = True
        return out

    def add_edge(
        self,
        from_id: str,
        to_id: str,
        relationship: str,
        properties: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        chk = self._check()
        if chk is not None:
            return chk
        eid = uuid.uuid4().hex
        try:
            props_json = json.dumps(properties or {})
            self._conn.execute(
                "MATCH (a:Node {id: $from_id}), (b:Node {id: $to_id}) "
                "CREATE (a)-[:Edge {id: $id, relationship: $rel, "
                "properties: $p, created_at: $t}]->(b)",
                {
                    "from_id": _safe_str(from_id),
                    "to_id": _safe_str(to_id),
                    "id": eid,
                    "rel": _safe_str(relationship),
                    "p": props_json,
                    "t": _NOW(),
                },
            )
            return {"ok": True, "id": eid}
        except Exception as e:
            log("graph_add_edge_error", error=str(e))
            return {"ok": False, "error": f"Graph backend unavailable: {e}"}

    def query(self, cypher: str, params: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        chk = self._check()
        if chk is not None:
            return chk
        try:
            res = self._conn.execute(cypher, params or {})
            cols = res.get_column_names()
            rows: list[dict[str, Any]] = []
            while res.has_next():
                vals = res.get_next()
                rows.append({c: _coerce(v) for c, v in zip(cols, vals)})
            return {"ok": True, "columns": cols, "rows": rows}
        except Exception as e:
            log("graph_query_error", error=str(e))
            return {"ok": False, "error": f"Graph backend unavailable: {e}"}

    def neighbors(self, node_id: str, depth: int = 1) -> dict[str, Any]:
        chk = self._check()
        if chk is not None:
            return chk
        depth = max(1, min(int(depth), 5))
        try:
            cypher = (
                f"MATCH (n:Node {{id: $id}})-[:Edge*1..{depth}]-(m:Node) "
                "RETURN DISTINCT m.id AS id, m.label AS label, m.properties AS properties"
            )
            res = self._conn.execute(cypher, {"id": _safe_str(node_id)})
            cols = res.get_column_names()
            rows: list[dict[str, Any]] = []
            while res.has_next():
                vals = res.get_next()
                rec = {c: _coerce(v) for c, v in zip(cols, vals)}
                rec["properties"] = _maybe_load_json(rec.get("properties"))
                rows.append(rec)
            return {"ok": True, "neighbors": rows}
        except Exception as e:
            log("graph_neighbors_error", error=str(e))
            return {"ok": False, "error": f"Graph backend unavailable: {e}"}

    def match(self, pattern: dict[str, Any]) -> dict[str, Any]:
        """Convenience matcher: pattern = {label, properties_subset}.
        Filters by label + JSON-substring of each properties_subset key/value."""
        chk = self._check()
        if chk is not None:
            return chk
        label = pattern.get("label")
        subset = pattern.get("properties_subset") or {}
        try:
            wheres = []
            params: dict[str, Any] = {}
            if label:
                wheres.append("n.label = $label")
                params["label"] = _safe_str(label)
            for i, (k, v) in enumerate(subset.items()):
                key = f"needle_{i}"
                wheres.append(f"n.properties CONTAINS ${key}")
                params[key] = f'"{k}": "{v}"'
            cypher = "MATCH (n:Node)"
            if wheres:
                cypher += " WHERE " + " AND ".join(wheres)
            cypher += " RETURN n.id AS id, n.label AS label, n.properties AS properties LIMIT 100"
            res = self._conn.execute(cypher, params)
            cols = res.get_column_names()
            rows: list[dict[str, Any]] = []
            while res.has_next():
                vals = res.get_next()
                rec = {c: _coerce(v) for c, v in zip(cols, vals)}
                rec["properties"] = _maybe_load_json(rec.get("properties"))
                rows.append(rec)
            return {"ok": True, "matches": rows}
        except Exception as e:
            log("graph_match_error", error=str(e))
            return {"ok": False, "error": f"Graph backend unavailable: {e}"}

    def shortest_path(self, from_id: str, to_id: str, max_depth: int = 6) -> dict[str, Any]:
        chk = self._check()
        if chk is not None:
            return chk
        depth = max(1, min(int(max_depth), 10))
        try:
            cypher = (
                f"MATCH p = (a:Node {{id: $from_id}})-[:Edge*1..{depth}]-(b:Node {{id: $to_id}}) "
                "RETURN nodes(p) AS path LIMIT 1"
            )
            res = self._conn.execute(
                cypher, {"from_id": _safe_str(from_id), "to_id": _safe_str(to_id)}
            )
            if not res.has_next():
                return {"ok": True, "path": []}
            row = res.get_next()
            nodes = row[0] or []
            ids = []
            for node in nodes:
                if isinstance(node, dict):
                    ids.append(node.get("id"))
            return {"ok": True, "path": ids}
        except Exception as e:
            log("graph_shortest_path_error", error=str(e))
            return {"ok": False, "error": f"Graph backend unavailable: {e}"}


def _coerce(v: Any) -> Any:
    """Coerce kuzu return values to JSON-serializable shapes."""
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, list):
        return [_coerce(x) for x in v]
    if isinstance(v, dict):
        return {k: _coerce(x) for k, x in v.items()}
    return str(v)


def _maybe_load_json(s: Any) -> Any:
    if isinstance(s, str) and s:
        try:
            return json.loads(s)
        except (ValueError, TypeError):
            return s
    return s or {}


# Module-level lazy singleton keyed by instance name.
_STORES: dict[str, KuzuStore] = {}


def store_for(instance: str, cfg: Config) -> KuzuStore:
    s = _STORES.get(instance)
    if s is None:
        s = KuzuStore(instance, cfg)
        _STORES[instance] = s
    return s


def reset_stores() -> None:
    """Test helper — clear singletons between tests."""
    for s in _STORES.values():
        try:
            if s._conn is not None:
                s._conn.close()
        except Exception:
            pass
        try:
            if s._db is not None:
                s._db.close()
        except Exception:
            pass
    _STORES.clear()


# -----------------------------------------------------------------------------
# Auto-edge helpers — invoked by tools.write_memory and wiki.write_page.
# Wrapped in try/except by callers so graph failures don't fail primary writes.
# -----------------------------------------------------------------------------


_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)
_TAGS_RE = re.compile(r"^tags:\s*(.+)$", re.MULTILINE)


def parse_wiki_frontmatter_tags(content: str) -> list[str]:
    """Cheap tags parse — finds 'tags: [a, b, c]' or 'tags: a, b, c' in
    the leading frontmatter block."""
    if not content:
        return []
    m = _FRONTMATTER_RE.search(content)
    if not m:
        return []
    fm = m.group(1)
    t = _TAGS_RE.search(fm)
    if not t:
        return []
    raw = t.group(1).strip()
    raw = raw.strip("[]")
    parts = [p.strip().strip('"').strip("'") for p in raw.split(",")]
    return [p for p in parts if p]


def auto_edges_from_memory(
    instance: str,
    cfg: Config,
    memory_text: str,
    payload: Optional[dict[str, Any]],
    source_path: str,
) -> dict[str, Any]:
    """Create graph nodes/edges from a memory write.

    - The memory entry itself becomes a node (label='memory_entry').
    - When payload['applies_to'] is present, each ref becomes an entity node
      (or is found if already present) and an edge mentions->entity is added.

    Returns the structured result; callers should swallow exceptions.
    """
    if not getattr(cfg, "graph_enabled", True):
        return {"ok": False, "error": "Graph backend disabled by config"}
    s = store_for(instance, cfg)
    chk = s._check()
    if chk is not None:
        return chk
    mem_text = (memory_text or "").strip()
    mem_props = {
        "preview": mem_text[:200],
        "source_path": source_path,
    }
    mem_res = s.add_node("memory_entry", mem_props)
    if not mem_res.get("ok"):
        return mem_res
    mem_id = mem_res["id"]
    edges: list[str] = []
    refs: list[str] = []
    if payload:
        applies = payload.get("applies_to")
        if isinstance(applies, str):
            refs = [r.strip() for r in re.split(r"[,\s]+", applies) if r.strip()]
        elif isinstance(applies, list):
            refs = [str(r).strip() for r in applies if str(r).strip()]
    for ref in refs:
        ent = s.find_or_create_node("entity", {"ref": ref})
        if ent.get("ok"):
            e = s.add_edge(mem_id, ent["id"], "applies_to", {"ref": ref})
            if e.get("ok"):
                edges.append(e["id"])
    return {"ok": True, "memory_node_id": mem_id, "edge_ids": edges, "refs": refs}


def auto_edges_from_wiki(
    instance: str,
    cfg: Config,
    page_path: str,
    content: str,
) -> dict[str, Any]:
    """Create graph nodes/edges from a wiki write.

    - The page becomes a node (label='wiki_page', ref=page_path).
    - Tags from frontmatter become topic-tag nodes (label='topic_tag', ref=tag)
      with edges page->tag (relationship='tagged').
    """
    if not getattr(cfg, "graph_enabled", True):
        return {"ok": False, "error": "Graph backend disabled by config"}
    s = store_for(instance, cfg)
    chk = s._check()
    if chk is not None:
        return chk
    page_props = {"ref": page_path, "preview": (content or "")[:200]}
    pg = s.find_or_create_node("wiki_page", page_props)
    if not pg.get("ok"):
        return pg
    page_id = pg["id"]
    tags = parse_wiki_frontmatter_tags(content)
    edges: list[str] = []
    for tag in tags:
        t = s.find_or_create_node("topic_tag", {"ref": tag})
        if t.get("ok"):
            e = s.add_edge(page_id, t["id"], "tagged", {"tag": tag})
            if e.get("ok"):
                edges.append(e["id"])
    return {"ok": True, "wiki_node_id": page_id, "tags": tags, "edge_ids": edges}
