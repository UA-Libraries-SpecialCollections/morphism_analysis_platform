#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone Morphism Analysis and Evidence Documentation Platform
================================================================

Python 3.10 desktop application for inspecting and analyzing compact/enriched
``morphism_comparison.pkl`` files produced by the Embedding Manifolds as Semantic
Morphisms pipeline.

Implemented components
----------------------
1. PKL loader + schema inspector
2. Match table/query workbench
3. Restored compact plot-cache graph views
4. Candidate evidence browser with Markdown/JSON evidence-packet export
5. Selected edge-match 3D visualizer with on-demand segment re-embedding

Performance notes
-----------------
Large compact pickles are opened in staged mode: deserialize and inspect schema first;
query tables, candidate tables, and graph rendering are run only on request.

The app intentionally does not import the pipeline modules.  It reads the compact
pickle payload directly, so it can be used as a standalone analysis tool.

Security note
-------------
Pickle is not a safe interchange format for untrusted data.  Only open .pkl files
that you created or that come from trusted collaborators.

Recommended dependencies
------------------------
Required: numpy, matplotlib
Optional: pandas (used for CSV export convenience; the app has a stdlib fallback)
Optional for 3D point clouds: sentence-transformers

Run
---
    python morphism_analysis_platform.py
"""

from __future__ import annotations

import csv
import json
import math
import os
import pickle
import re
import sys
import traceback
import threading
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - pandas is optional
    pd = None

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

try:
    import matplotlib
    # TkAgg is appropriate for this Tkinter desktop app.  If another backend was
    # already selected by the environment, matplotlib will generally keep it.
    try:
        matplotlib.use("TkAgg")
    except Exception:
        pass
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    from matplotlib.figure import Figure
    from matplotlib import cm, colors as mcolors
except Exception as ex:  # pragma: no cover
    raise RuntimeError(
        "This application requires matplotlib with a Tk-compatible backend. "
        "Install matplotlib and ensure tkinter is available."
    ) from ex


# -----------------------------------------------------------------------------
# Numeric / schema helpers
# -----------------------------------------------------------------------------

def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None or x == "":
            return float(default)
        v = float(x)
        return v if math.isfinite(v) else float(default)
    except Exception:
        return float(default)


def _safe_int(x: Any, default: int = -1) -> int:
    try:
        if x is None or x == "":
            return int(default)
        return int(x)
    except Exception:
        return int(default)


def _fmt_float(x: Any, digits: int = 4, blank_for_nan: bool = True) -> str:
    try:
        v = float(x)
        if not math.isfinite(v):
            return "" if blank_for_nan else "nan"
        return f"{v:.{digits}f}"
    except Exception:
        return ""


def _bytes_human(nbytes: int) -> str:
    n = float(max(0, int(nbytes)))
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024.0 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{nbytes} B"


def _array_nbytes(obj: Any) -> int:
    try:
        return int(np.asarray(obj).nbytes)
    except Exception:
        return 0


def _is_structured_array(arr: Any) -> bool:
    try:
        return isinstance(arr, np.ndarray) and arr.dtype.names is not None
    except Exception:
        return False


def _structured_field(records: np.ndarray, name: str, default: Any = 0, dtype: Any = float) -> np.ndarray:
    """Return a structured-array field or a default-filled array."""
    n = int(records.shape[0]) if isinstance(records, np.ndarray) else 0
    try:
        if _is_structured_array(records) and name in (records.dtype.names or ()):  # type: ignore[union-attr]
            return np.asarray(records[name], dtype=dtype)
    except Exception:
        pass
    return np.full(n, default, dtype=dtype)


def _dict_array_field(d: Dict[str, Any], name: str, n: int, default: Any = 0, dtype: Any = float) -> np.ndarray:
    try:
        if isinstance(d, dict) and name in d:
            arr = np.asarray(d[name], dtype=dtype)
            if arr.shape[0] == n:
                return arr
    except Exception:
        pass
    return np.full(n, default, dtype=dtype)


def _clip01(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    return np.clip(np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)


def _unit_vector(x: Any, eps: float = 1e-12) -> np.ndarray:
    """Return a finite unit vector or a zero vector with the same shape."""
    try:
        v = np.asarray(x, dtype=float).reshape(-1)
        if v.size == 0:
            return v
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        n = float(np.linalg.norm(v))
        if n <= eps:
            return np.zeros_like(v, dtype=float)
        return v / n
    except Exception:
        return np.zeros((0,), dtype=float)


def _vector_cosine(a: Any, b: Any, eps: float = 1e-12) -> float:
    """Cosine between two finite vectors, or NaN when unavailable."""
    try:
        va = np.asarray(a, dtype=float).reshape(-1)
        vb = np.asarray(b, dtype=float).reshape(-1)
        if va.size == 0 or vb.size == 0 or va.size != vb.size:
            return float("nan")
        va = np.nan_to_num(va, nan=0.0, posinf=0.0, neginf=0.0)
        vb = np.nan_to_num(vb, nan=0.0, posinf=0.0, neginf=0.0)
        na = float(np.linalg.norm(va))
        nb = float(np.linalg.norm(vb))
        if na <= eps or nb <= eps:
            return float("nan")
        return float(np.dot(va, vb) / (na * nb))
    except Exception:
        return float("nan")


def _vector_l2(a: Any, b: Any) -> float:
    """Euclidean distance between two finite vectors, or NaN when unavailable."""
    try:
        va = np.asarray(a, dtype=float).reshape(-1)
        vb = np.asarray(b, dtype=float).reshape(-1)
        if va.size == 0 or vb.size == 0 or va.size != vb.size:
            return float("nan")
        va = np.nan_to_num(va, nan=0.0, posinf=0.0, neginf=0.0)
        vb = np.nan_to_num(vb, nan=0.0, posinf=0.0, neginf=0.0)
        return float(np.linalg.norm(va - vb))
    except Exception:
        return float("nan")


def _label_equal(a: Any, b: Any) -> bool:
    """Robust comparison for cluster labels that may be int, np.int, or str."""
    try:
        return int(a) == int(b)
    except Exception:
        return str(a) == str(b)


def _label_display(label: Any) -> str:
    try:
        return str(int(label))
    except Exception:
        return str(label)


def _fit_pca_projection(points: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit a lightweight 3D PCA basis over selected high-dimensional points.

    Returns (center, basis) where basis has shape (dim, 3).  The implementation
    uses NumPy SVD only so the standalone app does not require scikit-learn.
    """
    arrays = []
    for p in points:
        try:
            arr = np.asarray(p, dtype=float)
            if arr.ndim == 1:
                arr = arr[None, :]
            if arr.ndim == 2 and arr.shape[0] and arr.shape[1]:
                finite = np.isfinite(arr).all(axis=1)
                if np.any(finite):
                    arrays.append(arr[finite])
        except Exception:
            pass
    if not arrays:
        return np.zeros((3,), dtype=float), np.eye(3, dtype=float)
    X = np.vstack(arrays)
    dim = int(X.shape[1])
    center = np.nanmean(X, axis=0)
    Xc = np.nan_to_num(X - center, nan=0.0, posinf=0.0, neginf=0.0)
    if Xc.shape[0] >= 2 and dim >= 1:
        try:
            _, _, vt = np.linalg.svd(Xc, full_matrices=False)
            basis = vt[: min(3, vt.shape[0])].T
        except Exception:
            basis = np.zeros((dim, 0), dtype=float)
    else:
        basis = np.zeros((dim, 0), dtype=float)

    # Complete to three orthonormal axes with standard basis vectors when the
    # selected points are nearly collinear or too sparse.
    cols = [basis[:, i] for i in range(basis.shape[1])] if basis.size else []
    for i in range(dim):
        if len(cols) >= 3:
            break
        e = np.zeros(dim, dtype=float)
        e[i] = 1.0
        for c in cols:
            e = e - np.dot(e, c) * c
        n = float(np.linalg.norm(e))
        if n > 1e-8:
            cols.append(e / n)
    while len(cols) < 3:
        z = np.zeros(dim, dtype=float)
        cols.append(z)
    return center, np.column_stack(cols[:3])


def _project_points(points: Any, center: np.ndarray, basis: np.ndarray) -> np.ndarray:
    arr = np.asarray(points, dtype=float)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.size == 0:
        return np.zeros((0, 3), dtype=float)
    return (np.nan_to_num(arr - center, nan=0.0, posinf=0.0, neginf=0.0) @ basis).astype(float)


def _build_space_preprocess_embeddings(E: Any, remove_top_components_n: int = 0) -> np.ndarray:
    """
    Reconstruct the embedding coordinate space used by the current CDM build.

    In the pipeline's mk_delta_manifold, SentenceTransformer embeddings are
    L2-normalized, then remove_top_components(..., n=0) is called.  Despite the
    name, n=0 is not a no-op in that implementation: it document-centers the
    embedding matrix and renormalizes each segment vector.  Stored cluster
    centroids in document_delta_dict.pkl are means in that build-space, so the
    analysis platform must apply the same document-level transform before
    comparing re-embedded point clouds to stored centroids.
    """
    X = np.asarray(E, dtype=np.float32)
    if X.ndim == 1:
        X = X[None, :]
    if X.size == 0 or X.shape[0] == 0:
        return np.zeros((0, 0), dtype=np.float32)
    Xc = X - np.nanmean(X, axis=0, keepdims=True)
    n = max(0, int(remove_top_components_n or 0))
    if n > 0 and Xc.shape[0] >= 2 and Xc.shape[1] >= 1:
        try:
            _, _, vt = np.linalg.svd(Xc, full_matrices=False)
            P = vt[:min(n, vt.shape[0])].T
            if P.size:
                Xc = Xc - Xc @ P @ P.T
        except Exception:
            pass
    Xc = Xc / (np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-12)
    return np.asarray(Xc, dtype=np.float32)


def _set_axes_equal_3d(ax: Any) -> None:
    """Set equal 3D axis scales for Matplotlib 3D axes."""
    try:
        x_limits = np.asarray(ax.get_xlim3d(), dtype=float)
        y_limits = np.asarray(ax.get_ylim3d(), dtype=float)
        z_limits = np.asarray(ax.get_zlim3d(), dtype=float)
        spans = np.array([np.diff(x_limits)[0], np.diff(y_limits)[0], np.diff(z_limits)[0]], dtype=float)
        centers = np.array([x_limits.mean(), y_limits.mean(), z_limits.mean()], dtype=float)
        radius = max(float(np.nanmax(np.abs(spans))) / 2.0, 1e-6)
        ax.set_xlim3d(centers[0] - radius, centers[0] + radius)
        ax.set_ylim3d(centers[1] - radius, centers[1] + radius)
        ax.set_zlim3d(centers[2] - radius, centers[2] + radius)
    except Exception:
        pass


def _match_type_label(x: Any) -> str:
    try:
        if isinstance(x, bytes):
            x = x.decode("utf-8", errors="replace")
        if isinstance(x, str):
            raw = x.strip().lower()
            if raw in {"aligned", "align", "0"}:
                return "aligned"
            if raw in {"pc1_only", "pc1only", "pc1-only", "1"}:
                return "pc1_only"
            return raw or "unknown"
        iv = int(x)
        return "aligned" if iv == 0 else "pc1_only" if iv == 1 else f"type_{iv}"
    except Exception:
        return "unknown"


def _normalize_payload(obj: Any) -> Dict[str, Any]:
    """
    Extract a morphism_comparison payload from either:
      - a direct payload: {'kind': 'morphism_comparison', ...}
      - an Analyze result wrapper: {'morphism_comparison': {...}}
      - a legacy wrapper created by save_morphism_comparison_pickle
    """
    if isinstance(obj, dict) and obj.get("kind") == "morphism_comparison":
        return obj
    if isinstance(obj, dict) and isinstance(obj.get("morphism_comparison"), dict):
        payload = obj.get("morphism_comparison")
        if payload.get("kind") == "morphism_comparison":
            return payload
    if isinstance(obj, dict) and obj.get("kind") == "morphism_comparison_legacy_result":
        legacy = obj.get("legacy_result")
        if isinstance(legacy, dict) and isinstance(legacy.get("morphism_comparison"), dict):
            return legacy["morphism_comparison"]
    raise ValueError(
        "The selected pickle does not look like a compact morphism_comparison payload. "
        "Expected a dict with kind='morphism_comparison' or a dict containing "
        "a 'morphism_comparison' key."
    )


def _load_pickle(path: str | os.PathLike[str]) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def _save_json(path: str | os.PathLike[str], data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=_json_default)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


# -----------------------------------------------------------------------------
# Lightweight lexical helpers for evidence browser
# -----------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*")
_STOPWORDS = {
    "the", "and", "for", "that", "with", "from", "this", "these", "those", "were", "was", "are", "is", "be",
    "been", "being", "have", "has", "had", "not", "but", "or", "nor", "than", "then", "there", "their", "they",
    "them", "its", "into", "onto", "over", "under", "between", "within", "without", "after", "before", "during",
    "while", "where", "when", "which", "what", "who", "whom", "whose", "can", "could", "may", "might", "must",
    "shall", "should", "will", "would", "also", "such", "only", "more", "most", "other", "some", "any", "all",
    "each", "both", "one", "two", "three", "his", "her", "him", "she", "you", "your", "our", "ours", "out",
    "about", "upon", "per", "via", "fig", "figure", "table", "vol", "no", "pp", "page", "pages", "et", "al",
}


def _tokens(texts: Sequence[str] | str, min_len: int = 3, remove_stopwords: bool = True) -> List[str]:
    if isinstance(texts, str):
        text = texts
    else:
        text = "\n".join(str(t or "") for t in texts)
    out: List[str] = []
    for tok in _TOKEN_RE.findall(text.lower()):
        tok = tok.strip("'-")
        if len(tok) < min_len:
            continue
        if remove_stopwords and tok in _STOPWORDS:
            continue
        out.append(tok)
    return out


def _counter(texts: Sequence[str] | str) -> Counter:
    return Counter(_tokens(texts))


def _top_shared(ca: Counter, cb: Counter, n: int = 30) -> List[Tuple[str, int, int]]:
    rows = [(t, ca[t], cb[t]) for t in (set(ca) & set(cb))]
    rows.sort(key=lambda r: (min(r[1], r[2]), r[1] + r[2], r[0]), reverse=True)
    return rows[:n]


def _top_distinctive(ca: Counter, cb: Counter, n: int = 30) -> List[Tuple[str, float]]:
    ta = sum(ca.values()) or 1
    tb = sum(cb.values()) or 1
    keys = set(ca) | set(cb)
    rows = []
    for k in keys:
        diff = ca.get(k, 0) / ta - cb.get(k, 0) / tb
        if diff:
            rows.append((k, diff))
    rows.sort(key=lambda x: abs(x[1]), reverse=True)
    return rows[:n]


def _lexical_metric_summary(texts_a: Sequence[str], texts_b: Sequence[str]) -> Dict[str, Any]:
    ca = _counter(texts_a)
    cb = _counter(texts_b)
    sa, sb = set(ca), set(cb)
    inter = sa & sb
    union = sa | sb
    total_a = sum(ca.values())
    total_b = sum(cb.values())
    if not union:
        return {
            "tokens_a": total_a, "tokens_b": total_b,
            "unique_a": len(sa), "unique_b": len(sb),
            "shared_unique": 0, "jaccard": 0.0, "overlap_coefficient": 0.0,
            "count_cosine": 0.0, "shared_terms": [], "distinctive_a_minus_b": [],
        }
    dot = sum(ca.get(t, 0) * cb.get(t, 0) for t in union)
    na = math.sqrt(sum(v * v for v in ca.values()))
    nb = math.sqrt(sum(v * v for v in cb.values()))
    return {
        "tokens_a": int(total_a),
        "tokens_b": int(total_b),
        "unique_a": int(len(sa)),
        "unique_b": int(len(sb)),
        "shared_unique": int(len(inter)),
        "jaccard": float(len(inter) / max(1, len(union))),
        "overlap_coefficient": float(len(inter) / max(1, min(len(sa), len(sb)))),
        "count_cosine": float(dot / max(1e-12, na * nb)) if na and nb else 0.0,
        "shared_terms": _top_shared(ca, cb),
        "distinctive_a_minus_b": _top_distinctive(ca, cb),
    }


# -----------------------------------------------------------------------------
# Data model
# -----------------------------------------------------------------------------

@dataclass
class QuerySpec:
    source_doc: str = ""
    target_doc: str = ""
    any_doc: str = ""
    match_type: str = "any"
    min_delta: str = ""
    min_pc1: str = ""
    min_quality: str = ""
    min_lex_div: str = ""
    max_lex_overlap: str = ""
    min_acuity: str = ""
    # Deprecated legacy filter kept for old saved UI state / old code paths.
    max_doc_cos: str = ""
    max_manifold_residual_doc_cos: str = ""
    max_raw_sbert_doc_cos: str = ""
    sort_by: str = "acuity_score"
    descending: bool = True
    limit: int = 1000


class MorphismComparisonModel:
    """In-memory accessor for compact/enriched morphism comparison payloads."""

    def __init__(self) -> None:
        self.path: Optional[str] = None
        self.raw_object: Any = None
        self.payload: Dict[str, Any] = {}
        self.records: np.ndarray = np.zeros((0,), dtype=[])
        self.edge_index: Dict[str, Any] = {}
        self.match_diagnostics: Dict[str, Any] = {}
        self.plot_cache: Dict[str, Any] = {}
        self.doc_ids: List[str] = []
        self.doc_code: np.ndarray = np.zeros((0,), dtype=np.int32)
        self.edge_src_label: np.ndarray = np.zeros((0,), dtype=np.int32)
        self.edge_dst_label: np.ndarray = np.zeros((0,), dtype=np.int32)
        self.document_delta_dict: Optional[Dict[Any, Any]] = None
        self.segments_by_doc: Optional[Dict[Any, Any]] = None
        self._doc_key_by_str: Dict[str, Any] = {}
        self._seg_key_by_str: Dict[str, Any] = {}
        self.last_query_indices: np.ndarray = np.zeros((0,), dtype=np.int64)
        # Row-aligned field caches.  These prevent repeated million-row
        # np.asarray/cast work while populating the table and evidence browser.
        self._record_fields: Dict[str, np.ndarray] = {}
        self._diag_fields: Dict[str, np.ndarray] = {}
        self._default_fields: Dict[Tuple[str, str, str], np.ndarray] = {}
        self._source_doc_code_cache: Optional[np.ndarray] = None
        self._target_doc_code_cache: Optional[np.ndarray] = None
        self._match_type_label_cache: Optional[np.ndarray] = None

    # ------------------------- loading -------------------------------------
    def load_comparison(self, path: str) -> None:
        obj = _load_pickle(path)
        payload = _normalize_payload(obj)
        records = np.asarray(payload.get("matches", np.zeros((0,), dtype=[])))
        if not isinstance(records, np.ndarray):
            records = np.asarray(records)
        edge_index = payload.get("edge_index") or {}
        if not isinstance(edge_index, dict):
            edge_index = {}

        self.path = path
        self.raw_object = obj
        self.payload = payload
        self.records = records
        self.edge_index = edge_index
        self.match_diagnostics = payload.get("match_diagnostics") if isinstance(payload.get("match_diagnostics"), dict) else {}
        self.plot_cache = payload.get("plot_cache") if isinstance(payload.get("plot_cache"), dict) else {}
        self.doc_ids = [str(d) for d in edge_index.get("doc_ids", [])]
        self.doc_code = np.asarray(edge_index.get("doc_code", []), dtype=np.int32)
        self.edge_src_label = np.asarray(edge_index.get("src_label", []), dtype=np.int32)
        self.edge_dst_label = np.asarray(edge_index.get("dst_label", []), dtype=np.int32)
        self.last_query_indices = np.zeros((0,), dtype=np.int64)
        self._build_field_cache()

    def _build_field_cache(self) -> None:
        """Cache row-aligned arrays as views so large files stay interactive."""
        self._record_fields = {}
        self._diag_fields = {}
        self._default_fields = {}
        self._source_doc_code_cache = None
        self._target_doc_code_cache = None
        self._match_type_label_cache = None
        if _is_structured_array(self.records):
            for name in self.records.dtype.names or ():
                try:
                    self._record_fields[str(name)] = np.asarray(self.records[name])
                except Exception:
                    pass
        if isinstance(self.match_diagnostics, dict):
            for name, value in self.match_diagnostics.items():
                try:
                    arr = np.asarray(value)
                    if arr.shape and int(arr.shape[0]) == self.n_matches:
                        self._diag_fields[str(name)] = arr
                except Exception:
                    pass

    def load_segments(self, path: str) -> None:
        obj = _load_pickle(path)
        if not isinstance(obj, dict):
            raise ValueError("segments_by_doc pickle must contain a dictionary")
        self.segments_by_doc = obj
        self._seg_key_by_str = {str(k): k for k in obj.keys()}

    def load_document_delta_dict(self, path: str) -> None:
        obj = _load_pickle(path)
        if not isinstance(obj, dict):
            raise ValueError("document_delta_dict pickle must contain a dictionary")
        self.document_delta_dict = obj
        self._doc_key_by_str = {str(k): k for k in obj.keys()}

    # ------------------------- basic properties ----------------------------
    @property
    def is_loaded(self) -> bool:
        return bool(self.payload) and isinstance(self.records, np.ndarray)

    @property
    def n_matches(self) -> int:
        try:
            return int(self.records.shape[0])
        except Exception:
            return 0

    @property
    def n_edges(self) -> int:
        try:
            return int(self.doc_code.shape[0])
        except Exception:
            return 0

    @property
    def n_docs(self) -> int:
        return len(self.doc_ids)

    def has_plot_cache(self) -> bool:
        return isinstance(self.plot_cache, dict) and "count" in self.plot_cache

    def has_match_diagnostics(self) -> bool:
        return isinstance(self.match_diagnostics, dict) and bool(self.match_diagnostics)

    # ------------------------- array access --------------------------------
    def _cached_default_array(self, name: str, default: Any = 0, dtype: Any = float) -> np.ndarray:
        try:
            dtype_key = str(np.dtype(dtype) if dtype is not object else "object")
        except Exception:
            dtype_key = str(dtype)
        key = (str(name), dtype_key, repr(default))
        arr = self._default_fields.get(key)
        if arr is None or arr.shape[0] != self.n_matches:
            arr = np.full(self.n_matches, default, dtype=dtype)
            self._default_fields[key] = arr
        return arr

    def record_field(self, name: str, default: Any = 0, dtype: Any = float) -> np.ndarray:
        arr = self._record_fields.get(str(name))
        if arr is not None and arr.shape[0] == self.n_matches:
            return arr
        return self._cached_default_array(name, default=default, dtype=dtype)

    def diag_field(self, name: str, default: Any = 0, dtype: Any = float) -> np.ndarray:
        arr = self._diag_fields.get(str(name))
        if arr is not None and arr.shape[0] == self.n_matches:
            return arr
        return self._cached_default_array(name, default=default, dtype=dtype)

    def metric_field(self, name: str, default: Any = 0, dtype: Any = float) -> np.ndarray:
        """Prefer match_diagnostics, then structured matches, without repeated whole-array conversion."""
        arr = self._diag_fields.get(str(name))
        if arr is not None and arr.shape[0] == self.n_matches:
            return arr
        arr = self._record_fields.get(str(name))
        if arr is not None and arr.shape[0] == self.n_matches:
            return arr
        return self._cached_default_array(name, default=default, dtype=dtype)

    def metric_array_if_present(self, name: str) -> Optional[np.ndarray]:
        """Return a row-aligned metric array only if the payload actually contains it."""
        arr = self._diag_fields.get(str(name))
        if arr is not None and arr.shape[0] == self.n_matches:
            return arr
        arr = self._record_fields.get(str(name))
        if arr is not None and arr.shape[0] == self.n_matches:
            return arr
        return None

    def metric_value(self, name: str, row: int, default: Any = "") -> Any:
        arr = self.metric_array_if_present(name)
        if arr is None:
            return default
        try:
            return arr[int(row)]
        except Exception:
            return default

    def metric_array_first_present(self, names: Sequence[str]) -> Tuple[Optional[np.ndarray], str]:
        """Return the first row-aligned metric array present from a list of aliases."""
        for name in names:
            arr = self.metric_array_if_present(str(name))
            if arr is not None:
                return arr, str(name)
        return None, ""

    def metric_value_first_present(self, names: Sequence[str], row: int, default: Any = "") -> Any:
        arr, _name = self.metric_array_first_present(names)
        if arr is None:
            return default
        try:
            return arr[int(row)]
        except Exception:
            return default

    def document_cosine_array(self, kind: str = "manifold_residual") -> Tuple[Optional[np.ndarray], str]:
        """
        Return a document-cosine metric array with backwards-compatible aliases.

        New comparison files distinguish:
          • manifold_residual_doc_cosine: cosine between document-centered manifold baselines
          • raw_sbert_doc_cosine: cosine between raw mean SBERT document anchors

        Older files used doc_embedding_cosine ambiguously.  The analysis platform
        treats that legacy field as the residual/manifold value unless the new
        explicit fields are present.
        """
        k = str(kind or "manifold_residual").strip().lower()
        if k in {"raw", "raw_sbert", "sbert"}:
            names = [
                "raw_sbert_doc_cosine",
                "raw_doc_embedding_cosine",
                "doc_embedding_cosine_raw_sbert",
            ]
        else:
            names = [
                "manifold_residual_doc_cosine",
                "residual_doc_cosine",
                "doc_embedding_cosine",  # legacy fallback
            ]
        return self.metric_array_first_present(names)

    def document_cosine_value(self, kind: str, row: int, default: Any = "") -> Any:
        arr, _name = self.document_cosine_array(kind)
        if arr is None:
            return default
        try:
            return arr[int(row)]
        except Exception:
            return default

    def _apply_max_any(self, mask: np.ndarray, fields: Sequence[str], max_text: str) -> np.ndarray:
        if not str(max_text or "").strip():
            return mask
        v = _safe_float(max_text, default=float("nan"))
        if not math.isfinite(v):
            return mask
        arr, _name = self.metric_array_first_present(fields)
        if arr is None:
            return mask
        try:
            return mask & (np.asarray(arr, dtype=float) <= v)
        except Exception:
            return mask

    def match_type_array(self) -> np.ndarray:
        if self._match_type_label_cache is not None and self._match_type_label_cache.shape[0] == self.n_matches:
            return self._match_type_label_cache
        raw = self.record_field("match_type", default=255, dtype=object)
        if raw.size == 0:
            self._match_type_label_cache = raw.astype(object)
        else:
            self._match_type_label_cache = np.asarray([_match_type_label(x) for x in raw], dtype=object)
        return self._match_type_label_cache

    def source_edges(self) -> np.ndarray:
        return self.record_field("src_edge", default=-1, dtype=np.int64)

    def target_edges(self) -> np.ndarray:
        return self.record_field("tgt_edge", default=-1, dtype=np.int64)

    def valid_edge_mask(self) -> np.ndarray:
        se = self.source_edges()
        te = self.target_edges()
        return (se >= 0) & (te >= 0) & (se < self.n_edges) & (te < self.n_edges)

    def edge_doc_code(self, edge_ids: np.ndarray) -> np.ndarray:
        edge_ids = np.asarray(edge_ids, dtype=np.int64)
        out = np.full(edge_ids.shape, -1, dtype=np.int32)
        valid = (edge_ids >= 0) & (edge_ids < self.n_edges) & (self.doc_code.size == self.n_edges)
        if np.any(valid):
            out[valid] = self.doc_code[edge_ids[valid]]
        return out

    def source_doc_codes_for_matches(self) -> np.ndarray:
        if self._source_doc_code_cache is None or self._source_doc_code_cache.shape[0] != self.n_matches:
            self._source_doc_code_cache = self.edge_doc_code(self.source_edges())
        return self._source_doc_code_cache

    def target_doc_codes_for_matches(self) -> np.ndarray:
        if self._target_doc_code_cache is None or self._target_doc_code_cache.shape[0] != self.n_matches:
            self._target_doc_code_cache = self.edge_doc_code(self.target_edges())
        return self._target_doc_code_cache

    def doc_id_from_code(self, code: int) -> str:
        try:
            if 0 <= int(code) < len(self.doc_ids):
                return self.doc_ids[int(code)]
        except Exception:
            pass
        return ""

    def edge_doc_id(self, edge_id: int) -> str:
        try:
            if 0 <= int(edge_id) < self.n_edges:
                return self.doc_id_from_code(int(self.doc_code[int(edge_id)]))
        except Exception:
            pass
        return ""

    def edge_labels(self, edge_id: int) -> Tuple[Any, Any]:
        try:
            edge_id = int(edge_id)
            if 0 <= edge_id < self.n_edges:
                src = self.edge_src_label[edge_id] if edge_id < self.edge_src_label.shape[0] else ""
                dst = self.edge_dst_label[edge_id] if edge_id < self.edge_dst_label.shape[0] else ""
                return int(src), int(dst)
        except Exception:
            pass
        return "", ""

    # ------------------------- companion geometry ---------------------------
    def document_cdm(self, doc_id: str) -> Optional[Any]:
        """Return the CDM tuple for a document id from companion document_delta_dict."""
        if not isinstance(self.document_delta_dict, dict):
            return None
        key_doc, _ = self._resolve_doc_key(doc_id)
        if key_doc in self.document_delta_dict:
            return self.document_delta_dict[key_doc]
        d = str(doc_id)
        for key in (d, doc_id):
            if key in self.document_delta_dict:
                return self.document_delta_dict[key]
        return None

    def _cluster_order(self, data: Any) -> List[Any]:
        try:
            order = data[1]
            return order.tolist() if hasattr(order, "tolist") else list(order)
        except Exception:
            return []

    def _cluster_index_for_label(self, data: Any, cluster_label: Any) -> Optional[int]:
        order = self._cluster_order(data)
        for i, lab in enumerate(order):
            if _label_equal(lab, cluster_label):
                return int(i)
        return None

    def _cluster_vector_from_payload(self, payload: Any, data: Any, cluster_label: Any) -> Optional[np.ndarray]:
        """Read a cluster-level vector from a dict or an order-aligned ndarray/list."""
        if payload is None:
            return None
        try:
            if isinstance(payload, dict):
                candidates = [cluster_label, str(cluster_label)]
                try:
                    candidates.append(int(cluster_label))
                except Exception:
                    pass
                for key in candidates:
                    if key in payload:
                        v = np.asarray(payload[key], dtype=float).reshape(-1)
                        return v if v.size else None
            idx = self._cluster_index_for_label(data, cluster_label)
            if idx is None:
                return None
            arr = np.asarray(payload, dtype=float)
            if arr.ndim == 1:
                return arr.reshape(-1)
            if 0 <= idx < arr.shape[0]:
                return np.asarray(arr[idx], dtype=float).reshape(-1)
        except Exception:
            return None
        return None

    def _cluster_quality_value(self, data: Any, cluster_label: Any) -> float:
        try:
            q_payload = data[6] if isinstance(data, (tuple, list)) and len(data) >= 7 else None
            if isinstance(q_payload, dict):
                candidates = [cluster_label, str(cluster_label)]
                try:
                    candidates.append(int(cluster_label))
                except Exception:
                    pass
                for key in candidates:
                    if key in q_payload:
                        rec = q_payload[key]
                        if isinstance(rec, dict):
                            return max(0.0, min(1.0, float(rec.get("quality", 1.0))))
                        return max(0.0, min(1.0, float(rec)))
            elif q_payload is not None:
                idx = self._cluster_index_for_label(data, cluster_label)
                if idx is not None:
                    arr = list(q_payload)
                    if 0 <= idx < len(arr):
                        rec = arr[idx]
                        if isinstance(rec, dict):
                            return max(0.0, min(1.0, float(rec.get("quality", 1.0))))
                        return max(0.0, min(1.0, float(rec)))
        except Exception:
            pass
        return 1.0

    def cluster_segment_indices(self, doc_id: str, cluster_label: Any) -> List[int]:
        """Return segment indices for one doc/cluster using companion labels."""
        data = self.document_cdm(doc_id)
        if data is None:
            return []
        try:
            labels = data[2]
            labels = labels.tolist() if hasattr(labels, "tolist") else list(labels)
            target = cluster_label
            return [int(i) for i, lab in enumerate(labels) if _label_equal(lab, target)]
        except Exception:
            return []

    def cluster_geometry(self, doc_id: str, cluster_label: Any) -> Dict[str, Any]:
        """Return centroid/PC1/Q/segment metadata for one cluster endpoint."""
        data = self.document_cdm(doc_id)
        if data is None:
            raise KeyError(f"document {doc_id!r} not found in document_delta_dict")
        if not isinstance(data, (tuple, list)) or len(data) < 6:
            raise ValueError(f"document {doc_id!r} CDM tuple must have at least 6 elements")
        idx = self._cluster_index_for_label(data, cluster_label)
        if idx is None:
            raise KeyError(f"cluster label {cluster_label!r} not found for document {doc_id!r}")
        centroid = self._cluster_vector_from_payload(data[4], data, cluster_label)
        pc1 = self._cluster_vector_from_payload(data[5], data, cluster_label)
        if centroid is None:
            raise ValueError(f"cluster centroid unavailable for {doc_id!r} cluster {cluster_label!r}")
        if pc1 is None or pc1.size != centroid.size:
            pc1 = np.zeros_like(centroid, dtype=float)
        texts = self.cluster_texts(doc_id, cluster_label)
        return {
            "doc_id": str(doc_id),
            "cluster_label": cluster_label,
            "cluster_label_display": _label_display(cluster_label),
            "cluster_index": int(idx),
            "centroid": np.asarray(centroid, dtype=float).reshape(-1),
            "pc1": _unit_vector(pc1),
            "quality": float(self._cluster_quality_value(data, cluster_label)),
            "segment_indices": self.cluster_segment_indices(doc_id, cluster_label),
            "segment_texts": texts,
            "segment_count": int(len(texts)),
        }

    def edge_pair_geometry(self, match_row: int) -> Dict[str, Any]:
        """Return source/target morphism endpoint geometry for a compact match row."""
        row = self.row_dict(match_row)
        source_src = self.cluster_geometry(row["src_doc"], row["src_from"])
        source_dst = self.cluster_geometry(row["src_doc"], row["src_to"])
        target_src = self.cluster_geometry(row["tgt_doc"], row["tgt_from"])
        target_dst = self.cluster_geometry(row["tgt_doc"], row["tgt_to"])

        def _edge(name: str, doc: str, src: Dict[str, Any], dst: Dict[str, Any], edge_id: int) -> Dict[str, Any]:
            delta = np.asarray(dst["centroid"], dtype=float) - np.asarray(src["centroid"], dtype=float)
            # Prefer stored Delta when available and shape-compatible.
            data = self.document_cdm(doc)
            try:
                if data is not None:
                    D = np.asarray(data[0], dtype=float)
                    i = int(src.get("cluster_index", -1))
                    j = int(dst.get("cluster_index", -1))
                    if D.ndim == 3 and 0 <= i < D.shape[0] and 0 <= j < D.shape[1]:
                        candidate = np.asarray(D[i, j], dtype=float).reshape(-1)
                        if candidate.shape == delta.shape:
                            delta = candidate
            except Exception:
                pass
            return {
                "name": name,
                "doc_id": str(doc),
                "edge_id": int(edge_id),
                "src": src,
                "dst": dst,
                "delta": np.asarray(delta, dtype=float).reshape(-1),
                "delta_norm": float(np.linalg.norm(delta)),
            }

        return {
            "match_row": int(match_row),
            "row": row,
            "source_edge": _edge("source morphism", row["src_doc"], source_src, source_dst, row.get("src_edge", -1)),
            "target_edge": _edge("target morphism", row["tgt_doc"], target_src, target_dst, row.get("tgt_edge", -1)),
        }

    # ------------------------- schema / summary ----------------------------
    def schema_lines(self) -> List[str]:
        if not self.is_loaded:
            return ["No morphism comparison file loaded."]
        p = self.payload
        lines = []
        lines.append(f"File: {self.path or ''}")
        lines.append(f"kind: {p.get('kind', '')}")
        lines.append(f"version: {p.get('version', '')}")
        lines.append(f"documents: {self.n_docs:,}")
        lines.append(f"directed morphism edges: {self.n_edges:,}")
        lines.append(f"retained matches: {self.n_matches:,}")
        lines.append(f"matches dtype: {self.records.dtype}")
        lines.append(f"matches memory: {_bytes_human(_array_nbytes(self.records))}")
        lines.append(f"edge_index keys: {', '.join(sorted(map(str, self.edge_index.keys())))}")
        lines.append(f"match_diagnostics: {'yes' if self.has_match_diagnostics() else 'no'}")
        doc_emb = p.get("document_embeddings") if isinstance(p.get("document_embeddings"), dict) else {}
        if doc_emb:
            lines.append(f"document_embeddings: yes; keys={', '.join(sorted(map(str, doc_emb.keys())))}")
            for name in ("manifold_residual", "raw_sbert"):
                table = doc_emb.get(name)
                if isinstance(table, dict):
                    vecs = table.get("vectors")
                    try:
                        arr = np.asarray(vecs)
                        lines.append(f"  {name}: vectors shape={tuple(arr.shape)} dtype={arr.dtype}")
                    except Exception:
                        lines.append(f"  {name}: table keys={', '.join(sorted(map(str, table.keys())))}")
        else:
            lines.append("document_embeddings: no explicit table")
        lines.append(f"plot_cache: {'yes' if self.has_plot_cache() else 'no'}")
        if self.has_plot_cache():
            c = np.asarray(self.plot_cache.get("count"))
            lines.append(f"plot_cache count grid shape: {tuple(c.shape)}; occupied cells: {int(np.count_nonzero(c)):,}")
            lines.append(f"plot_cache step: {self.plot_cache.get('step', '')}")
            tc = np.asarray(self.plot_cache.get("top_candidates", []))
            lines.append(f"top candidates cached: {int(tc.shape[0]) if isinstance(tc, np.ndarray) else 0:,}")
        summary = p.get("summary") if isinstance(p.get("summary"), dict) else {}
        if summary:
            lines.append("\nSummary:")
            for k in sorted(summary.keys()):
                lines.append(f"  {k}: {summary[k]}")
        params = p.get("params") if isinstance(p.get("params"), dict) else {}
        if params:
            lines.append("\nParameters:")
            for k in sorted(params.keys()):
                lines.append(f"  {k}: {params[k]}")
        diag = p.get("diagnostics") if isinstance(p.get("diagnostics"), dict) else {}
        if diag:
            lines.append("\nDiagnostics metadata:")
            for k in sorted(diag.keys()):
                v = diag[k]
                if isinstance(v, np.ndarray):
                    v = f"ndarray shape={v.shape} dtype={v.dtype}"
                lines.append(f"  {k}: {v}")
        return lines

    def schema_tree_rows(self, max_depth: int = 4, max_items_per_container: int = 80) -> List[Tuple[str, str, str]]:
        rows: List[Tuple[str, str, str]] = []

        def walk(name: str, obj: Any, depth: int) -> None:
            indent = "  " * depth
            if isinstance(obj, dict):
                rows.append((f"{indent}{name}", "dict", f"{len(obj)} keys"))
                if depth < max_depth:
                    keys = sorted(obj.keys(), key=lambda x: str(x))
                    for k in keys[:max_items_per_container]:
                        walk(str(k), obj[k], depth + 1)
                    if len(keys) > max_items_per_container:
                        rows.append((f"{indent}  ...", "truncated", f"{len(keys) - max_items_per_container} additional keys not shown"))
            elif isinstance(obj, np.ndarray):
                extra = f"shape={obj.shape}; dtype={obj.dtype}; memory={_bytes_human(int(obj.nbytes))}"
                if obj.dtype.names:
                    extra += f"; fields={', '.join(obj.dtype.names)}"
                rows.append((f"{indent}{name}", "ndarray", extra))
            elif isinstance(obj, (list, tuple)):
                rows.append((f"{indent}{name}", type(obj).__name__, f"len={len(obj)}"))
                if depth < max_depth and len(obj):
                    for i, item in enumerate(obj[:min(len(obj), 10)]):
                        walk(f"[{i}]", item, depth + 1)
                    if len(obj) > 10:
                        rows.append((f"{indent}  ...", "truncated", f"{len(obj) - 10} additional items not shown"))
            else:
                text = repr(obj)
                if len(text) > 160:
                    text = text[:157] + "..."
                rows.append((f"{indent}{name}", type(obj).__name__, text))

        if self.payload:
            walk("morphism_comparison", self.payload, 0)
        return rows

    # ------------------------- querying ------------------------------------
    def _doc_filter_codes(self, query: str) -> Optional[np.ndarray]:
        q = str(query or "").strip()
        if not q:
            return None
        q_lower = q.lower()
        exact = {d: i for i, d in enumerate(self.doc_ids)}
        if q in exact:
            return np.asarray([exact[q]], dtype=np.int32)
        codes = [i for i, d in enumerate(self.doc_ids) if q_lower in d.lower()]
        if not codes:
            return np.asarray([], dtype=np.int32)
        return np.asarray(codes, dtype=np.int32)

    def _apply_min(self, mask: np.ndarray, field: str, min_text: str) -> np.ndarray:
        if not str(min_text or "").strip():
            return mask
        v = _safe_float(min_text, default=float("nan"))
        if not math.isfinite(v):
            return mask
        arr = self.metric_field(field, default=0.0, dtype=float)
        return mask & (arr >= v)

    def _apply_max(self, mask: np.ndarray, field: str, max_text: str) -> np.ndarray:
        if not str(max_text or "").strip():
            return mask
        v = _safe_float(max_text, default=float("nan"))
        if not math.isfinite(v):
            return mask
        arr = self.metric_field(field, default=0.0, dtype=float)
        return mask & (arr <= v)

    def query_indices(self, spec: QuerySpec) -> np.ndarray:
        if not self.is_loaded:
            return np.zeros((0,), dtype=np.int64)
        n = self.n_matches
        mask = np.ones(n, dtype=bool)

        se = self.source_edges()
        te = self.target_edges()
        valid = self.valid_edge_mask()
        src_codes = self.source_doc_codes_for_matches()
        tgt_codes = self.target_doc_codes_for_matches()
        mask &= valid

        for attr, codes_source, field_name in [
            (spec.source_doc, src_codes, "source_doc"),
            (spec.target_doc, tgt_codes, "target_doc"),
        ]:
            codes = self._doc_filter_codes(attr)
            if codes is not None:
                if codes.size == 0:
                    return np.zeros((0,), dtype=np.int64)
                mask &= np.isin(codes_source, codes)

        any_codes = self._doc_filter_codes(spec.any_doc)
        if any_codes is not None:
            if any_codes.size == 0:
                return np.zeros((0,), dtype=np.int64)
            mask &= (np.isin(src_codes, any_codes) | np.isin(tgt_codes, any_codes))

        mt = str(spec.match_type or "any").strip().lower()
        if mt not in {"", "any", "all"}:
            labels = self.match_type_array()
            mt_norm = "pc1_only" if mt in {"pc1", "pc1only", "pc1-only"} else mt
            mask &= (labels == mt_norm)

        mask = self._apply_min(mask, "delta_cos", spec.min_delta)
        mask = self._apply_min(mask, "pc1_axis_value", spec.min_pc1)
        mask = self._apply_min(mask, "semantic_quality", spec.min_quality)
        mask = self._apply_min(mask, "lexical_divergence", spec.min_lex_div)
        mask = self._apply_max(mask, "lexical_overlap_coefficient", spec.max_lex_overlap)
        mask = self._apply_min(mask, "acuity_score", spec.min_acuity)
        # Explicit document-cosine filters.  The old max_doc_cos filter is
        # preserved as a residual/manifold fallback for older saved workflows.
        mask = self._apply_max_any(mask, ["manifold_residual_doc_cosine", "residual_doc_cosine", "doc_embedding_cosine"], spec.max_manifold_residual_doc_cos or spec.max_doc_cos)
        mask = self._apply_max_any(mask, ["raw_sbert_doc_cosine", "raw_doc_embedding_cosine", "doc_embedding_cosine_raw_sbert"], spec.max_raw_sbert_doc_cos)

        idx = np.where(mask)[0].astype(np.int64)
        if idx.size == 0:
            self.last_query_indices = idx
            return idx

        limit = max(1, int(spec.limit or 1000))
        sort_by = str(spec.sort_by or "").strip()
        descending = bool(spec.descending)
        if sort_by:
            vals = self.metric_field(sort_by, default=0.0, dtype=float)
            if vals.shape[0] == n:
                v = vals[idx]
                # Missing/NaN values sort last.
                v = np.nan_to_num(v, nan=-np.inf if descending else np.inf)
                if idx.size > limit:
                    if descending:
                        part = np.argpartition(-v, limit - 1)[:limit]
                        idx = idx[part]
                        v = v[part]
                        order = np.argsort(-v)
                    else:
                        part = np.argpartition(v, limit - 1)[:limit]
                        idx = idx[part]
                        v = v[part]
                        order = np.argsort(v)
                    idx = idx[order]
                else:
                    order = np.argsort(-v if descending else v)
                    idx = idx[order]
            else:
                idx = idx[:limit]
        else:
            idx = idx[:limit]

        if idx.size > limit:
            idx = idx[:limit]
        self.last_query_indices = idx
        return idx

    # ------------------------- row construction ----------------------------
    def _record_scalar(self, name: str, row: int, default: Any = "") -> Any:
        """Fast scalar read from the structured matches array.

        This deliberately avoids calling record_field()/np.asarray() for every
        table row.  On million-row files, repeated whole-column conversions are
        much slower than direct structured-array scalar indexing.
        """
        try:
            if (
                isinstance(self.records, np.ndarray)
                and self.records.dtype.names
                and name in self.records.dtype.names
                and 0 <= int(row) < self.records.shape[0]
            ):
                return self.records[name][int(row)]
        except Exception:
            pass
        return default

    def _diagnostic_scalar(self, name: str, row: int, default: Any = "") -> Any:
        """Fast scalar read from match_diagnostics without rebuilding arrays."""
        try:
            if isinstance(self.match_diagnostics, dict) and name in self.match_diagnostics:
                arr = self.match_diagnostics.get(name)
                if hasattr(arr, "shape") and getattr(arr, "shape")[0] == self.n_matches:
                    return arr[int(row)]
                if hasattr(arr, "__len__") and len(arr) == self.n_matches:  # type: ignore[arg-type]
                    return arr[int(row)]
        except Exception:
            pass
        return default

    def metric_scalar(self, name: str, row: int, default: Any = "") -> Any:
        """Prefer a diagnostic scalar, then a structured-match scalar."""
        val = self._diagnostic_scalar(name, row, default=None)
        if val is not None:
            return val
        return self._record_scalar(name, row, default=default)

    def row_dict(self, match_row: int, rank: Optional[int] = None) -> Dict[str, Any]:
        r = int(match_row)
        if r < 0 or r >= self.n_matches:
            return {"rank": rank if rank is not None else "", "match_row": r}

        se = _safe_int(self.metric_value("src_edge", r, -1), -1)
        te = _safe_int(self.metric_value("tgt_edge", r, -1), -1)
        s_doc = self.edge_doc_id(se)
        t_doc = self.edge_doc_id(te)
        s_from, s_to = self.edge_labels(se)
        t_from, t_to = self.edge_labels(te)
        mt = _match_type_label(self.metric_value("match_type", r, ""))

        out: Dict[str, Any] = {
            "rank": rank if rank is not None else "",
            "match_row": r,
            "match_type": mt,
            "src_doc": s_doc,
            "src_edge": se,
            "src_from": s_from,
            "src_to": s_to,
            "tgt_doc": t_doc,
            "tgt_edge": te,
            "tgt_from": t_from,
            "tgt_to": t_to,
        }
        for field in [
            "delta_cos", "src_pc1", "dst_pc1", "pc1_axis_value", "semantic_quality",
            "semantic_quality_min", "lexical_overlap_coefficient", "lexical_divergence",
            "alignment_core", "acuity_score", "acuity_score_count_cosine",
            "lexical_dst_count_cosine", "manifold_residual_doc_cosine",
            "raw_sbert_doc_cosine", "doc_embedding_cosine", "joint", "joint_min_4d",
            "detected_delta_thr", "detected_pc1_thr", "detected_quality_thr",
        ]:
            arr = self.metric_array_if_present(field)
            if arr is not None:
                try:
                    fv = float(arr[r])
                    out[field] = fv if math.isfinite(fv) else ""
                except Exception:
                    out[field] = ""
        # Backward/forward compatible aliases for document cosine naming.
        if "manifold_residual_doc_cosine" not in out:
            val = self.document_cosine_value("manifold_residual", r, default="")
            if val != "":
                try:
                    out["manifold_residual_doc_cosine"] = float(val)
                except Exception:
                    out["manifold_residual_doc_cosine"] = val
        if "raw_sbert_doc_cosine" not in out:
            val = self.document_cosine_value("raw_sbert", r, default="")
            if val != "":
                try:
                    out["raw_sbert_doc_cosine"] = float(val)
                except Exception:
                    out["raw_sbert_doc_cosine"] = val
        if "doc_embedding_cosine" not in out and "manifold_residual_doc_cosine" in out:
            out["doc_embedding_cosine"] = out.get("manifold_residual_doc_cosine")

        for field in [
            "manifold_residual_doc_available", "raw_sbert_doc_available",
            "doc_embedding_available", "lexical_available",
        ]:
            arr = self.metric_array_if_present(field)
            if arr is not None:
                try:
                    out[field] = bool(arr[r])
                except Exception:
                    pass
        return out

    def rows_for_indices(self, indices: Sequence[int]) -> List[Dict[str, Any]]:
        return [self.row_dict(int(i), rank=rank) for rank, i in enumerate(indices, 1)]

    def table_columns(self) -> List[str]:
        return [
            "rank", "match_row", "match_type",
            "src_doc", "src_from", "src_to",
            "tgt_doc", "tgt_from", "tgt_to",
            "delta_cos", "pc1_axis_value", "semantic_quality",
            "lexical_overlap_coefficient", "lexical_divergence",
            "alignment_core", "acuity_score",
            "manifold_residual_doc_cosine", "raw_sbert_doc_cosine",
            "src_edge", "tgt_edge",
        ]

    def format_table_value(self, key: str, value: Any) -> str:
        if key in {
            "delta_cos", "pc1_axis_value", "semantic_quality", "lexical_overlap_coefficient",
            "lexical_divergence", "alignment_core", "acuity_score",
            "manifold_residual_doc_cosine", "raw_sbert_doc_cosine", "doc_embedding_cosine",
        }:
            return _fmt_float(value, 4)
        return str(value)

    # ------------------------- candidate access ----------------------------
    def top_candidate_rows(self, limit: int = 5000) -> np.ndarray:
        """Return match-row indices ordered by cached top candidates or acuity."""
        limit = max(1, int(limit or 5000))
        tc = None
        if isinstance(self.plot_cache, dict) and "top_candidates" in self.plot_cache:
            try:
                arr = np.asarray(self.plot_cache.get("top_candidates"))
                if arr.size and arr.dtype.names and "match_row" in arr.dtype.names:
                    tc = np.asarray(arr["match_row"], dtype=np.int64)
                    tc = tc[(tc >= 0) & (tc < self.n_matches)]
            except Exception:
                tc = None
        if tc is not None and tc.size:
            return tc[:limit]
        # Fallback from diagnostics.
        acu = self.metric_field("acuity_score", default=0.0, dtype=float)
        if acu.shape[0] != self.n_matches:
            acu = self.metric_field("joint", default=0.0, dtype=float)
        if acu.size == 0:
            return np.zeros((0,), dtype=np.int64)
        k = min(limit, acu.shape[0])
        order = np.argpartition(-np.nan_to_num(acu, nan=-np.inf), k - 1)[:k] if acu.shape[0] > k else np.arange(acu.shape[0])
        order = order[np.argsort(-np.nan_to_num(acu[order], nan=-np.inf))]
        return order.astype(np.int64)

    # ------------------------- companion text ------------------------------
    def _resolve_doc_key(self, doc_id: str) -> Tuple[Any, Optional[Any]]:
        d = str(doc_id)
        key_doc = self._doc_key_by_str.get(d, d)
        key_seg = self._seg_key_by_str.get(d, d)
        return key_doc, key_seg

    def cluster_texts(self, doc_id: str, cluster_label: Any) -> List[str]:
        """Return segment texts for one doc/cluster if companion artifacts are loaded."""
        if not isinstance(self.document_delta_dict, dict) or not isinstance(self.segments_by_doc, dict):
            return []
        key_doc, key_seg = self._resolve_doc_key(doc_id)
        if key_doc not in self.document_delta_dict:
            return []
        if key_seg not in self.segments_by_doc:
            return []
        try:
            data = self.document_delta_dict[key_doc]
            labels = data[2]
            labels = labels.tolist() if hasattr(labels, "tolist") else list(labels)
            segs = list(self.segments_by_doc[key_seg])
            target = int(cluster_label)
            return [str(segs[i]) for i, lab in enumerate(labels) if i < len(segs) and int(lab) == target]
        except Exception:
            return []

    def evidence_packet(self, match_row: int, notes: str = "", include_text: bool = True) -> Dict[str, Any]:
        row = self.row_dict(match_row)
        source_src_texts = self.cluster_texts(row["src_doc"], row["src_from"]) if include_text else []
        source_dst_texts = self.cluster_texts(row["src_doc"], row["src_to"]) if include_text else []
        target_src_texts = self.cluster_texts(row["tgt_doc"], row["tgt_from"]) if include_text else []
        target_dst_texts = self.cluster_texts(row["tgt_doc"], row["tgt_to"]) if include_text else []

        lexical = {}
        if source_dst_texts and target_dst_texts:
            lexical["destination_cluster_comparison"] = _lexical_metric_summary(source_dst_texts, target_dst_texts)
        if source_src_texts and target_src_texts:
            lexical["source_cluster_comparison"] = _lexical_metric_summary(source_src_texts, target_src_texts)
        if (source_src_texts or source_dst_texts) and (target_src_texts or target_dst_texts):
            lexical["combined_edge_comparison"] = _lexical_metric_summary(
                source_src_texts + source_dst_texts,
                target_src_texts + target_dst_texts,
            )

        packet = {
            "platform": "standalone_morphism_analysis_platform",
            "comparison_file": self.path,
            "match": row,
            "notes": notes,
            "lexical_from_companion_text": lexical,
            "cluster_text_available": bool(source_src_texts or source_dst_texts or target_src_texts or target_dst_texts),
            "cluster_texts": {
                "source_document_source_cluster": source_src_texts,
                "source_document_destination_cluster": source_dst_texts,
                "target_document_source_cluster": target_src_texts,
                "target_document_destination_cluster": target_dst_texts,
            } if include_text else {},
        }
        return packet

    def packet_markdown(self, packet: Dict[str, Any], max_chars_per_cluster: int = 4000) -> str:
        m = packet.get("match", {})
        lines = []
        lines.append("# Morphism Match Evidence Packet")
        lines.append("")
        lines.append(f"Comparison file: `{packet.get('comparison_file', '')}`")
        lines.append("")
        lines.append("## Match summary")
        lines.append("")
        lines.append(f"- Match row: `{m.get('match_row', '')}`")
        lines.append(f"- Match type: `{m.get('match_type', '')}`")
        lines.append(f"- Source edge: `{m.get('src_doc', '')}` C{m.get('src_from', '')} → C{m.get('src_to', '')}")
        lines.append(f"- Target edge: `{m.get('tgt_doc', '')}` C{m.get('tgt_from', '')} → C{m.get('tgt_to', '')}")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|---|---:|")
        for key in [
            "delta_cos", "src_pc1", "dst_pc1", "pc1_axis_value", "semantic_quality",
            "semantic_quality_min", "lexical_overlap_coefficient", "lexical_divergence",
            "alignment_core", "acuity_score", "acuity_score_count_cosine",
            "manifold_residual_doc_cosine", "raw_sbert_doc_cosine", "doc_embedding_cosine",
        ]:
            if key in m:
                lines.append(f"| {key} | {_fmt_float(m.get(key), 5)} |")
        lines.append("")
        notes = str(packet.get("notes", "")).strip()
        if notes:
            lines.append("## Analyst notes")
            lines.append("")
            lines.append(notes)
            lines.append("")
        lex = packet.get("lexical_from_companion_text", {}) or {}
        if lex:
            lines.append("## Companion-text lexical checks")
            for name, metrics in lex.items():
                lines.append("")
                lines.append(f"### {name.replace('_', ' ').title()}")
                lines.append("")
                for key in ["tokens_a", "tokens_b", "unique_a", "unique_b", "shared_unique", "jaccard", "overlap_coefficient", "count_cosine"]:
                    if key in metrics:
                        val = metrics[key]
                        lines.append(f"- {key}: {_fmt_float(val, 5) if isinstance(val, float) else val}")
                shared = metrics.get("shared_terms") or []
                if shared:
                    lines.append("- top shared terms: " + ", ".join(f"{t} ({a}/{b})" for t, a, b in shared[:20]))
        texts = packet.get("cluster_texts", {}) or {}
        if texts:
            lines.append("")
            lines.append("## Cluster text excerpts")
            for name, text_list in texts.items():
                lines.append("")
                lines.append(f"### {name.replace('_', ' ').title()} ({len(text_list)} segments)")
                blob = "\n\n".join(str(t) for t in text_list)
                if len(blob) > max_chars_per_cluster:
                    blob = blob[:max_chars_per_cluster] + "\n\n[excerpt truncated]"
                lines.append("")
                lines.append(blob if blob.strip() else "(no companion text loaded)")
        return "\n".join(lines)


# -----------------------------------------------------------------------------
# UI utility frames
# -----------------------------------------------------------------------------

class ScrollText(ttk.Frame):
    def __init__(self, master: tk.Misc, **kwargs: Any) -> None:
        super().__init__(master, **kwargs)
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.text = tk.Text(self, wrap="word", undo=False)
        y = ttk.Scrollbar(self, orient="vertical", command=self.text.yview)
        self.text.configure(yscrollcommand=y.set)
        self.text.grid(row=0, column=0, sticky="nsew")
        y.grid(row=0, column=1, sticky="ns")

    def set_text(self, content: str) -> None:
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.insert("1.0", content)
        self.text.configure(state="normal")

    def get_text(self) -> str:
        return self.text.get("1.0", "end-1c")


class SchemaInspector(ttk.Frame):
    def __init__(self, master: tk.Misc, model: MorphismComparisonModel) -> None:
        super().__init__(master)
        self.model = model
        self.rowconfigure(1, weight=1)
        self.columnconfigure(0, weight=1)

        toolbar = ttk.Frame(self)
        toolbar.grid(row=0, column=0, sticky="ew", padx=8, pady=6)
        ttk.Button(toolbar, text="Refresh schema", command=self.refresh).pack(side="left")

        paned = ttk.Panedwindow(self, orient="vertical")
        paned.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))

        self.summary = ScrollText(paned)
        paned.add(self.summary, weight=1)

        tree_frame = ttk.Frame(paned)
        tree_frame.rowconfigure(0, weight=1)
        tree_frame.columnconfigure(0, weight=1)
        self.tree = ttk.Treeview(tree_frame, columns=("type", "details"), show="tree headings")
        self.tree.heading("#0", text="Key")
        self.tree.heading("type", text="Type")
        self.tree.heading("details", text="Details")
        self.tree.column("#0", width=340, anchor="w")
        self.tree.column("type", width=120, anchor="w")
        self.tree.column("details", width=760, anchor="w")
        y = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=y.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        y.grid(row=0, column=1, sticky="ns")
        paned.add(tree_frame, weight=2)

    def refresh(self) -> None:
        self.summary.set_text("\n".join(self.model.schema_lines()))
        for item in self.tree.get_children():
            self.tree.delete(item)
        for key, typ, details in self.model.schema_tree_rows():
            self.tree.insert("", "end", text=key, values=(typ, details))


class MatchQueryWorkbench(ttk.Frame):
    def __init__(
        self,
        master: tk.Misc,
        model: MorphismComparisonModel,
        on_select_match: Any = None,
        on_open_evidence: Any = None,
        on_open_3d: Any = None,
    ) -> None:
        super().__init__(master)
        self.model = model
        self.on_select_match = on_select_match
        self.on_open_evidence = on_open_evidence
        self.on_open_3d = on_open_3d
        self.selected_match_row: Optional[int] = None
        self.rowconfigure(2, weight=1)
        self.columnconfigure(0, weight=1)
        self.vars: Dict[str, tk.StringVar] = {}
        self.desc_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="Load a morphism_comparison.pkl file to begin.")
        self._build_controls()
        self._build_table()

    def _var(self, name: str, default: str = "") -> tk.StringVar:
        v = tk.StringVar(value=default)
        self.vars[name] = v
        return v

    def _build_controls(self) -> None:
        outer = ttk.LabelFrame(self, text="Match filters")
        outer.grid(row=0, column=0, sticky="ew", padx=8, pady=6)
        for c in range(12):
            outer.columnconfigure(c, weight=1)

        controls = [
            ("Any doc", "any_doc", ""),
            ("Source doc", "source_doc", ""),
            ("Target doc", "target_doc", ""),
            ("Min Δ", "min_delta", ""),
            ("Min PC1", "min_pc1", ""),
            ("Min Q", "min_quality", ""),
            ("Min lex div", "min_lex_div", ""),
            ("Max lex ov", "max_lex_overlap", ""),
            ("Min acuity", "min_acuity", ""),
            ("Max residual cos", "max_manifold_residual_doc_cos", ""),
            ("Max raw cos", "max_raw_sbert_doc_cos", ""),
        ]
        for i, (label, name, default) in enumerate(controls):
            r = 0 if i < 6 else 2
            c = (i % 6) * 2
            ttk.Label(outer, text=label).grid(row=r, column=c, sticky="w", padx=(6, 2), pady=3)
            ttk.Entry(outer, textvariable=self._var(name, default), width=16).grid(row=r, column=c + 1, sticky="ew", padx=(2, 6), pady=3)

        ttk.Label(outer, text="Type").grid(row=4, column=0, sticky="w", padx=(6, 2), pady=3)
        type_cb = ttk.Combobox(outer, textvariable=self._var("match_type", "any"), values=["any", "aligned", "pc1_only"], width=14, state="readonly")
        type_cb.grid(row=4, column=1, sticky="ew", padx=(2, 6), pady=3)

        ttk.Label(outer, text="Sort by").grid(row=4, column=2, sticky="w", padx=(6, 2), pady=3)
        sort_values = [
            "acuity_score", "alignment_core", "lexical_divergence", "lexical_overlap_coefficient",
            "manifold_residual_doc_cosine", "raw_sbert_doc_cosine", "doc_embedding_cosine",
            "delta_cos", "pc1_axis_value", "semantic_quality",
            "joint", "joint_min_4d", "match_row",
        ]
        sort_cb = ttk.Combobox(outer, textvariable=self._var("sort_by", "acuity_score"), values=sort_values, width=22)
        sort_cb.grid(row=4, column=3, sticky="ew", padx=(2, 6), pady=3)

        ttk.Checkbutton(outer, text="Descending", variable=self.desc_var).grid(row=4, column=4, sticky="w", padx=6, pady=3)

        ttk.Label(outer, text="Limit").grid(row=4, column=5, sticky="w", padx=(6, 2), pady=3)
        ttk.Entry(outer, textvariable=self._var("limit", "1000"), width=10).grid(row=4, column=6, sticky="ew", padx=(2, 6), pady=3)

        ttk.Button(outer, text="Run query", command=self.run_query).grid(row=4, column=7, padx=6, pady=3)
        ttk.Button(outer, text="Clear filters", command=self.clear_filters).grid(row=4, column=8, padx=6, pady=3)
        ttk.Button(outer, text="Export rows", command=self.export_rows).grid(row=4, column=9, padx=6, pady=3)

        ttk.Button(outer, text="Open selected evidence", command=self.open_selected_evidence).grid(row=5, column=7, padx=6, pady=(2, 5), sticky="ew")
        ttk.Button(outer, text="Open selected in 3D", command=self.open_selected_3d).grid(row=5, column=8, columnspan=2, padx=6, pady=(2, 5), sticky="ew")

        ttk.Label(self, textvariable=self.status_var).grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 4))

    def _build_table(self) -> None:
        frame = ttk.Frame(self)
        frame.grid(row=2, column=0, sticky="nsew", padx=8, pady=(0, 8))
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        columns = self.model.table_columns()
        self.tree = ttk.Treeview(frame, columns=columns, show="headings", selectmode="browse")
        for col in columns:
            self.tree.heading(col, text=col)
            width = 80
            if col in {"src_doc", "tgt_doc"}:
                width = 210
            elif col in {"match_type"}:
                width = 90
            elif col in {"lexical_overlap_coefficient", "lexical_divergence", "doc_embedding_cosine", "manifold_residual_doc_cosine", "raw_sbert_doc_cosine"}:
                width = 160
            self.tree.column(col, width=width, anchor="w")
        y = ttk.Scrollbar(frame, orient="vertical", command=self.tree.yview)
        x = ttk.Scrollbar(frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=y.set, xscrollcommand=x.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        y.grid(row=0, column=1, sticky="ns")
        x.grid(row=1, column=0, sticky="ew")
        self.tree.bind("<<TreeviewSelect>>", self._on_select)

    def clear_filters(self) -> None:
        for name, var in self.vars.items():
            if name == "match_type":
                var.set("any")
            elif name == "sort_by":
                var.set("acuity_score")
            elif name == "limit":
                var.set("1000")
            else:
                var.set("")
        self.desc_var.set(True)

    def clear_results(self, status: str = "File loaded. Set filters and click Run query.") -> None:
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.model.last_query_indices = np.zeros((0,), dtype=np.int64)
        self.status_var.set(status)

    def _spec(self) -> QuerySpec:
        return QuerySpec(
            source_doc=self.vars["source_doc"].get(),
            target_doc=self.vars["target_doc"].get(),
            any_doc=self.vars["any_doc"].get(),
            match_type=self.vars["match_type"].get(),
            min_delta=self.vars["min_delta"].get(),
            min_pc1=self.vars["min_pc1"].get(),
            min_quality=self.vars["min_quality"].get(),
            min_lex_div=self.vars["min_lex_div"].get(),
            max_lex_overlap=self.vars["max_lex_overlap"].get(),
            min_acuity=self.vars["min_acuity"].get(),
            max_doc_cos=self.vars.get("max_doc_cos", tk.StringVar(value="")).get(),
            max_manifold_residual_doc_cos=self.vars["max_manifold_residual_doc_cos"].get(),
            max_raw_sbert_doc_cos=self.vars["max_raw_sbert_doc_cos"].get(),
            sort_by=self.vars["sort_by"].get(),
            descending=bool(self.desc_var.get()),
            limit=max(1, _safe_int(self.vars["limit"].get(), 1000)),
        )

    def run_query(self) -> None:
        if not self.model.is_loaded:
            messagebox.showinfo("No file loaded", "Open a morphism_comparison.pkl file first.")
            return
        try:
            self.status_var.set("Running vectorized query ...")
            self.update_idletasks()
            idx = self.model.query_indices(self._spec())
            self.status_var.set(f"Populating {idx.size:,} displayed rows ...")
            self.update_idletasks()
            self.populate(idx)
            self.status_var.set(f"Showing {idx.size:,} rows from {self.model.n_matches:,} retained matches.")
        except Exception as ex:
            messagebox.showerror("Query failed", f"{ex}\n\n{traceback.format_exc(limit=3)}")

    def clear_table(self, status: str = "") -> None:
        for item in self.tree.get_children():
            self.tree.delete(item)
        if status:
            self.status_var.set(status)

    def populate(self, indices: Sequence[int]) -> None:
        for item in self.tree.get_children():
            self.tree.delete(item)
        rows = self.model.rows_for_indices(indices)
        columns = self.model.table_columns()
        for row in rows:
            vals = [self.model.format_table_value(c, row.get(c, "")) for c in columns]
            self.tree.insert("", "end", iid=str(row.get("match_row", "")), values=vals)

    def _selected_row_from_table(self) -> Optional[int]:
        sel = self.tree.selection()
        if not sel:
            return self.selected_match_row
        try:
            return int(sel[0])
        except Exception:
            return self.selected_match_row

    def _on_select(self, _event: Any = None) -> None:
        row = self._selected_row_from_table()
        if row is None:
            return
        self.selected_match_row = int(row)
        if self.on_select_match:
            try:
                # Selection updates downstream state, but it no longer forces a
                # tab switch.  Use the explicit buttons for evidence or 3D.
                self.on_select_match(int(row))
            except Exception:
                pass

    def open_selected_evidence(self) -> None:
        row = self._selected_row_from_table()
        if row is None:
            messagebox.showinfo("No match selected", "Select a match row in the query table first.")
            return
        self.selected_match_row = int(row)
        if self.on_open_evidence:
            self.on_open_evidence(int(row))
        elif self.on_select_match:
            self.on_select_match(int(row))

    def open_selected_3d(self) -> None:
        row = self._selected_row_from_table()
        if row is None:
            messagebox.showinfo("No match selected", "Select a match row in the query table first.")
            return
        self.selected_match_row = int(row)
        if not self.on_open_3d:
            messagebox.showinfo("3D view unavailable", "The application shell did not register a 3D view callback.")
            return
        self.on_open_3d(int(row))

    def export_rows(self) -> None:
        if not self.model.last_query_indices.size:
            messagebox.showinfo("No rows", "Run a query first.")
            return
        path = filedialog.asksaveasfilename(
            title="Export current match rows",
            defaultextension=".tsv",
            filetypes=[("TSV files", "*.tsv"), ("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        rows = self.model.rows_for_indices(self.model.last_query_indices)
        columns = self.model.table_columns()
        delimiter = "," if path.lower().endswith(".csv") else "\t"
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns, delimiter=delimiter, extrasaction="ignore")
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        messagebox.showinfo("Export complete", f"Wrote {len(rows):,} rows to:\n{path}")


class PlotCacheView(ttk.Frame):
    def __init__(self, master: tk.Misc, model: MorphismComparisonModel) -> None:
        super().__init__(master)
        self.model = model
        self.rowconfigure(1, weight=1)
        self.columnconfigure(0, weight=1)
        self.view_var = tk.StringVar(value="count")
        self.q_floor_var = tk.DoubleVar(value=0.0)
        self.max_points_var = tk.StringVar(value="150000")
        self.status_var = tk.StringVar(value="Load a comparison file with plot_cache to restore enriched graph views.")
        self.canvas: Optional[FigureCanvasTkAgg] = None
        self.toolbar: Optional[NavigationToolbar2Tk] = None
        self._build_controls()
        self.plot_frame = ttk.Frame(self)
        self.plot_frame.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))
        self.plot_frame.rowconfigure(0, weight=1)
        self.plot_frame.columnconfigure(0, weight=1)

    def _build_controls(self) -> None:
        bar = ttk.LabelFrame(self, text="Compact plot-cache graph views")
        bar.grid(row=0, column=0, sticky="ew", padx=8, pady=6)
        views = [
            "count", "mean_lexical_overlap", "lexical_divergence", "peak_acuity",
            "acute_candidate", "lexical_z_quality",
            "mean_manifold_residual_doc_cosine", "mean_raw_sbert_doc_cosine",
            "mean_doc_embedding_cosine", "anchor_edge",
        ]
        ttk.Label(bar, text="View").pack(side="left", padx=(8, 2))
        ttk.Combobox(bar, textvariable=self.view_var, values=views, width=28, state="readonly").pack(side="left", padx=(0, 8))
        ttk.Label(bar, text="Q floor").pack(side="left", padx=(8, 2))
        qscale = ttk.Scale(bar, from_=0.0, to=1.0, orient="horizontal", variable=self.q_floor_var, length=180)
        qscale.pack(side="left", padx=(0, 8))
        ttk.Button(bar, text="Render", command=self.render).pack(side="left", padx=6)
        ttk.Label(bar, text="Max points").pack(side="left", padx=(10, 2))
        ttk.Entry(bar, textvariable=self.max_points_var, width=10).pack(side="left", padx=(0, 8))
        ttk.Label(bar, textvariable=self.status_var).pack(side="left", padx=8)

    def refresh(self) -> None:
        if not self.model.is_loaded:
            self.status_var.set("No comparison file loaded.")
        elif not self.model.has_plot_cache():
            self.status_var.set("No enriched plot_cache found; count fallback can be rendered from matches.")
        else:
            c = np.asarray(self.model.plot_cache.get("count"))
            self.status_var.set(f"plot_cache ready: grid={tuple(c.shape)} occupied={int(np.count_nonzero(c)):,}")

    def _clear_plot(self) -> None:
        if self.toolbar is not None:
            try:
                self.toolbar.destroy()
            except Exception:
                pass
            self.toolbar = None
        if self.canvas is not None:
            try:
                self.canvas.get_tk_widget().destroy()
            except Exception:
                pass
            self.canvas = None

    def _thresholds(self, n_bins: int, step: float) -> np.ndarray:
        cache = self.model.plot_cache if isinstance(self.model.plot_cache, dict) else {}
        try:
            thr = np.asarray(cache.get("thresholds"), dtype=float)
            if thr.shape[0] == n_bins:
                return thr
        except Exception:
            pass
        return np.round(np.linspace(1.0, 0.0, n_bins), 6)

    def _ensure_grid_cache(self) -> Dict[str, Any]:
        """Return plot_cache or build a count-only fallback from matches."""
        if self.model.has_plot_cache():
            return self.model.plot_cache
        records = self.model.records
        step = 0.01
        n_bins = int(round(1.0 / step)) + 1
        delta = _clip01(self.model.metric_field("delta_cos", 0.0, float))
        pc = _clip01(self.model.metric_field("pc1_axis_value", 1.0, float))
        q = _clip01(self.model.metric_field("semantic_quality", 1.0, float))
        bi = np.clip(np.rint((1.0 - delta) / step).astype(int), 0, n_bins - 1)
        bj = np.clip(np.rint((1.0 - pc) / step).astype(int), 0, n_bins - 1)
        bk = np.clip(np.rint((1.0 - q) / step).astype(int), 0, n_bins - 1)
        count = np.zeros((n_bins, n_bins, n_bins), dtype=np.uint32)
        if records.size:
            np.add.at(count, (bi, bj, bk), 1)
        return {"step": step, "thresholds": np.round(np.linspace(1.0, 0.0, n_bins), 6), "count": count, "empty": False}

    def _grid_points(self, value_grid: np.ndarray, support_grid: np.ndarray, thr: np.ndarray, q_floor: float, max_points: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        support = np.asarray(support_grid)
        values = np.asarray(value_grid, dtype=float)
        mask = support > 0
        if mask.ndim != 3:
            return (np.array([]), np.array([]), np.array([]), np.array([]), np.array([]))
        q_keep = thr >= float(q_floor)
        if q_keep.shape[0] == mask.shape[2]:
            mask[:, :, ~q_keep] = False
        idx = np.where(mask)
        if len(idx[0]) == 0:
            return (np.array([]), np.array([]), np.array([]), np.array([]), np.array([]))
        vals = np.nan_to_num(values[idx], nan=0.0, posinf=0.0, neginf=0.0)
        counts = np.asarray(support[idx], dtype=float)
        if vals.size > max_points:
            score = vals if np.any(vals > 0) else counts
            order = np.argpartition(-score, max_points - 1)[:max_points]
            idx = tuple(a[order] for a in idx)
            vals = vals[order]
            counts = counts[order]
        x = thr[idx[0]]
        y = thr[idx[1]]
        z = thr[idx[2]]
        return x, y, z, vals, counts

    def render(self) -> None:
        if not self.model.is_loaded:
            messagebox.showinfo("No file loaded", "Open a morphism_comparison.pkl file first.")
            return
        try:
            cache = self._ensure_grid_cache()
            count = np.asarray(cache.get("count"), dtype=np.uint32)
            if count.size == 0:
                messagebox.showinfo("No plot data", "No count grid or match records are available.")
                return
            step = float(cache.get("step", 0.01) or 0.01)
            n_bins = count.shape[0]
            thr = self._thresholds(n_bins, step)
            q_floor = float(np.clip(self.q_floor_var.get(), 0.0, 1.0))
            max_points = max(100, _safe_int(self.max_points_var.get(), 150000))
            view = self.view_var.get()

            fig = Figure(figsize=(9.5, 7.2), dpi=100)
            ax = fig.add_subplot(111, projection="3d")
            ax.set_xlabel("Δ direction cosine threshold")
            ax.set_ylabel("PC1-axis concordance threshold")
            ax.set_zlabel("Semantic quality Q threshold")
            ax.set_title(view.replace("_", " ").title())

            color_label = "value"
            x = y = z = cvals = sizes = None

            if view == "mean_lexical_overlap":
                lex_sum = np.asarray(cache.get("lex_sum", np.zeros_like(count, dtype=np.float32)), dtype=float)
                lex_n = np.asarray(cache.get("lex_n", np.zeros_like(count, dtype=np.uint32)), dtype=np.uint32)
                with np.errstate(divide="ignore", invalid="ignore"):
                    grid = np.divide(lex_sum, np.maximum(lex_n, 1), where=(lex_n > 0))
                x, y, z, cvals, sizes = self._grid_points(grid, lex_n, thr, q_floor, max_points)
                color_label = "mean lexical overlap"
            elif view == "lexical_divergence":
                div_sum = np.asarray(cache.get("div_sum", np.zeros_like(count, dtype=np.float32)), dtype=float)
                lex_n = np.asarray(cache.get("lex_n", np.zeros_like(count, dtype=np.uint32)), dtype=np.uint32)
                with np.errstate(divide="ignore", invalid="ignore"):
                    grid = np.divide(div_sum, np.maximum(lex_n, 1), where=(lex_n > 0))
                x, y, z, cvals, sizes = self._grid_points(grid, lex_n, thr, q_floor, max_points)
                color_label = "mean lexical divergence"
            elif view == "peak_acuity":
                grid = np.asarray(cache.get("acuity_max", np.zeros_like(count, dtype=np.float32)), dtype=float)
                x, y, z, cvals, sizes = self._grid_points(grid, count, thr, q_floor, max_points)
                color_label = "peak acuity"
            elif view in {"mean_manifold_residual_doc_cosine", "mean_doc_embedding_cosine"}:
                # New files use explicit manifold-residual names; older files used
                # the ambiguous doc_embedding_cosine cache keys.
                s = np.asarray(cache.get(
                    "manifold_residual_doc_cosine_sum",
                    cache.get("doc_embedding_cosine_sum", np.zeros_like(count, dtype=np.float32))
                ), dtype=float)
                n = np.asarray(cache.get(
                    "manifold_residual_doc_cosine_n",
                    cache.get("doc_embedding_cosine_n", np.zeros_like(count, dtype=np.uint32))
                ), dtype=np.uint32)
                with np.errstate(divide="ignore", invalid="ignore"):
                    grid = np.divide(s, np.maximum(n, 1), where=(n > 0))
                x, y, z, cvals, sizes = self._grid_points(grid, n, thr, q_floor, max_points)
                color_label = "mean manifold-residual document cosine"
            elif view == "mean_raw_sbert_doc_cosine":
                s = np.asarray(cache.get("raw_sbert_doc_cosine_sum", np.zeros_like(count, dtype=np.float32)), dtype=float)
                n = np.asarray(cache.get("raw_sbert_doc_cosine_n", np.zeros_like(count, dtype=np.uint32)), dtype=np.uint32)
                with np.errstate(divide="ignore", invalid="ignore"):
                    grid = np.divide(s, np.maximum(n, 1), where=(n > 0))
                x, y, z, cvals, sizes = self._grid_points(grid, n, thr, q_floor, max_points)
                color_label = "mean raw SBERT document cosine"
            elif view == "acute_candidate":
                tc = np.asarray(cache.get("top_candidates", []))
                if tc.size and tc.dtype.names:
                    qv = np.asarray(tc["semantic_quality"], dtype=float)
                    keep = qv >= q_floor
                    rows = tc[keep]
                    if rows.size > max_points:
                        score = np.asarray(rows["acuity_score"], dtype=float)
                        order = np.argpartition(-score, max_points - 1)[:max_points]
                        rows = rows[order]
                    x = np.asarray(rows["delta_cos"], dtype=float)
                    y = np.asarray(rows["pc1_axis_value"], dtype=float)
                    z = np.asarray(rows["semantic_quality"], dtype=float)
                    cvals = np.asarray(rows["acuity_score"], dtype=float)
                    sizes = np.ones_like(cvals)
                    color_label = "acuity score"
                else:
                    x = y = z = cvals = sizes = np.array([])
            elif view == "lexical_z_quality":
                lex_sum = np.asarray(cache.get("lex_sum", np.zeros_like(count, dtype=np.float32)), dtype=float)
                lex_n = np.asarray(cache.get("lex_n", np.zeros_like(count, dtype=np.uint32)), dtype=np.uint32)
                acu = np.asarray(cache.get("acuity_max", np.zeros_like(count, dtype=np.float32)), dtype=float)
                with np.errstate(divide="ignore", invalid="ignore"):
                    lex_mean = np.divide(lex_sum, np.maximum(lex_n, 1), where=(lex_n > 0))
                xi, yi, zi, vals, sizes = self._grid_points(acu, lex_n, thr, q_floor, max_points)
                # Recompute z as mean lexical overlap at the same occupied cells.
                mask = lex_n > 0
                q_keep = thr >= q_floor
                if q_keep.shape[0] == mask.shape[2]:
                    mask[:, :, ~q_keep] = False
                idx = np.where(mask)
                if len(idx[0]):
                    vals = acu[idx]
                    zz = np.clip(np.nan_to_num(lex_mean[idx], nan=0.0), 0.0, 1.0)
                    if vals.size > max_points:
                        order = np.argpartition(-vals, max_points - 1)[:max_points]
                        idx = tuple(a[order] for a in idx)
                        vals = vals[order]
                        zz = zz[order]
                    x = thr[idx[0]]; y = thr[idx[1]]; z = zz; cvals = vals; sizes = np.asarray(lex_n[idx], dtype=float)
                else:
                    x = y = z = cvals = sizes = np.array([])
                ax.set_zlabel("Mean destination lexical overlap")
                color_label = "peak acuity"
            elif view == "anchor_edge":
                tc = np.asarray(cache.get("top_candidates", []))
                es = cache.get("edge_stats", {}) if isinstance(cache.get("edge_stats"), dict) else {}
                if tc.size and tc.dtype.names:
                    qv = np.asarray(tc["semantic_quality"], dtype=float)
                    keep = qv >= q_floor
                    rows = tc[keep]
                    if rows.size > max_points:
                        score = np.asarray(rows["acuity_score"], dtype=float)
                        order = np.argpartition(-score, max_points - 1)[:max_points]
                        rows = rows[order]
                    x = np.asarray(rows["delta_cos"], dtype=float)
                    y = np.asarray(rows["pc1_axis_value"], dtype=float)
                    z = np.asarray(rows["semantic_quality"], dtype=float)
                    cvals = np.asarray(rows["acuity_score"], dtype=float)
                    high_counts = np.asarray(es.get("high_acuity_count_by_src_edge", []), dtype=float)
                    sizes = np.ones_like(cvals)
                    if high_counts.size:
                        src_edges = np.asarray(rows["src_edge"], dtype=int)
                        valid = (src_edges >= 0) & (src_edges < high_counts.shape[0])
                        sizes[valid] = np.maximum(1.0, high_counts[src_edges[valid]])
                    color_label = "best/source-edge acuity"
                else:
                    x = y = z = cvals = sizes = np.array([])
            else:
                x, y, z, cvals, sizes = self._grid_points(count, count, thr, q_floor, max_points)
                color_label = "match count"

            if x is None or len(x) == 0:
                ax.text2D(0.05, 0.95, "No points for this view/Q floor.", transform=ax.transAxes)
            else:
                cvals = np.nan_to_num(np.asarray(cvals, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
                sizes = np.nan_to_num(np.asarray(sizes, dtype=float), nan=1.0, posinf=1.0, neginf=1.0)
                # Scaled marker areas; log keeps high-density bins visible without
                # letting them dominate the plot.
                marker_sizes = 12.0 + 12.0 * np.log1p(np.maximum(0.0, sizes))
                norm = mcolors.Normalize(vmin=float(np.nanmin(cvals)), vmax=float(np.nanmax(cvals)) if np.nanmax(cvals) > np.nanmin(cvals) else float(np.nanmin(cvals) + 1.0))
                sc = ax.scatter(x, y, z, c=cvals, s=marker_sizes, cmap="viridis", norm=norm, alpha=0.82, depthshade=True)
                cb = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.10)
                cb.set_label(color_label)
                self.status_var.set(f"Rendered {len(x):,} points; Q floor={q_floor:.2f}; view={view}")
            ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_zlim(0, 1)
            ax.view_init(elev=24, azim=-55)
            fig.tight_layout()

            self._clear_plot()
            self.canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
            self.canvas.draw()
            self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
            self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame, pack_toolbar=False)
            self.toolbar.update()
            self.toolbar.grid(row=1, column=0, sticky="ew")
        except Exception as ex:
            messagebox.showerror("Plot render failed", f"{ex}\n\n{traceback.format_exc(limit=6)}")


class EvidenceBrowser(ttk.Frame):
    def __init__(self, master: tk.Misc, model: MorphismComparisonModel, select_match_callback: Any = None) -> None:
        super().__init__(master)
        self.model = model
        self.selected_match_row: Optional[int] = None
        self.rowconfigure(1, weight=1)
        self.columnconfigure(0, weight=1)
        self.limit_var = tk.StringVar(value="1000")
        self.status_var = tk.StringVar(value="Load a comparison file, then refresh candidates.")
        self._build_controls()
        self._build_body()

    def _build_controls(self) -> None:
        bar = ttk.LabelFrame(self, text="Candidate evidence browser")
        bar.grid(row=0, column=0, sticky="ew", padx=8, pady=6)
        ttk.Label(bar, text="Candidate limit").pack(side="left", padx=(8, 2))
        ttk.Entry(bar, textvariable=self.limit_var, width=10).pack(side="left", padx=(0, 8))
        ttk.Button(bar, text="Refresh candidates", command=self.refresh_candidates).pack(side="left", padx=6)
        ttk.Button(bar, text="Save Markdown packet", command=self.save_markdown_packet).pack(side="left", padx=6)
        ttk.Button(bar, text="Save JSON packet", command=self.save_json_packet).pack(side="left", padx=6)
        ttk.Label(bar, textvariable=self.status_var).pack(side="left", padx=8)

    def _build_body(self) -> None:
        paned = ttk.Panedwindow(self, orient="horizontal")
        paned.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))

        left = ttk.Frame(paned)
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)
        cols = ["rank", "match_row", "acuity", "src_doc", "src_edge", "tgt_doc", "tgt_edge", "delta", "pc1", "Q", "lex_div", "resid_doc_cos", "raw_doc_cos"]
        self.candidate_tree = ttk.Treeview(left, columns=cols, show="headings", selectmode="browse")
        for col in cols:
            self.candidate_tree.heading(col, text=col)
            width = 80
            if col in {"src_doc", "tgt_doc"}:
                width = 180
            self.candidate_tree.column(col, width=width, anchor="w")
        y = ttk.Scrollbar(left, orient="vertical", command=self.candidate_tree.yview)
        x = ttk.Scrollbar(left, orient="horizontal", command=self.candidate_tree.xview)
        self.candidate_tree.configure(yscrollcommand=y.set, xscrollcommand=x.set)
        self.candidate_tree.grid(row=0, column=0, sticky="nsew")
        y.grid(row=0, column=1, sticky="ns")
        x.grid(row=1, column=0, sticky="ew")
        self.candidate_tree.bind("<<TreeviewSelect>>", self._on_candidate_select)
        paned.add(left, weight=2)

        right = ttk.Frame(paned)
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)
        notes_box = ttk.LabelFrame(right, text="Analyst notes")
        notes_box.grid(row=0, column=0, sticky="ew", padx=0, pady=(0, 6))
        notes_box.columnconfigure(0, weight=1)
        self.notes = tk.Text(notes_box, height=5, wrap="word")
        self.notes.grid(row=0, column=0, sticky="ew", padx=4, pady=4)

        self.notebook = ttk.Notebook(right)
        self.notebook.grid(row=1, column=0, sticky="nsew")
        self.summary_text = ScrollText(self.notebook)
        self.texts_text = ScrollText(self.notebook)
        self.terms_text = ScrollText(self.notebook)
        self.packet_text = ScrollText(self.notebook)
        self.notebook.add(self.summary_text, text="Scores")
        self.notebook.add(self.texts_text, text="Cluster text")
        self.notebook.add(self.terms_text, text="Lexical terms")
        self.notebook.add(self.packet_text, text="Markdown preview")
        paned.add(right, weight=3)

    def clear_candidates(self, status: str = "") -> None:
        for item in self.candidate_tree.get_children():
            self.candidate_tree.delete(item)
        self.selected_match_row = None
        self.summary_text.set_text("")
        self.texts_text.set_text("")
        self.terms_text.set_text("")
        self.packet_text.set_text("")
        if status:
            self.status_var.set(status)

    def refresh_candidates(self) -> None:
        if not self.model.is_loaded:
            messagebox.showinfo("No file loaded", "Open a morphism_comparison.pkl file first.")
            return
        try:
            limit = max(1, _safe_int(self.limit_var.get(), 1000))
            self.status_var.set("Selecting top candidate rows ...")
            self.update_idletasks()
            rows = self.model.top_candidate_rows(limit)
            self.status_var.set(f"Populating {rows.size:,} candidate rows ...")
            self.update_idletasks()
            for item in self.candidate_tree.get_children():
                self.candidate_tree.delete(item)
            for rank, match_row in enumerate(rows, 1):
                rd = self.model.row_dict(int(match_row), rank=rank)
                vals = [
                    rank,
                    int(match_row),
                    _fmt_float(rd.get("acuity_score"), 4),
                    rd.get("src_doc", ""),
                    f"C{rd.get('src_from')}→C{rd.get('src_to')}",
                    rd.get("tgt_doc", ""),
                    f"C{rd.get('tgt_from')}→C{rd.get('tgt_to')}",
                    _fmt_float(rd.get("delta_cos"), 4),
                    _fmt_float(rd.get("pc1_axis_value"), 4),
                    _fmt_float(rd.get("semantic_quality"), 4),
                    _fmt_float(rd.get("lexical_divergence"), 4),
                    _fmt_float(rd.get("manifold_residual_doc_cosine"), 4),
                    _fmt_float(rd.get("raw_sbert_doc_cosine"), 4),
                ]
                self.candidate_tree.insert("", "end", iid=str(match_row), values=vals)
            self.status_var.set(f"Loaded {rows.size:,} candidate rows.")
        except Exception as ex:
            messagebox.showerror("Candidate refresh failed", f"{ex}\n\n{traceback.format_exc(limit=4)}")

    def select_match(self, match_row: int) -> None:
        self.selected_match_row = int(match_row)
        try:
            if str(match_row) in self.candidate_tree.get_children(""):
                self.candidate_tree.selection_set(str(match_row))
                self.candidate_tree.see(str(match_row))
        except Exception:
            pass
        self._render_evidence()

    def _on_candidate_select(self, _event: Any = None) -> None:
        sel = self.candidate_tree.selection()
        if not sel:
            return
        try:
            self.selected_match_row = int(sel[0])
            self._render_evidence()
        except Exception:
            pass

    def _render_evidence(self) -> None:
        if self.selected_match_row is None:
            return
        try:
            notes = self.notes.get("1.0", "end-1c")
            packet = self.model.evidence_packet(self.selected_match_row, notes=notes, include_text=True)
            row = packet.get("match", {})
            lines = []
            lines.append(f"Match row: {row.get('match_row')}")
            lines.append(f"Match type: {row.get('match_type')}")
            lines.append(f"Source: {row.get('src_doc')} C{row.get('src_from')} → C{row.get('src_to')}  [edge {row.get('src_edge')}]")
            lines.append(f"Target: {row.get('tgt_doc')} C{row.get('tgt_from')} → C{row.get('tgt_to')}  [edge {row.get('tgt_edge')}]")
            lines.append("")
            for key in [
                "delta_cos", "src_pc1", "dst_pc1", "pc1_axis_value", "semantic_quality",
                "semantic_quality_min", "lexical_overlap_coefficient", "lexical_divergence",
                "alignment_core", "acuity_score", "acuity_score_count_cosine",
                "lexical_dst_count_cosine", "manifold_residual_doc_cosine",
                "raw_sbert_doc_cosine", "doc_embedding_cosine", "joint", "joint_min_4d",
            ]:
                if key in row:
                    lines.append(f"{key}: {_fmt_float(row.get(key), 6)}")
            lines.append("")
            lines.append(f"lexical_available: {row.get('lexical_available', '')}")
            lines.append(f"manifold_residual_doc_available: {row.get('manifold_residual_doc_available', row.get('doc_embedding_available', ''))}")
            lines.append(f"raw_sbert_doc_available: {row.get('raw_sbert_doc_available', '')}")
            lines.append(f"doc_embedding_available (legacy): {row.get('doc_embedding_available', '')}")
            self.summary_text.set_text("\n".join(lines))

            texts = packet.get("cluster_texts", {}) or {}
            tlines: List[str] = []
            for name, lst in texts.items():
                tlines.append("=" * 78)
                tlines.append(f"{name.replace('_', ' ').title()} ({len(lst)} segments)")
                tlines.append("=" * 78)
                tlines.append("\n\n".join(str(t) for t in lst) if lst else "(no companion text loaded)")
                tlines.append("")
            if not tlines:
                tlines.append("Load document_delta_dict.pkl and segments_by_doc.pkl to show cluster texts.")
            self.texts_text.set_text("\n".join(tlines))

            lex = packet.get("lexical_from_companion_text", {}) or {}
            llines: List[str] = []
            if lex:
                for name, metrics in lex.items():
                    llines.append("=" * 78)
                    llines.append(name.replace("_", " ").title())
                    llines.append("=" * 78)
                    for key in ["tokens_a", "tokens_b", "unique_a", "unique_b", "shared_unique", "jaccard", "overlap_coefficient", "count_cosine"]:
                        if key in metrics:
                            val = metrics[key]
                            llines.append(f"{key}: {_fmt_float(val, 5) if isinstance(val, float) else val}")
                    shared = metrics.get("shared_terms") or []
                    if shared:
                        llines.append("\nTop shared terms:")
                        for t, a, b in shared[:30]:
                            llines.append(f"  {t:<24} A={a:<5} B={b:<5}")
                    distinctive = metrics.get("distinctive_a_minus_b") or []
                    if distinctive:
                        llines.append("\nMost distinctive A-minus-B normalized-frequency terms:")
                        for t, diff in distinctive[:30]:
                            llines.append(f"  {t:<24} {diff:+.6f}")
                    llines.append("")
            else:
                llines.append("Companion text lexical checks are unavailable until document_delta_dict.pkl and segments_by_doc.pkl are loaded.")
            self.terms_text.set_text("\n".join(llines))
            self.packet_text.set_text(self.model.packet_markdown(packet))
        except Exception as ex:
            messagebox.showerror("Evidence render failed", f"{ex}\n\n{traceback.format_exc(limit=4)}")

    def _packet_for_save(self) -> Optional[Dict[str, Any]]:
        if self.selected_match_row is None:
            messagebox.showinfo("No candidate selected", "Select a candidate row first.")
            return None
        notes = self.notes.get("1.0", "end-1c")
        return self.model.evidence_packet(self.selected_match_row, notes=notes, include_text=True)

    def save_markdown_packet(self) -> None:
        packet = self._packet_for_save()
        if packet is None:
            return
        row = packet.get("match", {})
        default = f"morphism_evidence_match_{row.get('match_row', 'unknown')}.md"
        path = filedialog.asksaveasfilename(
            title="Save evidence packet Markdown",
            defaultextension=".md",
            initialfile=default,
            filetypes=[("Markdown", "*.md"), ("Text", "*.txt"), ("All files", "*.*")],
        )
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.model.packet_markdown(packet))
        messagebox.showinfo("Saved", f"Evidence packet written to:\n{path}")

    def save_json_packet(self) -> None:
        packet = self._packet_for_save()
        if packet is None:
            return
        row = packet.get("match", {})
        default = f"morphism_evidence_match_{row.get('match_row', 'unknown')}.json"
        path = filedialog.asksaveasfilename(
            title="Save evidence packet JSON",
            defaultextension=".json",
            initialfile=default,
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        _save_json(path, packet)
        messagebox.showinfo("Saved", f"Evidence packet written to:\n{path}")


# -----------------------------------------------------------------------------
# Selected edge-match 3D visualizer
# -----------------------------------------------------------------------------

class EdgeMatch3DView(ttk.Frame):
    """
    Query-driven 3D morphism-pair visualizer.

    The view uses compact match rows to identify two directed cluster edges,
    reads centroid/PC1/displacement geometry from document_delta_dict.pkl, then
    re-embeds only the selected endpoint cluster segments from segments_by_doc.pkl
    to draw local point-cloud extents.  No collection-scale segment embedding
    artifact is required.
    """

    PALETTE = {
        # Distinct cluster palette for the four endpoint clusters:
        "A_src": "#d81b60", # magenta
        "A_dst": "#00acc1", # cyan
        "B_src": "#43a047", # green
        "B_dst": "#c99700", # ocher
    }
    EDGE_COLORS = {"A": "#6a3d9a", "B": "#111111"}
    COMPLETE_GRAPH_PALETTE = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173",
        "#3182bd", "#31a354", "#756bb1", "#636363", "#e6550d",
    ]

    def __init__(self, master: tk.Misc, model: MorphismComparisonModel) -> None:
        super().__init__(master)
        self.model = model
        self.selected_match_row: Optional[int] = None
        self.match_row_var = tk.StringVar(value="")
        self.model_name_var = tk.StringVar(value="all-MiniLM-L6-v2")
        self.device_var = tk.StringVar(value="auto")
        self.batch_size_var = tk.StringVar(value="128")
        self.max_points_var = tk.StringVar(value="600")
        self.delta_scale_var = tk.StringVar(value="1.0")
        self.pc1_scale_var = tk.StringVar(value="0.45")
        self.coordinate_mode_var = tk.StringVar(value="residual_cdm_space")
        self.projection_fit_scope_var = tk.StringVar(value="selected_edge_endpoints")
        self.anchor_residual_scale_var = tk.StringVar(value="auto")
        self.canonical_pc1_var = tk.BooleanVar(value=True)
        self.complete_graph_var = tk.BooleanVar(value=False)
        self.complete_graph_show_delta_var = tk.BooleanVar(value=True)
        self.complete_graph_show_pc1_var = tk.BooleanVar(value=True)
        self.complete_graph_show_labels_var = tk.BooleanVar(value=True)
        self.complete_graph_label_mode_var = tk.StringVar(value="selected")  # none | selected | all
        self.show_cloud_means_var = tk.BooleanVar(value=False)
        self.complete_graph_edge_filter_var = tk.StringVar(value="all")
        self.complete_graph_max_arrows_var = tk.StringVar(value="all")
        self.complete_graph_edge_alpha_var = tk.StringVar(value="0.16")
        self.complete_graph_edge_width_var = tk.StringVar(value="0.65")
        self.selected_delta_width_var = tk.StringVar(value="3.0")
        self.selected_delta_alpha_var = tk.StringVar(value="1.0")
        self.pc1_width_var = tk.StringVar(value="1.4")
        self.point_size_var = tk.StringVar(value="12")
        self.point_alpha_var = tk.StringVar(value="1.0")
        self.selected_centroid_size_var = tk.StringVar(value="190")
        self.nonselected_centroid_size_var = tk.StringVar(value="95")
        self.label_font_size_var = tk.StringVar(value="8")
        self.status_var = tk.StringVar(value="Select a query row, then open it here for 3D inspection.")
        self.canvas: Optional[FigureCanvasTkAgg] = None
        self.toolbar: Optional[NavigationToolbar2Tk] = None
        self.current_figure: Optional[Figure] = None
        self._render_thread: Optional[threading.Thread] = None
        self._embedder_cache: Dict[Tuple[str, str], Any] = {}
        self._cloud_cache: Dict[Tuple[Any, ...], np.ndarray] = {}
        self._doc_embedding_cache: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
        self.rowconfigure(1, weight=1)
        self.columnconfigure(0, weight=1)
        self._build_controls()
        self._build_body()

    def _build_controls(self) -> None:
        bar = ttk.LabelFrame(self, text="Selected match edge-pair 3D visualization")
        bar.grid(row=0, column=0, sticky="ew", padx=8, pady=6)
        for i in range(20):
            bar.columnconfigure(i, weight=0)
        bar.columnconfigure(19, weight=1)

        ttk.Label(bar, text="Match row").grid(row=0, column=0, padx=(8, 2), pady=3, sticky="w")
        ttk.Entry(bar, textvariable=self.match_row_var, width=12).grid(row=0, column=1, padx=(2, 8), pady=3, sticky="w")
        ttk.Button(bar, text="Render", command=self.render_async).grid(row=0, column=2, padx=4, pady=3)
        ttk.Button(bar, text="Save PNG", command=self.save_png).grid(row=0, column=3, padx=4, pady=3)
        ttk.Button(bar, text="Clear", command=self.clear_view).grid(row=0, column=4, padx=4, pady=3)

        ttk.Label(bar, text="SBERT model").grid(row=1, column=0, padx=(8, 2), pady=3, sticky="w")
        ttk.Entry(bar, textvariable=self.model_name_var, width=24).grid(row=1, column=1, columnspan=2, padx=(2, 8), pady=3, sticky="ew")
        ttk.Label(bar, text="device").grid(row=1, column=3, padx=(8, 2), pady=3, sticky="w")
        ttk.Combobox(bar, textvariable=self.device_var, values=["auto", "cpu", "cuda"], width=8).grid(row=1, column=4, padx=(2, 8), pady=3, sticky="w")
        ttk.Label(bar, text="batch").grid(row=1, column=5, padx=(8, 2), pady=3, sticky="w")
        ttk.Entry(bar, textvariable=self.batch_size_var, width=8).grid(row=1, column=6, padx=(2, 8), pady=3, sticky="w")
        ttk.Label(bar, text="max pts/cluster").grid(row=1, column=7, padx=(8, 2), pady=3, sticky="w")
        ttk.Entry(bar, textvariable=self.max_points_var, width=8).grid(row=1, column=8, padx=(2, 8), pady=3, sticky="w")
        ttk.Label(bar, text="Δ scale").grid(row=1, column=9, padx=(8, 2), pady=3, sticky="w")
        ttk.Entry(bar, textvariable=self.delta_scale_var, width=7).grid(row=1, column=10, padx=(2, 8), pady=3, sticky="w")
        ttk.Label(bar, text="PC1 scale").grid(row=1, column=11, padx=(8, 2), pady=3, sticky="w")
        ttk.Entry(bar, textvariable=self.pc1_scale_var, width=7).grid(row=1, column=12, padx=(2, 8), pady=3, sticky="w")
        ttk.Checkbutton(bar, text="canonicalize PC1 to edge flow", variable=self.canonical_pc1_var).grid(row=1, column=13, columnspan=3, padx=8, pady=3, sticky="w")

        ttk.Label(bar, text="Coordinate mode").grid(row=2, column=0, padx=(8, 2), pady=3, sticky="w")
        ttk.Combobox(
            bar,
            textvariable=self.coordinate_mode_var,
            values=["residual_cdm_space", "anchored_residual_raw_sbert_origins"],
            width=34,
            state="readonly",
        ).grid(row=2, column=1, columnspan=3, padx=(2, 8), pady=3, sticky="ew")
        ttk.Label(bar, text="projection fit").grid(row=5, column=0, padx=(8, 2), pady=3, sticky="w")
        ttk.Combobox(
            bar,
            textvariable=self.projection_fit_scope_var,
            values=["selected_edge_endpoints", "displayed_objects", "complete_selected_documents"],
            width=28,
            state="readonly",
        ).grid(row=5, column=1, columnspan=3, padx=(2, 8), pady=3, sticky="ew")
        ttk.Label(bar, text="anchor residual scale").grid(row=2, column=4, padx=(8, 2), pady=3, sticky="w")
        ttk.Entry(bar, textvariable=self.anchor_residual_scale_var, width=10).grid(row=2, column=5, padx=(2, 8), pady=3, sticky="w")
        ttk.Checkbutton(bar, text="Complete document graph for selected match", variable=self.complete_graph_var).grid(row=2, column=6, columnspan=5, padx=8, pady=3, sticky="w")
        ttk.Label(bar, text="labels").grid(row=2, column=11, padx=(8, 2), pady=3, sticky="e")
        ttk.Combobox(bar, textvariable=self.complete_graph_label_mode_var, values=["none", "selected", "all"], width=9, state="readonly").grid(row=2, column=12, padx=(2, 8), pady=3, sticky="w")
        ttk.Checkbutton(bar, text="cloud mean overlay", variable=self.show_cloud_means_var).grid(row=2, column=13, columnspan=2, padx=4, pady=3, sticky="w")

        ttk.Checkbutton(bar, text="show all Δ", variable=self.complete_graph_show_delta_var).grid(row=3, column=0, padx=(8, 2), pady=3, sticky="w")
        ttk.Checkbutton(bar, text="show all PC1", variable=self.complete_graph_show_pc1_var).grid(row=3, column=1, padx=(2, 8), pady=3, sticky="w")
        ttk.Label(bar, text="edge filter").grid(row=3, column=2, padx=(8, 2), pady=3, sticky="w")
        ttk.Combobox(
            bar,
            textvariable=self.complete_graph_edge_filter_var,
            values=["all", "top_length", "top_quality", "top_flow"],
            width=12,
            state="readonly",
        ).grid(row=3, column=3, padx=(2, 8), pady=3, sticky="w")
        ttk.Label(bar, text="max arrows/doc").grid(row=3, column=4, padx=(8, 2), pady=3, sticky="w")
        ttk.Entry(bar, textvariable=self.complete_graph_max_arrows_var, width=8).grid(row=3, column=5, padx=(2, 8), pady=3, sticky="w")
        ttk.Label(bar, text="all-Δ alpha").grid(row=3, column=6, padx=(8, 2), pady=3, sticky="w")
        ttk.Entry(bar, textvariable=self.complete_graph_edge_alpha_var, width=7).grid(row=3, column=7, padx=(2, 8), pady=3, sticky="w")
        ttk.Label(bar, text="all-Δ width").grid(row=3, column=8, padx=(8, 2), pady=3, sticky="w")
        ttk.Entry(bar, textvariable=self.complete_graph_edge_width_var, width=7).grid(row=3, column=9, padx=(2, 8), pady=3, sticky="w")
        ttk.Label(bar, text="selected Δ width").grid(row=3, column=10, padx=(8, 2), pady=3, sticky="w")
        ttk.Entry(bar, textvariable=self.selected_delta_width_var, width=7).grid(row=3, column=11, padx=(2, 8), pady=3, sticky="w")
        ttk.Label(bar, text="selected Δ alpha").grid(row=3, column=12, padx=(8, 2), pady=3, sticky="w")
        ttk.Entry(bar, textvariable=self.selected_delta_alpha_var, width=7).grid(row=3, column=13, padx=(2, 8), pady=3, sticky="w")

        ttk.Label(bar, text="point size").grid(row=4, column=0, padx=(8, 2), pady=3, sticky="w")
        ttk.Entry(bar, textvariable=self.point_size_var, width=7).grid(row=4, column=1, padx=(2, 8), pady=3, sticky="w")
        ttk.Label(bar, text="point alpha").grid(row=4, column=2, padx=(8, 2), pady=3, sticky="w")
        ttk.Entry(bar, textvariable=self.point_alpha_var, width=7).grid(row=4, column=3, padx=(2, 8), pady=3, sticky="w")
        ttk.Label(bar, text="PC1 width").grid(row=4, column=4, padx=(8, 2), pady=3, sticky="w")
        ttk.Entry(bar, textvariable=self.pc1_width_var, width=7).grid(row=4, column=5, padx=(2, 8), pady=3, sticky="w")
        ttk.Label(bar, text="sel centroid").grid(row=4, column=6, padx=(8, 2), pady=3, sticky="w")
        ttk.Entry(bar, textvariable=self.selected_centroid_size_var, width=7).grid(row=4, column=7, padx=(2, 8), pady=3, sticky="w")
        ttk.Label(bar, text="other centroid").grid(row=4, column=8, padx=(8, 2), pady=3, sticky="w")
        ttk.Entry(bar, textvariable=self.nonselected_centroid_size_var, width=7).grid(row=4, column=9, padx=(2, 8), pady=3, sticky="w")
        ttk.Label(bar, text="label font").grid(row=4, column=10, padx=(8, 2), pady=3, sticky="w")
        ttk.Entry(bar, textvariable=self.label_font_size_var, width=7).grid(row=4, column=11, padx=(2, 8), pady=3, sticky="w")

        ttk.Label(bar, textvariable=self.status_var).grid(row=6, column=0, columnspan=20, sticky="ew", padx=8, pady=(2, 4))

    def _build_body(self) -> None:
        paned = ttk.Panedwindow(self, orient="horizontal")
        paned.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))

        self.plot_frame = ttk.Frame(paned)
        self.plot_frame.rowconfigure(0, weight=1)
        self.plot_frame.columnconfigure(0, weight=1)
        paned.add(self.plot_frame, weight=4)

        self.summary = ScrollText(paned)
        paned.add(self.summary, weight=1)
        self.summary.set_text(
            "This view renders one selected match.\n\n"
            "Required companion artifacts:\n"
            "  • document_delta_dict.pkl for centroids, Δ, and PC1 vectors\n"
            "  • segments_by_doc.pkl for endpoint cluster segment text\n\n"
            "Point clouds are generated on demand by re-embedding the two selected documents, "
            "reconstructing the pipeline's document-centered CDM build-space, "
            "and then plotting either the four endpoint clusters or, with complete graph enabled, "
            "all clusters and all directed cluster-to-cluster displacement arrows for the selected documents. "
            "Use projection fit = selected_edge_endpoints to keep the selected-edge PCA frame stable while adding complete-graph context; "
            "use displayed_objects to fit PCA to exactly what is drawn; use complete_selected_documents to fit over all two-document cluster clouds and centroids."
        )

    def clear_view(self) -> None:
        self._clear_plot()
        self.current_figure = None
        self.summary.set_text("")
        self.status_var.set("3D view cleared.")

    def _clear_plot(self) -> None:
        if self.canvas is not None:
            try:
                self.canvas.get_tk_widget().destroy()
            except Exception:
                pass
            self.canvas = None
        if self.toolbar is not None:
            try:
                self.toolbar.destroy()
            except Exception:
                pass
            self.toolbar = None

    def set_match(self, match_row: int, render: bool = False) -> None:
        self.selected_match_row = int(match_row)
        self.match_row_var.set(str(int(match_row)))
        self.status_var.set(f"Selected match row {int(match_row)}. Click Render to build the edge-pair 3D view.")
        if render:
            self.render_async()

    def _row_from_entry(self) -> Optional[int]:
        txt = str(self.match_row_var.get() or "").strip()
        if not txt:
            return self.selected_match_row
        try:
            return int(txt)
        except Exception:
            return self.selected_match_row

    def _get_embedder(self, model_name: str, device: str) -> Any:
        model_name = str(model_name or "all-MiniLM-L6-v2").strip() or "all-MiniLM-L6-v2"
        device = str(device or "auto").strip().lower() or "auto"
        key = (model_name, device)
        if key not in self._embedder_cache:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
            except Exception as ex:
                raise RuntimeError(
                    "The 3D point-cloud view requires sentence-transformers. "
                    "Install it with: pip install sentence-transformers"
                ) from ex
            if device in {"", "auto"}:
                self._embedder_cache[key] = SentenceTransformer(model_name)
            else:
                self._embedder_cache[key] = SentenceTransformer(model_name, device=device)
        return self._embedder_cache[key]

    def _sample_texts(self, texts: Sequence[str], max_points: int) -> Tuple[List[str], List[int]]:
        texts = [str(t or "") for t in texts]
        n = len(texts)
        if n == 0:
            return [], []
        max_points = max(1, int(max_points or 1))
        if n <= max_points:
            return texts, list(range(n))
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(n, size=max_points, replace=False))
        return [texts[int(i)] for i in idx], [int(i) for i in idx]

    def _encode_cluster_texts(
        self,
        doc_id: str,
        cluster_label: Any,
        texts: Sequence[str],
        model_name: str,
        device: str,
        batch_size: int,
        max_points: int,
    ) -> Tuple[np.ndarray, int, int]:
        sampled_texts, sampled_idx = self._sample_texts(texts, max_points=max_points)
        if not sampled_texts:
            return np.zeros((0, 0), dtype=np.float32), 0, 0
        cache_key = (
            str(model_name), str(device), str(doc_id), str(cluster_label),
            len(texts), sum(len(str(t or "")) for t in texts), len(sampled_texts),
        )
        if cache_key in self._cloud_cache:
            return self._cloud_cache[cache_key], len(texts), len(sampled_texts)
        embedder = self._get_embedder(model_name, device)
        arr = embedder.encode(
            sampled_texts,
            batch_size=max(1, int(batch_size or 1)),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        self._cloud_cache[cache_key] = arr
        return arr, len(texts), len(sampled_texts)

    def _document_segments_and_labels(self, doc_id: str) -> Tuple[List[str], List[Any]]:
        """Return full document segments and labels aligned to one another."""
        if not isinstance(self.model.document_delta_dict, dict) or not isinstance(self.model.segments_by_doc, dict):
            return [], []
        key_doc, key_seg = self.model._resolve_doc_key(str(doc_id))
        if key_doc not in self.model.document_delta_dict or key_seg not in self.model.segments_by_doc:
            return [], []
        try:
            data = self.model.document_delta_dict[key_doc]
            labels = data[2]
            labels = labels.tolist() if hasattr(labels, "tolist") else list(labels)
            segs = list(self.model.segments_by_doc[key_seg])
            n = min(len(segs), len(labels))
            return [str(s or "") for s in segs[:n]], list(labels[:n])
        except Exception:
            return [], []

    def _encode_document_spaces(
        self,
        doc_id: str,
        model_name: str,
        device: str,
        batch_size: int,
    ) -> Dict[str, Any]:
        """
        Re-embed a whole selected document and return both raw and build spaces.

        The raw SBERT vectors provide document-origin anchors.  The build-space
        vectors reconstruct the document-centered / row-renormalized coordinate
        system used to compute stored CDM cluster centroids, Δ vectors, and PC1
        directions.  Keeping both spaces lets the anchored view translate local
        residual morphism glyphs to raw document origins without saving a
        collection-scale segment-embedding store.
        """
        segs, labels = self._document_segments_and_labels(doc_id)
        if not segs or not labels:
            return {
                "raw": np.zeros((0, 0), dtype=np.float32),
                "build_space": np.zeros((0, 0), dtype=np.float32),
                "labels": [],
                "segments": [],
                "raw_anchor": np.zeros((0,), dtype=float),
                "raw_anchor_available": False,
                "raw_anchor_source": "unavailable",
            }
        cache_key = (
            "doc_spaces_v2", str(model_name), str(device), str(doc_id),
            len(segs), sum(len(str(t or "")) for t in segs),
        )
        if cache_key in self._doc_embedding_cache:
            return self._doc_embedding_cache[cache_key]
        embedder = self._get_embedder(model_name, device)
        raw = embedder.encode(
            segs,
            batch_size=max(1, int(batch_size or 1)),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        raw = np.asarray(raw, dtype=np.float32)
        if raw.ndim == 1:
            raw = raw[None, :]
        build_space = _build_space_preprocess_embeddings(raw, remove_top_components_n=0)
        raw_anchor = _unit_vector(np.nanmean(raw, axis=0)) if raw.size else np.zeros((0,), dtype=float)
        payload = {
            "raw": raw,
            "build_space": build_space,
            "labels": labels,
            "segments": segs,
            "raw_anchor": raw_anchor,
            "raw_anchor_available": bool(raw_anchor.size),
            "raw_anchor_source": "reembedded_raw_mean_sbert",
        }
        self._doc_embedding_cache[cache_key] = payload
        return payload

    def _encode_document_build_space(
        self,
        doc_id: str,
        model_name: str,
        device: str,
        batch_size: int,
    ) -> Tuple[np.ndarray, List[Any], List[str]]:
        spaces = self._encode_document_spaces(doc_id, model_name, device, batch_size)
        return (
            np.asarray(spaces.get("build_space", np.zeros((0, 0), dtype=np.float32)), dtype=np.float32),
            list(spaces.get("labels", [])),
            list(spaces.get("segments", [])),
        )

    def _cdm_document_embedding(self, doc_id: str, kind: str = "raw_sbert") -> Tuple[np.ndarray, str]:
        """Read a document-level embedding vector from a loaded CDM tuple when available."""
        data = self.model.document_cdm(str(doc_id))
        if not isinstance(data, (tuple, list)) or len(data) < 8 or not isinstance(data[7], dict):
            return np.zeros((0,), dtype=float), ""
        payload = data[7]
        if str(kind).lower().startswith("raw"):
            keys = ["raw_sbert_document_embedding", "raw_document_embedding", "raw_sbert_doc_embedding"]
        else:
            keys = ["manifold_residual_document_embedding", "document_embedding", "residual_document_embedding"]
        for key in keys:
            if key in payload:
                v = _unit_vector(payload.get(key))
                if v.size:
                    return v, f"cdm_payload:{key}"
        return np.zeros((0,), dtype=float), ""

    def _raw_document_anchor(
        self,
        doc_id: str,
        model_name: str,
        device: str,
        batch_size: int,
    ) -> Tuple[np.ndarray, str]:
        """Return a raw mean-SBERT document anchor, preferring stored CDM payloads."""
        v, source = self._cdm_document_embedding(doc_id, kind="raw_sbert")
        if v.size:
            return v, source
        spaces = self._encode_document_spaces(doc_id, model_name, device, batch_size)
        v = _unit_vector(spaces.get("raw_anchor"))
        if v.size:
            return v, str(spaces.get("raw_anchor_source", "reembedded_raw_mean_sbert"))
        return np.zeros((0,), dtype=float), "unavailable"

    def _cluster_color_for_complete_graph(
        self,
        edge_code: str,
        cluster_index: int,
        selected_key: str = "",
    ) -> str:
        """Return a stable display color for a cluster in complete-graph mode."""
        if selected_key and selected_key in self.PALETTE:
            return self.PALETTE[selected_key]
        palette = self.COMPLETE_GRAPH_PALETTE
        offset = 0 if str(edge_code) == "A" else 7
        return palette[(int(cluster_index) + offset) % len(palette)]

    def _all_cluster_geometries_for_doc(self, doc_id: str) -> List[Dict[str, Any]]:
        """Return cluster geometries for every cluster in a loaded document CDM."""
        data = self.model.document_cdm(str(doc_id))
        if data is None:
            raise KeyError(f"document {doc_id!r} not found in document_delta_dict")
        order = self.model._cluster_order(data)
        out: List[Dict[str, Any]] = []
        for lab in order:
            try:
                out.append(self.model.cluster_geometry(str(doc_id), lab))
            except Exception:
                continue
        return out

    def _complete_graph_edge_specs_for_doc(
        self,
        edge_code: str,
        edge_label: str,
        doc_id: str,
        cluster_entries: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Build all directed cluster-to-cluster edge specs for one document."""
        data = self.model.document_cdm(str(doc_id))
        if data is None:
            return []
        by_doc_cluster: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for c in cluster_entries:
            if str(c.get("edge_code")) != str(edge_code):
                continue
            by_doc_cluster[(str(c.get("doc_id")), str(c.get("cluster_label_display")))] = c
        specs: List[Dict[str, Any]] = []
        try:
            D = np.asarray(data[0], dtype=float)
            order = self.model._cluster_order(data)
            if D.ndim != 3:
                return []
            n = min(len(order), D.shape[0], D.shape[1])
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    src_lab = order[i]
                    dst_lab = order[j]
                    src_key = f"{edge_code}_C{_label_display(src_lab)}"
                    dst_key = f"{edge_code}_C{_label_display(dst_lab)}"
                    src_rec = next((c for c in cluster_entries if c.get("key") == src_key), None)
                    dst_rec = next((c for c in cluster_entries if c.get("key") == dst_key), None)
                    if src_rec is None or dst_rec is None:
                        continue
                    delta_vec = np.asarray(D[i, j], dtype=float).reshape(-1)
                    delta_norm = float(np.linalg.norm(delta_vec)) if delta_vec.size else 0.0
                    src_q = float(src_rec.get("quality", 1.0)) if src_rec is not None else 1.0
                    dst_q = float(dst_rec.get("quality", 1.0)) if dst_rec is not None else 1.0
                    edge_quality = max(0.0, min(1.0, min(src_q, dst_q)))
                    flow_score = 0.0
                    try:
                        v = _unit_vector(delta_vec)
                        s = _unit_vector(src_rec.get("pc1_oriented", src_rec.get("pc1", [])))
                        t = _unit_vector(dst_rec.get("pc1_oriented", dst_rec.get("pc1", [])))
                        if v.size == s.size == t.size and v.size:
                            a1 = max(0.0, float(np.dot(v, s)))
                            a2 = max(0.0, float(np.dot(-v, t)))
                            a3 = abs(float(np.dot(s, t)))
                            flow_score = float(a1 * a2 * math.sqrt(max(0.0, a3)))
                    except Exception:
                        flow_score = 0.0
                    specs.append({
                        "edge_code": edge_code,
                        "edge_label": edge_label,
                        "doc_id": str(doc_id),
                        "src_key": src_key,
                        "dst_key": dst_key,
                        "src_label": src_lab,
                        "dst_label": dst_lab,
                        "delta": delta_vec,
                        "delta_norm": delta_norm,
                        "edge_quality": edge_quality,
                        "flow_score": flow_score,
                        "color": self.EDGE_COLORS.get(edge_code, "#333333"),
                    })
        except Exception:
            return []
        return specs

    def _encode_cluster_build_space_points(
        self,
        doc_id: str,
        cluster_label: Any,
        model_name: str,
        device: str,
        batch_size: int,
        max_points: int,
    ) -> Tuple[np.ndarray, int, int, str]:
        """Return selected cluster point cloud in reconstructed CDM build-space."""
        E, labels, _segs = self._encode_document_build_space(doc_id, model_name, device, batch_size)
        if E.size == 0 or not labels:
            return np.zeros((0, 0), dtype=np.float32), 0, 0, "unavailable"
        try:
            target = int(cluster_label)
            mask = np.asarray([int(lab) == target for lab in labels], dtype=bool)
        except Exception:
            mask = np.asarray([str(lab) == str(cluster_label) for lab in labels], dtype=bool)
        idx = np.where(mask)[0]
        n_total = int(idx.size)
        if n_total == 0:
            return np.zeros((0, E.shape[1]), dtype=np.float32), 0, 0, "build_space_doc_centered_n0"
        max_points = max(1, int(max_points or 1))
        if idx.size > max_points:
            rng = np.random.default_rng(0)
            idx = np.sort(rng.choice(idx, size=max_points, replace=False))
        return np.asarray(E[idx], dtype=np.float32), n_total, int(idx.size), "build_space_doc_centered_n0"

    def render_async(self) -> None:
        row = self._row_from_entry()
        if row is None:
            messagebox.showinfo("No match selected", "Select a query row first or enter a match row number.")
            return
        if not self.model.is_loaded:
            messagebox.showinfo("No comparison loaded", "Open a morphism_comparison.pkl file first.")
            return
        if not isinstance(self.model.document_delta_dict, dict):
            messagebox.showinfo("document_delta_dict required", "Open companion document_delta_dict.pkl before rendering edge geometry.")
            return
        if not isinstance(self.model.segments_by_doc, dict):
            messagebox.showinfo("segments_by_doc required", "Open companion segments_by_doc.pkl before rendering re-embedded point clouds.")
            return
        if self._render_thread is not None and self._render_thread.is_alive():
            messagebox.showinfo("Render in progress", "An edge-match 3D render is already running.")
            return

        settings = {
            "match_row": int(row),
            "model_name": str(self.model_name_var.get() or "all-MiniLM-L6-v2").strip() or "all-MiniLM-L6-v2",
            "device": str(self.device_var.get() or "auto").strip() or "auto",
            "batch_size": max(1, _safe_int(self.batch_size_var.get(), 128)),
            "max_points": max(1, _safe_int(self.max_points_var.get(), 600)),
            "delta_scale": max(0.0, _safe_float(self.delta_scale_var.get(), 1.0)),
            "pc1_scale": max(0.0, _safe_float(self.pc1_scale_var.get(), 0.45)),
            "coordinate_mode": str(self.coordinate_mode_var.get() or "residual_cdm_space"),
            "projection_fit_scope": str(self.projection_fit_scope_var.get() or "selected_edge_endpoints"),
            "anchor_residual_scale": str(self.anchor_residual_scale_var.get() or "auto"),
            "complete_graph": bool(self.complete_graph_var.get()),
            "complete_graph_show_delta": bool(self.complete_graph_show_delta_var.get()),
            "complete_graph_show_pc1": bool(self.complete_graph_show_pc1_var.get()),
            "complete_graph_show_labels": bool(self.complete_graph_show_labels_var.get()),
            "complete_graph_label_mode": str(self.complete_graph_label_mode_var.get() or "selected").strip().lower(),
            "show_cloud_means": bool(self.show_cloud_means_var.get()),
            "complete_graph_edge_filter": str(self.complete_graph_edge_filter_var.get() or "all"),
            "complete_graph_max_arrows": str(self.complete_graph_max_arrows_var.get() or "all"),
            "complete_graph_edge_alpha": min(1.0, max(0.0, _safe_float(self.complete_graph_edge_alpha_var.get(), 0.16))),
            "complete_graph_edge_width": max(0.0, _safe_float(self.complete_graph_edge_width_var.get(), 0.65)),
            "selected_delta_width": max(0.0, _safe_float(self.selected_delta_width_var.get(), 3.0)),
            "selected_delta_alpha": min(1.0, max(0.0, _safe_float(self.selected_delta_alpha_var.get(), 1.0))),
            "pc1_width": max(0.0, _safe_float(self.pc1_width_var.get(), 1.4)),
            "point_size": max(1.0, _safe_float(self.point_size_var.get(), 12.0)),
            "point_alpha": min(1.0, max(0.0, _safe_float(self.point_alpha_var.get(), 1.0))),
            "selected_centroid_size": max(1.0, _safe_float(self.selected_centroid_size_var.get(), 190.0)),
            "nonselected_centroid_size": max(1.0, _safe_float(self.nonselected_centroid_size_var.get(), 95.0)),
            "label_font_size": max(1.0, _safe_float(self.label_font_size_var.get(), 8.0)),
            "canonical_pc1": bool(self.canonical_pc1_var.get()),
        }
        self.selected_match_row = int(row)
        mode_msg = "complete document graphs" if bool(settings.get("complete_graph")) else "selected endpoint clusters"
        self.status_var.set(
            f"Preparing match row {row}: reading CDM geometry and re-embedding {mode_msg} ..."
        )
        self.update_idletasks()

        def worker() -> None:
            try:
                t0 = time.perf_counter()
                data = self._prepare_render_data(settings)
                elapsed = time.perf_counter() - t0
                self.after(0, lambda data=data, elapsed=elapsed: self._draw_render_data(data, elapsed))
            except Exception as ex:
                err = str(ex)
                tb = traceback.format_exc(limit=8)
                self.after(0, lambda err=err, tb=tb: (
                    self.status_var.set("3D render failed."),
                    messagebox.showerror("3D render failed", f"{err}\n\n{tb}")
                ))

        self._render_thread = threading.Thread(target=worker, daemon=True)
        self._render_thread.start()

    def _prepare_render_data(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        match_row = int(settings["match_row"])
        geom = self.model.edge_pair_geometry(match_row)
        row = geom["row"]
        source_edge = geom["source_edge"]
        target_edge = geom["target_edge"]
        edge_entries = [
            ("A", source_edge, "source morphism"),
            ("B", target_edge, "target morphism"),
        ]
        complete_graph = bool(settings.get("complete_graph", False))
        cluster_entries: List[Dict[str, Any]] = []
        complete_edge_specs: List[Dict[str, Any]] = []
        warnings: List[str] = []

        base_delta = np.asarray([source_edge.get("delta_norm", 0.0), target_edge.get("delta_norm", 0.0)], dtype=float)
        base_delta = base_delta[np.isfinite(base_delta) & (base_delta > 1e-12)]
        base_len = float(np.median(base_delta)) if base_delta.size else 0.10
        pc1_len = float(settings["pc1_scale"]) * base_len

        def _append_cluster_entry(
            edge_code: str,
            edge_label: str,
            endpoint_name: str,
            endpoint_role: str,
            c_in: Dict[str, Any],
            delta_ref: np.ndarray,
            key: str,
            color: str,
            is_selected_endpoint: bool = False,
        ) -> None:
            c = dict(c_in)
            pc1 = _unit_vector(c.get("pc1"))
            # PC1 signs are ambiguous.  For the selected source/destination
            # endpoints, retain the morphism-flow canonicalization used in the
            # pair view.  For non-selected complete-graph clusters, keep the
            # stored document-level orientation because the cluster participates
            # in many directed edges at once.
            if bool(settings.get("canonical_pc1", True)) and bool(is_selected_endpoint) and pc1.size == delta_ref.size and delta_ref.size:
                ref = delta_ref if endpoint_name == "src" else -delta_ref
                if float(np.dot(pc1, ref)) < 0.0:
                    pc1 = -pc1
            texts = c.get("segment_texts") or []
            # Prefer reconstructed CDM build-space clouds.  This encodes the
            # whole selected document so the document-centering step matches
            # the pipeline's build-time centroid coordinate system.  Fallback
            # to cluster-only raw SBERT points only when full-document
            # companion labels/segments are unavailable.
            cloud, n_total, n_sampled, cloud_coordinate_space = self._encode_cluster_build_space_points(
                c.get("doc_id", ""),
                c.get("cluster_label", ""),
                model_name=str(settings["model_name"]),
                device=str(settings["device"]),
                batch_size=int(settings["batch_size"]),
                max_points=int(settings["max_points"]),
            )
            if not cloud.size:
                cloud, n_total, n_sampled = self._encode_cluster_texts(
                    c.get("doc_id", ""),
                    c.get("cluster_label", ""),
                    texts,
                    model_name=str(settings["model_name"]),
                    device=str(settings["device"]),
                    batch_size=int(settings["batch_size"]),
                    max_points=int(settings["max_points"]),
                )
                cloud_coordinate_space = "raw_cluster_only_sbert_fallback"
            centroid = np.asarray(c.get("centroid"), dtype=float).reshape(-1)
            if cloud.size and cloud.shape[1] != centroid.shape[0]:
                raise ValueError(
                    "Re-embedded segment dimension does not match stored CDM centroid dimension. "
                    f"cloud={cloud.shape[1]}, centroid={centroid.shape[0]}. "
                    "Use the same SBERT model used by the build pipeline."
                )
            if not n_total:
                warnings.append(
                    f"No segment text found for {edge_label} {endpoint_role}: "
                    f"{c.get('doc_id')} C{c.get('cluster_label_display', c.get('cluster_label'))}."
                )

            # Centroid diagnostic basis:
            # The stored CDM centroid was computed during the build pipeline as
            # the arithmetic mean of build-space segment embeddings.  The point
            # cloud shown here is reconstructed by re-embedding segment text on
            # demand.  If both coordinate populations agree, the stored centroid
            # should coincide with the mean of the re-embedded cloud under any
            # linear projection.
            cloud_mean = np.zeros_like(centroid, dtype=float)
            cloud_mean_available = False
            if cloud.size and cloud.ndim == 2 and cloud.shape[0] > 0 and cloud.shape[1] == centroid.shape[0]:
                cloud_mean = np.nanmean(np.asarray(cloud, dtype=float), axis=0).reshape(-1)
                cloud_mean_available = bool(cloud_mean.size == centroid.size)

            centroid_norm = float(np.linalg.norm(centroid)) if centroid.size else float("nan")
            cloud_mean_norm = float(np.linalg.norm(cloud_mean)) if cloud_mean_available else float("nan")
            centroid_cloud_cos = _vector_cosine(centroid, cloud_mean) if cloud_mean_available else float("nan")
            centroid_cloud_l2 = _vector_l2(centroid, cloud_mean) if cloud_mean_available else float("nan")
            centroid_cloud_unit_l2 = _vector_l2(_unit_vector(centroid), _unit_vector(cloud_mean)) if cloud_mean_available else float("nan")

            if cloud_mean_available and np.isfinite(centroid_cloud_cos) and centroid_cloud_cos < 0.995:
                warnings.append(
                    f"Stored centroid vs re-embedded cloud mean divergence for {key} "
                    f"({c.get('doc_id')} C{c.get('cluster_label_display', c.get('cluster_label'))}): "
                    f"cos={centroid_cloud_cos:.6f}, L2={centroid_cloud_l2:.6f}."
                )

            c.update({
                "key": key,
                "edge_code": edge_code,
                "edge_label": edge_label,
                "endpoint_name": endpoint_name,
                "endpoint_role": endpoint_role,
                "is_selected_endpoint": bool(is_selected_endpoint),
                "pc1_oriented": pc1,
                "cloud": cloud,
                "cloud_coordinate_space": cloud_coordinate_space,
                "cloud_mean": cloud_mean,
                "cloud_mean_available": bool(cloud_mean_available),
                "centroid_norm": centroid_norm,
                "cloud_mean_norm": cloud_mean_norm,
                "centroid_cloud_cosine": centroid_cloud_cos,
                "centroid_cloud_l2": centroid_cloud_l2,
                "centroid_cloud_unit_l2": centroid_cloud_unit_l2,
                "n_total_segments": int(n_total),
                "n_sampled_segments": int(n_sampled),
                "color": color,
            })
            cluster_entries.append(c)

        if complete_graph:
            selected_pairs = {
                "A": {
                    "src": source_edge.get("src", {}).get("cluster_label"),
                    "dst": source_edge.get("dst", {}).get("cluster_label"),
                    "delta": np.asarray(source_edge.get("delta", []), dtype=float).reshape(-1),
                },
                "B": {
                    "src": target_edge.get("src", {}).get("cluster_label"),
                    "dst": target_edge.get("dst", {}).get("cluster_label"),
                    "delta": np.asarray(target_edge.get("delta", []), dtype=float).reshape(-1),
                },
            }
            for edge_code, edge, edge_label in edge_entries:
                doc_id = str(edge.get("doc_id", ""))
                doc_clusters = self._all_cluster_geometries_for_doc(doc_id)
                selected = selected_pairs.get(edge_code, {})
                for idx, c in enumerate(doc_clusters):
                    endpoint_name = "cluster"
                    endpoint_role = "non-selected cluster"
                    selected_key = ""
                    is_selected_endpoint = False
                    lab = c.get("cluster_label")
                    if _label_equal(lab, selected.get("src")):
                        endpoint_name = "src"
                        endpoint_role = "selected source endpoint"
                        selected_key = f"{edge_code}_src"
                        is_selected_endpoint = True
                    elif _label_equal(lab, selected.get("dst")):
                        endpoint_name = "dst"
                        endpoint_role = "selected destination endpoint"
                        selected_key = f"{edge_code}_dst"
                        is_selected_endpoint = True
                    key = f"{edge_code}_C{c.get('cluster_label_display', _label_display(lab))}"
                    color = self._cluster_color_for_complete_graph(edge_code, idx, selected_key=selected_key)
                    _append_cluster_entry(
                        edge_code=edge_code,
                        edge_label=edge_label,
                        endpoint_name=endpoint_name,
                        endpoint_role=endpoint_role,
                        c_in=c,
                        delta_ref=np.asarray(selected.get("delta", []), dtype=float).reshape(-1),
                        key=key,
                        color=color,
                        is_selected_endpoint=is_selected_endpoint,
                    )
                complete_edge_specs.extend(
                    self._complete_graph_edge_specs_for_doc(edge_code, edge_label, doc_id, cluster_entries)
                )
        else:
            for edge_code, edge, edge_label in edge_entries:
                delta = np.asarray(edge["delta"], dtype=float).reshape(-1)
                for endpoint_name, endpoint_role in [("src", "source endpoint"), ("dst", "destination endpoint")]:
                    c = dict(edge[endpoint_name])
                    key = f"{edge_code}_{endpoint_name}"
                    _append_cluster_entry(
                        edge_code=edge_code,
                        edge_label=edge_label,
                        endpoint_name=endpoint_name,
                        endpoint_role=endpoint_role,
                        c_in=c,
                        delta_ref=delta,
                        key=key,
                        color=self.PALETTE.get(key, "#666666"),
                        is_selected_endpoint=True,
                    )

        if complete_graph and complete_edge_specs:
            # Presentation legibility: optionally keep only the strongest complete-graph
            # underlay arrows per document.  The selected match arrows are always drawn
            # separately in the highlighted overlay, so this filter only reduces the
            # contextual all-edges underlay.
            edge_filter = str(settings.get("complete_graph_edge_filter", "all") or "all").strip().lower()
            max_text = str(settings.get("complete_graph_max_arrows", "all") or "all").strip().lower()
            max_per_doc = None
            if max_text not in {"", "all", "none", "0"}:
                try:
                    max_per_doc = max(1, int(float(max_text)))
                except Exception:
                    max_per_doc = None
            if max_per_doc is not None or edge_filter in {"top_length", "top_quality", "top_flow"}:
                def _score_complete_edge(spec: Dict[str, Any]) -> float:
                    if edge_filter == "top_quality":
                        return _safe_float(spec.get("edge_quality"), 0.0)
                    if edge_filter == "top_flow":
                        return _safe_float(spec.get("flow_score"), 0.0)
                    return _safe_float(spec.get("delta_norm"), 0.0)
                grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
                for spec in complete_edge_specs:
                    grouped[(str(spec.get("edge_code", "")), str(spec.get("doc_id", "")))].append(spec)
                filtered_specs: List[Dict[str, Any]] = []
                for _group_key, specs in grouped.items():
                    specs = list(specs)
                    if max_per_doc is None:
                        filtered_specs.extend(specs)
                    else:
                        specs.sort(key=_score_complete_edge, reverse=True)
                        filtered_specs.extend(specs[:max_per_doc])
                if len(filtered_specs) < len(complete_edge_specs):
                    warnings.append(
                        f"Complete-graph underlay arrows filtered from {len(complete_edge_specs)} to {len(filtered_specs)} "
                        f"using filter={edge_filter}, max_arrows/doc={max_text}."
                    )
                complete_edge_specs = filtered_specs

        coordinate_mode = str(settings.get("coordinate_mode") or "residual_cdm_space")
        anchored_mode = coordinate_mode == "anchored_residual_raw_sbert_origins"

        # Raw SBERT document anchors for the anchored residual view.  These are
        # global base points; the cluster clouds/centroids/arrows remain residual
        # CDM geometry translated to those base points.
        doc_anchors: Dict[str, Dict[str, Any]] = {}
        if anchored_mode:
            for edge_code, edge, edge_label in edge_entries:
                doc_id = str(edge.get("doc_id", ""))
                anchor, anchor_source = self._raw_document_anchor(
                    doc_id,
                    model_name=str(settings["model_name"]),
                    device=str(settings["device"]),
                    batch_size=int(settings["batch_size"]),
                )
                doc_anchors[edge_code] = {
                    "edge_code": edge_code,
                    "edge_label": edge_label,
                    "doc_id": doc_id,
                    "anchor": anchor,
                    "anchor_source": anchor_source,
                    "color": self.EDGE_COLORS.get(edge_code, "#333333"),
                }
            anchor_dims = [int(np.asarray(v.get("anchor")).reshape(-1).size) for v in doc_anchors.values() if np.asarray(v.get("anchor")).size]
            residual_dims = [int(np.asarray(c.get("centroid")).reshape(-1).size) for c in cluster_entries]
            if not anchor_dims or not residual_dims or any(d != residual_dims[0] for d in anchor_dims):
                warnings.append("Raw SBERT document anchors are unavailable or dimension-mismatched; using residual CDM space instead.")
                anchored_mode = False
                coordinate_mode = "residual_cdm_space"

        # Resolve residual glyph scale for anchored mode.  'auto' scales the
        # local CDM glyphs so their median endpoint-cloud radius is readable
        # relative to raw document-anchor separation without swamping it.
        resolved_anchor_residual_scale = 1.0
        if anchored_mode:
            scale_text = str(settings.get("anchor_residual_scale", "auto") or "auto").strip().lower()
            if scale_text not in {"", "auto"}:
                resolved_anchor_residual_scale = max(0.0, _safe_float(scale_text, 1.0))
            else:
                anchors = [np.asarray(v.get("anchor"), dtype=float).reshape(-1) for v in doc_anchors.values() if np.asarray(v.get("anchor")).size]
                anchor_dist = float(np.linalg.norm(anchors[0] - anchors[1])) if len(anchors) >= 2 and anchors[0].shape == anchors[1].shape else 1.0
                radii = []
                for c in cluster_entries:
                    cloud = np.asarray(c.get("cloud"), dtype=float)
                    centroid = np.asarray(c.get("centroid"), dtype=float).reshape(1, -1)
                    if cloud.ndim == 2 and cloud.shape[0] > 0 and cloud.shape[1] == centroid.shape[1]:
                        rr = np.sqrt(np.sum((cloud - centroid) ** 2, axis=1))
                        if rr.size:
                            radii.append(float(np.median(rr)))
                med_radius = float(np.median(radii)) if radii else max(base_len, 1e-3)
                if not np.isfinite(anchor_dist) or anchor_dist <= 1e-8:
                    anchor_dist = 1.0
                if not np.isfinite(med_radius) or med_radius <= 1e-8:
                    med_radius = max(base_len, 1e-3)
                resolved_anchor_residual_scale = float(np.clip(0.40 * anchor_dist / med_radius, 0.05, 2.50))
        settings["coordinate_mode"] = coordinate_mode
        settings["resolved_anchor_residual_scale"] = resolved_anchor_residual_scale

        def _anchor_for_cluster(c: Dict[str, Any]) -> np.ndarray:
            if not anchored_mode:
                return np.zeros((0,), dtype=float)
            rec = doc_anchors.get(str(c.get("edge_code")), {})
            return np.asarray(rec.get("anchor", np.zeros((0,), dtype=float)), dtype=float).reshape(-1)

        def _display_point(c: Dict[str, Any], residual_point: Any) -> np.ndarray:
            rp = np.asarray(residual_point, dtype=float).reshape(-1)
            if anchored_mode:
                anchor = _anchor_for_cluster(c)
                if anchor.size == rp.size:
                    return anchor + resolved_anchor_residual_scale * rp
            return rp

        def _display_cloud(c: Dict[str, Any], residual_cloud: Any) -> np.ndarray:
            cloud = np.asarray(residual_cloud, dtype=float)
            if cloud.ndim == 1:
                cloud = cloud[None, :]
            if cloud.size == 0:
                dim = int(np.asarray(c.get("centroid"), dtype=float).reshape(-1).size)
                return np.zeros((0, dim), dtype=float)
            if anchored_mode:
                anchor = _anchor_for_cluster(c)
                if anchor.size == cloud.shape[1]:
                    return anchor.reshape(1, -1) + resolved_anchor_residual_scale * cloud
            return cloud

        def _cluster_key_for_edge_endpoint(edge_code: str, edge: Dict[str, Any], endpoint_name: str) -> str:
            if complete_graph:
                endpoint = edge.get(endpoint_name, {}) if isinstance(edge, dict) else {}
                lab = endpoint.get("cluster_label_display", endpoint.get("cluster_label", endpoint_name))
                return f"{edge_code}_C{_label_display(lab)}"
            return f"{edge_code}_{endpoint_name}"

        for c in cluster_entries:
            centroid = np.asarray(c["centroid"], dtype=float).reshape(-1)
            cloud = np.asarray(c.get("cloud", np.zeros((0, centroid.size))), dtype=float)
            cloud_mean = np.asarray(c.get("cloud_mean", np.zeros_like(centroid)), dtype=float).reshape(-1)
            c["display_centroid"] = _display_point(c, centroid)
            c["display_cloud"] = _display_cloud(c, cloud)
            c["display_cloud_mean"] = _display_point(c, cloud_mean) if bool(c.get("cloud_mean_available")) else np.full_like(c["display_centroid"], np.nan, dtype=float)

        projection_fit_scope = str(settings.get("projection_fit_scope", "selected_edge_endpoints") or "selected_edge_endpoints").strip().lower()
        projection_fit_aliases = {
            "selected": "selected_edge_endpoints",
            "selected_edge": "selected_edge_endpoints",
            "selected_edge_only": "selected_edge_endpoints",
            "edge_only": "selected_edge_endpoints",
            "displayed": "displayed_objects",
            "all_displayed": "displayed_objects",
            "complete": "complete_selected_documents",
            "complete_documents": "complete_selected_documents",
            "two_documents": "complete_selected_documents",
        }
        projection_fit_scope = projection_fit_aliases.get(projection_fit_scope, projection_fit_scope)
        if projection_fit_scope not in {"selected_edge_endpoints", "displayed_objects", "complete_selected_documents"}:
            projection_fit_scope = "selected_edge_endpoints"
        settings["projection_fit_scope"] = projection_fit_scope

        def _projection_include_cluster(c: Dict[str, Any]) -> bool:
            if projection_fit_scope == "selected_edge_endpoints":
                return bool(c.get("is_selected_endpoint"))
            # displayed_objects and complete_selected_documents both include the
            # complete two-document cluster clouds/centroids when complete graph
            # mode is active.  In edge-only mode, every displayed cluster is an
            # endpoint cluster, so this is equivalent to the selected frame.
            return True

        def _append_projection_cluster_points(c: Dict[str, Any], include_cloud: bool = True, include_pc1: bool = True) -> None:
            centroid_disp = np.asarray(c.get("display_centroid"), dtype=float).reshape(1, -1)
            if centroid_disp.size:
                fit_points.append(centroid_disp)
            if include_cloud:
                cloud_disp = np.asarray(c.get("display_cloud"), dtype=float)
                if cloud_disp.size:
                    fit_points.append(cloud_disp)
            if include_pc1:
                pc1 = np.asarray(c.get("pc1_oriented"), dtype=float).reshape(-1)
                centroid_resid = np.asarray(c.get("centroid"), dtype=float).reshape(-1)
                if pc1.size == centroid_resid.size and float(np.linalg.norm(pc1)) > 0.0:
                    fit_points.append(_display_point(c, centroid_resid + pc1 * pc1_len).reshape(1, -1))
                    fit_points.append(_display_point(c, centroid_resid - pc1 * pc1_len).reshape(1, -1))

        fit_points: List[np.ndarray] = []
        if anchored_mode:
            for rec in doc_anchors.values():
                anchor = np.asarray(rec.get("anchor"), dtype=float).reshape(-1)
                if anchor.size:
                    fit_points.append(anchor.reshape(1, -1))

        for c in cluster_entries:
            if _projection_include_cluster(c):
                _append_projection_cluster_points(c, include_cloud=True, include_pc1=True)

        # Always keep the selected matched arrows in the fit.  This makes
        # projection_fit_scope=selected_edge_endpoints reproduce the visual
        # frame of the endpoint-only figure even when complete graph context is
        # drawn afterward.
        for edge_code, edge, _edge_label in edge_entries:
            src_key = _cluster_key_for_edge_endpoint(edge_code, edge, "src")
            src_c_rec = next((cc for cc in cluster_entries if cc.get("key") == src_key), None)
            if src_c_rec is None:
                continue
            src_c = np.asarray(edge["src"]["centroid"], dtype=float).reshape(-1)
            delta = np.asarray(edge["delta"], dtype=float).reshape(-1)
            if src_c.size == delta.size:
                fit_points.append(_display_point(src_c_rec, src_c + delta * float(settings["delta_scale"])).reshape(1, -1))

        # Add the complete-graph underlay arrows to the PCA fit only when the
        # requested fit scope is explicitly the displayed object set.  Leaving
        # them out for selected_edge_endpoints keeps the selected morphism match
        # visually stable; leaving them out for complete_selected_documents fits
        # to the full two-document cloud/centroid population rather than to a
        # potentially filtered arrow set.
        if projection_fit_scope == "displayed_objects":
            for spec in complete_edge_specs:
                src_c_rec = next((cc for cc in cluster_entries if cc.get("key") == spec.get("src_key")), None)
                if src_c_rec is None:
                    continue
                src_c = np.asarray(src_c_rec.get("centroid"), dtype=float).reshape(-1)
                delta = np.asarray(spec.get("delta"), dtype=float).reshape(-1)
                if src_c.size == delta.size:
                    fit_points.append(_display_point(src_c_rec, src_c + delta * float(settings["delta_scale"])).reshape(1, -1))

        center, basis = _fit_pca_projection(fit_points)

        document_anchor_draw: List[Dict[str, Any]] = []
        if anchored_mode:
            for rec in doc_anchors.values():
                anchor = np.asarray(rec.get("anchor"), dtype=float).reshape(-1)
                if anchor.size:
                    rec = dict(rec)
                    rec["anchor_3d"] = _project_points(anchor, center, basis)[0]
                    document_anchor_draw.append(rec)
            if len(document_anchor_draw) >= 2:
                a0 = np.asarray(document_anchor_draw[0].get("anchor"), dtype=float)
                a1 = np.asarray(document_anchor_draw[1].get("anchor"), dtype=float)
                if a0.shape == a1.shape and a0.size:
                    raw_anchor_cos = _vector_cosine(a0, a1)
                    raw_anchor_l2 = _vector_l2(a0, a1)
                else:
                    raw_anchor_cos = float("nan")
                    raw_anchor_l2 = float("nan")
            else:
                raw_anchor_cos = float("nan")
                raw_anchor_l2 = float("nan")
        else:
            raw_anchor_cos = float("nan")
            raw_anchor_l2 = float("nan")

        cluster_by_key = {c["key"]: c for c in cluster_entries}
        for c in cluster_entries:
            centroid = np.asarray(c["centroid"], dtype=float).reshape(-1)
            pc1 = np.asarray(c.get("pc1_oriented"), dtype=float).reshape(-1)
            centroid_disp = np.asarray(c.get("display_centroid"), dtype=float).reshape(-1)
            c["centroid_3d"] = _project_points(centroid_disp, center, basis)[0]
            c["cloud_3d"] = _project_points(c.get("display_cloud", np.zeros((0, centroid_disp.size))), center, basis)

            if bool(c.get("cloud_mean_available")):
                cloud_mean_disp = np.asarray(c.get("display_cloud_mean"), dtype=float).reshape(-1)
                c["cloud_mean_3d"] = _project_points(cloud_mean_disp, center, basis)[0]
                off3 = np.asarray(c["cloud_mean_3d"], dtype=float) - np.asarray(c["centroid_3d"], dtype=float)
                c["centroid_cloud_projected_l2"] = float(np.linalg.norm(off3))
                P3 = np.asarray(c.get("cloud_3d", np.zeros((0, 3))), dtype=float)
                if P3.ndim == 2 and P3.shape[0] > 0:
                    dif = P3 - np.asarray(c["cloud_mean_3d"], dtype=float).reshape(1, 3)
                    rms = float(np.sqrt(np.mean(np.sum(dif * dif, axis=1))))
                    mx = float(np.max(np.sqrt(np.sum(dif * dif, axis=1))))
                else:
                    rms = float("nan")
                    mx = float("nan")
                c["cloud_projected_rms_radius"] = rms
                c["cloud_projected_max_radius"] = mx
                c["centroid_cloud_projected_l2_over_rms"] = (
                    float(c["centroid_cloud_projected_l2"] / rms)
                    if np.isfinite(rms) and rms > 1e-12 else float("nan")
                )
            else:
                c["cloud_mean_3d"] = np.full(3, np.nan, dtype=float)
                c["centroid_cloud_projected_l2"] = float("nan")
                c["cloud_projected_rms_radius"] = float("nan")
                c["cloud_projected_max_radius"] = float("nan")
                c["centroid_cloud_projected_l2_over_rms"] = float("nan")

            if pc1.size == centroid.size and float(np.linalg.norm(pc1)) > 0.0:
                p1 = _project_points(_display_point(c, centroid + pc1 * pc1_len), center, basis)[0]
                c["pc1_arrow_3d"] = p1 - c["centroid_3d"]
            else:
                c["pc1_arrow_3d"] = np.zeros(3, dtype=float)

        complete_edge_draw = []
        if complete_graph:
            for spec in complete_edge_specs:
                src_c_rec = cluster_by_key.get(str(spec.get("src_key")))
                dst_c_rec = cluster_by_key.get(str(spec.get("dst_key")))
                if src_c_rec is None or dst_c_rec is None:
                    continue
                src_centroid = np.asarray(src_c_rec.get("centroid"), dtype=float).reshape(-1)
                delta = np.asarray(spec.get("delta"), dtype=float).reshape(-1)
                start3 = np.asarray(src_c_rec.get("centroid_3d"), dtype=float)
                if src_centroid.size == delta.size:
                    end_disp = _display_point(src_c_rec, src_centroid + delta * float(settings["delta_scale"]))
                    end3 = _project_points(end_disp, center, basis)[0]
                    vec3 = end3 - start3
                else:
                    vec3 = np.asarray(dst_c_rec.get("centroid_3d"), dtype=float) - start3
                complete_edge_draw.append({
                    "edge_code": spec.get("edge_code"),
                    "edge_label": spec.get("edge_label"),
                    "doc_id": spec.get("doc_id"),
                    "src_key": spec.get("src_key"),
                    "dst_key": spec.get("dst_key"),
                    "start_3d": start3,
                    "delta_arrow_3d": vec3,
                    "color": spec.get("color", "#888888"),
                })

        edge_draw = []
        for edge_code, edge, edge_label in edge_entries:
            src_key = _cluster_key_for_edge_endpoint(edge_code, edge, "src")
            dst_key = _cluster_key_for_edge_endpoint(edge_code, edge, "dst")
            src_c_rec = cluster_by_key[src_key]
            dst_c_rec = cluster_by_key[dst_key]
            src_centroid = np.asarray(edge["src"]["centroid"], dtype=float).reshape(-1)
            delta = np.asarray(edge["delta"], dtype=float).reshape(-1)
            start3 = np.asarray(src_c_rec["centroid_3d"], dtype=float)
            if src_centroid.size == delta.size:
                end_disp = _display_point(src_c_rec, src_centroid + delta * float(settings["delta_scale"]))
                end3 = _project_points(end_disp, center, basis)[0]
                vec3 = end3 - start3
            else:
                vec3 = np.asarray(dst_c_rec["centroid_3d"], dtype=float) - start3
            edge_draw.append({
                "edge_code": edge_code,
                "edge_label": edge_label,
                "edge_id": edge.get("edge_id", -1),
                "src_key": src_key,
                "dst_key": dst_key,
                "start_3d": start3,
                "delta_arrow_3d": vec3,
                "delta_norm": float(edge.get("delta_norm", 0.0)),
                "color": self.EDGE_COLORS.get(edge_code, "#333333"),
            })

        return {
            "settings": settings,
            "row": row,
            "clusters": cluster_entries,
            "edges": edge_draw,
            "complete_edges": complete_edge_draw,
            "document_anchors": document_anchor_draw,
            "raw_anchor_cosine_computed": raw_anchor_cos,
            "raw_anchor_l2_computed": raw_anchor_l2,
            "projection_center_norm": float(np.linalg.norm(center)) if center.size else 0.0,
            "projection_basis_shape": tuple(basis.shape),
            "warnings": warnings,
        }

    def _draw_render_data(self, data: Dict[str, Any], elapsed: float) -> None:
        row = data.get("row", {})
        fig = Figure(figsize=(10.5, 8.2), dpi=100)
        ax = fig.add_subplot(111, projection="3d")
        coord_mode = str(data.get("settings", {}).get("coordinate_mode", "residual_cdm_space"))
        mode_label = "anchored residual / raw SBERT origins" if coord_mode == "anchored_residual_raw_sbert_origins" else "residual CDM space"
        projection_fit_scope = str(data.get("settings", {}).get("projection_fit_scope", "selected_edge_endpoints"))
        projection_fit_label = {
            "selected_edge_endpoints": "fit: selected edge",
            "displayed_objects": "fit: displayed objects",
            "complete_selected_documents": "fit: complete docs",
        }.get(projection_fit_scope, f"fit: {projection_fit_scope}")
        if bool(data.get("settings", {}).get("complete_graph")):
            mode_label = mode_label + " · complete document graph"
        mode_label = mode_label + " · " + projection_fit_label
        ax.set_title(
            f"Match {row.get('match_row')} · {row.get('match_type')} · {mode_label} · "
            f"Δ={_fmt_float(row.get('delta_cos'), 3)} PC1={_fmt_float(row.get('pc1_axis_value'), 3)} "
            f"Q={_fmt_float(row.get('semantic_quality'), 3)}",
            fontsize=11,
        )

        anchors = data.get("document_anchors") or []
        for a in anchors:
            A = np.asarray(a.get("anchor_3d", np.full(3, np.nan)), dtype=float).reshape(-1)
            if A.size == 3 and np.isfinite(A).all():
                color = a.get("color", "#333333")
                ax.scatter([A[0]], [A[1]], [A[2]], s=280, marker="s", color=color, edgecolors="black", linewidths=1.5, depthshade=True, label=f"{a.get('edge_code')} raw SBERT document anchor")
                ax.text(A[0], A[1], A[2], f" {a.get('edge_code')} raw anchor\n {a.get('doc_id')}", fontsize=8, weight="bold")
        if len(anchors) >= 2:
            A0 = np.asarray(anchors[0].get("anchor_3d", np.full(3, np.nan)), dtype=float).reshape(-1)
            A1 = np.asarray(anchors[1].get("anchor_3d", np.full(3, np.nan)), dtype=float).reshape(-1)
            if A0.size == 3 and A1.size == 3 and np.isfinite(A0).all() and np.isfinite(A1).all():
                ax.plot([A0[0], A1[0]], [A0[1], A1[1]], [A0[2], A1[2]], color="0.35", linestyle="--", linewidth=1.8, label="raw document-anchor separation")

        settings = data.get("settings", {}) if isinstance(data.get("settings"), dict) else {}
        complete_graph_enabled = bool(settings.get("complete_graph"))

        # In complete-graph mode, draw every directed cluster-to-cluster
        # displacement first as a tunable underlay.  The selected matched
        # morphisms are drawn later with heavier arrows.
        if bool(settings.get("complete_graph_show_delta", True)):
            all_delta_alpha = min(1.0, max(0.0, _safe_float(settings.get("complete_graph_edge_alpha"), 0.16)))
            all_delta_width = max(0.0, _safe_float(settings.get("complete_graph_edge_width"), 0.65))
            for e in data.get("complete_edges", []):
                S = np.asarray(e.get("start_3d"), dtype=float)
                V = np.asarray(e.get("delta_arrow_3d"), dtype=float)
                if S.size == 3 and V.size == 3 and float(np.linalg.norm(V)) > 1e-10:
                    ax.quiver(
                        S[0], S[1], S[2], V[0], V[1], V[2],
                        color=e.get("color", "#777777"), linewidth=all_delta_width, alpha=all_delta_alpha,
                        arrow_length_ratio=0.08,
                    )

        point_size = max(1.0, _safe_float(settings.get("point_size"), 12.0))
        point_alpha = min(1.0, max(0.0, _safe_float(settings.get("point_alpha"), 1.0)))
        selected_centroid_size = max(1.0, _safe_float(settings.get("selected_centroid_size"), 190.0))
        nonselected_centroid_size = max(1.0, _safe_float(settings.get("nonselected_centroid_size"), 95.0))
        label_font_size = max(1.0, _safe_float(settings.get("label_font_size"), 8.0))
        pc1_width = max(0.0, _safe_float(settings.get("pc1_width"), 1.4))
        label_mode = str(settings.get("complete_graph_label_mode", "selected") or "selected").strip().lower()
        if label_mode not in {"none", "selected", "all"}:
            label_mode = "selected" if bool(settings.get("complete_graph_show_labels", True)) else "none"
        show_labels = label_mode != "none"
        show_cloud_means = bool(settings.get("show_cloud_means", False))
        show_all_pc1 = bool(settings.get("complete_graph_show_pc1", True))

        for c in data.get("clusters", []):
            color = c.get("color", "#666666")
            P = np.asarray(c.get("cloud_3d", np.zeros((0, 3))), dtype=float)
            label = (
                f"{c.get('edge_code')} {c.get('endpoint_name')} "
                f"{c.get('doc_id')} C{c.get('cluster_label_display')} "
                f"segments {c.get('n_sampled_segments')}/{c.get('n_total_segments')}"
            )
            if P.size:
                ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=point_size, alpha=point_alpha, color=color, depthshade=False, label=label)
            C = np.asarray(c.get("centroid_3d"), dtype=float)
            if c.get("endpoint_name") == "src":
                marker = "o"
            elif c.get("endpoint_name") == "dst":
                marker = "^"
            else:
                marker = "D"
            is_selected = bool(c.get("is_selected_endpoint"))
            centroid_size = selected_centroid_size if is_selected else nonselected_centroid_size
            centroid_lw = 1.35 if is_selected else 0.65
            ax.scatter([C[0]], [C[1]], [C[2]], s=centroid_size, color=color, edgecolors="black", linewidths=centroid_lw, marker=marker, depthshade=True)

            M = np.asarray(c.get("cloud_mean_3d", np.full(3, np.nan)), dtype=float).reshape(-1)
            if show_cloud_means and M.size == 3 and np.isfinite(M).all():
                ax.plot(
                    [C[0], M[0]], [C[1], M[1]], [C[2], M[2]],
                    color="black", linestyle="--", linewidth=1.2, alpha=0.90,
                )
                ax.scatter(
                    [M[0]], [M[1]], [M[2]],
                    s=120, marker="X", color="black", edgecolors=color, linewidths=1.4,
                    depthshade=False, label=f"{c.get('key')} re-embedded cloud mean",
                )

            label_this_cluster = show_labels and (label_mode == "all" or bool(c.get("is_selected_endpoint")))
            if label_this_cluster:
                ax.text(
                    C[0], C[1], C[2],
                    f" {c.get('edge_code')}-{c.get('endpoint_name')}\n {c.get('doc_id')}\n C{c.get('cluster_label_display')}",
                    fontsize=label_font_size,
                    weight="bold" if bool(c.get("is_selected_endpoint")) else "normal",
                )
            pc1 = np.asarray(c.get("pc1_arrow_3d", np.zeros(3)), dtype=float)
            if float(np.linalg.norm(pc1)) > 1e-10 and ((not complete_graph_enabled) or show_all_pc1 or bool(c.get("is_selected_endpoint"))):
                pc1_alpha = 1.0 if bool(c.get("is_selected_endpoint")) else max(0.15, min(1.0, _safe_float(settings.get("complete_graph_edge_alpha"), 0.16) + 0.20))
                ax.quiver(
                    C[0], C[1], C[2], pc1[0], pc1[1], pc1[2],
                    color=color,
                    linewidth=(pc1_width if bool(c.get("is_selected_endpoint")) else max(0.25, pc1_width * 0.70)),
                    alpha=pc1_alpha,
                    arrow_length_ratio=0.18,
                )

        for e in data.get("edges", []):
            S = np.asarray(e.get("start_3d"), dtype=float)
            V = np.asarray(e.get("delta_arrow_3d"), dtype=float)
            if float(np.linalg.norm(V)) > 1e-10:
                ax.quiver(
                    S[0], S[1], S[2], V[0], V[1], V[2],
                    color=e.get("color", "#333333"),
                    linewidth=max(0.0, _safe_float(settings.get("selected_delta_width"), 3.0)),
                    alpha=min(1.0, max(0.0, _safe_float(settings.get("selected_delta_alpha"), 1.0))),
                    arrow_length_ratio=0.12,
                    label=f"{e.get('edge_code')} selected Δ displacement",
                )

        if coord_mode == "anchored_residual_raw_sbert_origins":
            ax.set_xlabel("anchored raw-SBERT PCA 1")
            ax.set_ylabel("anchored raw-SBERT PCA 2")
            ax.set_zlabel("anchored raw-SBERT PCA 3")
            ax.text2D(
                0.015, 0.02,
                "Raw anchors: cos=" + _fmt_float(row.get("raw_sbert_doc_cosine", data.get("raw_anchor_cosine_computed")), 4) +
                " · residual cos=" + _fmt_float(row.get("manifold_residual_doc_cosine", row.get("doc_embedding_cosine", "")), 4),
                transform=ax.transAxes,
                fontsize=8,
            )
        else:
            ax.set_xlabel("local residual-CDM PCA 1")
            ax.set_ylabel("local residual-CDM PCA 2")
            ax.set_zlabel("local residual-CDM PCA 3")
        ax.view_init(elev=24, azim=-52)
        _set_axes_equal_3d(ax)
        try:
            ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=7)
        except Exception:
            pass
        fig.tight_layout()

        self._clear_plot()
        self.current_figure = fig
        self.canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.grid(row=1, column=0, sticky="ew")
        self.summary.set_text(self._format_summary(data, elapsed))
        self.status_var.set(f"Rendered match {row.get('match_row')} in {elapsed:.2f}s using {mode_label}.")

    def _format_summary(self, data: Dict[str, Any], elapsed: float) -> str:
        row = data.get("row", {})
        settings = data.get("settings", {})
        lines = []
        lines.append("Selected edge-pair 3D visualization")
        lines.append("=" * 42)
        lines.append(f"Match row: {row.get('match_row')}")
        lines.append(f"Match type: {row.get('match_type')}")
        lines.append(f"Source morphism: {row.get('src_doc')} C{row.get('src_from')} → C{row.get('src_to')}  [edge {row.get('src_edge')}]" )
        lines.append(f"Target morphism: {row.get('tgt_doc')} C{row.get('tgt_from')} → C{row.get('tgt_to')}  [edge {row.get('tgt_edge')}]" )
        lines.append("")
        lines.append("Scores")
        for key in [
            "delta_cos", "src_pc1", "dst_pc1", "pc1_axis_value", "semantic_quality",
            "semantic_quality_min", "lexical_overlap_coefficient", "lexical_divergence",
            "alignment_core", "acuity_score", "manifold_residual_doc_cosine",
            "raw_sbert_doc_cosine", "doc_embedding_cosine",
        ]:
            if key in row:
                lines.append(f"  {key}: {_fmt_float(row.get(key), 6)}")
        lines.append("")
        lines.append("Rendering settings")
        lines.append(f"  SBERT model: {settings.get('model_name')}")
        lines.append(f"  device: {settings.get('device')}")
        lines.append(f"  batch size: {settings.get('batch_size')}")
        lines.append(f"  max points / cluster: {settings.get('max_points')}")
        lines.append(f"  Δ scale: {settings.get('delta_scale')}")
        lines.append(f"  PC1 scale × median(|Δ|): {settings.get('pc1_scale')}")
        lines.append(f"  coordinate mode: {settings.get('coordinate_mode')}")
        lines.append(f"  projection fit scope: {settings.get('projection_fit_scope')}")
        lines.append(f"  complete graph: {settings.get('complete_graph')}")
        if settings.get("complete_graph"):
            lines.append(f"  complete graph show all Δ: {settings.get('complete_graph_show_delta')}")
            lines.append(f"  complete graph show all PC1: {settings.get('complete_graph_show_pc1')}")
            lines.append(f"  complete graph labels: {settings.get('complete_graph_label_mode', 'selected')}")
            lines.append(f"  complete graph edge filter: {settings.get('complete_graph_edge_filter')}")
            lines.append(f"  complete graph max arrows/doc: {settings.get('complete_graph_max_arrows')}")
            lines.append(f"  all-Δ alpha/width: {settings.get('complete_graph_edge_alpha')} / {settings.get('complete_graph_edge_width')}")
        lines.append(f"  show cloud means: {settings.get('show_cloud_means')}")
        lines.append(f"  point size/alpha: {settings.get('point_size')} / {settings.get('point_alpha')}")
        lines.append(f"  selected Δ width/alpha: {settings.get('selected_delta_width')} / {settings.get('selected_delta_alpha')}")
        lines.append(f"  PC1 width: {settings.get('pc1_width')}")
        lines.append(f"  centroid sizes selected/other: {settings.get('selected_centroid_size')} / {settings.get('nonselected_centroid_size')}")
        lines.append(f"  label font size: {settings.get('label_font_size')}")
        lines.append(f"  anchor residual scale: {settings.get('resolved_anchor_residual_scale', settings.get('anchor_residual_scale'))}")
        lines.append(f"  PC1 canonicalized to edge flow: {settings.get('canonical_pc1')}")
        lines.append(f"  elapsed: {elapsed:.2f}s")
        if settings.get("coordinate_mode") == "anchored_residual_raw_sbert_origins":
            lines.append("")
            lines.append("Raw SBERT document anchors")
            lines.append(f"  computed raw-anchor cosine: {_fmt_float(data.get('raw_anchor_cosine_computed'), 6)}")
            lines.append(f"  computed raw-anchor L2 distance: {_fmt_float(data.get('raw_anchor_l2_computed'), 6)}")
            for a in data.get("document_anchors", []) or []:
                lines.append(
                    f"  {a.get('edge_code')}: {a.get('doc_id')} source={a.get('anchor_source')} "
                    f"norm={_fmt_float(np.linalg.norm(np.asarray(a.get('anchor'), dtype=float)), 6)}"
                )
        lines.append("")
        if settings.get("complete_graph"):
            docs = sorted({str(c.get("doc_id")) for c in data.get("clusters", [])})
            lines.append("Complete graph")
            lines.append(f"  documents rendered: {len(docs)}")
            lines.append(f"  clusters rendered: {len(data.get('clusters', []))}")
            lines.append(f"  faint directed Δ arrows rendered: {len(data.get('complete_edges', []))}")
            lines.append(f"  selected matched Δ arrows highlighted: {len(data.get('edges', []))}")
            lines.append("")
        lines.append("Endpoint clusters" if not settings.get("complete_graph") else "Rendered clusters")
        for c in data.get("clusters", []):
            lines.append(
                f"  {c.get('key')}: {c.get('doc_id')} C{c.get('cluster_label_display')} "
                f"Q={_fmt_float(c.get('quality'), 4)} segments plotted={c.get('n_sampled_segments')}/{c.get('n_total_segments')} "
                f"space={c.get('cloud_coordinate_space', '')}"
            )
            lines.append(
                f"      stored centroid norm={_fmt_float(c.get('centroid_norm'), 6)}; "
                f"re-embedded mean norm={_fmt_float(c.get('cloud_mean_norm'), 6)}"
            )
            lines.append(
                f"      stored centroid ↔ re-embedded mean: "
                f"cos={_fmt_float(c.get('centroid_cloud_cosine'), 6)}; "
                f"L2={_fmt_float(c.get('centroid_cloud_l2'), 6)}; "
                f"unit-L2={_fmt_float(c.get('centroid_cloud_unit_l2'), 6)}"
            )
            lines.append(
                f"      projected offset={_fmt_float(c.get('centroid_cloud_projected_l2'), 6)}; "
                f"cloud RMS radius={_fmt_float(c.get('cloud_projected_rms_radius'), 6)}; "
                f"offset/RMS={_fmt_float(c.get('centroid_cloud_projected_l2_over_rms'), 4)}"
            )
        lines.append("")
        lines.append("Legend")
        if settings.get("coordinate_mode") == "anchored_residual_raw_sbert_origins":
            lines.append("  large squares = raw mean-SBERT document origin anchors")
            lines.append("  dashed gray line = raw document-anchor separation")
            lines.append("  cluster points/centroids/arrows = residual CDM geometry translated to each raw anchor")
        if settings.get("complete_graph"):
            lines.append("  diamonds = non-selected cluster centroids in complete-document graph mode")
            lines.append("  faint thin arrows = all directed cluster-to-cluster Δ displacements")
            lines.append("  thick arrows = selected matched source/target Δ displacements")
        lines.append("  circles = stored source endpoint centroids; triangles = stored destination endpoint centroids")
        lines.append("  opaque colored points = re-embedded segment point clouds in reconstructed CDM build-space")
        lines.append("  black X markers = means of the re-embedded plotted point clouds")
        lines.append("  dashed black lines = stored centroid → re-embedded cloud-mean diagnostic offsets")
        lines.append("  thick arrows = source/destination cluster displacement vectors Δ")
        lines.append("  thin arrows = endpoint PC1 directions, optionally sign-canonicalized to the edge flow")
        warnings = data.get("warnings") or []
        if warnings:
            lines.append("")
            lines.append("Warnings")
            lines.extend(f"  - {w}" for w in warnings)
        return "\n".join(lines)

    def save_png(self) -> None:
        if self.current_figure is None:
            messagebox.showinfo("No figure", "Render an edge-match 3D figure first.")
            return
        row = self._row_from_entry()
        default = f"edge_match_3d_{row if row is not None else 'selected'}.png"
        path = filedialog.asksaveasfilename(
            title="Save edge-match 3D figure",
            defaultextension=".png",
            initialfile=default,
            filetypes=[("PNG image", "*.png"), ("SVG vector", "*.svg"), ("PDF", "*.pdf"), ("All files", "*.*")],
        )
        if not path:
            return
        self.current_figure.savefig(path, dpi=220, bbox_inches="tight")
        messagebox.showinfo("Saved", f"3D figure written to:\n{path}")



# -----------------------------------------------------------------------------
# Arrangement experiment helpers and UI
# -----------------------------------------------------------------------------

def _arr_unit(x: Any, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(x, dtype=float).reshape(-1)
    if v.size == 0:
        return v
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    n = float(np.linalg.norm(v))
    if n <= eps:
        return np.zeros_like(v, dtype=float)
    return v / n


def _arr_align_signs_to_flow(v: np.ndarray, s: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    s2 = np.asarray(s, dtype=float).copy()
    t2 = np.asarray(t, dtype=float).copy()
    if float(np.dot(s2, v)) < 0.0:
        s2 = -s2
    if float(np.dot(t2, -v)) < 0.0:
        t2 = -t2
    return v, s2, t2


def _arr_spherical_bin(v: np.ndarray, n_az: int = 12, n_el: int = 6) -> int:
    vv = _arr_unit(np.asarray(v, dtype=float).reshape(-1)[:3])
    if vv.size < 3:
        tmp = np.zeros(3, dtype=float)
        tmp[:vv.size] = vv
        vv = _arr_unit(tmp)
    phi = math.atan2(float(vv[1]), float(vv[0]))
    if phi < 0:
        phi += 2.0 * math.pi
    theta = math.asin(max(-1.0, min(1.0, float(vv[2]))))
    az_bin = min(max(1, int(n_az)) - 1, int((phi / (2.0 * math.pi)) * max(1, int(n_az))))
    el_bin = min(max(1, int(n_el)) - 1, int(((theta + math.pi / 2.0) / math.pi) * max(1, int(n_el))))
    return int(el_bin * max(1, int(n_az)) + az_bin)


def _arr_cluster_quality(cdm_tuple: Any, label: Any, default: float = 1.0) -> float:
    try:
        q_payload = cdm_tuple[6] if isinstance(cdm_tuple, (tuple, list)) and len(cdm_tuple) >= 7 else None
        candidates: List[Any] = [label, str(label)]
        try:
            candidates.append(int(label))
        except Exception:
            pass
        if isinstance(q_payload, dict):
            for key in candidates:
                if key in q_payload:
                    rec = q_payload[key]
                    val = rec.get('quality', default) if isinstance(rec, dict) else rec
                    return max(0.0, min(1.0, float(val)))
        elif q_payload is not None:
            order = list(cdm_tuple[1])
            for i, lab in enumerate(order):
                if _label_equal(lab, label):
                    arr = list(q_payload)
                    if 0 <= i < len(arr):
                        rec = arr[i]
                        val = rec.get('quality', default) if isinstance(rec, dict) else rec
                        return max(0.0, min(1.0, float(val)))
    except Exception:
        pass
    return float(default)


def _arr_extract_shapes_from_cdm_dict(
    document_delta_dict: Dict[Any, Any],
    *,
    max_edges_per_doc: Optional[int] = 2000,
    min_weight: float = 0.0,
    include_length: bool = True,
    include_v_bin: bool = False,
    vbin_az: int = 12,
    vbin_el: int = 6,
    align_signs: bool = True,
    dir_weight_beta: float = 1.0,
    status_callback: Any = None,
) -> Tuple[List[Dict[str, Any]], np.ndarray, List[str]]:
    rows: List[Dict[str, Any]] = []
    feats: List[List[float]] = []
    errors: List[str] = []
    docs = list(document_delta_dict.keys()) if isinstance(document_delta_dict, dict) else []
    nbins = max(1, int(vbin_az)) * max(1, int(vbin_el)) if include_v_bin else 0
    feat_dim = 3 + (1 if include_length else 0) + nbins

    for doc_pos, doc_id in enumerate(docs):
        if status_callback and (doc_pos == 0 or (doc_pos + 1) % 50 == 0 or doc_pos + 1 == len(docs)):
            try:
                status_callback(f"Extracting shape records from document {doc_pos + 1:,}/{len(docs):,} ...")
            except Exception:
                pass
        try:
            tup = document_delta_dict[doc_id]
            if not isinstance(tup, (tuple, list)) or len(tup) < 6:
                errors.append(f"{doc_id}: CDM tuple has fewer than 6 elements")
                continue
            Delta, cluster_order, _seg_labels, _topic_dists, _cluster_embeddings, cluster_dirs = tup[:6]
            Delta = np.asarray(Delta, dtype=float)
            D = np.asarray(cluster_dirs, dtype=float)
            order = list(cluster_order.tolist() if hasattr(cluster_order, 'tolist') else cluster_order)
            if Delta.ndim != 3 or Delta.shape[0] != Delta.shape[1] or Delta.shape[0] != len(order):
                errors.append(f"{doc_id}: malformed delta tensor/order")
                continue
            n = int(Delta.shape[0])
            if D.ndim != 2 or D.shape[0] != n:
                D = np.zeros((n, Delta.shape[2]), dtype=float)

            norms: List[float] = []
            for i in range(n):
                for j in range(n):
                    if i != j:
                        ln = float(np.linalg.norm(Delta[i, j]))
                        if np.isfinite(ln) and ln > 0.0:
                            norms.append(ln)
            weight_scale = max(1e-8, float(np.median(norms)) if norms else 1.0)

            candidates: List[Tuple[float, int, int, float, np.ndarray, np.ndarray, np.ndarray]] = []
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    dvec = np.asarray(Delta[i, j], dtype=float).reshape(-1)
                    ln = float(np.linalg.norm(dvec))
                    if not np.isfinite(ln) or ln <= 0.0:
                        continue
                    v = _arr_unit(dvec)
                    s = _arr_unit(D[i]) if D.size else np.zeros_like(v)
                    t = _arr_unit(D[j]) if D.size else np.zeros_like(v)
                    if align_signs:
                        v, s, t = _arr_align_signs_to_flow(v, s, t)
                    a1 = max(0.0, float(np.dot(v, s)))
                    a2 = max(0.0, float(np.dot(-v, t)))
                    a3 = abs(float(np.dot(s, t)))
                    dir_score = a1 * a2 * math.sqrt(max(0.0, a3))
                    base_w = math.exp(-ln / weight_scale)
                    w = base_w * ((dir_score + 1e-12) ** float(dir_weight_beta))
                    if w >= float(min_weight):
                        candidates.append((float(w), i, j, float(ln), v, s, t))
            if max_edges_per_doc is not None and int(max_edges_per_doc) > 0 and len(candidates) > int(max_edges_per_doc):
                candidates.sort(key=lambda x: x[0], reverse=True)
                candidates = candidates[: int(max_edges_per_doc)]

            for w, i, j, ln, v, s, t in candidates:
                cos_vs = float(np.dot(v, s))
                cos_vt_in = float(np.dot(-v, t))
                cos_st = float(np.dot(s, t))
                feat: List[float] = [cos_vs, cos_vt_in, cos_st]
                if include_length:
                    feat.append(float(math.log(ln + 1e-9)))
                vbin = -1
                if include_v_bin:
                    vbin = _arr_spherical_bin(v, n_az=int(vbin_az), n_el=int(vbin_el))
                    oh = [0.0] * nbins
                    if 0 <= vbin < nbins:
                        oh[vbin] = 1.0
                    feat.extend(oh)
                if len(feat) != feat_dim:
                    feat = (feat + [0.0] * feat_dim)[:feat_dim]
                src_lab = order[i]
                dst_lab = order[j]
                q_src = _arr_cluster_quality(tup, src_lab)
                q_dst = _arr_cluster_quality(tup, dst_lab)
                rows.append({
                    'shape_row': len(rows),
                    'doc_id': str(doc_id),
                    'src_label': int(src_lab) if str(src_lab).lstrip('-').isdigit() else src_lab,
                    'dst_label': int(dst_lab) if str(dst_lab).lstrip('-').isdigit() else dst_lab,
                    'w': float(w),
                    'len_delta': float(ln),
                    'cos_vs': cos_vs,
                    'cos_vt_in': cos_vt_in,
                    'cos_st': cos_st,
                    'vbin': int(vbin),
                    'src_quality': float(q_src),
                    'dst_quality': float(q_dst),
                    'edge_quality': float(min(q_src, q_dst)),
                })
                feats.append(feat)
        except Exception as ex:
            errors.append(f"{doc_id}: {ex}")

    X = np.asarray(feats, dtype=np.float32) if feats else np.zeros((0, feat_dim), dtype=np.float32)
    return rows, X, errors


def _arr_kmeans_np(X: np.ndarray, k: int, n_iter: int = 50, random_state: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        X = X.reshape(-1, X.shape[-1]) if X.size else np.zeros((0, 0), dtype=float)
    n, d = X.shape
    if n == 0:
        return np.zeros((0,), dtype=int), np.zeros((max(1, int(k)), d), dtype=float)
    k_eff = max(1, min(int(k), n))
    rng = np.random.default_rng(int(random_state))
    centroids = np.empty((k_eff, d), dtype=float)
    first = int(rng.integers(0, n))
    centroids[0] = X[first]
    dist_sq = np.sum((X - centroids[0]) ** 2, axis=1)
    for c in range(1, k_eff):
        dist_sq = np.maximum(dist_sq, 0.0)
        total = float(dist_sq.sum())
        if (not np.isfinite(total)) or total <= 0.0:
            idx = int(rng.integers(0, n))
        else:
            probs = dist_sq / total
            if not np.isfinite(probs).all() or float(probs.sum()) <= 0.0:
                idx = int(rng.integers(0, n))
            else:
                probs = probs / float(probs.sum())
                idx = int(rng.choice(n, p=probs))
        centroids[c] = X[idx]
        new_dist_sq = np.sum((X - centroids[c]) ** 2, axis=1)
        dist_sq = np.minimum(dist_sq, new_dist_sq)
    labels = np.zeros(n, dtype=int)
    for it in range(max(1, int(n_iter))):
        dists = np.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(dists, axis=1)
        if it > 0 and np.all(new_labels == labels):
            break
        labels = new_labels
        for c in range(k_eff):
            mask = labels == c
            if np.any(mask):
                centroids[c] = X[mask].mean(axis=0)
            else:
                centroids[c] = X[int(rng.integers(0, n))]
    return labels.astype(int), centroids.astype(np.float32)


def _arr_doc_membership(rows: List[Dict[str, Any]], shape_labels: np.ndarray, k: int, weight_mode: str = 'weighted', normalize: bool = True) -> Tuple[np.ndarray, List[str]]:
    doc_ids: List[str] = []
    doc_to_idx: Dict[str, int] = {}
    for r in rows:
        d = str(r.get('doc_id', ''))
        if d not in doc_to_idx:
            doc_to_idx[d] = len(doc_ids)
            doc_ids.append(d)
    M = np.zeros((len(doc_ids), max(1, int(k))), dtype=np.float32)
    for r, lab in zip(rows, np.asarray(shape_labels, dtype=int)):
        d = str(r.get('doc_id', ''))
        if d not in doc_to_idx or not (0 <= int(lab) < M.shape[1]):
            continue
        weight = 1.0 if str(weight_mode).lower().strip() == 'count' else float(r.get('w', 1.0) or 0.0)
        M[doc_to_idx[d], int(lab)] += float(weight)
    if normalize and M.size:
        row_sum = M.sum(axis=1, keepdims=True)
        M = np.divide(M, np.maximum(row_sum, 1e-12), out=np.zeros_like(M), where=row_sum > 0)
    return M, doc_ids


def _arr_collection_labels(doc_ids: Sequence[str], source: str, regex: str = r'^([^_]+_[^_]+)_', csv_path: str = '', unknown: str = 'Unknown') -> List[str]:
    source = str(source or 'doc_id_regex').strip().lower()
    unknown = str(unknown or 'Unknown')
    mapping: Dict[str, str] = {}
    if source == 'metadata_csv' and csv_path:
        try:
            with open(csv_path, 'r', encoding='utf-8-sig', newline='') as f:
                reader = csv.DictReader(f)
                if reader.fieldnames:
                    fields = list(reader.fieldnames)
                    doc_col = 'doc_id' if 'doc_id' in fields else fields[0]
                    type_col = 'collection_type' if 'collection_type' in fields else (fields[1] if len(fields) > 1 else fields[0])
                    for row in reader:
                        mapping[str(row.get(doc_col, '')).strip()] = str(row.get(type_col, unknown)).strip() or unknown
        except Exception:
            mapping = {}
    labels: List[str] = []
    pat = None
    if source == 'doc_id_regex':
        try:
            pat = re.compile(regex or r'^([^_]+_[^_]+)_')
        except Exception:
            pat = re.compile(r'^([^_]+_[^_]+)_')
    for d in doc_ids:
        ds = str(d)
        if source == 'metadata_csv':
            labels.append(mapping.get(ds, unknown))
        elif source == 'none':
            labels.append(unknown)
        elif source == 'doc_id_regex':
            m = pat.search(ds) if pat is not None else None
            labels.append((m.group(1) if m else unknown).strip() or unknown)
        else:
            labels.append(ds.split('_', 1)[0] if '_' in ds else unknown)
    return labels


def _arr_cosine_similarity_matrix(M: np.ndarray) -> np.ndarray:
    X = np.asarray(M, dtype=float)
    if X.ndim != 2 or X.shape[0] == 0:
        return np.zeros((0, 0), dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    Xn = np.divide(X, np.maximum(norm, 1e-12), out=np.zeros_like(X), where=norm > 0)
    S = np.clip(Xn @ Xn.T, -1.0, 1.0)
    return S.astype(np.float32)


def _arr_js_similarity_matrix(M: np.ndarray) -> np.ndarray:
    X = np.asarray(M, dtype=float)
    if X.ndim != 2 or X.shape[0] == 0:
        return np.zeros((0, 0), dtype=np.float32)
    X = np.maximum(np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    row = X.sum(axis=1, keepdims=True)
    P = np.divide(X, np.maximum(row, 1e-12), out=np.zeros_like(X), where=row > 0)
    n = P.shape[0]
    S = np.eye(n, dtype=float)
    eps = 1e-12
    for i in range(n):
        p = P[i]
        for j in range(i + 1, n):
            q = P[j]
            m = 0.5 * (p + q)
            kl_pm = np.sum(np.where(p > 0, p * (np.log(p + eps) - np.log(m + eps)), 0.0))
            kl_qm = np.sum(np.where(q > 0, q * (np.log(q + eps) - np.log(m + eps)), 0.0))
            js = 0.5 * (kl_pm + kl_qm)
            sim = 1.0 - math.sqrt(max(0.0, js) / math.log(2.0))
            S[i, j] = S[j, i] = max(0.0, min(1.0, sim))
    return S.astype(np.float32)


def _arr_pca_2d(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] == 0:
        return np.zeros((0, 2), dtype=float)
    if X.shape[0] == 1:
        return np.zeros((1, 2), dtype=float)
    Xc = X - np.nanmean(X, axis=0, keepdims=True)
    Xc = np.nan_to_num(Xc, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        _, _, vt = np.linalg.svd(Xc, full_matrices=False)
        B = vt[: min(2, vt.shape[0])].T
        Y = Xc @ B
        if Y.shape[1] == 1:
            Y = np.column_stack([Y[:, 0], np.zeros(Y.shape[0])])
        return Y[:, :2]
    except Exception:
        return np.column_stack([np.arange(X.shape[0], dtype=float), np.zeros(X.shape[0])])


def _arr_graph_edges_from_similarity(S: np.ndarray, top_k: int = 5, threshold: float = 0.0) -> List[Tuple[int, int, float]]:
    S = np.asarray(S, dtype=float)
    n = S.shape[0] if S.ndim == 2 else 0
    edges: Dict[Tuple[int, int], float] = {}
    for i in range(n):
        vals = S[i].copy()
        vals[i] = -np.inf
        if int(top_k) > 0:
            cand = np.argsort(vals)[::-1][: int(top_k)]
        else:
            cand = np.where(vals >= float(threshold))[0]
        for j in cand:
            w = float(vals[j])
            if not np.isfinite(w) or w < float(threshold):
                continue
            a, b = sorted((int(i), int(j)))
            if a == b:
                continue
            edges[(a, b)] = max(edges.get((a, b), -np.inf), w)
    return [(a, b, w) for (a, b), w in sorted(edges.items(), key=lambda x: x[1], reverse=True)]




def _arr_shape_summary_lookup(shape_summary: Sequence[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    """Return shape_id -> summary row for neighbor display/enrichment."""
    out: Dict[int, Dict[str, Any]] = {}
    for r in shape_summary or []:
        try:
            out[int(r.get('shape_id', -1))] = dict(r)
        except Exception:
            continue
    return out


def _arr_pairwise_cosine_rows(X: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity between rows of X, with zero rows guarded."""
    A = np.asarray(X, dtype=float)
    if A.ndim != 2 or A.shape[0] == 0:
        return np.zeros((0, 0), dtype=np.float32)
    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    nrm = np.linalg.norm(A, axis=1, keepdims=True)
    An = np.divide(A, np.maximum(nrm, 1e-12), out=np.zeros_like(A), where=nrm > 0)
    return np.clip(An @ An.T, -1.0, 1.0).astype(np.float32)


def _arr_pairwise_l2_rows(X: np.ndarray) -> np.ndarray:
    """Pairwise Euclidean distance between rows of X."""
    A = np.asarray(X, dtype=float)
    if A.ndim != 2 or A.shape[0] == 0:
        return np.zeros((0, 0), dtype=np.float32)
    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    diff = A[:, None, :] - A[None, :, :]
    return np.sqrt(np.maximum(0.0, np.sum(diff * diff, axis=2))).astype(np.float32)


def _arr_collection_profiles(M: np.ndarray, collection_labels: Sequence[str]) -> Tuple[np.ndarray, List[str]]:
    """
    Build shape × collection profile matrix from document × shape membership.

    profile[s, c] = mean document membership in shape s for documents from
    collection label c.  These profiles let us ask whether two shape categories
    are distributed across collection types in similar ways, independent of
    whether their centroids are geometrically close.
    """
    X = np.asarray(M, dtype=float)
    if X.ndim != 2 or X.shape[1] == 0:
        return np.zeros((0, 0), dtype=np.float32), []
    coll = [str(c) if str(c).strip() else 'Unknown' for c in collection_labels]
    types = sorted(set(coll)) if coll else []
    P = np.zeros((X.shape[1], len(types)), dtype=np.float32)
    for j, t in enumerate(types):
        mask = np.asarray([c == t for c in coll], dtype=bool)
        if mask.shape[0] == X.shape[0] and np.any(mask):
            P[:, j] = np.nan_to_num(X[mask].mean(axis=0), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return P, types



def _arr_collection_profile_analysis(experiment: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute collection-level summaries over document × shape membership.

    Outputs:
      - dominant_shape_distribution: for each collection, how many documents have
        each shape as their dominant membership category, plus mean membership.
      - collection_profile_similarity: collection × collection similarity matrices
        using both cosine and Jensen-Shannon similarity over average shape profiles.

    These tables make the membership heatmap's collection-scale bands auditable:
    users can distinguish display-order effects from actual profile similarity.
    """
    if not isinstance(experiment, dict):
        return {}
    M = np.asarray(experiment.get('doc_shape_membership', []), dtype=float)
    docs = [str(d) for d in experiment.get('doc_ids', [])]
    coll = [str(c) if str(c).strip() else 'Unknown' for c in experiment.get('collection_type', [])]
    if M.ndim != 2 or M.size == 0 or not docs:
        return {}
    n_docs, k = M.shape
    if len(coll) < n_docs:
        coll = coll + ['Unknown'] * (n_docs - len(coll))
    coll = coll[:n_docs]
    collection_types = sorted(set(coll))
    c_count = len(collection_types)
    type_to_idx = {c: i for i, c in enumerate(collection_types)}

    profile = np.zeros((c_count, k), dtype=np.float32)
    n_docs_by_type = np.zeros(c_count, dtype=np.int32)
    dom_count = np.zeros((c_count, k), dtype=np.int32)
    dom_shape = np.argmax(M, axis=1) if k else np.zeros(n_docs, dtype=np.int32)

    for c in collection_types:
        ci = type_to_idx[c]
        mask = np.asarray([x == c for x in coll], dtype=bool)
        n_docs_by_type[ci] = int(mask.sum())
        if np.any(mask):
            profile[ci] = np.nan_to_num(M[mask].mean(axis=0), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            for sid, cnt in Counter(int(x) for x in dom_shape[mask]).items():
                if 0 <= sid < k:
                    dom_count[ci, sid] = int(cnt)

    dom_fraction = np.divide(
        dom_count.astype(float),
        np.maximum(n_docs_by_type[:, None].astype(float), 1.0),
        out=np.zeros_like(dom_count, dtype=float),
        where=n_docs_by_type[:, None] > 0,
    ).astype(np.float32)

    rows: List[Dict[str, Any]] = []
    for ci, c in enumerate(collection_types):
        mean_order = np.argsort(profile[ci])[::-1]
        dom_order = np.argsort(dom_fraction[ci])[::-1]
        mean_rank = {int(s): int(r + 1) for r, s in enumerate(mean_order)}
        dom_rank = {int(s): int(r + 1) for r, s in enumerate(dom_order)}
        for sid in range(k):
            rows.append({
                'collection_type': c,
                'shape_id': int(sid),
                'dominant_doc_count': int(dom_count[ci, sid]),
                'dominant_doc_fraction': float(dom_fraction[ci, sid]),
                'mean_membership': float(profile[ci, sid]),
                'rank_by_mean_membership': int(mean_rank.get(sid, k)),
                'rank_by_dominant_fraction': int(dom_rank.get(sid, k)),
                'n_docs': int(n_docs_by_type[ci]),
            })

    cosine = _arr_cosine_similarity_matrix(profile)
    js = _arr_js_similarity_matrix(profile)
    pair_rows: List[Dict[str, Any]] = []
    for i in range(c_count):
        for j in range(i + 1, c_count):
            pair_rows.append({
                'collection_a': collection_types[i],
                'collection_b': collection_types[j],
                'cosine_similarity': float(cosine[i, j]) if cosine.size else 0.0,
                'jensen_shannon_similarity': float(js[i, j]) if js.size else 0.0,
                'n_docs_a': int(n_docs_by_type[i]),
                'n_docs_b': int(n_docs_by_type[j]),
            })
    pair_rows.sort(key=lambda r: (float(r.get('cosine_similarity', 0.0)), float(r.get('jensen_shannon_similarity', 0.0))), reverse=True)

    return {
        'dominant_shape_distribution': {
            'collection_types': collection_types,
            'shape_ids': [int(i) for i in range(k)],
            'dominant_count_matrix': dom_count.astype(np.int32),
            'dominant_fraction_matrix': dom_fraction.astype(np.float32),
            'mean_membership_matrix': profile.astype(np.float32),
            'n_docs_by_collection': n_docs_by_type.astype(np.int32),
            'rows': rows,
        },
        'collection_profile_similarity': {
            'collection_types': collection_types,
            'shape_ids': [int(i) for i in range(k)],
            'profile_matrix': profile.astype(np.float32),
            'cosine_matrix': cosine.astype(np.float32),
            'jensen_shannon_matrix': js.astype(np.float32),
            'pair_rows': pair_rows,
        },
    }



# -----------------------------------------------------------------------------
# Arrangement ROC / separability helpers
# -----------------------------------------------------------------------------

def _arr_cosine_score_vector(M: np.ndarray, profile: np.ndarray) -> np.ndarray:
    """Cosine similarity between each row of M and one profile vector."""
    X = np.asarray(M, dtype=float)
    p = np.asarray(profile, dtype=float).reshape(-1)
    if X.ndim != 2 or p.size != (X.shape[1] if X.ndim == 2 else -1):
        return np.zeros((X.shape[0] if X.ndim == 2 else 0,), dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    pn = float(np.linalg.norm(p))
    xn = np.linalg.norm(X, axis=1)
    if pn <= 1e-12:
        return np.zeros((X.shape[0],), dtype=np.float32)
    out = np.divide(X @ p, np.maximum(xn * pn, 1e-12), out=np.zeros((X.shape[0],), dtype=float), where=(xn > 0))
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def _arr_js_similarity_to_profile(M: np.ndarray, profile: np.ndarray) -> np.ndarray:
    """Jensen-Shannon similarity between each nonnegative row of M and one profile."""
    X = np.asarray(M, dtype=float)
    p = np.asarray(profile, dtype=float).reshape(-1)
    if X.ndim != 2 or p.size != (X.shape[1] if X.ndim == 2 else -1):
        return np.zeros((X.shape[0] if X.ndim == 2 else 0,), dtype=np.float32)
    X = np.maximum(np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    p = np.maximum(np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    row = X.sum(axis=1, keepdims=True)
    P = np.divide(X, np.maximum(row, 1e-12), out=np.zeros_like(X), where=row > 0)
    ps = float(p.sum())
    q = p / max(ps, 1e-12) if ps > 0 else np.zeros_like(p)
    Mmid = 0.5 * (P + q[None, :])
    eps = 1e-12
    kl_pm = np.sum(np.where(P > 0, P * (np.log(P + eps) - np.log(Mmid + eps)), 0.0), axis=1)
    kl_qm = np.sum(np.where(q[None, :] > 0, q[None, :] * (np.log(q[None, :] + eps) - np.log(Mmid + eps)), 0.0), axis=1)
    js = 0.5 * (kl_pm + kl_qm)
    sim = 1.0 - np.sqrt(np.maximum(0.0, js) / math.log(2.0))
    return np.clip(sim, 0.0, 1.0).astype(np.float32)


def _arr_js_similarity_pair(a: np.ndarray, b: np.ndarray) -> float:
    """Jensen-Shannon similarity for two nonnegative membership vectors."""
    pa = np.maximum(np.nan_to_num(np.asarray(a, dtype=float).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    pb = np.maximum(np.nan_to_num(np.asarray(b, dtype=float).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    if pa.size == 0 or pa.size != pb.size:
        return 0.0
    sa = float(pa.sum()); sb = float(pb.sum())
    if sa <= 0.0 or sb <= 0.0:
        return 0.0
    pa = pa / sa; pb = pb / sb
    m = 0.5 * (pa + pb)
    eps = 1e-12
    kl_a = float(np.sum(np.where(pa > 0, pa * (np.log(pa + eps) - np.log(m + eps)), 0.0)))
    kl_b = float(np.sum(np.where(pb > 0, pb * (np.log(pb + eps) - np.log(m + eps)), 0.0)))
    js = 0.5 * (kl_a + kl_b)
    return float(max(0.0, min(1.0, 1.0 - math.sqrt(max(0.0, js) / math.log(2.0)))))




def _arr_pairwise_js_similarity_rows(X: np.ndarray) -> np.ndarray:
    """Pairwise Jensen-Shannon similarity for nonnegative row vectors."""
    A = np.asarray(X, dtype=float)
    if A.ndim != 2 or A.size == 0:
        return np.zeros((0, 0), dtype=np.float32)
    A = np.maximum(np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    rs = A.sum(axis=1, keepdims=True)
    P = np.divide(A, np.maximum(rs, 1e-12), out=np.zeros_like(A), where=rs > 0)
    n = P.shape[0]
    out = np.eye(n, dtype=np.float32)
    eps = 1e-12
    log2 = math.log(2.0)
    for i in range(n):
        pi = P[i]
        for j in range(i + 1, n):
            pj = P[j]
            m = 0.5 * (pi + pj)
            kl_i = float(np.sum(np.where(pi > 0, pi * (np.log(pi + eps) - np.log(m + eps)), 0.0)))
            kl_j = float(np.sum(np.where(pj > 0, pj * (np.log(pj + eps) - np.log(m + eps)), 0.0)))
            js = 0.5 * (kl_i + kl_j)
            sim = 1.0 - math.sqrt(max(0.0, js) / log2)
            out[i, j] = out[j, i] = np.float32(max(0.0, min(1.0, sim)))
    return out


def _arr_rankdata_average(x: np.ndarray) -> np.ndarray:
    """Small scipy-free average-rank helper for Spearman correlations."""
    a = np.asarray(x, dtype=float).reshape(-1)
    if a.size == 0:
        return a
    order = np.argsort(a, kind='mergesort')
    ranks = np.empty(a.size, dtype=float)
    i = 0
    while i < a.size:
        j = i + 1
        while j < a.size and a[order[j]] == a[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def _arr_corr_pair(a: Any, b: Any, method: str = 'pearson') -> Tuple[float, int]:
    """Correlation between two score vectors with finite filtering."""
    x = np.asarray(a, dtype=float).reshape(-1)
    y = np.asarray(b, dtype=float).reshape(-1)
    n = min(x.size, y.size)
    if n <= 1:
        return float('nan'), 0
    x = x[:n]; y = y[:n]
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]; y = y[mask]
    if x.size < 3:
        return float('nan'), int(x.size)
    if str(method).lower().startswith('spear'):
        x = _arr_rankdata_average(x)
        y = _arr_rankdata_average(y)
    sx = float(np.std(x)); sy = float(np.std(y))
    if sx <= 1e-12 or sy <= 1e-12:
        return float('nan'), int(x.size)
    return float(np.corrcoef(x, y)[0, 1]), int(x.size)


def _arr_doc_embedding_payload_vector(cdm_tuple: Any, kind: str) -> Tuple[np.ndarray, bool]:
    """Extract a unit-normalized raw/residual document vector from a CDM tuple payload."""
    if not isinstance(cdm_tuple, (tuple, list)) or len(cdm_tuple) < 8 or not isinstance(cdm_tuple[7], dict):
        return np.zeros((0,), dtype=np.float32), False
    payload = cdm_tuple[7]
    if str(kind).lower().startswith('raw'):
        keys = ['raw_sbert_document_embedding', 'raw_document_embedding', 'raw_sbert_doc_embedding']
    else:
        keys = ['manifold_residual_document_embedding', 'residual_document_embedding', 'document_embedding']
    for key in keys:
        if key in payload:
            v = _unit_vector(payload.get(key)).astype(np.float32, copy=False)
            if v.size:
                return v, True
    return np.zeros((0,), dtype=np.float32), False


def _arr_document_embedding_tables_from_cdm_dict(document_delta_dict: Dict[Any, Any], doc_ids: Sequence[str]) -> Dict[str, Any]:
    """Build doc-aligned raw SBERT and manifold-residual embedding tables from CDM payloads."""
    by_str = {str(k): k for k in document_delta_dict.keys()} if isinstance(document_delta_dict, dict) else {}
    out: Dict[str, Any] = {
        'kind': 'arrangement_document_embedding_tables',
        'version': 1,
        'doc_ids': [str(d) for d in doc_ids],
    }
    for kind in ('raw_sbert', 'manifold_residual'):
        vecs: List[np.ndarray] = []
        avail: List[int] = []
        dim = 0
        extracted: List[Tuple[np.ndarray, bool]] = []
        for doc in doc_ids:
            key = by_str.get(str(doc), str(doc))
            cdm = document_delta_dict.get(key) if isinstance(document_delta_dict, dict) else None
            v, ok = _arr_doc_embedding_payload_vector(cdm, kind)
            if ok and v.size and dim == 0:
                dim = int(v.size)
            extracted.append((v, ok))
        for v, ok in extracted:
            if ok and dim and v.size == dim:
                vecs.append(v.astype(np.float32, copy=False)); avail.append(1)
            else:
                vecs.append(np.zeros((dim,), dtype=np.float32)); avail.append(0)
        if dim == 0:
            arr = np.zeros((len(doc_ids), 0), dtype=np.float32)
        else:
            arr = np.vstack(vecs).astype(np.float32, copy=False)
        out[kind] = {
            'vectors': arr,
            'available': np.asarray(avail, dtype=np.uint8),
            'n_available': int(np.sum(avail)),
            'dim': int(dim),
            'source': 'document_delta_dict_cdm_payload',
        }
    return out


def _arr_get_document_embedding_matrix(experiment: Dict[str, Any], kind: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
    """Return doc-aligned embedding matrix and availability mask from an arrangement artifact."""
    emb = experiment.get('document_embeddings') if isinstance(experiment, dict) else None
    if not isinstance(emb, dict):
        return None, None, ''
    k = 'raw_sbert' if str(kind).lower().startswith('raw') else 'manifold_residual'
    table = emb.get(k)
    if not isinstance(table, dict):
        return None, None, ''
    try:
        V = np.asarray(table.get('vectors', []), dtype=float)
    except Exception:
        return None, None, ''
    if V.ndim != 2 or V.shape[0] == 0 or V.shape[1] == 0:
        return None, None, ''
    avail = np.asarray(table.get('available', np.ones((V.shape[0],), dtype=np.uint8))).reshape(-1)
    if avail.size < V.shape[0]:
        tmp = np.zeros((V.shape[0],), dtype=np.uint8)
        tmp[:avail.size] = avail.astype(np.uint8)
        avail = tmp
    avail = (avail[:V.shape[0]].astype(bool) & np.isfinite(V).all(axis=1) & (np.linalg.norm(V, axis=1) > 1e-12))
    V = np.nan_to_num(V, nan=0.0, posinf=0.0, neginf=0.0)
    return V.astype(np.float32, copy=False), avail, str(table.get('source', k))


def _arr_profile_scores(
    X: np.ndarray,
    coll: Sequence[str],
    target: str,
    *,
    score_mode: str = 'cosine',
    availability: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Leave-one-out one-vs-rest scores for a representation matrix."""
    A = np.asarray(X, dtype=float)
    n = A.shape[0] if A.ndim == 2 else 0
    scores = np.full((n,), np.nan, dtype=np.float32)
    if A.ndim != 2 or n == 0:
        return scores
    labels = np.asarray([str(c) == str(target) for c in coll[:n]], dtype=bool)
    avail = np.ones((n,), dtype=bool) if availability is None else np.asarray(availability, dtype=bool).reshape(-1)[:n]
    if avail.size < n:
        tmp = np.zeros((n,), dtype=bool); tmp[:avail.size] = avail; avail = tmp
    pos_avail = labels & avail
    n_pos = int(pos_avail.sum())
    if n_pos <= 0:
        return scores
    pos_sum = A[pos_avail].sum(axis=0)
    full_profile = pos_sum / max(1, n_pos)
    for i in range(n):
        if not avail[i]:
            continue
        if labels[i] and n_pos > 1:
            profile = (pos_sum - A[i]) / max(1, n_pos - 1)
        else:
            profile = full_profile
        if str(score_mode).lower().startswith('js') or 'jensen' in str(score_mode).lower():
            scores[i] = float(_arr_js_similarity_to_profile(A[i:i+1], profile)[0])
        else:
            scores[i] = float(_arr_cosine_score_vector(A[i:i+1], profile)[0])
    return scores


def _arr_representation_specs(experiment: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Representations available for collection ROC baseline comparison."""
    specs: Dict[str, Dict[str, Any]] = {}
    M = np.asarray(experiment.get('doc_shape_membership', []), dtype=float) if isinstance(experiment, dict) else np.zeros((0, 0))
    if M.ndim == 2 and M.size:
        M = np.maximum(np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
        specs['shape_membership_cosine'] = {
            'label': 'Shape membership cosine', 'matrix': M, 'score_mode': 'cosine',
            'available': np.ones((M.shape[0],), dtype=bool), 'family': 'shape_membership',
        }
        specs['shape_membership_jensen_shannon'] = {
            'label': 'Shape membership Jensen-Shannon', 'matrix': M, 'score_mode': 'jensen_shannon',
            'available': np.ones((M.shape[0],), dtype=bool), 'family': 'shape_membership',
        }
    raw, raw_avail, raw_src = _arr_get_document_embedding_matrix(experiment, 'raw_sbert')
    if raw is not None and raw_avail is not None and int(np.sum(raw_avail)) >= 2:
        specs['raw_sbert_cosine'] = {
            'label': 'Raw SBERT doc cosine', 'matrix': raw, 'score_mode': 'cosine',
            'available': raw_avail, 'family': 'raw_sbert', 'source': raw_src,
        }
    resid, resid_avail, resid_src = _arr_get_document_embedding_matrix(experiment, 'manifold_residual')
    if resid is not None and resid_avail is not None and int(np.sum(resid_avail)) >= 2:
        specs['manifold_residual_cosine'] = {
            'label': 'Manifold-residual doc cosine', 'matrix': resid, 'score_mode': 'cosine',
            'available': resid_avail, 'family': 'manifold_residual', 'source': resid_src,
        }
    return specs


def _arr_pair_score_from_spec(spec: Dict[str, Any], i: int, j: int) -> float:
    X = np.asarray(spec.get('matrix', []), dtype=float)
    if X.ndim != 2 or i >= X.shape[0] or j >= X.shape[0]:
        return float('nan')
    avail = np.asarray(spec.get('available', np.ones((X.shape[0],), dtype=bool)), dtype=bool).reshape(-1)
    if avail.size < X.shape[0] or not (bool(avail[i]) and bool(avail[j])):
        return float('nan')
    if str(spec.get('score_mode', 'cosine')).lower().startswith('jensen'):
        return float(_arr_js_similarity_pair(X[i], X[j]))
    return float(_arr_cosine_score_vector(X[i:i+1], X[j])[0])


def _arr_representation_pair_scores(experiment: Dict[str, Any], pair_idx: Sequence[Tuple[int, int]]) -> Dict[str, np.ndarray]:
    specs = _arr_representation_specs(experiment)
    out: Dict[str, np.ndarray] = {}
    for name, spec in specs.items():
        vals = [ _arr_pair_score_from_spec(spec, int(i), int(j)) for i, j in pair_idx ]
        out[name] = np.asarray(vals, dtype=np.float32)
    return out


def _arr_representation_similarity_correlations(experiment: Dict[str, Any], pair_idx: Sequence[Tuple[int, int]]) -> Dict[str, Any]:
    """Compare document-pair similarity vectors across representation domains."""
    scores = _arr_representation_pair_scores(experiment, pair_idx)
    names = list(scores.keys())
    pearson = np.eye(len(names), dtype=np.float32)
    spearman = np.eye(len(names), dtype=np.float32)
    rows: List[Dict[str, Any]] = []
    for a_i, a in enumerate(names):
        for b_i, b in enumerate(names):
            if b_i < a_i:
                pearson[a_i, b_i] = pearson[b_i, a_i]
                spearman[a_i, b_i] = spearman[b_i, a_i]
                continue
            if a == b:
                corr_p, n_p = 1.0, int(np.isfinite(scores[a]).sum())
                corr_s, n_s = 1.0, int(np.isfinite(scores[a]).sum())
            else:
                corr_p, n_p = _arr_corr_pair(scores[a], scores[b], method='pearson')
                corr_s, n_s = _arr_corr_pair(scores[a], scores[b], method='spearman')
            pearson[a_i, b_i] = np.float32(corr_p) if np.isfinite(corr_p) else np.float32(np.nan)
            spearman[a_i, b_i] = np.float32(corr_s) if np.isfinite(corr_s) else np.float32(np.nan)
            if b_i >= a_i and a != b:
                rows.append({
                    'representation_a': a,
                    'representation_b': b,
                    'pearson_correlation': float(corr_p),
                    'spearman_correlation': float(corr_s),
                    'n_pairs': int(min(n_p, n_s)),
                })
    return {
        'representations': names,
        'pearson_matrix': pearson,
        'spearman_matrix': spearman,
        'rows': rows,
        'n_pairs_used': int(len(pair_idx)),
        'pair_scores': scores,  # kept in-memory / pickle artifact; small for capped pair samples
    }

def _arr_roc_curve(y_true: Any, scores: Any) -> Dict[str, Any]:
    """
    Compute a basic ROC curve and AUC without sklearn.

    Higher scores are treated as stronger evidence for the positive class.
    Returns arrays plus summary fields, including Youden-J best threshold.
    """
    y = np.asarray(y_true, dtype=bool).reshape(-1)
    s = np.asarray(scores, dtype=float).reshape(-1)
    if y.size != s.size or y.size == 0:
        return {'fpr': np.zeros((0,), dtype=np.float32), 'tpr': np.zeros((0,), dtype=np.float32), 'thresholds': np.zeros((0,), dtype=np.float32), 'auc': float('nan'), 'n_pos': int(y.sum()), 'n_neg': int((~y).sum())}
    finite = np.isfinite(s)
    y = y[finite]; s = s[finite]
    n_pos = int(y.sum()); n_neg = int((~y).sum())
    if y.size == 0 or n_pos == 0 or n_neg == 0:
        return {'fpr': np.zeros((0,), dtype=np.float32), 'tpr': np.zeros((0,), dtype=np.float32), 'thresholds': np.zeros((0,), dtype=np.float32), 'auc': float('nan'), 'n_pos': n_pos, 'n_neg': n_neg}
    order = np.argsort(s)[::-1]
    y = y[order]; s = s[order]
    fpr = [0.0]; tpr = [0.0]; thresholds = [float('inf')]
    tp = 0; fp = 0; idx = 0; n = int(s.size)
    while idx < n:
        val = float(s[idx])
        j = idx
        pos_count = 0; neg_count = 0
        while j < n and float(s[j]) == val:
            if bool(y[j]):
                pos_count += 1
            else:
                neg_count += 1
            j += 1
        tp += pos_count; fp += neg_count
        tpr.append(tp / max(1, n_pos))
        fpr.append(fp / max(1, n_neg))
        thresholds.append(val)
        idx = j
    if fpr[-1] < 1.0 or tpr[-1] < 1.0:
        fpr.append(1.0); tpr.append(1.0); thresholds.append(float('-inf'))
    f = np.asarray(fpr, dtype=float); t = np.asarray(tpr, dtype=float)
    auc = float(np.trapz(t, f)) if f.size >= 2 else float('nan')
    j_scores = t - f
    best_idx = int(np.nanargmax(j_scores)) if j_scores.size else 0
    pos_scores = s[y]
    neg_scores = s[~y]
    return {
        'fpr': f.astype(np.float32),
        'tpr': t.astype(np.float32),
        'thresholds': np.asarray(thresholds, dtype=np.float32),
        'auc': auc,
        'n_pos': n_pos,
        'n_neg': n_neg,
        'best_threshold_youden_j': float(thresholds[best_idx]) if best_idx < len(thresholds) else float('nan'),
        'best_youden_j': float(j_scores[best_idx]) if best_idx < j_scores.size else float('nan'),
        'best_tpr': float(t[best_idx]) if best_idx < t.size else float('nan'),
        'best_fpr': float(f[best_idx]) if best_idx < f.size else float('nan'),
        'mean_positive_score': float(np.mean(pos_scores)) if pos_scores.size else float('nan'),
        'mean_negative_score': float(np.mean(neg_scores)) if neg_scores.size else float('nan'),
    }


def _arr_sample_pair_indices(collection_labels: Sequence[str], max_pairs: int = 250000, random_state: int = 0) -> List[Tuple[int, int]]:
    """Return document-pair indices, using full enumeration for small n and stratified sampling for large n."""
    coll = [str(c) if str(c).strip() else 'Unknown' for c in collection_labels]
    n = len(coll)
    if n < 2:
        return []
    total_pairs = n * (n - 1) // 2
    max_pairs = max(1, int(max_pairs or 250000))
    if total_pairs <= max_pairs:
        return [(i, j) for i in range(n) for j in range(i + 1, n)]

    rng = np.random.default_rng(int(random_state))
    by_type: Dict[str, List[int]] = defaultdict(list)
    for i, c in enumerate(coll):
        by_type[c].append(i)
    same_types = [t for t, idx in by_type.items() if len(idx) >= 2]
    diff_types = [t for t in by_type.keys()]
    pairs: List[Tuple[int, int]] = []
    seen = set()
    target_same = max_pairs // 2
    target_diff = max_pairs - target_same

    # Positive/same-collection pairs.
    attempts = 0
    while len(pairs) < target_same and same_types and attempts < target_same * 20:
        attempts += 1
        weights = np.asarray([len(by_type[t]) * (len(by_type[t]) - 1) / 2 for t in same_types], dtype=float)
        weights = weights / max(float(weights.sum()), 1e-12)
        t = str(rng.choice(same_types, p=weights))
        idxs = by_type[t]
        a, b = rng.choice(idxs, size=2, replace=False)
        i, j = (int(a), int(b)) if int(a) < int(b) else (int(b), int(a))
        key = (i, j)
        if key not in seen:
            seen.add(key); pairs.append(key)

    # Negative/cross-collection pairs.
    attempts = 0
    while len(pairs) < max_pairs and len(diff_types) >= 2 and attempts < target_diff * 30:
        attempts += 1
        i = int(rng.integers(0, n))
        other_types = [t for t in diff_types if t != coll[i] and by_type.get(t)]
        if not other_types:
            continue
        t = str(rng.choice(other_types))
        j = int(rng.choice(by_type[t]))
        if i == j:
            continue
        a, b = (i, j) if i < j else (j, i)
        key = (a, b)
        if key not in seen:
            seen.add(key); pairs.append(key)
    return pairs


def _arr_collection_roc_analysis(experiment: Dict[str, Any], max_pairwise_pairs: int = 250000) -> Dict[str, Any]:
    """
    Build ROC-style separability diagnostics across representation domains.

    Representations currently supported:
      • shape_membership_cosine
      • shape_membership_jensen_shannon
      • raw_sbert_cosine, when raw document embeddings are available
      • manifold_residual_cosine, when residual document embeddings are available

    One-vs-rest curves score each document by similarity to a leave-one-out
    collection profile. Pairwise curves score document pairs and label pairs as
    same-collection vs cross-collection. Similarity-correlation diagnostics
    compare the document-pair score vectors across representation domains.
    """
    if not isinstance(experiment, dict):
        return {}
    M = np.asarray(experiment.get('doc_shape_membership', []), dtype=float)
    docs = [str(d) for d in experiment.get('doc_ids', [])]
    coll = [str(c) if str(c).strip() else 'Unknown' for c in experiment.get('collection_type', [])]
    if M.ndim != 2 or M.size == 0 or not docs:
        return {}
    n, k = M.shape
    if len(coll) < n:
        coll = coll + ['Unknown'] * (n - len(coll))
    coll = coll[:n]
    types = sorted(set(coll))
    rows: List[Dict[str, Any]] = []
    specs = _arr_representation_specs(experiment)
    one_vs_rest: Dict[str, Dict[str, Any]] = {}

    for mode, spec in specs.items():
        one_vs_rest[mode] = {'label': str(spec.get('label', mode)), 'curves': {}, 'family': str(spec.get('family', ''))}

    for t in types:
        mask_all = np.asarray([c == t for c in coll], dtype=bool)
        for mode, spec in specs.items():
            X = np.asarray(spec.get('matrix', []), dtype=float)
            avail = np.asarray(spec.get('available', np.ones((n,), dtype=bool)), dtype=bool).reshape(-1)
            if X.ndim != 2 or X.shape[0] != n:
                continue
            scores = _arr_profile_scores(
                X,
                coll,
                t,
                score_mode=str(spec.get('score_mode', 'cosine')),
                availability=avail,
            )
            curve = _arr_roc_curve(mask_all, scores)
            if int(curve.get('n_pos', 0) or 0) <= 0 or int(curve.get('n_neg', 0) or 0) <= 0:
                continue
            one_vs_rest[mode]['curves'][t] = curve
            rows.append({
                'roc_type': 'one_vs_rest',
                'collection_type': t,
                'score_mode': mode,
                'representation': mode,
                'n_pos': int(curve.get('n_pos', 0)),
                'n_neg': int(curve.get('n_neg', 0)),
                'auc': float(curve.get('auc', float('nan'))),
                'best_threshold_youden_j': float(curve.get('best_threshold_youden_j', float('nan'))),
                'best_youden_j': float(curve.get('best_youden_j', float('nan'))),
                'best_tpr': float(curve.get('best_tpr', float('nan'))),
                'best_fpr': float(curve.get('best_fpr', float('nan'))),
                'mean_positive_score': float(curve.get('mean_positive_score', float('nan'))),
                'mean_negative_score': float(curve.get('mean_negative_score', float('nan'))),
                'n_samples': int(n),
                'profile_mode': 'leave_one_out_for_positive_docs',
            })

    # Backwards-compatible aliases expected by earlier UI/export code.
    if 'shape_membership_cosine' in one_vs_rest:
        one_vs_rest['cosine_profile'] = one_vs_rest['shape_membership_cosine']
    if 'shape_membership_jensen_shannon' in one_vs_rest:
        one_vs_rest['jensen_shannon_profile'] = one_vs_rest['shape_membership_jensen_shannon']

    # Pairwise same-collection vs cross-collection curves.
    max_pairs = int((experiment.get('params') or {}).get('roc_max_pairwise_pairs', max_pairwise_pairs))
    pair_idx = _arr_sample_pair_indices(coll, max_pairs=max_pairs, random_state=int((experiment.get('params') or {}).get('random_state', 0)))
    y_pair = np.asarray([bool(coll[int(i)] == coll[int(j)]) for i, j in pair_idx], dtype=bool)
    pair_scores = _arr_representation_pair_scores(experiment, pair_idx)
    pairwise = {
        'sampling': 'full' if (n * (n - 1) // 2) <= max_pairs else 'stratified_sample',
        'max_pairwise_pairs': int(max_pairs),
        'n_pairs_evaluated': int(len(pair_idx)),
        'curves': {},
    }
    for mode, scores in pair_scores.items():
        curve = _arr_roc_curve(y_pair, np.asarray(scores, dtype=float))
        pairwise['curves'][mode] = curve
        rows.append({
            'roc_type': 'pairwise_same_vs_cross',
            'collection_type': 'same_collection_pair',
            'score_mode': mode,
            'representation': mode,
            'n_pos': int(curve.get('n_pos', 0)),
            'n_neg': int(curve.get('n_neg', 0)),
            'auc': float(curve.get('auc', float('nan'))),
            'best_threshold_youden_j': float(curve.get('best_threshold_youden_j', float('nan'))),
            'best_youden_j': float(curve.get('best_youden_j', float('nan'))),
            'best_tpr': float(curve.get('best_tpr', float('nan'))),
            'best_fpr': float(curve.get('best_fpr', float('nan'))),
            'mean_positive_score': float(curve.get('mean_positive_score', float('nan'))),
            'mean_negative_score': float(curve.get('mean_negative_score', float('nan'))),
            'n_samples': int(len(pair_idx)),
            'profile_mode': pairwise['sampling'],
        })
    # Backwards-compatible pairwise aliases.
    if 'shape_membership_cosine' in pairwise['curves']:
        pairwise['curves']['pairwise_cosine'] = pairwise['curves']['shape_membership_cosine']
    if 'shape_membership_jensen_shannon' in pairwise['curves']:
        pairwise['curves']['pairwise_jensen_shannon'] = pairwise['curves']['shape_membership_jensen_shannon']

    # Macro summaries over one-vs-rest classes for each representation.
    for mode, block in list(one_vs_rest.items()):
        if mode in {'cosine_profile', 'jensen_shannon_profile'}:
            continue
        mode_rows = [r for r in rows if r.get('roc_type') == 'one_vs_rest' and r.get('score_mode') == mode and np.isfinite(float(r.get('auc', float('nan'))))]
        if mode_rows:
            aucs = np.asarray([float(r['auc']) for r in mode_rows], dtype=float)
            weights = np.asarray([max(1, int(r.get('n_pos', 1))) for r in mode_rows], dtype=float)
            block['macro_auc'] = float(np.mean(aucs))
            block['weighted_macro_auc'] = float(np.average(aucs, weights=weights))
        else:
            block['macro_auc'] = float('nan')
            block['weighted_macro_auc'] = float('nan')
    if 'cosine_profile' in one_vs_rest and 'shape_membership_cosine' in one_vs_rest:
        one_vs_rest['cosine_profile']['macro_auc'] = one_vs_rest['shape_membership_cosine'].get('macro_auc', float('nan'))
        one_vs_rest['cosine_profile']['weighted_macro_auc'] = one_vs_rest['shape_membership_cosine'].get('weighted_macro_auc', float('nan'))
    if 'jensen_shannon_profile' in one_vs_rest and 'shape_membership_jensen_shannon' in one_vs_rest:
        one_vs_rest['jensen_shannon_profile']['macro_auc'] = one_vs_rest['shape_membership_jensen_shannon'].get('macro_auc', float('nan'))
        one_vs_rest['jensen_shannon_profile']['weighted_macro_auc'] = one_vs_rest['shape_membership_jensen_shannon'].get('weighted_macro_auc', float('nan'))

    corr = _arr_representation_similarity_correlations(experiment, pair_idx)
    # Do not keep per-pair score vectors in saved artifacts by default; they can be recomputed.
    corr_export = dict(corr)
    corr_export.pop('pair_scores', None)

    available = {name: {'label': str(spec.get('label', name)), 'family': str(spec.get('family', '')), 'n_available': int(np.asarray(spec.get('available', []), dtype=bool).sum())} for name, spec in specs.items()}
    return {
        'collection_roc': {
            'kind': 'collection_roc_analysis',
            'version': 2,
            'collection_types': types,
            'n_docs': int(n),
            'n_shapes': int(k),
            'available_representations': available,
            'one_vs_rest': one_vs_rest,
            'pairwise': pairwise,
            'similarity_matrix_correlations': corr_export,
            'rows': rows,
        }
    }

def _arr_ensure_collection_roc_analysis(experiment: Dict[str, Any]) -> None:
    """Backfill ROC diagnostics for newly built or older arrangement artifacts."""
    if not isinstance(experiment, dict):
        return
    roc = experiment.get('collection_roc')
    if isinstance(roc, dict) and isinstance(roc.get('one_vs_rest'), dict) and isinstance(roc.get('pairwise'), dict):
        if int(roc.get('version', 0) or 0) >= 2 and isinstance(roc.get('similarity_matrix_correlations'), dict):
            return
    try:
        analysis = _arr_collection_roc_analysis(experiment)
        experiment.update(analysis)
    except Exception as ex:
        experiment.setdefault('summary', {})['collection_roc_error'] = str(ex)

def _arr_ensure_collection_profile_analysis(experiment: Dict[str, Any]) -> None:
    """Backfill collection-profile analysis for newly built or older experiments."""
    if not isinstance(experiment, dict):
        return
    if isinstance(experiment.get('dominant_shape_distribution'), dict) and isinstance(experiment.get('collection_profile_similarity'), dict):
        return
    try:
        analysis = _arr_collection_profile_analysis(experiment)
        experiment.update(analysis)
    except Exception as ex:
        experiment.setdefault('summary', {})['collection_profile_analysis_error'] = str(ex)


def _arr_compute_shape_neighbors(experiment: Dict[str, Any], top_k: int = 10) -> Dict[str, Any]:
    """
    Compute three shape-neighbor families for arrangement interpretation.

    1. centroid neighbors: nearest shape-cluster centroids in the feature space
       used for shape clustering.
    2. cooccurrence neighbors: shapes that co-occur in the same documents above
       the support threshold.
    3. collection-profile neighbors: shapes whose average membership profiles
       across collection labels are similar.
    """
    if not isinstance(experiment, dict):
        return {}
    M = np.asarray(experiment.get('doc_shape_membership', []), dtype=float)
    centroids = np.asarray(experiment.get('shape_centroids', []), dtype=float)
    coll = list(experiment.get('collection_type', []))
    shape_summary = list(experiment.get('shape_summary', []))
    summary_by_id = _arr_shape_summary_lookup(shape_summary)
    k_candidates = []
    if M.ndim == 2:
        k_candidates.append(M.shape[1])
    if centroids.ndim == 2:
        k_candidates.append(centroids.shape[0])
    if shape_summary:
        try:
            k_candidates.append(max(int(r.get('shape_id', -1)) for r in shape_summary) + 1)
        except Exception:
            pass
    k = max([x for x in k_candidates if x is not None and x > 0], default=0)
    top_k = max(1, int(top_k or 10))

    def _summary_fields(sid: int) -> Dict[str, Any]:
        r = summary_by_id.get(int(sid), {})
        return {
            'n_edges': int(r.get('n_edges', 0) or 0),
            'n_docs': int(r.get('n_docs', 0) or 0),
            'support_docs': int(r.get('support_docs', 0) or 0),
            'top_collection_type': str(r.get('top_collection_type', '')),
            'collection_type_entropy': float(r.get('collection_type_entropy', 0.0) or 0.0),
            'mean_edge_quality': float(r.get('mean_edge_quality', 0.0) or 0.0),
            'mean_weight': float(r.get('mean_weight', 0.0) or 0.0),
        }

    # Centroid-neighbor matrices.
    if centroids.ndim == 2 and centroids.shape[0] > 0:
        C = centroids[:k] if k and centroids.shape[0] >= k else centroids
        centroid_distance = _arr_pairwise_l2_rows(C)
        centroid_cosine = _arr_pairwise_cosine_rows(C)
    else:
        centroid_distance = np.zeros((k, k), dtype=np.float32)
        centroid_cosine = np.zeros((k, k), dtype=np.float32)

    centroid_neighbors: Dict[int, List[Dict[str, Any]]] = {}
    for sid in range(k):
        rows: List[Dict[str, Any]] = []
        if centroid_distance.shape == (k, k):
            vals = centroid_distance[sid].copy()
            vals[sid] = np.inf
            order = np.argsort(vals)[: min(top_k, max(0, k - 1))]
            for rank, nid in enumerate(order, start=1):
                nid = int(nid)
                if not np.isfinite(vals[nid]):
                    continue
                row = {
                    'rank': int(rank),
                    'shape_id': int(nid),
                    'centroid_distance': float(centroid_distance[sid, nid]),
                    'centroid_cosine': float(centroid_cosine[sid, nid]) if centroid_cosine.shape == (k, k) else float('nan'),
                }
                row.update(_summary_fields(nid))
                rows.append(row)
        centroid_neighbors[int(sid)] = rows

    # Co-occurrence neighbors.
    co = experiment.get('shape_cooccurrence') or {}
    co_matrix = np.asarray(co.get('matrix', []), dtype=float)
    count_matrix = np.asarray(co.get('count_matrix', []), dtype=float)
    jaccard_matrix = np.asarray(co.get('jaccard_matrix', []), dtype=float)
    lift_matrix = np.asarray(co.get('lift_matrix', []), dtype=float)
    if co_matrix.shape != (k, k):
        co_matrix = np.zeros((k, k), dtype=np.float32)
    if count_matrix.shape != (k, k):
        count_matrix = np.zeros((k, k), dtype=np.float32)
    if jaccard_matrix.shape != (k, k):
        jaccard_matrix = np.zeros((k, k), dtype=np.float32)
    if lift_matrix.shape != (k, k):
        lift_matrix = np.zeros((k, k), dtype=np.float32)
    cooccurrence_neighbors: Dict[int, List[Dict[str, Any]]] = {}
    for sid in range(k):
        vals = co_matrix[sid].copy()
        vals[sid] = -np.inf
        # Prefer the active co-occurrence weight; use Jaccard/count as tie support.
        order = np.argsort(vals)[::-1][: min(top_k, max(0, k - 1))]
        rows = []
        for rank, nid in enumerate(order, start=1):
            nid = int(nid)
            score = float(vals[nid])
            cnt = int(count_matrix[sid, nid]) if count_matrix.size else 0
            if not np.isfinite(score) or (score <= 0 and cnt <= 0):
                continue
            row = {
                'rank': int(rank),
                'shape_id': nid,
                'cooccurrence_score': score,
                'cooccurring_docs': cnt,
                'jaccard': float(jaccard_matrix[sid, nid]) if jaccard_matrix.size else 0.0,
                'lift': float(lift_matrix[sid, nid]) if lift_matrix.size else 0.0,
            }
            row.update(_summary_fields(nid))
            rows.append(row)
        cooccurrence_neighbors[int(sid)] = rows

    # Collection-profile neighbors.
    profiles, collection_types = _arr_collection_profiles(M, coll)
    profile_cosine = _arr_pairwise_cosine_rows(profiles)
    profile_l2 = _arr_pairwise_l2_rows(profiles)
    collection_profile_neighbors: Dict[int, List[Dict[str, Any]]] = {}
    for sid in range(k):
        rows = []
        if profile_cosine.shape == (k, k):
            vals = profile_cosine[sid].copy()
            vals[sid] = -np.inf
            order = np.argsort(vals)[::-1][: min(top_k, max(0, k - 1))]
            for rank, nid in enumerate(order, start=1):
                nid = int(nid)
                sim = float(vals[nid])
                if not np.isfinite(sim):
                    continue
                row = {
                    'rank': int(rank),
                    'shape_id': nid,
                    'profile_cosine': sim,
                    'profile_l2': float(profile_l2[sid, nid]) if profile_l2.shape == (k, k) else float('nan'),
                }
                if profiles.ndim == 2 and profiles.shape[0] > nid and profiles.shape[1] > 0:
                    top_idx = int(np.argmax(profiles[nid]))
                    row['profile_top_collection_type'] = collection_types[top_idx] if top_idx < len(collection_types) else ''
                    row['profile_top_membership'] = float(profiles[nid, top_idx])
                row.update(_summary_fields(nid))
                rows.append(row)
        collection_profile_neighbors[int(sid)] = rows

    return {
        'top_k': int(top_k),
        'centroid': {
            'metric': 'euclidean_distance_ascending',
            'distance_matrix': centroid_distance.astype(np.float32),
            'cosine_matrix': centroid_cosine.astype(np.float32),
            'neighbors': centroid_neighbors,
        },
        'cooccurrence': {
            'metric': str((experiment.get('params') or {}).get('shape_cooccurrence_weight', 'jaccard')),
            'matrix': co_matrix.astype(np.float32),
            'count_matrix': count_matrix.astype(np.float32),
            'jaccard_matrix': jaccard_matrix.astype(np.float32),
            'lift_matrix': lift_matrix.astype(np.float32),
            'neighbors': cooccurrence_neighbors,
        },
        'collection_profile': {
            'metric': 'cosine_similarity_descending',
            'collection_types': list(collection_types),
            'profiles': profiles.astype(np.float32),
            'cosine_matrix': profile_cosine.astype(np.float32),
            'l2_matrix': profile_l2.astype(np.float32),
            'neighbors': collection_profile_neighbors,
        },
    }

def _arr_normalized_entropy(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = np.maximum(np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    total = float(arr.sum())
    if total <= 0 or arr.size <= 1:
        return 0.0
    p = arr / total
    positive = p[p > 0]
    if positive.size == 0:
        return 0.0
    return float(-np.sum(positive * np.log(positive)) / math.log(arr.size))


def _arr_build_experiment(
    document_delta_dict: Dict[Any, Any],
    params: Dict[str, Any],
    status_callback: Any = None,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    shape_k = max(1, int(params.get('shape_k', 48)))
    rows, X, errors = _arr_extract_shapes_from_cdm_dict(
        document_delta_dict,
        max_edges_per_doc=(None if int(params.get('max_edges_per_doc', 2000)) <= 0 else int(params.get('max_edges_per_doc', 2000))),
        min_weight=float(params.get('min_weight', 0.0)),
        include_length=bool(params.get('include_length', True)),
        include_v_bin=bool(params.get('include_v_bin', False)),
        vbin_az=int(params.get('vbin_az', 12)),
        vbin_el=int(params.get('vbin_el', 6)),
        align_signs=True,
        dir_weight_beta=float(params.get('dir_weight_beta', 1.0)),
        status_callback=status_callback,
    )
    if status_callback:
        status_callback(f"Clustering {X.shape[0]:,} morphism-shape records into k={shape_k} categories ...")
    if X.shape[0] == 0:
        labels = np.zeros((0,), dtype=np.int32)
        centroids = np.zeros((shape_k, X.shape[1] if X.ndim == 2 else 0), dtype=np.float32)
    else:
        labels, centroids = _arr_kmeans_np(X, k=shape_k, n_iter=50, random_state=int(params.get('random_state', 0)))
    k_eff = int(centroids.shape[0]) if centroids.ndim == 2 and centroids.shape[0] else shape_k
    if status_callback:
        status_callback("Building document × shape-category membership matrix ...")
    M, doc_ids = _arr_doc_membership(
        rows,
        labels,
        k=k_eff,
        weight_mode=str(params.get('weight_mode', 'weighted')),
        normalize=bool(params.get('normalize_membership', True)),
    )
    document_embeddings = _arr_document_embedding_tables_from_cdm_dict(document_delta_dict, doc_ids)
    coll = _arr_collection_labels(
        doc_ids,
        source=str(params.get('collection_label_source', 'doc_id_regex')),
        regex=str(params.get('collection_label_regex', r'^([^_]+_[^_]+)_')),
        csv_path=str(params.get('metadata_csv_path', '')),
        unknown=str(params.get('unknown_label', 'Unknown')),
    )
    sim_metric = str(params.get('similarity_metric', 'cosine')).lower()
    S = _arr_js_similarity_matrix(M) if sim_metric.startswith('js') or 'jensen' in sim_metric else _arr_cosine_similarity_matrix(M)
    doc_edges = _arr_graph_edges_from_similarity(
        S,
        top_k=int(params.get('doc_top_k_neighbors', 5)),
        threshold=float(params.get('doc_graph_threshold', 0.0)),
    )

    supports = M >= float(params.get('shape_support_threshold', 0.03))
    support_counts = supports.sum(axis=0).astype(int) if supports.size else np.zeros(k_eff, dtype=int)
    cooc = (supports.astype(np.int32).T @ supports.astype(np.int32)).astype(np.float32) if supports.size else np.zeros((k_eff, k_eff), dtype=np.float32)
    if cooc.size:
        np.fill_diagonal(cooc, 0.0)
    union = support_counts[:, None] + support_counts[None, :] - cooc
    jacc = np.divide(cooc, np.maximum(union, 1.0), out=np.zeros_like(cooc), where=union > 0)
    n_docs = max(1, len(doc_ids))
    lift = np.divide(cooc * n_docs, np.maximum(support_counts[:, None] * support_counts[None, :], 1.0), out=np.zeros_like(cooc), where=(support_counts[:, None] * support_counts[None, :]) > 0)
    co_weight_mode = str(params.get('shape_cooccurrence_weight', 'jaccard')).lower()
    co_mat = lift if co_weight_mode == 'lift' else jacc if co_weight_mode == 'jaccard' else cooc
    min_co_docs = int(params.get('min_cooccurring_docs', 3))
    shape_co_edges: List[Tuple[int, int, float, int]] = []
    for i in range(k_eff):
        for j in range(i + 1, k_eff):
            cnt = int(cooc[i, j])
            w = float(co_mat[i, j])
            if cnt >= min_co_docs and np.isfinite(w) and w > 0:
                shape_co_edges.append((i, j, w, cnt))
    shape_co_edges.sort(key=lambda x: x[2], reverse=True)

    # Shape summaries and representatives.
    shape_rows: List[Dict[str, Any]] = []
    labels_arr = np.asarray(labels, dtype=int)
    coll_by_doc = {d: c for d, c in zip(doc_ids, coll)}
    rows_by_shape: Dict[int, List[Dict[str, Any]]] = {i: [] for i in range(k_eff)}
    for r, lab in zip(rows, labels_arr):
        if 0 <= int(lab) < k_eff:
            rr = dict(r)
            rr['shape_id'] = int(lab)
            rr['collection_type'] = coll_by_doc.get(str(r.get('doc_id', '')), '')
            rows_by_shape[int(lab)].append(rr)
    rep_edges: List[Dict[str, Any]] = []
    for sid in range(k_eff):
        members = rows_by_shape.get(sid, [])
        docs_in = sorted({str(r.get('doc_id', '')) for r in members})
        coll_counts = Counter(coll_by_doc.get(d, 'Unknown') for d in docs_in)
        top_coll = coll_counts.most_common(1)[0][0] if coll_counts else ''
        weights = np.asarray([float(r.get('w', 0.0)) for r in members], dtype=float)
        qs = np.asarray([float(r.get('edge_quality', 1.0)) for r in members], dtype=float)
        lens = np.asarray([float(r.get('len_delta', 0.0)) for r in members], dtype=float)
        means = {k: float(np.mean([float(r.get(k, 0.0)) for r in members])) if members else 0.0 for k in ['cos_vs', 'cos_vt_in', 'cos_st']}
        shape_rows.append({
            'shape_id': sid,
            'n_edges': len(members),
            'n_docs': len(docs_in),
            'support_docs': int(support_counts[sid]) if sid < len(support_counts) else 0,
            'top_collection_type': top_coll,
            'collection_type_entropy': _arr_normalized_entropy(list(coll_counts.values())),
            'mean_weight': float(np.mean(weights)) if weights.size else 0.0,
            'mean_edge_quality': float(np.mean(qs)) if qs.size else 0.0,
            'mean_len_delta': float(np.mean(lens)) if lens.size else 0.0,
            'cos_vs_centroid': float(centroids[sid, 0]) if centroids.ndim == 2 and centroids.shape[1] > 0 and sid < centroids.shape[0] else means['cos_vs'],
            'cos_vt_in_centroid': float(centroids[sid, 1]) if centroids.ndim == 2 and centroids.shape[1] > 1 and sid < centroids.shape[0] else means['cos_vt_in'],
            'cos_st_centroid': float(centroids[sid, 2]) if centroids.ndim == 2 and centroids.shape[1] > 2 and sid < centroids.shape[0] else means['cos_st'],
            'mean_cos_vs': means['cos_vs'],
            'mean_cos_vt_in': means['cos_vt_in'],
            'mean_cos_st': means['cos_st'],
        })
        if members:
            if X.shape[0] == len(rows) and centroids.ndim == 2 and sid < centroids.shape[0]:
                member_indices = [int(r['shape_row']) for r in members if 0 <= int(r['shape_row']) < X.shape[0]]
                if member_indices:
                    D = np.sum((X[member_indices] - centroids[sid]) ** 2, axis=1)
                    order = [member_indices[int(i)] for i in np.argsort(D)[: min(10, len(member_indices))]]
                    by_idx = {int(r['shape_row']): r for r in members}
                    chosen = [by_idx[i] for i in order if i in by_idx]
                else:
                    chosen = sorted(members, key=lambda r: (-float(r.get('edge_quality', 0.0)), -float(r.get('w', 0.0))))[:10]
            else:
                chosen = sorted(members, key=lambda r: (-float(r.get('edge_quality', 0.0)), -float(r.get('w', 0.0))))[:10]
            for rank, r in enumerate(chosen, start=1):
                out = dict(r)
                out['rep_rank'] = rank
                rep_edges.append(out)

    # Collection summaries.
    collection_rows: List[Dict[str, Any]] = []
    coll_values = sorted(set(coll))
    for c in coll_values:
        mask = np.asarray([x == c for x in coll], dtype=bool)
        sub = M[mask] if M.size else np.zeros((0, k_eff), dtype=float)
        prof = sub.mean(axis=0) if sub.size else np.zeros(k_eff, dtype=float)
        top = np.argsort(prof)[::-1][: min(8, prof.size)] if prof.size else []
        dominant = np.argmax(M[mask], axis=1) if sub.size else np.asarray([], dtype=int)
        dom_counts = Counter(int(x) for x in dominant)
        collection_rows.append({
            'collection_type': c,
            'n_docs': int(mask.sum()),
            'top_shapes': '; '.join(f"S{int(i)}={prof[int(i)]:.4f}" for i in top),
            'dominant_shape_counts': '; '.join(f"S{k}:{v}" for k, v in dom_counts.most_common(8)),
            'membership_entropy_mean': float(np.mean([_arr_normalized_entropy(row) for row in sub])) if sub.size else 0.0,
        })

    # Experiment metrics.
    same_type_topk = []
    topk = max(1, int(params.get('doc_top_k_neighbors', 5)))
    if S.size and len(doc_ids) > 1:
        for i in range(len(doc_ids)):
            vals = S[i].copy()
            vals[i] = -np.inf
            nn = np.argsort(vals)[::-1][: min(topk, len(doc_ids) - 1)]
            same_type_topk.append(float(np.mean([coll[int(j)] == coll[i] for j in nn])) if len(nn) else 0.0)
    within_vals: List[float] = []
    between_vals: List[float] = []
    if S.size:
        for i in range(len(doc_ids)):
            for j in range(i + 1, len(doc_ids)):
                if coll[i] == coll[j]:
                    within_vals.append(float(S[i, j]))
                else:
                    between_vals.append(float(S[i, j]))
    summary = {
        'n_docs': int(len(doc_ids)),
        'n_collection_types': int(len(coll_values)),
        'n_shape_records': int(len(rows)),
        'shape_feature_dim': int(X.shape[1]) if X.ndim == 2 else 0,
        'shape_k_effective': int(k_eff),
        'doc_shape_membership_shape': tuple(M.shape),
        'same_type_topk_neighbor_rate': float(np.mean(same_type_topk)) if same_type_topk else 0.0,
        'within_type_mean_similarity': float(np.mean(within_vals)) if within_vals else 0.0,
        'between_type_mean_similarity': float(np.mean(between_vals)) if between_vals else 0.0,
        'within_minus_between_similarity': (float(np.mean(within_vals)) - float(np.mean(between_vals))) if within_vals and between_vals else 0.0,
        'build_elapsed_seconds': float(time.perf_counter() - t0),
        'errors': errors[:100],
        'n_errors': len(errors),
    }
    # Shape-neighbor families make substituted top-shape IDs inspectable.
    # They are recomputed here so saved arrangement experiment artifacts carry
    # centroid, co-occurrence, and collection-profile neighbor tables.
    neighbor_stub = {
        'params': dict(params),
        'doc_ids': list(doc_ids),
        'collection_type': list(coll),
        'shape_centroids': centroids.astype(np.float32),
        'doc_shape_membership': M.astype(np.float32),
        'shape_cooccurrence': {
            'matrix': co_mat.astype(np.float32),
            'count_matrix': cooc.astype(np.float32),
            'jaccard_matrix': jacc.astype(np.float32),
            'lift_matrix': lift.astype(np.float32),
            'edges': shape_co_edges,
        },
        'shape_summary': shape_rows,
    }
    shape_neighbors = _arr_compute_shape_neighbors(neighbor_stub, top_k=int(params.get('shape_neighbor_top_k', 10)))
    collection_analysis_stub = {
        'doc_ids': list(doc_ids),
        'collection_type': list(coll),
        'doc_shape_membership': M.astype(np.float32),
    }
    collection_analysis = _arr_collection_profile_analysis(collection_analysis_stub)
    # ROC/separability diagnostics quantify how well shape-membership profiles
    # recover collection labels in one-vs-rest and pairwise same-vs-cross tests.
    roc_analysis_stub = {
        'params': dict(params),
        'doc_ids': list(doc_ids),
        'collection_type': list(coll),
        'doc_shape_membership': M.astype(np.float32),
        'document_embeddings': document_embeddings,
    }
    roc_analysis = _arr_collection_roc_analysis(roc_analysis_stub)

    return {
        'kind': 'morphism_arrangement_experiment',
        'version': 1,
        'params': dict(params),
        'summary': summary,
        'doc_ids': list(doc_ids),
        'collection_type': list(coll),
        'shape_records': rows,
        'shape_features': X,
        'shape_labels': labels_arr.astype(np.int32),
        'shape_centroids': centroids.astype(np.float32),
        'doc_shape_membership': M.astype(np.float32),
        'document_embeddings': document_embeddings,
        'doc_similarity': {'metric': sim_metric, 'matrix': S.astype(np.float32), 'edges': doc_edges},
        'shape_cooccurrence': {'matrix': co_mat.astype(np.float32), 'count_matrix': cooc.astype(np.float32), 'jaccard_matrix': jacc.astype(np.float32), 'lift_matrix': lift.astype(np.float32), 'edges': shape_co_edges},
        'shape_neighbors': shape_neighbors,
        'shape_summary': shape_rows,
        'representative_edges': rep_edges,
        'collection_summary': collection_rows,
        'dominant_shape_distribution': collection_analysis.get('dominant_shape_distribution', {}),
        'collection_profile_similarity': collection_analysis.get('collection_profile_similarity', {}),
        'collection_roc': roc_analysis.get('collection_roc', {}),
        'created_at_unix': time.time(),
    }


def _arr_as_shape_id(value: Any, default: Optional[int] = None) -> Optional[int]:
    """Parse shape labels such as 12, '12', or 'S12'."""
    try:
        if value is None:
            return default
        s = str(value).strip()
        if not s:
            return default
        if s.lower().startswith('s'):
            s = s[1:]
        return int(float(s))
    except Exception:
        return default


def _arr_shape_member_rows(experiment: Dict[str, Any], shape_id: int) -> List[Dict[str, Any]]:
    """Return arrangement shape-record rows assigned to one shape bin, enriched with labels/distances."""
    if not isinstance(experiment, dict):
        return []
    rows = list(experiment.get('shape_records', []) or [])
    labels = np.asarray(experiment.get('shape_labels', []), dtype=int)
    X = np.asarray(experiment.get('shape_features', []), dtype=float)
    C = np.asarray(experiment.get('shape_centroids', []), dtype=float)
    doc_ids = [str(d) for d in experiment.get('doc_ids', [])]
    coll = [str(c) for c in experiment.get('collection_type', [])]
    coll_by_doc = {d: (coll[i] if i < len(coll) else '') for i, d in enumerate(doc_ids)}
    sid = int(shape_id)
    out: List[Dict[str, Any]] = []
    if labels.size == len(rows):
        iterator = [(i, r, int(labels[i])) for i, r in enumerate(rows)]
    else:
        iterator = []
        for i, r in enumerate(rows):
            iterator.append((i, r, _arr_as_shape_id(r.get('shape_id'), -1) or -1))
    for i, r, lab in iterator:
        if int(lab) != sid:
            continue
        rr = dict(r)
        rr['shape_id'] = sid
        rr['collection_type'] = coll_by_doc.get(str(rr.get('doc_id', '')), str(rr.get('collection_type', '')))
        if X.ndim == 2 and 0 <= i < X.shape[0] and C.ndim == 2 and 0 <= sid < C.shape[0] and C.shape[1] == X.shape[1]:
            try:
                rr['prototype_distance'] = float(np.linalg.norm(X[i] - C[sid]))
            except Exception:
                rr['prototype_distance'] = ''
        else:
            rr.setdefault('prototype_distance', '')
        out.append(rr)
    return out


def _arr_shape_bin_basic_stats(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray([float(v) for v in values if np.isfinite(float(v))], dtype=float) if values else np.asarray([], dtype=float)
    if arr.size == 0:
        return {'mean': 0.0, 'median': 0.0, 'std': 0.0, 'iqr': 0.0, 'min': 0.0, 'max': 0.0, 'p95': 0.0}
    q25, q75 = np.percentile(arr, [25, 75])
    return {
        'mean': float(np.mean(arr)), 'median': float(np.median(arr)), 'std': float(np.std(arr)),
        'iqr': float(q75 - q25), 'min': float(np.min(arr)), 'max': float(np.max(arr)),
        'p95': float(np.percentile(arr, 95)),
    }


def _arr_pairwise_mean_cosine(vectors: np.ndarray, max_pairs: int = 50000, seed: int = 0) -> Tuple[float, int]:
    V = np.asarray(vectors, dtype=float)
    if V.ndim != 2 or V.shape[0] < 2:
        return float('nan'), 0
    norms = np.linalg.norm(V, axis=1, keepdims=True) + 1e-12
    V = V / norms
    n = V.shape[0]
    total_pairs = n * (n - 1) // 2
    if total_pairs <= max_pairs:
        vals = []
        for i in range(n):
            vals.extend((V[i + 1:] @ V[i]).tolist())
        return float(np.mean(vals)) if vals else float('nan'), len(vals)
    rng = np.random.default_rng(int(seed))
    vals = []
    seen = set()
    while len(vals) < int(max_pairs) and len(seen) < total_pairs:
        a = int(rng.integers(0, n)); b = int(rng.integers(0, n - 1))
        if b >= a:
            b += 1
        i, j = (a, b) if a < b else (b, a)
        if (i, j) in seen:
            continue
        seen.add((i, j)); vals.append(float(np.dot(V[i], V[j])))
    return float(np.mean(vals)) if vals else float('nan'), len(vals)


def _arr_doc_embedding_vectors_for_docs(experiment: Dict[str, Any], docs: Sequence[str], kind: str) -> np.ndarray:
    """Return document embedding vectors aligned to docs from arrangement experiment tables."""
    emb = experiment.get('document_embeddings') if isinstance(experiment, dict) else None
    if not isinstance(emb, dict):
        return np.zeros((0, 0), dtype=float)
    key = 'raw_sbert' if str(kind).lower().startswith('raw') else 'manifold_residual'
    table = emb.get(key)
    if not isinstance(table, dict):
        return np.zeros((0, 0), dtype=float)
    table_docs = [str(d) for d in table.get('doc_ids', experiment.get('doc_ids', []))]
    vecs = np.asarray(table.get('vectors', []), dtype=float)
    avail = np.asarray(table.get('available', np.ones((vecs.shape[0],), dtype=bool)), dtype=bool) if vecs.ndim == 2 else np.asarray([], dtype=bool)
    if vecs.ndim != 2 or not table_docs:
        return np.zeros((0, 0), dtype=float)
    by_doc = {d: i for i, d in enumerate(table_docs)}
    out = []
    for d in docs:
        idx = by_doc.get(str(d))
        if idx is None or idx >= vecs.shape[0] or (avail.size and idx < avail.size and not bool(avail[idx])):
            continue
        v = np.asarray(vecs[idx], dtype=float).reshape(-1)
        if v.size and np.isfinite(v).all():
            out.append(v)
    return np.vstack(out) if out else np.zeros((0, vecs.shape[1] if vecs.ndim == 2 else 0), dtype=float)


def _arr_shape_bin_field_summary(experiment: Dict[str, Any], shape_id: int) -> Dict[str, Any]:
    """Compute NSF-facing diagnostic metrics for one shape bin field."""
    members = _arr_shape_member_rows(experiment, int(shape_id))
    sid = int(shape_id)
    if not members:
        return {'shape_id': sid, 'n_edges': 0, 'n_docs': 0, 'n_collections': 0}
    docs = [str(r.get('doc_id', '')) for r in members]
    doc_set = sorted(set(docs))
    coll = [str(r.get('collection_type', '') or 'Unknown') for r in members]
    coll_counts = Counter(coll)
    top_coll, top_coll_n = coll_counts.most_common(1)[0] if coll_counts else ('', 0)
    coll_entropy = _arr_normalized_entropy(list(coll_counts.values()))
    top_coll_share = float(top_coll_n / max(1, len(members)))
    specificity = float(1.0 - coll_entropy)
    feature_keys = ['cos_vs', 'cos_vt_in', 'cos_st', 'len_delta', 'edge_quality', 'src_quality', 'dst_quality', 'w', 'prototype_distance']
    stats = {k: _arr_shape_bin_basic_stats([_safe_float(r.get(k), float('nan')) for r in members]) for k in feature_keys}
    # Coherence: smaller prototype distance and lower feature variance -> higher score.
    pd_mean = stats['prototype_distance']['mean'] if np.isfinite(stats['prototype_distance']['mean']) else 0.0
    pd_std = stats['prototype_distance']['std'] if np.isfinite(stats['prototype_distance']['std']) else 0.0
    coherence = float(1.0 / (1.0 + max(0.0, pd_mean + pd_std)))
    # Transition asymmetry: source departure vs destination arrival balance.
    gaps = np.asarray([_safe_float(r.get('cos_vs'), 0.0) - _safe_float(r.get('cos_vt_in'), 0.0) for r in members], dtype=float)
    abs_gaps = np.abs(gaps)
    asymmetry = float(np.mean(abs_gaps)) if abs_gaps.size else 0.0
    balance = float(np.mean([min(_safe_float(r.get('cos_vs'), 0.0), _safe_float(r.get('cos_vt_in'), 0.0)) for r in members])) if members else 0.0
    # Collection-stratification: how different are per-collection feature means inside the same bin?
    per_coll_means: Dict[str, List[float]] = {}
    for c in sorted(coll_counts.keys()):
        cmembers = [r for r in members if str(r.get('collection_type', '') or 'Unknown') == c]
        per_coll_means[c] = [float(np.mean([_safe_float(r.get(k), 0.0) for r in cmembers])) if cmembers else 0.0 for k in ['cos_vs', 'cos_vt_in', 'cos_st', 'len_delta']]
    if len(per_coll_means) >= 2:
        mat = np.asarray(list(per_coll_means.values()), dtype=float)
        stratification = float(np.mean(np.std(mat, axis=0)))
    else:
        stratification = 0.0
    # Semantic breadth: low mean raw/residual doc cosine among contributing documents implies broader content span.
    raw_vecs = _arr_doc_embedding_vectors_for_docs(experiment, doc_set, 'raw')
    res_vecs = _arr_doc_embedding_vectors_for_docs(experiment, doc_set, 'residual')
    raw_mean_cos, raw_pairs = _arr_pairwise_mean_cosine(raw_vecs)
    res_mean_cos, res_pairs = _arr_pairwise_mean_cosine(res_vecs)
    semantic_breadth = float(1.0 - raw_mean_cos) if np.isfinite(raw_mean_cos) else float('nan')
    quality = stats['edge_quality']['mean']
    structural_breadth_score = float(coherence * max(0.0, min(1.0, quality)) * max(0.0, min(2.0, semantic_breadth if np.isfinite(semantic_breadth) else 0.0)) / 2.0)
    # Simple interpretive labels.
    if specificity >= 0.65:
        genericity_label = 'collection-specific'
    elif specificity <= 0.25:
        genericity_label = 'generic/shared'
    else:
        genericity_label = 'mixed/shared-with-enrichment'
    if coherence >= 0.75:
        coherence_label = 'tight'
    elif coherence >= 0.45:
        coherence_label = 'moderate'
    else:
        coherence_label = 'diffuse'
    if structural_breadth_score >= 0.35:
        breadth_label = 'semantically broad + structurally stable candidate'
    elif np.isfinite(semantic_breadth) and semantic_breadth >= 0.35:
        breadth_label = 'semantically broad but needs coherence/quality review'
    else:
        breadth_label = 'semantically narrow or unavailable'
    if asymmetry >= 0.25:
        asymmetry_label = 'asymmetric source/destination transition'
    elif balance >= 0.20:
        asymmetry_label = 'balanced flow-aligned transition'
    else:
        asymmetry_label = 'weak endpoint-flow alignment'
    if stratification >= 0.10 and len(per_coll_means) >= 2:
        strat_label = 'collection-stratified variants likely'
    else:
        strat_label = 'little collection stratification detected'
    out: Dict[str, Any] = {
        'shape_id': sid,
        'n_edges': int(len(members)),
        'n_docs': int(len(doc_set)),
        'n_collections': int(len(coll_counts)),
        'top_collection_type': top_coll,
        'top_collection_edge_share': float(top_coll_share),
        'collection_entropy': float(coll_entropy),
        'collection_specificity_score': float(specificity),
        'bin_coherence_score': coherence,
        'transition_asymmetry_index': asymmetry,
        'endpoint_transition_balance': balance,
        'collection_stratification_score': stratification,
        'mean_raw_sbert_doc_cosine_among_supporting_docs': raw_mean_cos,
        'mean_manifold_residual_doc_cosine_among_supporting_docs': res_mean_cos,
        'raw_doc_pairs_used': raw_pairs,
        'residual_doc_pairs_used': res_pairs,
        'semantic_breadth_score': semantic_breadth,
        'structural_breadth_score': structural_breadth_score,
        'genericity_label': genericity_label,
        'coherence_label': coherence_label,
        'semantic_breadth_label': breadth_label,
        'transition_asymmetry_label': asymmetry_label,
        'collection_stratification_label': strat_label,
    }
    for k, st in stats.items():
        for stat_name, val in st.items():
            out[f'{k}_{stat_name}'] = val
    return out


def _arr_shape_bin_field_summary_rows(experiment: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not isinstance(experiment, dict):
        return []
    C = np.asarray(experiment.get('shape_centroids', []), dtype=float)
    k = int(C.shape[0]) if C.ndim == 2 and C.shape[0] else int((experiment.get('summary') or {}).get('shape_k_effective', 0) or 0)
    if k <= 0:
        labels = np.asarray(experiment.get('shape_labels', []), dtype=int)
        k = int(labels.max() + 1) if labels.size else 0
    return [_arr_shape_bin_field_summary(experiment, sid) for sid in range(k)]


def _arr_canonical_vectors_from_shape_features(cos_vs: float, cos_vt_in: float, cos_st: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return v, source-PC1, destination-PC1 in a canonical v=(+x) frame.

    cos_vs = dot(v, s), cos_vt_in = dot(-v, t), cos_st = dot(s, t).
    Destination PC1 is represented as a vector whose incoming-flow alignment is
    dot(-v, t)=cos_vt_in, so t_x = -cos_vt_in.
    """
    a = max(-1.0, min(1.0, float(cos_vs)))
    b = max(-1.0, min(1.0, float(cos_vt_in)))
    c = max(-1.0, min(1.0, float(cos_st)))
    v = np.asarray([1.0, 0.0, 0.0], dtype=float)
    sy = math.sqrt(max(0.0, 1.0 - a * a))
    s = np.asarray([a, sy, 0.0], dtype=float)
    tx = -b
    if sy > 1e-8:
        ty = (c + a * b) / sy
    else:
        ty = 0.0
    max_ty = math.sqrt(max(0.0, 1.0 - tx * tx))
    ty = max(-max_ty, min(max_ty, ty))
    tz = math.sqrt(max(0.0, 1.0 - tx * tx - ty * ty))
    t = np.asarray([tx, ty, tz], dtype=float)
    return v, _arr_unit(s), _arr_unit(t)


class ArrangementExperimentView(ttk.Frame):
    """Simple mixed-collection arrangement experiment over morphism shape membership."""

    def __init__(self, master: tk.Misc, model: MorphismComparisonModel) -> None:
        super().__init__(master)
        self.model = model
        self.experiment: Optional[Dict[str, Any]] = None
        self._worker: Optional[threading.Thread] = None
        self.status_var = tk.StringVar(value="Load document_delta_dict.pkl, configure labels, then build the arrangement experiment.")
        self.vars: Dict[str, tk.Variable] = {}
        self.canvas: Dict[str, Optional[FigureCanvasTkAgg]] = {
            'heatmap': None, 'docgraph': None, 'shapegraph': None,
            'bars': None, 'collection_similarity': None, 'shapeneighbors': None, 'roc': None,
            'roc_baseline': None, 'roc_pairwise_baseline': None, 'roc_corr': None,
            'shape_bin_canonical': None, 'shape_bin_residual': None, 'shape_bin_distributions': None,
            'shape_bin_composition': None,
        }
        self.toolbar: Dict[str, Optional[NavigationToolbar2Tk]] = {
            'heatmap': None, 'docgraph': None, 'shapegraph': None,
            'bars': None, 'collection_similarity': None, 'shapeneighbors': None, 'roc': None,
            'roc_baseline': None, 'roc_pairwise_baseline': None, 'roc_corr': None,
            'shape_bin_canonical': None, 'shape_bin_residual': None, 'shape_bin_distributions': None,
            'shape_bin_composition': None,
        }
        self.rowconfigure(1, weight=1)
        self.columnconfigure(0, weight=1)
        self._build_controls()
        self._build_body()

    def _var(self, name: str, default: Any) -> tk.Variable:
        if name not in self.vars:
            if isinstance(default, bool):
                self.vars[name] = tk.BooleanVar(value=default)
            else:
                self.vars[name] = tk.StringVar(value=str(default))
        return self.vars[name]

    def _build_controls(self) -> None:
        outer = ttk.LabelFrame(self, text="Arrangement experiment: documents by shared morphism-shape category membership")
        outer.grid(row=0, column=0, sticky="ew", padx=8, pady=6)
        for i in range(12):
            outer.columnconfigure(i, weight=0)
        outer.columnconfigure(11, weight=1)

        ttk.Label(outer, text="shape_k").grid(row=0, column=0, sticky="w", padx=(8, 2), pady=3)
        ttk.Entry(outer, textvariable=self._var('shape_k', '48'), width=7).grid(row=0, column=1, sticky="w", padx=(2, 8), pady=3)
        ttk.Label(outer, text="max edges/doc").grid(row=0, column=2, sticky="w", padx=(8, 2), pady=3)
        ttk.Entry(outer, textvariable=self._var('max_edges_per_doc', '2000'), width=9).grid(row=0, column=3, sticky="w", padx=(2, 8), pady=3)
        ttk.Label(outer, text="min weight").grid(row=0, column=4, sticky="w", padx=(8, 2), pady=3)
        ttk.Entry(outer, textvariable=self._var('min_weight', '0.0'), width=8).grid(row=0, column=5, sticky="w", padx=(2, 8), pady=3)
        ttk.Label(outer, text="dir β").grid(row=0, column=6, sticky="w", padx=(8, 2), pady=3)
        ttk.Entry(outer, textvariable=self._var('dir_weight_beta', '1.0'), width=7).grid(row=0, column=7, sticky="w", padx=(2, 8), pady=3)
        ttk.Checkbutton(outer, text="include length", variable=self._var('include_length', True)).grid(row=0, column=8, sticky="w", padx=8, pady=3)
        ttk.Checkbutton(outer, text="include v-bin", variable=self._var('include_v_bin', False)).grid(row=0, column=9, sticky="w", padx=8, pady=3)

        ttk.Label(outer, text="weight mode").grid(row=1, column=0, sticky="w", padx=(8, 2), pady=3)
        ttk.Combobox(outer, textvariable=self._var('weight_mode', 'weighted'), values=['weighted', 'count'], width=10, state='readonly').grid(row=1, column=1, sticky="w", padx=(2, 8), pady=3)
        ttk.Checkbutton(outer, text="normalize membership", variable=self._var('normalize_membership', True)).grid(row=1, column=2, columnspan=2, sticky="w", padx=8, pady=3)
        ttk.Label(outer, text="similarity").grid(row=1, column=4, sticky="w", padx=(8, 2), pady=3)
        ttk.Combobox(outer, textvariable=self._var('similarity_metric', 'cosine'), values=['cosine', 'jensen_shannon'], width=15, state='readonly').grid(row=1, column=5, sticky="w", padx=(2, 8), pady=3)
        ttk.Label(outer, text="doc top-k").grid(row=1, column=6, sticky="w", padx=(8, 2), pady=3)
        ttk.Entry(outer, textvariable=self._var('doc_top_k_neighbors', '5'), width=7).grid(row=1, column=7, sticky="w", padx=(2, 8), pady=3)
        ttk.Label(outer, text="support ≥").grid(row=1, column=8, sticky="w", padx=(8, 2), pady=3)
        ttk.Entry(outer, textvariable=self._var('shape_support_threshold', '0.03'), width=8).grid(row=1, column=9, sticky="w", padx=(2, 8), pady=3)

        ttk.Label(outer, text="label source").grid(row=2, column=0, sticky="w", padx=(8, 2), pady=3)
        ttk.Combobox(outer, textvariable=self._var('collection_label_source', 'doc_id_regex'), values=['prefix_before_underscore', 'doc_id_regex', 'metadata_csv', 'none'], width=24, state='readonly').grid(row=2, column=1, columnspan=2, sticky="w", padx=(2, 8), pady=3)
        ttk.Label(outer, text="regex").grid(row=2, column=3, sticky="w", padx=(8, 2), pady=3)
        ttk.Entry(outer, textvariable=self._var('collection_label_regex', r'^([^_]+_[^_]+)_'), width=24).grid(row=2, column=4, columnspan=2, sticky="ew", padx=(2, 8), pady=3)
        ttk.Button(outer, text="Metadata CSV...", command=self.choose_metadata_csv).grid(row=2, column=6, padx=6, pady=3)
        ttk.Label(outer, textvariable=self._var('metadata_csv_path', '')).grid(row=2, column=7, columnspan=5, sticky="w", padx=6, pady=3)

        ttk.Button(outer, text="Build experiment", command=self.build_async).grid(row=3, column=0, padx=8, pady=5, sticky="ew")
        ttk.Button(outer, text="Load experiment PKL", command=self.load_experiment).grid(row=3, column=1, padx=6, pady=5, sticky="ew")
        ttk.Button(outer, text="Save experiment PKL", command=self.save_experiment).grid(row=3, column=2, padx=6, pady=5, sticky="ew")
        ttk.Button(outer, text="Export tables", command=self.export_tables).grid(row=3, column=3, padx=6, pady=5, sticky="ew")
        ttk.Button(outer, text="Refresh views", command=self.refresh_all_views).grid(row=3, column=4, padx=6, pady=5, sticky="ew")
        ttk.Label(outer, textvariable=self.status_var).grid(row=3, column=5, columnspan=7, sticky="ew", padx=8, pady=5)

    def _build_body(self) -> None:
        self.nb = ttk.Notebook(self)
        self.nb.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))
        self.summary_text = ScrollText(self.nb)
        self.nb.add(self.summary_text, text="Summary")
        self._build_shape_tab()
        self._build_membership_tab()
        self._build_doc_graph_tab()
        self._build_shape_graph_tab()
        self._build_shape_neighbors_tab()
        self._build_shape_bin_field_tab()
        self._build_collection_tab()
        self._build_roc_tab()
        self.summary_text.set_text(
            "This tab runs a simple arrangement experiment: cluster intra-document morphism shapes, "
            "build a document × shape-category membership matrix, then compare that arrangement to collection-type labels.\n\n"
            "Required: companion document_delta_dict.pkl. Optional: metadata CSV with columns doc_id, collection_type."
        )

    def _build_shape_tab(self) -> None:
        tab = ttk.Panedwindow(self.nb, orient='horizontal')
        self.nb.add(tab, text="Shape atlas / edges")
        left = ttk.Frame(tab); right = ttk.Frame(tab)
        left.rowconfigure(0, weight=1); left.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1); right.columnconfigure(0, weight=1)
        tab.add(left, weight=2); tab.add(right, weight=3)
        cols = ['shape_id', 'n_edges', 'n_docs', 'support_docs', 'top_collection_type', 'collection_type_entropy', 'mean_weight', 'mean_edge_quality', 'mean_len_delta', 'cos_vs_centroid', 'cos_vt_in_centroid', 'cos_st_centroid']
        self.shape_tree = ttk.Treeview(left, columns=cols, show='headings', selectmode='browse')
        for c in cols:
            self.shape_tree.heading(c, text=c)
            self.shape_tree.column(c, width=120 if c != 'top_collection_type' else 160, anchor='w')
        self.shape_tree.grid(row=0, column=0, sticky='nsew')
        sy = ttk.Scrollbar(left, orient='vertical', command=self.shape_tree.yview)
        self.shape_tree.configure(yscrollcommand=sy.set); sy.grid(row=0, column=1, sticky='ns')
        self.shape_tree.bind('<<TreeviewSelect>>', self._on_shape_select)
        ttk.Label(right, text="Representative edges for selected shape category").grid(row=0, column=0, sticky='w', padx=4, pady=4)
        ecols = ['rep_rank', 'shape_id', 'doc_id', 'collection_type', 'src_label', 'dst_label', 'w', 'edge_quality', 'len_delta', 'cos_vs', 'cos_vt_in', 'cos_st']
        self.rep_tree = ttk.Treeview(right, columns=ecols, show='headings')
        for c in ecols:
            self.rep_tree.heading(c, text=c)
            self.rep_tree.column(c, width=100 if c not in {'doc_id'} else 220, anchor='w')
        self.rep_tree.grid(row=1, column=0, sticky='nsew')
        ry = ttk.Scrollbar(right, orient='vertical', command=self.rep_tree.yview)
        self.rep_tree.configure(yscrollcommand=ry.set); ry.grid(row=1, column=1, sticky='ns')

    def _build_membership_tab(self) -> None:
        tab = ttk.Frame(self.nb)
        tab.rowconfigure(1, weight=1); tab.columnconfigure(0, weight=1)
        self.nb.add(tab, text="Membership heatmap")
        bar = ttk.Frame(tab)
        bar.grid(row=0, column=0, sticky='ew', padx=4, pady=4)
        ttk.Button(bar, text='Render heatmap', command=self.render_heatmap).pack(side='left', padx=4)
        ttk.Label(bar, text='row order').pack(side='left', padx=(10, 3))
        self.heatmap_row_order_var = self._var('heatmap_row_order', 'collection_dominant_shape')
        ttk.Combobox(
            bar,
            textvariable=self.heatmap_row_order_var,
            values=[
                'collection_dominant_shape',
                'collection_doc_id',
                'doc_id',
                'dominant_shape',
                'membership_pca',
                'collection_membership_pca',
                'random_within_collection',
            ],
            width=26,
            state='readonly',
        ).pack(side='left', padx=3)
        ttk.Button(bar, text='Render collection dominant-shape bars', command=self.render_dominant_bars).pack(side='left', padx=4)
        ttk.Label(bar, text='Selected doc:').pack(side='left', padx=(16, 3))
        self.selected_doc_var = tk.StringVar(value='')
        self.doc_combo = ttk.Combobox(bar, textvariable=self.selected_doc_var, values=[], width=42)
        self.doc_combo.pack(side='left', padx=3)
        ttk.Button(bar, text='Show similar docs', command=self.populate_similar_docs).pack(side='left', padx=4)
        paned = ttk.Panedwindow(tab, orient='horizontal')
        paned.grid(row=1, column=0, sticky='nsew')
        plot_frame = ttk.Frame(paned); plot_frame.rowconfigure(0, weight=1); plot_frame.columnconfigure(0, weight=1)
        self.heatmap_frame = plot_frame
        paned.add(plot_frame, weight=4)
        right = ttk.Frame(paned); right.rowconfigure(0, weight=1); right.columnconfigure(0, weight=1)
        sim_cols = ['rank', 'doc_id', 'collection_type', 'similarity', 'same_collection_type', 'top_shared_shapes', 'top_differing_shapes']
        self.similar_tree = ttk.Treeview(right, columns=sim_cols, show='headings')
        for c in sim_cols:
            self.similar_tree.heading(c, text=c)
            self.similar_tree.column(c, width=110 if c not in {'doc_id', 'top_shared_shapes', 'top_differing_shapes'} else 210, anchor='w')
        self.similar_tree.grid(row=0, column=0, sticky='nsew')
        sim_y = ttk.Scrollbar(right, orient='vertical', command=self.similar_tree.yview)
        self.similar_tree.configure(yscrollcommand=sim_y.set); sim_y.grid(row=0, column=1, sticky='ns')
        paned.add(right, weight=2)

    def _build_doc_graph_tab(self) -> None:
        tab = ttk.Frame(self.nb)
        tab.rowconfigure(1, weight=1); tab.columnconfigure(0, weight=1)
        self.nb.add(tab, text="Document graph")
        bar = ttk.Frame(tab); bar.grid(row=0, column=0, sticky='ew', padx=4, pady=4)
        ttk.Button(bar, text='Render document graph', command=self.render_doc_graph).pack(side='left', padx=4)
        ttk.Label(bar, text='Node color = collection type; edges = top-k similarity over membership rows').pack(side='left', padx=12)
        self.docgraph_frame = ttk.Frame(tab); self.docgraph_frame.rowconfigure(0, weight=1); self.docgraph_frame.columnconfigure(0, weight=1)
        self.docgraph_frame.grid(row=1, column=0, sticky='nsew')

    def _build_shape_graph_tab(self) -> None:
        tab = ttk.Frame(self.nb)
        tab.rowconfigure(1, weight=1); tab.columnconfigure(0, weight=1)
        self.nb.add(tab, text="Shape co-occurrence graph")
        bar = ttk.Frame(tab); bar.grid(row=0, column=0, sticky='ew', padx=4, pady=4)
        ttk.Button(bar, text='Render shape co-occurrence graph', command=self.render_shape_graph).pack(side='left', padx=4)
        ttk.Label(bar, text='Nodes = shape categories; edges = co-occurrence in documents above support threshold').pack(side='left', padx=12)
        self.shapegraph_frame = ttk.Frame(tab); self.shapegraph_frame.rowconfigure(0, weight=1); self.shapegraph_frame.columnconfigure(0, weight=1)
        self.shapegraph_frame.grid(row=1, column=0, sticky='nsew')

    def _build_shape_neighbors_tab(self) -> None:
        tab = ttk.Panedwindow(self.nb, orient='horizontal')
        self.nb.add(tab, text="Shape neighbor inspector")

        left = ttk.Frame(tab)
        left.rowconfigure(1, weight=1)
        left.columnconfigure(0, weight=1)
        right = ttk.Frame(tab)
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)
        tab.add(left, weight=3)
        tab.add(right, weight=2)

        bar = ttk.Frame(left)
        bar.grid(row=0, column=0, sticky='ew', padx=4, pady=4)
        ttk.Label(bar, text='Selected shape').pack(side='left', padx=(4, 2))
        self.selected_shape_neighbor_var = tk.StringVar(value='')
        self.shape_neighbor_combo = ttk.Combobox(bar, textvariable=self.selected_shape_neighbor_var, values=[], width=12)
        self.shape_neighbor_combo.pack(side='left', padx=3)
        self.shape_neighbor_combo.bind('<<ComboboxSelected>>', lambda _e: self.populate_shape_neighbors())
        ttk.Button(bar, text='Refresh neighbor tables', command=self.populate_shape_neighbors).pack(side='left', padx=4)
        ttk.Button(bar, text='Render neighbor graph', command=self.render_shape_neighbor_graph).pack(side='left', padx=4)
        ttk.Label(bar, text='Compare centroid geometry, same-document co-occurrence, and collection-profile similarity.').pack(side='left', padx=12)

        inner = ttk.Panedwindow(left, orient='vertical')
        inner.grid(row=1, column=0, sticky='nsew')

        def _make_neighbor_tree(title: str, cols: List[str]) -> ttk.Treeview:
            lf = ttk.LabelFrame(inner, text=title)
            lf.rowconfigure(0, weight=1)
            lf.columnconfigure(0, weight=1)
            tree = ttk.Treeview(lf, columns=cols, show='headings', height=7)
            for c in cols:
                tree.heading(c, text=c)
                width = 95
                if c in {'top_collection_type', 'profile_top_collection_type'}:
                    width = 165
                elif c == 'shape_id':
                    width = 70
                tree.column(c, width=width, anchor='w')
            tree.grid(row=0, column=0, sticky='nsew')
            y = ttk.Scrollbar(lf, orient='vertical', command=tree.yview)
            tree.configure(yscrollcommand=y.set)
            y.grid(row=0, column=1, sticky='ns')
            inner.add(lf, weight=1)
            return tree

        self.centroid_neighbor_tree = _make_neighbor_tree(
            'Shape-centroid nearest neighbors',
            ['rank', 'shape_id', 'centroid_distance', 'centroid_cosine', 'support_docs', 'n_edges', 'top_collection_type'],
        )
        self.cooccurrence_neighbor_tree = _make_neighbor_tree(
            'Shape co-occurrence neighbors',
            ['rank', 'shape_id', 'cooccurrence_score', 'cooccurring_docs', 'jaccard', 'lift', 'support_docs', 'top_collection_type'],
        )
        self.profile_neighbor_tree = _make_neighbor_tree(
            'Collection-profile neighbors',
            ['rank', 'shape_id', 'profile_cosine', 'profile_l2', 'profile_top_collection_type', 'profile_top_membership', 'support_docs', 'top_collection_type'],
        )

        graph_bar = ttk.Frame(right)
        graph_bar.grid(row=0, column=0, sticky='ew', padx=4, pady=4)
        ttk.Label(graph_bar, text='Neighbor graph: selected shape + union of top neighbors').pack(side='left', padx=4)
        self.shapeneighbor_frame = ttk.Frame(right)
        self.shapeneighbor_frame.rowconfigure(0, weight=1)
        self.shapeneighbor_frame.columnconfigure(0, weight=1)
        self.shapeneighbor_frame.grid(row=1, column=0, sticky='nsew')


    def _build_shape_bin_field_tab(self) -> None:
        """Build Shape Bin Field workspace for selected morphism-shape categories."""
        tab = ttk.Panedwindow(self.nb, orient='vertical')
        self.nb.add(tab, text="Shape Bin Field")
        self.shape_bin_field_tab = tab

        top = ttk.Frame(tab)
        top.columnconfigure(9, weight=1)
        tab.add(top, weight=0)
        self.selected_shape_bin_var = tk.StringVar(value='')
        self.shape_bin_member_filter_var = tk.StringVar(value='all')
        self.shape_bin_color_by_var = tk.StringVar(value='collection')
        self.shape_bin_length_mode_var = tk.StringVar(value='preserve_length')
        self.shape_bin_residual_mode_var = tk.StringVar(value='centroid_positions')
        self.shape_bin_max_members_var = tk.StringVar(value='500')
        self.shape_bin_max_context_var = tk.StringVar(value='24')
        self.shape_bin_pc1_scale_var = tk.StringVar(value='0.28')
        self.shape_bin_show_prototype_var = tk.BooleanVar(value=True)
        self.shape_bin_show_cones_var = tk.BooleanVar(value=True)
        self.shape_bin_q_threshold_var = tk.StringVar(value='0.0')

        ttk.Label(top, text='shape bin').grid(row=0, column=0, padx=(6, 2), pady=3, sticky='w')
        self.shape_bin_combo = ttk.Combobox(top, textvariable=self.selected_shape_bin_var, values=[], width=10)
        self.shape_bin_combo.grid(row=0, column=1, padx=(2, 8), pady=3, sticky='w')
        self.shape_bin_combo.bind('<<ComboboxSelected>>', lambda _e: self.populate_shape_bin_field())
        ttk.Button(top, text='Refresh field', command=self.populate_shape_bin_field).grid(row=0, column=2, padx=4, pady=3)
        ttk.Button(top, text='Render canonical field', command=self.render_shape_bin_canonical_field).grid(row=0, column=3, padx=4, pady=3)
        ttk.Button(top, text='Render residual context', command=self.render_shape_bin_residual_context).grid(row=0, column=4, padx=4, pady=3)
        ttk.Button(top, text='Render distributions', command=self.render_shape_bin_distributions).grid(row=0, column=5, padx=4, pady=3)
        ttk.Button(top, text='Render composition', command=self.render_shape_bin_composition).grid(row=0, column=6, padx=4, pady=3)

        ttk.Label(top, text='filter').grid(row=1, column=0, padx=(6, 2), pady=3, sticky='w')
        self.shape_bin_filter_combo = ttk.Combobox(top, textvariable=self.shape_bin_member_filter_var, values=['all'], width=22)
        self.shape_bin_filter_combo.grid(row=1, column=1, padx=(2, 8), pady=3, sticky='w')
        ttk.Label(top, text='color by').grid(row=1, column=2, padx=(6, 2), pady=3, sticky='w')
        ttk.Combobox(top, textvariable=self.shape_bin_color_by_var, values=['collection', 'edge_quality', 'prototype_distance', 'len_delta', 'cos_vs', 'cos_vt_in'], width=18, state='readonly').grid(row=1, column=3, padx=(2, 8), pady=3, sticky='w')
        ttk.Label(top, text='length').grid(row=1, column=4, padx=(6, 2), pady=3, sticky='w')
        ttk.Combobox(top, textvariable=self.shape_bin_length_mode_var, values=['preserve_length', 'normalize_length'], width=17, state='readonly').grid(row=1, column=5, padx=(2, 8), pady=3, sticky='w')
        ttk.Label(top, text='residual mode').grid(row=1, column=6, padx=(6, 2), pady=3, sticky='w')
        ttk.Combobox(top, textvariable=self.shape_bin_residual_mode_var, values=['centroid_positions', 'centroid_directions'], width=19, state='readonly').grid(row=1, column=7, padx=(2, 8), pady=3, sticky='w')

        ttk.Label(top, text='max canonical').grid(row=2, column=0, padx=(6, 2), pady=3, sticky='w')
        ttk.Entry(top, textvariable=self.shape_bin_max_members_var, width=8).grid(row=2, column=1, padx=(2, 8), pady=3, sticky='w')
        ttk.Label(top, text='max context').grid(row=2, column=2, padx=(6, 2), pady=3, sticky='w')
        ttk.Entry(top, textvariable=self.shape_bin_max_context_var, width=8).grid(row=2, column=3, padx=(2, 8), pady=3, sticky='w')
        ttk.Label(top, text='PC1 scale').grid(row=2, column=4, padx=(6, 2), pady=3, sticky='w')
        ttk.Entry(top, textvariable=self.shape_bin_pc1_scale_var, width=8).grid(row=2, column=5, padx=(2, 8), pady=3, sticky='w')
        ttk.Label(top, text='min Q').grid(row=2, column=6, padx=(6, 2), pady=3, sticky='w')
        ttk.Entry(top, textvariable=self.shape_bin_q_threshold_var, width=8).grid(row=2, column=7, padx=(2, 8), pady=3, sticky='w')
        ttk.Checkbutton(top, text='prototype', variable=self.shape_bin_show_prototype_var).grid(row=2, column=8, padx=6, pady=3, sticky='w')
        ttk.Checkbutton(top, text='PC1 cones', variable=self.shape_bin_show_cones_var).grid(row=2, column=9, padx=6, pady=3, sticky='w')

        body = ttk.Notebook(tab)
        tab.add(body, weight=1)

        self.shape_bin_report = ScrollText(body)
        body.add(self.shape_bin_report, text='Scientific report')

        canonical_tab = ttk.Frame(body); canonical_tab.rowconfigure(0, weight=1); canonical_tab.columnconfigure(0, weight=1)
        self.shape_bin_canonical_frame = canonical_tab
        body.add(canonical_tab, text='Canonicalized field')

        residual_tab = ttk.Frame(body); residual_tab.rowconfigure(0, weight=1); residual_tab.columnconfigure(0, weight=1)
        self.shape_bin_residual_frame = residual_tab
        body.add(residual_tab, text='Residual-space context')

        dist_tab = ttk.Frame(body); dist_tab.rowconfigure(0, weight=1); dist_tab.columnconfigure(0, weight=1)
        self.shape_bin_dist_frame = dist_tab
        body.add(dist_tab, text='Feature distributions')

        comp_tab = ttk.Frame(body); comp_tab.rowconfigure(0, weight=1); comp_tab.columnconfigure(0, weight=1)
        self.shape_bin_comp_frame = comp_tab
        body.add(comp_tab, text='Collection composition')

        tables = ttk.Panedwindow(body, orient='horizontal')
        body.add(tables, text='Members / metrics')
        metric_frame = ttk.Frame(tables); metric_frame.rowconfigure(0, weight=1); metric_frame.columnconfigure(0, weight=1)
        member_frame = ttk.Frame(tables); member_frame.rowconfigure(0, weight=1); member_frame.columnconfigure(0, weight=1)
        tables.add(metric_frame, weight=1); tables.add(member_frame, weight=2)
        mcols = ['metric', 'value', 'interpretation']
        self.shape_bin_metric_tree = ttk.Treeview(metric_frame, columns=mcols, show='headings')
        for c in mcols:
            self.shape_bin_metric_tree.heading(c, text=c)
            self.shape_bin_metric_tree.column(c, width=160 if c != 'interpretation' else 360, anchor='w')
        self.shape_bin_metric_tree.grid(row=0, column=0, sticky='nsew')
        my = ttk.Scrollbar(metric_frame, orient='vertical', command=self.shape_bin_metric_tree.yview)
        self.shape_bin_metric_tree.configure(yscrollcommand=my.set); my.grid(row=0, column=1, sticky='ns')
        cols = ['shape_row', 'doc_id', 'collection_type', 'src_label', 'dst_label', 'w', 'edge_quality', 'len_delta', 'cos_vs', 'cos_vt_in', 'cos_st', 'prototype_distance']
        self.shape_bin_member_tree = ttk.Treeview(member_frame, columns=cols, show='headings')
        for c in cols:
            self.shape_bin_member_tree.heading(c, text=c)
            self.shape_bin_member_tree.column(c, width=110 if c not in {'doc_id', 'collection_type'} else 190, anchor='w')
        self.shape_bin_member_tree.grid(row=0, column=0, sticky='nsew')
        yy = ttk.Scrollbar(member_frame, orient='vertical', command=self.shape_bin_member_tree.yview)
        self.shape_bin_member_tree.configure(yscrollcommand=yy.set); yy.grid(row=0, column=1, sticky='ns')
        xx = ttk.Scrollbar(member_frame, orient='horizontal', command=self.shape_bin_member_tree.xview)
        self.shape_bin_member_tree.configure(xscrollcommand=xx.set); xx.grid(row=1, column=0, sticky='ew')
        self.shape_bin_report.set_text('Build or load an arrangement experiment, select a shape bin, then render the canonicalized morphism field and residual-space context views.')

    def update_shape_bin_field_selector(self) -> None:
        if not hasattr(self, 'shape_bin_combo'):
            return
        vals: List[str] = []
        collections = ['all']
        if isinstance(self.experiment, dict):
            C = np.asarray(self.experiment.get('shape_centroids', []), dtype=float)
            k = int(C.shape[0]) if C.ndim == 2 and C.shape[0] else int((self.experiment.get('summary') or {}).get('shape_k_effective', 0) or 0)
            vals = [f'S{i}' for i in range(k)]
            collections += sorted({str(c) for c in self.experiment.get('collection_type', []) if str(c).strip()})
        self.shape_bin_combo.configure(values=vals)
        self.shape_bin_filter_combo.configure(values=collections)
        if vals and not self.selected_shape_bin_var.get():
            self.selected_shape_bin_var.set(vals[0])

    def _selected_shape_bin_id(self) -> Optional[int]:
        if not hasattr(self, 'selected_shape_bin_var'):
            return None
        return _arr_as_shape_id(self.selected_shape_bin_var.get())

    def _shape_bin_members_filtered(self) -> List[Dict[str, Any]]:
        if not isinstance(self.experiment, dict):
            return []
        sid = self._selected_shape_bin_id()
        if sid is None:
            return []
        members = _arr_shape_member_rows(self.experiment, sid)
        min_q = _safe_float(self.shape_bin_q_threshold_var.get() if hasattr(self, 'shape_bin_q_threshold_var') else '0', 0.0)
        if min_q > 0:
            members = [r for r in members if _safe_float(r.get('edge_quality'), 0.0) >= min_q]
        filt = str(self.shape_bin_member_filter_var.get() if hasattr(self, 'shape_bin_member_filter_var') else 'all')
        if filt and filt != 'all':
            members = [r for r in members if str(r.get('collection_type', '')) == filt]
        return members

    def populate_shape_bin_field(self) -> None:
        if not hasattr(self, 'shape_bin_report'):
            return
        self.update_shape_bin_field_selector()
        if not isinstance(self.experiment, dict):
            self.shape_bin_report.set_text('No arrangement experiment is loaded.')
            return
        sid = self._selected_shape_bin_id()
        if sid is None:
            self.shape_bin_report.set_text('Select a shape bin.')
            return
        members = self._shape_bin_members_filtered()
        all_summary = _arr_shape_bin_field_summary(self.experiment, sid)
        filt_summary = _arr_shape_bin_field_summary({**self.experiment, 'shape_records': members, 'shape_labels': np.asarray([sid] * len(members), dtype=int), 'shape_features': np.zeros((len(members), 0))}, 0) if False else all_summary
        # Metrics table.
        if hasattr(self, 'shape_bin_metric_tree'):
            for item in self.shape_bin_metric_tree.get_children():
                self.shape_bin_metric_tree.delete(item)
            metric_rows = [
                ('member edges', all_summary.get('n_edges'), 'number of directed morphism edges assigned to this bin'),
                ('supporting documents', all_summary.get('n_docs'), 'unique documents contributing at least one edge'),
                ('supporting collections', all_summary.get('n_collections'), 'collection types represented among member edges'),
                ('top collection', all_summary.get('top_collection_type'), 'collection contributing the most member edges'),
                ('collection specificity', all_summary.get('collection_specificity_score'), '1 - normalized collection entropy; high means collection-specific'),
                ('coherence score', all_summary.get('bin_coherence_score'), 'inverse prototype-distance dispersion; high means a tighter shape family'),
                ('mean Δ length', all_summary.get('len_delta_mean'), 'average transition scale; length is part of the current shape basis'),
                ('mean cos(v,src PC1)', all_summary.get('cos_vs_mean'), 'source-flow alignment'),
                ('mean cos(-v,dst PC1)', all_summary.get('cos_vt_in_mean'), 'destination incoming-flow alignment'),
                ('mean cos(src,dst PC1)', all_summary.get('cos_st_mean'), 'endpoint orientation concordance'),
                ('mean edge Q', all_summary.get('edge_quality_mean'), 'semantic quality of bin member edges'),
                ('asymmetry index', all_summary.get('transition_asymmetry_index'), 'mean absolute source/destination alignment gap'),
                ('semantic breadth', all_summary.get('semantic_breadth_score'), '1 - mean raw SBERT cosine among supporting docs, when available'),
                ('structural breadth', all_summary.get('structural_breadth_score'), 'coherence × quality × semantic breadth compound score'),
                ('collection stratification', all_summary.get('collection_stratification_score'), 'feature separation of collection variants inside the bin'),
            ]
            for i, (m, v, desc) in enumerate(metric_rows):
                val = _fmt_float(v, 5) if isinstance(v, (float, np.floating)) else str(v)
                self.shape_bin_metric_tree.insert('', 'end', iid=str(i), values=[m, val, desc])
        # Member table.
        if hasattr(self, 'shape_bin_member_tree'):
            for item in self.shape_bin_member_tree.get_children():
                self.shape_bin_member_tree.delete(item)
            cols = list(self.shape_bin_member_tree['columns'])
            rows = sorted(members, key=lambda r: (_safe_float(r.get('prototype_distance'), 9999.0), -_safe_float(r.get('edge_quality'), 0.0)))
            for i, r in enumerate(rows[:5000]):
                vals = []
                for c in cols:
                    v = r.get(c, '')
                    vals.append(_fmt_float(v, 5) if isinstance(v, (float, np.floating)) else str(v))
                self.shape_bin_member_tree.insert('', 'end', iid=str(i), values=vals)
        lines = self._shape_bin_report_lines(sid, all_summary)
        self.shape_bin_report.set_text('\n'.join(lines))
        self.status_var.set(f'Shape Bin Field: S{sid}; {len(members):,} displayed/filter-matching members; {all_summary.get("n_edges", 0):,} total bin edges.')

    def _shape_bin_report_lines(self, sid: int, s: Dict[str, Any]) -> List[str]:
        p = self.experiment.get('params', {}) if isinstance(self.experiment, dict) else {}
        lines = [
            f'Shape Bin Field report: S{sid}',
            '',
            'Basis interpretation',
            f'  dir_weight_beta: {p.get("dir_weight_beta", "")}',
            f'  include_length: {p.get("include_length", "")}',
            f'  include_v_bin: {p.get("include_v_bin", "")}',
            '  With include_v_bin=false, the bin is defined by edge-local transition form rather than absolute residual-space direction.',
            '  With include_length=true, displacement scale is part of the shape definition.',
            '  With dir_weight_beta=0.0, PC1-flow coherence is measured in this workspace but not used as a membership weight gate.',
            '',
            'Core metrics',
        ]
        keys = [
            ('n_edges', 'member edges'), ('n_docs', 'supporting documents'), ('n_collections', 'supporting collections'),
            ('top_collection_type', 'top collection'), ('collection_specificity_score', 'collection specificity'),
            ('bin_coherence_score', 'coherence'), ('len_delta_mean', 'mean Δ length'),
            ('cos_vs_mean', 'source-flow alignment'), ('cos_vt_in_mean', 'destination-flow alignment'),
            ('cos_st_mean', 'endpoint concordance'), ('edge_quality_mean', 'mean edge Q'),
            ('transition_asymmetry_index', 'asymmetry'), ('semantic_breadth_score', 'semantic breadth'),
            ('structural_breadth_score', 'structural breadth'), ('collection_stratification_score', 'collection stratification'),
        ]
        for k, label in keys:
            v = s.get(k, '')
            lines.append(f'  {label}: {_fmt_float(v, 5) if isinstance(v, (float, np.floating)) else v}')
        lines += [
            '',
            'Scientific characterization',
            f'  Generic vs collection-specific: {s.get("genericity_label", "")}',
            f'  Tight vs loose shape family: {s.get("coherence_label", "")}',
            f'  Semantically broad but structurally stable: {s.get("semantic_breadth_label", "")}',
            f'  Recurring asymmetry: {s.get("transition_asymmetry_label", "")}',
            f'  Bin stratification by collection family: {s.get("collection_stratification_label", "")}',
            '',
            'View guidance',
            '  Canonicalized field: aligns all member edges into a common edge-local frame; best for inspecting the defining transition form.',
            '  Residual-space context: projects actual CDM centroid locations for representative edges; best for seeing where the form occurred in documents.',
        ]
        return lines

    def _shape_bin_sample_members(self, members: List[Dict[str, Any]], max_n: int) -> List[Dict[str, Any]]:
        if len(members) <= max_n:
            return list(members)
        # Prefer central/prototype-near members while preserving collection diversity.
        by_coll: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in members:
            by_coll[str(r.get('collection_type', 'Unknown'))].append(r)
        out: List[Dict[str, Any]] = []
        per_coll = max(1, int(max_n / max(1, len(by_coll))))
        for _c, rows in sorted(by_coll.items()):
            rows = sorted(rows, key=lambda r: (_safe_float(r.get('prototype_distance'), 9999.0), -_safe_float(r.get('edge_quality'), 0.0)))
            out.extend(rows[:per_coll])
        if len(out) < max_n:
            chosen_ids = {id(r) for r in out}
            rest = [r for r in sorted(members, key=lambda r: (_safe_float(r.get('prototype_distance'), 9999.0), -_safe_float(r.get('edge_quality'), 0.0))) if id(r) not in chosen_ids]
            out.extend(rest[: max_n - len(out)])
        return out[:max_n]

    def render_shape_bin_canonical_field(self) -> None:
        if not isinstance(self.experiment, dict):
            messagebox.showinfo('No experiment', 'Build or load an arrangement experiment first.')
            return
        sid = self._selected_shape_bin_id()
        if sid is None:
            messagebox.showinfo('No shape selected', 'Select a shape bin first.')
            return
        members = self._shape_bin_members_filtered()
        if not members:
            messagebox.showinfo('No members', f'No members found for S{sid} under current filters.')
            return
        max_n = max(1, _safe_int(self.shape_bin_max_members_var.get(), 500))
        rows = self._shape_bin_sample_members(members, max_n)
        color_by = str(self.shape_bin_color_by_var.get() or 'collection')
        length_mode = str(self.shape_bin_length_mode_var.get() or 'preserve_length')
        pc1_scale = max(0.01, _safe_float(self.shape_bin_pc1_scale_var.get(), 0.28))
        lens = np.asarray([_safe_float(r.get('len_delta'), 1.0) for r in rows], dtype=float)
        default_L = float(np.median(lens[lens > 0])) if np.any(lens > 0) else 1.0
        # Palette / numeric color setup.
        collections = sorted({str(r.get('collection_type', 'Unknown')) for r in rows})
        cmap = cm.get_cmap('tab20', max(1, len(collections)))
        coll_color = {c: cmap(i) for i, c in enumerate(collections)}
        numeric_vals = None
        norm = None
        if color_by != 'collection':
            numeric_vals = np.asarray([_safe_float(r.get(color_by), 0.0) for r in rows], dtype=float)
            vmin = float(np.nanmin(numeric_vals)) if numeric_vals.size else 0.0
            vmax = float(np.nanmax(numeric_vals)) if numeric_vals.size else 1.0
            if vmax <= vmin:
                vmax = vmin + 1.0
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        fig = Figure(figsize=(10.8, 8.0), dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        all_src = []; all_dst = []; all_s = []; all_t = []
        for idx, r in enumerate(rows):
            L = 2.0 if length_mode == 'normalize_length' else max(0.05, _safe_float(r.get('len_delta'), default_L))
            Ldisp = L / max(default_L, 1e-9) if length_mode == 'preserve_length' else 1.0
            src = np.asarray([-0.5 * Ldisp, 0.0, 0.0], dtype=float)
            dst = np.asarray([0.5 * Ldisp, 0.0, 0.0], dtype=float)
            _v, svec, tvec = _arr_canonical_vectors_from_shape_features(_safe_float(r.get('cos_vs'), 0.0), _safe_float(r.get('cos_vt_in'), 0.0), _safe_float(r.get('cos_st'), 0.0))
            if color_by == 'collection':
                color = coll_color.get(str(r.get('collection_type', 'Unknown')), (0.5, 0.5, 0.5, 1.0))
            else:
                color = cm.viridis(norm(numeric_vals[idx])) if norm is not None and numeric_vals is not None else (0.3, 0.3, 0.3, 1.0)
            alpha = 0.18 if len(rows) > 120 else 0.28
            ax.plot([src[0], dst[0]], [src[1], dst[1]], [src[2], dst[2]], color=color, alpha=alpha, linewidth=0.7)
            ax.scatter([src[0]], [src[1]], [src[2]], s=8, color=color, alpha=0.45, depthshade=False)
            ax.scatter([dst[0]], [dst[1]], [dst[2]], s=8, color=color, alpha=0.45, depthshade=False, marker='^')
            ax.quiver(src[0], src[1], src[2], svec[0] * pc1_scale, svec[1] * pc1_scale, svec[2] * pc1_scale, color=color, alpha=0.22, linewidth=0.55, arrow_length_ratio=0.22)
            ax.quiver(dst[0], dst[1], dst[2], tvec[0] * pc1_scale, tvec[1] * pc1_scale, tvec[2] * pc1_scale, color=color, alpha=0.22, linewidth=0.55, arrow_length_ratio=0.22)
            all_src.append(src); all_dst.append(dst); all_s.append(svec); all_t.append(tvec)
        # Prototype / mean field.
        if bool(self.shape_bin_show_prototype_var.get()) and rows:
            Lm = 2.0 if length_mode == 'normalize_length' else float(np.mean([max(0.05, _safe_float(r.get('len_delta'), default_L)) for r in rows]) / max(default_L, 1e-9))
            psrc = np.asarray([-0.5 * Lm, 0.0, 0.0]); pdst = np.asarray([0.5 * Lm, 0.0, 0.0])
            av = {k: float(np.mean([_safe_float(r.get(k), 0.0) for r in rows])) for k in ['cos_vs', 'cos_vt_in', 'cos_st']}
            _v, ps, pt = _arr_canonical_vectors_from_shape_features(av['cos_vs'], av['cos_vt_in'], av['cos_st'])
            ax.plot([psrc[0], pdst[0]], [0, 0], [0, 0], color='black', linewidth=4.0, alpha=0.95, label='bin prototype')
            ax.scatter([psrc[0]], [0], [0], s=170, color='#d81b60', edgecolors='black', linewidths=1.0, depthshade=False, label='prototype source')
            ax.scatter([pdst[0]], [0], [0], s=170, color='#00acc1', edgecolors='black', linewidths=1.0, marker='^', depthshade=False, label='prototype destination')
            ax.quiver(psrc[0], 0, 0, ps[0] * pc1_scale * 1.5, ps[1] * pc1_scale * 1.5, ps[2] * pc1_scale * 1.5, color='#d81b60', linewidth=2.6, arrow_length_ratio=0.18)
            ax.quiver(pdst[0], 0, 0, pt[0] * pc1_scale * 1.5, pt[1] * pc1_scale * 1.5, pt[2] * pc1_scale * 1.5, color='#00acc1', linewidth=2.6, arrow_length_ratio=0.18)
        # PC1 cone hints: draw one-standard-deviation angular text, not mesh cones.
        if bool(self.shape_bin_show_cones_var.get()) and rows:
            src_angles = [math.degrees(math.acos(max(-1.0, min(1.0, _safe_float(r.get('cos_vs'), 0.0))))) for r in rows]
            dst_angles = [math.degrees(math.acos(max(-1.0, min(1.0, _safe_float(r.get('cos_vt_in'), 0.0))))) for r in rows]
            txt = f'source PC1 angle to flow: μ={np.mean(src_angles):.1f}° σ={np.std(src_angles):.1f}°\n' \
                  f'destination PC1 angle to incoming flow: μ={np.mean(dst_angles):.1f}° σ={np.std(dst_angles):.1f}°'
            ax.text2D(0.02, 0.95, txt, transform=ax.transAxes, fontsize=8, va='top')
        ax.set_title(f'Canonicalized morphism field for S{sid} ({len(rows):,}/{len(members):,} displayed)')
        ax.set_xlabel('canonical flow axis +Δ')
        ax.set_ylabel('endpoint PC1 plane / variation')
        ax.set_zlabel('residual PC1 variation')
        try:
            if color_by == 'collection' and len(collections) <= 14:
                for c in collections:
                    ax.plot([], [], [], color=coll_color[c], label=c)
            elif color_by != 'collection' and norm is not None:
                sm = cm.ScalarMappable(norm=norm, cmap='viridis'); sm.set_array([])
                fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.08, label=color_by)
            ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize=7)
        except Exception:
            pass
        ax.view_init(elev=22, azim=-58)
        _set_axes_equal_3d(ax)
        fig.tight_layout()
        self._draw_figure('shape_bin_canonical', self.shape_bin_canonical_frame, fig)
        self.status_var.set(f'Rendered canonicalized morphism field for S{sid}.')

    def render_shape_bin_residual_context(self) -> None:
        if not isinstance(self.experiment, dict):
            messagebox.showinfo('No experiment', 'Build or load an arrangement experiment first.')
            return
        if not isinstance(self.model.document_delta_dict, dict) or not self.model.document_delta_dict:
            messagebox.showinfo('document_delta_dict required', 'Open companion document_delta_dict.pkl to render residual-space context locations.')
            return
        sid = self._selected_shape_bin_id()
        if sid is None:
            return
        members = self._shape_bin_members_filtered()
        if not members:
            return
        max_n = max(1, _safe_int(self.shape_bin_max_context_var.get(), 24))
        rows = self._shape_bin_sample_members(members, max_n)
        mode = str(self.shape_bin_residual_mode_var.get() or 'centroid_positions')
        pc1_scale = max(0.01, _safe_float(self.shape_bin_pc1_scale_var.get(), 0.28))
        entries = []
        fit_points = []
        warnings = []
        for r in rows:
            try:
                sg = self.model.cluster_geometry(str(r.get('doc_id')), r.get('src_label'))
                dg = self.model.cluster_geometry(str(r.get('doc_id')), r.get('dst_label'))
                sc = np.asarray(sg.get('centroid'), dtype=float).reshape(-1)
                dc = np.asarray(dg.get('centroid'), dtype=float).reshape(-1)
                sp = np.asarray(sg.get('pc1'), dtype=float).reshape(-1)
                dp = np.asarray(dg.get('pc1'), dtype=float).reshape(-1)
                if mode == 'centroid_directions':
                    sc0 = _arr_unit(sc); dc0 = _arr_unit(dc)
                else:
                    sc0 = sc; dc0 = dc
                delta = dc0 - sc0
                v = _arr_unit(delta)
                if sp.size == sc.size and np.dot(sp, v) < 0:
                    sp = -sp
                if dp.size == dc.size and np.dot(dp, -v) < 0:
                    dp = -dp
                entries.append({'row': r, 'src': sc0, 'dst': dc0, 'src_pc1': sp, 'dst_pc1': dp, 'collection': str(r.get('collection_type', 'Unknown'))})
                for pnt in (sc0, dc0):
                    fit_points.append(pnt.reshape(1, -1))
                if sp.size == sc0.size:
                    fit_points.append((sc0 + sp * pc1_scale).reshape(1, -1))
                if dp.size == dc0.size:
                    fit_points.append((dc0 + dp * pc1_scale).reshape(1, -1))
            except Exception as ex:
                warnings.append(f"{r.get('doc_id')} {r.get('src_label')}→{r.get('dst_label')}: {ex}")
        if not entries:
            messagebox.showinfo('No residual geometry', 'Could not resolve selected bin member edges in document_delta_dict.')
            return
        center, basis = _fit_pca_projection(fit_points)
        collections = sorted({e['collection'] for e in entries})
        cmap = cm.get_cmap('tab20', max(1, len(collections)))
        colors = {c: cmap(i) for i, c in enumerate(collections)}
        fig = Figure(figsize=(10.6, 8.0), dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        # Visual guide: draw a faint unit circle in the first two display axes.
        th = np.linspace(0, 2 * math.pi, 160)
        ax.plot(np.cos(th), np.sin(th), np.zeros_like(th), color='0.86', linewidth=0.8, alpha=0.6, label='visual unit circle guide')
        for e in entries:
            color = colors[e['collection']]
            s3 = _project_points(np.asarray(e['src']), center, basis)[0]
            d3 = _project_points(np.asarray(e['dst']), center, basis)[0]
            ax.plot([s3[0], d3[0]], [s3[1], d3[1]], [s3[2], d3[2]], color=color, alpha=0.60, linewidth=1.6)
            ax.scatter([s3[0]], [s3[1]], [s3[2]], s=55, color=color, edgecolors='black', linewidths=0.5, marker='o', depthshade=False)
            ax.scatter([d3[0]], [d3[1]], [d3[2]], s=55, color=color, edgecolors='black', linewidths=0.5, marker='^', depthshade=False)
            sp = np.asarray(e['src_pc1'], dtype=float).reshape(-1)
            dp = np.asarray(e['dst_pc1'], dtype=float).reshape(-1)
            if sp.size == np.asarray(e['src']).size:
                sp3 = _project_points(np.asarray(e['src']) + sp * pc1_scale, center, basis)[0] - s3
                ax.quiver(s3[0], s3[1], s3[2], sp3[0], sp3[1], sp3[2], color=color, alpha=0.45, linewidth=0.9, arrow_length_ratio=0.18)
            if dp.size == np.asarray(e['dst']).size:
                dp3 = _project_points(np.asarray(e['dst']) + dp * pc1_scale, center, basis)[0] - d3
                ax.quiver(d3[0], d3[1], d3[2], dp3[0], dp3[1], dp3[2], color=color, alpha=0.45, linewidth=0.9, arrow_length_ratio=0.18)
        for c in collections:
            ax.plot([], [], [], color=colors[c], label=c)
        ax.set_title(f'Residual-space morphism ball context for S{sid}\nmode={mode}; local PCA projection of actual CDM centroid/PC1 locations')
        ax.set_xlabel('residual context PCA 1'); ax.set_ylabel('residual context PCA 2'); ax.set_zlabel('residual context PCA 3')
        ax.view_init(elev=23, azim=-53)
        _set_axes_equal_3d(ax)
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize=7)
        fig.tight_layout()
        self._draw_figure('shape_bin_residual', self.shape_bin_residual_frame, fig)
        if warnings:
            self.status_var.set(f'Rendered residual context for S{sid} with {len(entries)} edges; {len(warnings)} geometry warnings.')
        else:
            self.status_var.set(f'Rendered residual context for S{sid} with {len(entries)} representative edges.')

    def render_shape_bin_distributions(self) -> None:
        if not isinstance(self.experiment, dict):
            return
        sid = self._selected_shape_bin_id()
        if sid is None:
            return
        members = self._shape_bin_members_filtered()
        if not members:
            return
        keys = [('cos_vs', 'cos(v, source PC1)'), ('cos_vt_in', 'cos(-v, destination PC1)'), ('cos_st', 'cos(source PC1, destination PC1)'), ('len_delta', 'Δ length'), ('edge_quality', 'edge Q'), ('prototype_distance', 'prototype distance')]
        fig = Figure(figsize=(11.2, 8.2), dpi=100)
        for idx, (k, title) in enumerate(keys, start=1):
            ax = fig.add_subplot(2, 3, idx)
            vals = np.asarray([_safe_float(r.get(k), float('nan')) for r in members], dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size:
                ax.hist(vals, bins=min(40, max(8, int(math.sqrt(vals.size)))), alpha=0.88)
                ax.axvline(float(np.mean(vals)), color='black', linestyle='--', linewidth=1.2, label='mean')
                ax.axvline(float(np.median(vals)), color='0.35', linestyle=':', linewidth=1.2, label='median')
            ax.set_title(title, fontsize=9)
            ax.set_ylabel('member edges')
            if idx == 1:
                ax.legend(fontsize=7)
        fig.suptitle(f'Shape-bin feature / quality distributions for S{sid}', fontsize=11)
        fig.tight_layout()
        self._draw_figure('shape_bin_distributions', self.shape_bin_dist_frame, fig)

    def render_shape_bin_composition(self) -> None:
        if not isinstance(self.experiment, dict):
            return
        sid = self._selected_shape_bin_id()
        if sid is None:
            return
        members = _arr_shape_member_rows(self.experiment, sid)
        if not members:
            return
        coll_counts = Counter(str(r.get('collection_type', 'Unknown')) for r in members)
        coll_weight = defaultdict(float)
        for r in members:
            coll_weight[str(r.get('collection_type', 'Unknown'))] += _safe_float(r.get('w'), 0.0)
        types = [c for c, _ in coll_counts.most_common()]
        counts = np.asarray([coll_counts[c] for c in types], dtype=float)
        weights = np.asarray([coll_weight[c] for c in types], dtype=float)
        fig = Figure(figsize=(11.0, 7.4), dpi=100)
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        y = np.arange(len(types))
        ax1.barh(y, counts)
        ax1.set_yticks(y); ax1.set_yticklabels(types, fontsize=8)
        ax1.invert_yaxis(); ax1.set_xlabel('edge count'); ax1.set_title(f'S{sid} collection composition by member-edge count')
        ax2.barh(y, weights)
        ax2.set_yticks(y); ax2.set_yticklabels(types, fontsize=8)
        ax2.invert_yaxis(); ax2.set_xlabel('summed edge membership weight'); ax2.set_title('collection composition by summed member weights')
        fig.tight_layout()
        self._draw_figure('shape_bin_composition', self.shape_bin_comp_frame, fig)

    def _build_collection_tab(self) -> None:
        tab = ttk.Panedwindow(self.nb, orient='horizontal')
        self.nb.add(tab, text="Collection summary")
        self.collection_tab = tab

        left = ttk.Frame(tab)
        left.rowconfigure(1, weight=1)
        left.columnconfigure(0, weight=1)
        right = ttk.Frame(tab)
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)
        tab.add(left, weight=3)
        tab.add(right, weight=3)

        topbar = ttk.Frame(left)
        topbar.grid(row=0, column=0, sticky='ew', padx=4, pady=4)
        ttk.Button(topbar, text='Render dominant-shape stacked bars', command=self.render_dominant_bars).pack(side='left', padx=4)
        ttk.Button(topbar, text='Render collection-profile similarity', command=self.render_collection_similarity_matrix).pack(side='left', padx=4)
        ttk.Label(topbar, text='Dominant distribution = per-document argmax shape within each collection.').pack(side='left', padx=10)

        cols = ['collection_type', 'n_docs', 'membership_entropy_mean', 'top_shapes', 'dominant_shape_counts']
        self.collection_tree = ttk.Treeview(left, columns=cols, show='headings')
        for c in cols:
            self.collection_tree.heading(c, text=c)
            self.collection_tree.column(c, width=160 if c not in {'top_shapes', 'dominant_shape_counts'} else 320, anchor='w')
        self.collection_tree.grid(row=1, column=0, sticky='nsew')
        cy = ttk.Scrollbar(left, orient='vertical', command=self.collection_tree.yview)
        self.collection_tree.configure(yscrollcommand=cy.set)
        cy.grid(row=1, column=1, sticky='ns')

        self.collection_right_nb = ttk.Notebook(right)
        self.collection_right_nb.grid(row=0, column=0, sticky='nsew')

        # Dominant-shape distribution: stacked bar figure + inspection table.
        dom_tab = ttk.Panedwindow(self.collection_right_nb, orient='vertical')
        self.collection_right_nb.add(dom_tab, text='Dominant-shape distribution')
        dom_plot = ttk.Frame(dom_tab)
        dom_plot.rowconfigure(0, weight=1)
        dom_plot.columnconfigure(0, weight=1)
        self.bars_frame = dom_plot
        dom_tab.add(dom_plot, weight=3)
        dom_table_frame = ttk.Frame(dom_tab)
        dom_table_frame.rowconfigure(0, weight=1)
        dom_table_frame.columnconfigure(0, weight=1)
        dom_cols = ['collection_type', 'shape_id', 'dominant_doc_count', 'dominant_doc_fraction', 'mean_membership', 'rank_by_mean_membership', 'rank_by_dominant_fraction', 'n_docs']
        self.dominant_dist_tree = ttk.Treeview(dom_table_frame, columns=dom_cols, show='headings', height=8)
        for c in dom_cols:
            self.dominant_dist_tree.heading(c, text=c)
            self.dominant_dist_tree.column(c, width=130 if c != 'collection_type' else 190, anchor='w')
        self.dominant_dist_tree.grid(row=0, column=0, sticky='nsew')
        dy = ttk.Scrollbar(dom_table_frame, orient='vertical', command=self.dominant_dist_tree.yview)
        self.dominant_dist_tree.configure(yscrollcommand=dy.set)
        dy.grid(row=0, column=1, sticky='ns')
        dom_tab.add(dom_table_frame, weight=2)

        # Collection profile similarity: dual matrix figure + pairwise table.
        sim_tab = ttk.Panedwindow(self.collection_right_nb, orient='vertical')
        self.collection_right_nb.add(sim_tab, text='Collection-profile similarity')
        sim_plot = ttk.Frame(sim_tab)
        sim_plot.rowconfigure(0, weight=1)
        sim_plot.columnconfigure(0, weight=1)
        self.collection_similarity_frame = sim_plot
        sim_tab.add(sim_plot, weight=3)
        sim_table_frame = ttk.Frame(sim_tab)
        sim_table_frame.rowconfigure(0, weight=1)
        sim_table_frame.columnconfigure(0, weight=1)
        sim_cols = ['collection_a', 'collection_b', 'cosine_similarity', 'jensen_shannon_similarity', 'n_docs_a', 'n_docs_b']
        self.collection_similarity_tree = ttk.Treeview(sim_table_frame, columns=sim_cols, show='headings', height=8)
        for c in sim_cols:
            self.collection_similarity_tree.heading(c, text=c)
            self.collection_similarity_tree.column(c, width=145 if c not in {'collection_a', 'collection_b'} else 190, anchor='w')
        self.collection_similarity_tree.grid(row=0, column=0, sticky='nsew')
        sy = ttk.Scrollbar(sim_table_frame, orient='vertical', command=self.collection_similarity_tree.yview)
        self.collection_similarity_tree.configure(yscrollcommand=sy.set)
        sy.grid(row=0, column=1, sticky='ns')
        sim_tab.add(sim_table_frame, weight=2)


    def _build_roc_tab(self) -> None:
        """Build Collection ROC / separability tab inside Arrangement Experiment."""
        tab = ttk.Panedwindow(self.nb, orient='vertical')
        self.nb.add(tab, text="Collection ROC")
        self.roc_tab = tab

        plot_panel = ttk.Frame(tab)
        plot_panel.rowconfigure(1, weight=1)
        plot_panel.columnconfigure(0, weight=1)
        tab.add(plot_panel, weight=4)
        bar = ttk.Frame(plot_panel)
        bar.grid(row=0, column=0, sticky='ew', padx=4, pady=4)
        ttk.Button(bar, text='Render Collection ROC', command=self.render_collection_roc).pack(side='left', padx=4)
        ttk.Button(bar, text='Refresh ROC table', command=self.populate_roc_table).pack(side='left', padx=4)
        ttk.Label(
            bar,
            text='One-vs-rest uses leave-one-out collection profiles; pairwise tests same-collection vs cross-collection document pairs.'
        ).pack(side='left', padx=12)
        self.roc_plot_nb = ttk.Notebook(plot_panel)
        self.roc_plot_nb.grid(row=1, column=0, sticky='nsew')

        self.roc_frame = ttk.Frame(self.roc_plot_nb)
        self.roc_frame.rowconfigure(0, weight=1)
        self.roc_frame.columnconfigure(0, weight=1)
        self.roc_plot_nb.add(self.roc_frame, text='Shape ROC curves')

        self.roc_baseline_frame = ttk.Frame(self.roc_plot_nb)
        self.roc_baseline_frame.rowconfigure(0, weight=1)
        self.roc_baseline_frame.columnconfigure(0, weight=1)
        self.roc_plot_nb.add(self.roc_baseline_frame, text='One-vs-rest baselines')

        self.roc_pairwise_baseline_frame = ttk.Frame(self.roc_plot_nb)
        self.roc_pairwise_baseline_frame.rowconfigure(0, weight=1)
        self.roc_pairwise_baseline_frame.columnconfigure(0, weight=1)
        self.roc_plot_nb.add(self.roc_pairwise_baseline_frame, text='Pairwise baselines')

        self.roc_corr_frame = ttk.Frame(self.roc_plot_nb)
        self.roc_corr_frame.rowconfigure(0, weight=1)
        self.roc_corr_frame.columnconfigure(0, weight=1)
        self.roc_plot_nb.add(self.roc_corr_frame, text='Similarity correlations')

        table_panel = ttk.Frame(tab)
        table_panel.rowconfigure(0, weight=1)
        table_panel.columnconfigure(0, weight=1)
        tab.add(table_panel, weight=2)
        roc_cols = [
            'roc_type', 'collection_type', 'score_mode', 'representation', 'auc', 'n_pos', 'n_neg',
            'best_threshold_youden_j', 'best_youden_j', 'best_tpr', 'best_fpr',
            'mean_positive_score', 'mean_negative_score', 'n_samples', 'profile_mode'
        ]
        self.roc_tree = ttk.Treeview(table_panel, columns=roc_cols, show='headings', height=10)
        for c in roc_cols:
            self.roc_tree.heading(c, text=c)
            width = 125
            if c in {'collection_type', 'score_mode', 'profile_mode'}:
                width = 200
            elif c == 'roc_type':
                width = 170
            self.roc_tree.column(c, width=width, anchor='w')
        self.roc_tree.grid(row=0, column=0, sticky='nsew')
        ry = ttk.Scrollbar(table_panel, orient='vertical', command=self.roc_tree.yview)
        self.roc_tree.configure(yscrollcommand=ry.set)
        ry.grid(row=0, column=1, sticky='ns')
        rx = ttk.Scrollbar(table_panel, orient='horizontal', command=self.roc_tree.xview)
        self.roc_tree.configure(xscrollcommand=rx.set)
        rx.grid(row=1, column=0, sticky='ew')

    def choose_metadata_csv(self) -> None:
        path = filedialog.askopenfilename(title='Open collection metadata CSV', filetypes=[('CSV files', '*.csv'), ('All files', '*.*')])
        if path:
            self._var('metadata_csv_path', '').set(path)
            self._var('collection_label_source', 'metadata_csv').set('metadata_csv')

    def _params(self) -> Dict[str, Any]:
        return {
            'shape_k': max(1, _safe_int(self._var('shape_k', '48').get(), 48)),
            'include_length': bool(self._var('include_length', True).get()),
            'include_v_bin': bool(self._var('include_v_bin', False).get()),
            'vbin_az': 12,
            'vbin_el': 6,
            'dir_weight_beta': _safe_float(self._var('dir_weight_beta', '1.0').get(), 1.0),
            'max_edges_per_doc': _safe_int(self._var('max_edges_per_doc', '2000').get(), 2000),
            'min_weight': _safe_float(self._var('min_weight', '0.0').get(), 0.0),
            'weight_mode': str(self._var('weight_mode', 'weighted').get() or 'weighted'),
            'normalize_membership': bool(self._var('normalize_membership', True).get()),
            'random_state': 0,
            'collection_label_source': str(self._var('collection_label_source', 'doc_id_regex').get()),
            'collection_label_regex': str(self._var('collection_label_regex', r'^([^_]+_[^_]+)_').get()),
            'metadata_csv_path': str(self._var('metadata_csv_path', '').get()),
            'unknown_label': 'Unknown',
            'similarity_metric': str(self._var('similarity_metric', 'cosine').get()),
            'doc_top_k_neighbors': max(1, _safe_int(self._var('doc_top_k_neighbors', '5').get(), 5)),
            'doc_graph_threshold': 0.0,
            'shape_support_threshold': _safe_float(self._var('shape_support_threshold', '0.03').get(), 0.03),
            'shape_cooccurrence_weight': 'jaccard',
            'min_cooccurring_docs': 3,
        }

    def build_async(self) -> None:
        if not isinstance(self.model.document_delta_dict, dict) or not self.model.document_delta_dict:
            messagebox.showinfo('document_delta_dict required', 'Open companion document_delta_dict.pkl before building the arrangement experiment.')
            return
        if self._worker is not None and self._worker.is_alive():
            messagebox.showinfo('Build in progress', 'An arrangement experiment build is already running.')
            return
        params = self._params()
        self.status_var.set('Starting arrangement experiment build ...')
        self.update_idletasks()
        doc_delta = self.model.document_delta_dict

        def status(msg: str) -> None:
            try:
                self.after(0, lambda msg=msg: self.status_var.set(msg))
            except Exception:
                pass

        def worker() -> None:
            try:
                exp = _arr_build_experiment(doc_delta, params, status_callback=status)
                self.after(0, lambda exp=exp: self._install_experiment(exp))
            except Exception as ex:
                err = str(ex); tb = traceback.format_exc(limit=8)
                self.after(0, lambda err=err, tb=tb: messagebox.showerror('Arrangement build failed', f'{err}\n\n{tb}'))
                self.after(0, lambda: self.status_var.set('Arrangement experiment build failed.'))
        self._worker = threading.Thread(target=worker, daemon=True)
        self._worker.start()

    def _install_experiment(self, exp: Dict[str, Any]) -> None:
        self.experiment = exp
        self._ensure_shape_neighbors()
        self._ensure_collection_profile_analysis()
        self._ensure_collection_roc_analysis()
        s = exp.get('summary', {}) if isinstance(exp, dict) else {}
        self.status_var.set(
            f"Built arrangement: {s.get('n_docs', 0):,} docs; {s.get('n_shape_records', 0):,} shape records; "
            f"k={s.get('shape_k_effective', 0)}; elapsed={_fmt_float(s.get('build_elapsed_seconds'), 2)}s."
        )
        self.refresh_all_views()

    def _ensure_shape_neighbors(self) -> None:
        """Backfill shape-neighbor structures for newly loaded or older artifacts."""
        if not isinstance(self.experiment, dict):
            return
        existing = self.experiment.get('shape_neighbors')
        if isinstance(existing, dict) and existing.get('centroid') and existing.get('cooccurrence') and existing.get('collection_profile'):
            return
        try:
            top_k = int((self.experiment.get('params') or {}).get('shape_neighbor_top_k', 10))
            self.experiment['shape_neighbors'] = _arr_compute_shape_neighbors(self.experiment, top_k=top_k)
        except Exception as ex:
            self.experiment.setdefault('summary', {})['shape_neighbor_error'] = str(ex)

    def _ensure_collection_profile_analysis(self) -> None:
        """Backfill dominant-shape distribution and collection similarity structures."""
        if not isinstance(self.experiment, dict):
            return
        _arr_ensure_collection_profile_analysis(self.experiment)

    def _ensure_collection_roc_analysis(self) -> None:
        """Backfill one-vs-rest and pairwise ROC/separability structures."""
        if not isinstance(self.experiment, dict):
            return
        # Older saved arrangement experiments may not include document embedding
        # tables. If a matching document_delta_dict is loaded, backfill raw SBERT
        # and manifold-residual document vectors so ROC baselines can be computed.
        if not isinstance(self.experiment.get('document_embeddings'), dict) and isinstance(self.model.document_delta_dict, dict):
            try:
                self.experiment['document_embeddings'] = _arr_document_embedding_tables_from_cdm_dict(
                    self.model.document_delta_dict,
                    [str(d) for d in self.experiment.get('doc_ids', [])],
                )
                # Force ROC v2 recompute after adding baselines.
                self.experiment.pop('collection_roc', None)
            except Exception as ex:
                self.experiment.setdefault('summary', {})['document_embedding_backfill_error'] = str(ex)
        _arr_ensure_collection_roc_analysis(self.experiment)

    def load_experiment(self) -> None:
        path = filedialog.askopenfilename(title='Open morphism_arrangement_experiment.pkl', filetypes=[('Pickle files', '*.pkl'), ('All files', '*.*')])
        if not path:
            return
        try:
            obj = _load_pickle(path)
            if not isinstance(obj, dict) or obj.get('kind') != 'morphism_arrangement_experiment':
                raise ValueError('Selected pickle is not a morphism_arrangement_experiment artifact.')
            self._install_experiment(obj)
            self.status_var.set(f"Loaded arrangement experiment: {os.path.basename(path)}")
        except Exception as ex:
            messagebox.showerror('Load arrangement failed', f'{ex}\n\n{traceback.format_exc(limit=6)}')

    def save_experiment(self) -> None:
        if not isinstance(self.experiment, dict):
            messagebox.showinfo('No experiment', 'Build or load an arrangement experiment first.')
            return
        self._ensure_collection_profile_analysis()
        self._ensure_collection_roc_analysis()
        path = filedialog.asksaveasfilename(title='Save morphism_arrangement_experiment.pkl', defaultextension='.pkl', initialfile='morphism_arrangement_experiment.pkl', filetypes=[('Pickle files', '*.pkl'), ('All files', '*.*')])
        if not path:
            return
        with open(path, 'wb') as f:
            pickle.dump(self.experiment, f, protocol=pickle.HIGHEST_PROTOCOL)
        messagebox.showinfo('Saved', f'Arrangement experiment written to:\n{path}')

    def export_tables(self) -> None:
        if not isinstance(self.experiment, dict):
            messagebox.showinfo('No experiment', 'Build or load an arrangement experiment first.')
            return
        folder = filedialog.askdirectory(title='Choose folder for arrangement experiment CSV exports')
        if not folder:
            return
        base = Path(folder)
        def write_rows(name: str, rows: List[Dict[str, Any]]) -> None:
            if not rows:
                return
            keys: List[str] = []
            for r in rows:
                for k in r.keys():
                    if k not in keys:
                        keys.append(k)
            with open(base / name, 'w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
                w.writeheader(); w.writerows(rows)
        write_rows('shape_summary.csv', list(self.experiment.get('shape_summary', [])))
        try:
            write_rows('shape_bin_field_summary.csv', _arr_shape_bin_field_summary_rows(self.experiment))
        except Exception:
            pass
        write_rows('representative_edges.csv', list(self.experiment.get('representative_edges', [])))
        write_rows('collection_summary.csv', list(self.experiment.get('collection_summary', [])))
        # Shape-neighbor families as flat tables.
        neighbors = self.experiment.get('shape_neighbors') or {}
        def _neighbor_rows(kind: str) -> List[Dict[str, Any]]:
            block = (neighbors.get(kind) or {}).get('neighbors') or {}
            out: List[Dict[str, Any]] = []
            for sid, rows0 in block.items():
                try:
                    src_sid = int(sid)
                except Exception:
                    src_sid = sid
                for r in rows0 or []:
                    rr = dict(r)
                    rr['source_shape_id'] = src_sid
                    rr['neighbor_kind'] = kind
                    out.append(rr)
            return out
        write_rows('shape_centroid_neighbors.csv', _neighbor_rows('centroid'))
        write_rows('shape_cooccurrence_neighbors.csv', _neighbor_rows('cooccurrence'))
        write_rows('shape_collection_profile_neighbors.csv', _neighbor_rows('collection_profile'))
        self._ensure_collection_profile_analysis()
        dom_dist = self.experiment.get('dominant_shape_distribution') or {}
        write_rows('dominant_shape_distribution.csv', list(dom_dist.get('rows', [])))
        coll_sim = self.experiment.get('collection_profile_similarity') or {}
        write_rows('collection_profile_similarity_pairs.csv', list(coll_sim.get('pair_rows', [])))
        self._ensure_collection_roc_analysis()
        roc = self.experiment.get('collection_roc') or {}
        roc_rows = list(roc.get('rows', []))
        write_rows('collection_roc_all_rows.csv', roc_rows)
        write_rows('collection_roc_one_vs_rest.csv', [r for r in roc_rows if r.get('roc_type') == 'one_vs_rest'])
        write_rows('collection_roc_pairwise.csv', [r for r in roc_rows if r.get('roc_type') == 'pairwise_same_vs_cross'])
        corr_rows = list(((roc.get('similarity_matrix_correlations') or {}).get('rows') or []))
        write_rows('collection_roc_similarity_correlations.csv', corr_rows)
        # Collection profile matrix.
        try:
            types = list(coll_sim.get('collection_types', []))
            profile = np.asarray(coll_sim.get('profile_matrix', []), dtype=float)
            if profile.ndim == 2 and types:
                prof_rows = []
                for i, t in enumerate(types):
                    r = {'collection_type': t}
                    for s in range(profile.shape[1]):
                        r[f'shape_{s}'] = float(profile[i, s])
                    prof_rows.append(r)
                write_rows('collection_shape_profile_matrix.csv', prof_rows)
        except Exception:
            pass
        # Membership table.
        doc_ids = list(self.experiment.get('doc_ids', []))
        coll = list(self.experiment.get('collection_type', []))
        M = np.asarray(self.experiment.get('doc_shape_membership', []), dtype=float)
        if M.ndim == 2 and doc_ids:
            rows = []
            for i, d in enumerate(doc_ids):
                r = {'doc_id': d, 'collection_type': coll[i] if i < len(coll) else ''}
                for s in range(M.shape[1]):
                    r[f'shape_{s}'] = float(M[i, s])
                rows.append(r)
            write_rows('doc_shape_membership.csv', rows)
        messagebox.showinfo('Export complete', f'Arrangement experiment tables exported to:\n{folder}')

    def clear_for_new_document_delta(self) -> None:
        self.status_var.set('document_delta_dict loaded. Build an arrangement experiment when ready.')

    def refresh_all_views(self) -> None:
        self.populate_summary()
        self.populate_shape_table()
        self._ensure_collection_profile_analysis()
        self.populate_collection_table()
        self.populate_dominant_distribution_table()
        self.populate_collection_similarity_table()
        self._ensure_collection_roc_analysis()
        self.populate_roc_table()
        self.update_doc_selector()
        self.update_shape_neighbor_selector()
        self.populate_shape_neighbors()
        self.update_shape_bin_field_selector()
        self.populate_shape_bin_field()

    def populate_summary(self) -> None:
        if not isinstance(self.experiment, dict):
            n_docs = len(self.model.document_delta_dict) if isinstance(self.model.document_delta_dict, dict) else 0
            self.summary_text.set_text(
                f"No arrangement experiment built yet.\n\nLoaded document_delta_dict documents: {n_docs:,}\n\n"
                "Use the controls above to cluster morphism-shape records and build document × shape-category membership."
            )
            return
        s = self.experiment.get('summary', {})
        p = self.experiment.get('params', {})
        lines = ['Arrangement experiment summary', '']
        for k in ['n_docs', 'n_collection_types', 'n_shape_records', 'shape_feature_dim', 'shape_k_effective', 'doc_shape_membership_shape', 'same_type_topk_neighbor_rate', 'within_type_mean_similarity', 'between_type_mean_similarity', 'within_minus_between_similarity', 'build_elapsed_seconds', 'n_errors']:
            v = s.get(k, '')
            lines.append(f"{k}: {_fmt_float(v, 5) if isinstance(v, float) else v}")
        lines.append('\nParameters:')
        for k in sorted(p.keys()):
            lines.append(f"  {k}: {p[k]}")
        errs = s.get('errors') or []
        if errs:
            lines.append('\nFirst extraction warnings/errors:')
            lines.extend(f"  - {e}" for e in errs[:20])
        self.summary_text.set_text('\n'.join(lines))

    def populate_shape_table(self) -> None:
        for t in [self.shape_tree, self.rep_tree]:
            for item in t.get_children():
                t.delete(item)
        if not isinstance(self.experiment, dict):
            return
        for r in self.experiment.get('shape_summary', []):
            vals = []
            for c in self.shape_tree['columns']:
                v = r.get(c, '')
                vals.append(_fmt_float(v, 4) if isinstance(v, (float, np.floating)) else str(v))
            self.shape_tree.insert('', 'end', iid=str(r.get('shape_id', '')), values=vals)
        # Show reps for first shape.
        if self.shape_tree.get_children():
            first = self.shape_tree.get_children()[0]
            self.shape_tree.selection_set(first)
            self._populate_reps_for_shape(int(first))

    def _on_shape_select(self, _event: Any = None) -> None:
        sel = self.shape_tree.selection()
        if not sel:
            return
        try:
            sid = int(sel[0])
            self._populate_reps_for_shape(sid)
            if hasattr(self, 'selected_shape_neighbor_var'):
                self.selected_shape_neighbor_var.set(f'S{sid}')
                self.populate_shape_neighbors()
            if hasattr(self, 'selected_shape_bin_var'):
                self.selected_shape_bin_var.set(f'S{sid}')
                self.populate_shape_bin_field()
        except Exception:
            pass

    def _populate_reps_for_shape(self, shape_id: int) -> None:
        for item in self.rep_tree.get_children():
            self.rep_tree.delete(item)
        if not isinstance(self.experiment, dict):
            return
        rows = [r for r in self.experiment.get('representative_edges', []) if int(r.get('shape_id', -1)) == int(shape_id)]
        for idx, r in enumerate(rows[:200]):
            vals = []
            for c in self.rep_tree['columns']:
                v = r.get(c, '')
                vals.append(_fmt_float(v, 5) if isinstance(v, (float, np.floating)) else str(v))
            self.rep_tree.insert('', 'end', iid=str(idx), values=vals)

    def _shape_id_from_neighbor_var(self) -> Optional[int]:
        raw = str(getattr(self, 'selected_shape_neighbor_var', tk.StringVar(value='')).get() or '').strip()
        if not raw and hasattr(self, 'shape_tree'):
            sel = self.shape_tree.selection()
            raw = str(sel[0]) if sel else ''
        if raw.upper().startswith('S'):
            raw = raw[1:]
        try:
            sid = int(raw)
            return sid if sid >= 0 else None
        except Exception:
            return None

    def update_shape_neighbor_selector(self) -> None:
        if not hasattr(self, 'shape_neighbor_combo'):
            return
        if not isinstance(self.experiment, dict):
            self.shape_neighbor_combo.configure(values=[])
            return
        k = 0
        M = np.asarray(self.experiment.get('doc_shape_membership', []), dtype=float)
        if M.ndim == 2:
            k = max(k, M.shape[1])
        try:
            summaries = list(self.experiment.get('shape_summary', []))
            if summaries:
                k = max(k, max(int(r.get('shape_id', -1)) for r in summaries) + 1)
        except Exception:
            pass
        values = [f'S{i}' for i in range(max(0, k))]
        self.shape_neighbor_combo.configure(values=values)
        if values and not str(self.selected_shape_neighbor_var.get() or '').strip():
            self.selected_shape_neighbor_var.set(values[0])

    def _clear_neighbor_tree(self, tree: ttk.Treeview) -> None:
        for item in tree.get_children():
            tree.delete(item)

    def _insert_neighbor_rows(self, tree: ttk.Treeview, rows: Sequence[Dict[str, Any]]) -> None:
        self._clear_neighbor_tree(tree)
        cols = list(tree['columns'])
        for i, row in enumerate(rows or []):
            vals = []
            for c in cols:
                v = row.get(c, '')
                if isinstance(v, (float, np.floating)):
                    vals.append(_fmt_float(v, 6))
                else:
                    vals.append(str(v))
            tree.insert('', 'end', iid=str(i), values=vals)

    def populate_shape_neighbors(self) -> None:
        if not all(hasattr(self, name) for name in ('centroid_neighbor_tree', 'cooccurrence_neighbor_tree', 'profile_neighbor_tree')):
            return
        for tree in (self.centroid_neighbor_tree, self.cooccurrence_neighbor_tree, self.profile_neighbor_tree):
            self._clear_neighbor_tree(tree)
        if not isinstance(self.experiment, dict):
            return
        self._ensure_shape_neighbors()
        sid = self._shape_id_from_neighbor_var()
        if sid is None:
            return
        neighbors = self.experiment.get('shape_neighbors') or {}
        centroid_rows = ((neighbors.get('centroid') or {}).get('neighbors') or {}).get(sid, [])
        if not centroid_rows:
            centroid_rows = ((neighbors.get('centroid') or {}).get('neighbors') or {}).get(str(sid), [])
        co_rows = ((neighbors.get('cooccurrence') or {}).get('neighbors') or {}).get(sid, [])
        if not co_rows:
            co_rows = ((neighbors.get('cooccurrence') or {}).get('neighbors') or {}).get(str(sid), [])
        prof_rows = ((neighbors.get('collection_profile') or {}).get('neighbors') or {}).get(sid, [])
        if not prof_rows:
            prof_rows = ((neighbors.get('collection_profile') or {}).get('neighbors') or {}).get(str(sid), [])
        self._insert_neighbor_rows(self.centroid_neighbor_tree, centroid_rows)
        self._insert_neighbor_rows(self.cooccurrence_neighbor_tree, co_rows)
        self._insert_neighbor_rows(self.profile_neighbor_tree, prof_rows)
        self.status_var.set(
            f'Shape-neighbor inspector: S{sid}; '
            f'{len(centroid_rows)} centroid, {len(co_rows)} co-occurrence, {len(prof_rows)} collection-profile neighbors.'
        )

    def render_shape_neighbor_graph(self) -> None:
        if not isinstance(self.experiment, dict):
            messagebox.showinfo('No experiment', 'Build or load an arrangement experiment first.')
            return
        self._ensure_shape_neighbors()
        sid = self._shape_id_from_neighbor_var()
        if sid is None:
            messagebox.showinfo('No shape selected', 'Select a shape category first.')
            return
        neighbors = self.experiment.get('shape_neighbors') or {}
        summary_by_id = _arr_shape_summary_lookup(list(self.experiment.get('shape_summary', [])))

        def _rows(kind: str) -> List[Dict[str, Any]]:
            block = (neighbors.get(kind) or {}).get('neighbors') or {}
            return list(block.get(sid, block.get(str(sid), [])) or [])

        centroid_rows = _rows('centroid')[:10]
        co_rows = _rows('cooccurrence')[:10]
        profile_rows = _rows('collection_profile')[:10]
        node_ids = {int(sid)}
        for rows in (centroid_rows, co_rows, profile_rows):
            for r in rows:
                try:
                    node_ids.add(int(r.get('shape_id')))
                except Exception:
                    pass
        node_ids = sorted(node_ids)
        if not node_ids:
            return

        centroids = np.asarray(self.experiment.get('shape_centroids', []), dtype=float)
        if centroids.ndim == 2 and centroids.shape[0] > max(node_ids):
            Y_all = _arr_pca_2d(centroids)
            Y = {i: Y_all[i] for i in node_ids if i < Y_all.shape[0]}
        else:
            M = np.asarray(self.experiment.get('doc_shape_membership', []), dtype=float)
            if M.ndim == 2 and M.shape[1] > max(node_ids):
                Y_all = _arr_pca_2d(M.T)
                Y = {i: Y_all[i] for i in node_ids if i < Y_all.shape[0]}
            else:
                ang = np.linspace(0, 2 * math.pi, len(node_ids), endpoint=False)
                Y = {i: np.asarray([math.cos(a), math.sin(a)]) for i, a in zip(node_ids, ang)}
        # Re-center selected shape if possible for readability.
        if sid in Y:
            origin = np.asarray(Y[sid], dtype=float)
            Y = {i: np.asarray(v, dtype=float) - origin for i, v in Y.items()}

        coll_types = sorted({str((summary_by_id.get(i) or {}).get('top_collection_type', 'Unknown') or 'Unknown') for i in node_ids})
        cmap = cm.get_cmap('tab20', max(1, len(coll_types)))
        color_map = {t: cmap(i) for i, t in enumerate(coll_types)}

        fig = Figure(figsize=(8.2, 7.0), dpi=100)
        ax = fig.add_subplot(111)

        style_specs = [
            ('centroid', centroid_rows, '#1f77b4', 'solid', 'centroid nearest'),
            ('cooccurrence', co_rows, '#ff7f0e', 'dashed', 'co-occurrence'),
            ('collection_profile', profile_rows, '#2ca02c', 'dotted', 'collection profile'),
        ]
        # Draw guide legend handles.
        for _kind, _rows0, color, ls, label in style_specs:
            ax.plot([], [], color=color, linestyle=ls, linewidth=2.0, label=label)
        for kind, rows, color, ls, _label in style_specs:
            for r in rows:
                try:
                    nid = int(r.get('shape_id'))
                except Exception:
                    continue
                if nid not in Y or sid not in Y:
                    continue
                lw = 1.6
                if kind == 'cooccurrence':
                    lw = 1.0 + min(3.0, float(r.get('cooccurrence_score', 0.0) or 0.0) * 4.0)
                elif kind == 'collection_profile':
                    lw = 1.0 + min(3.0, max(0.0, float(r.get('profile_cosine', 0.0) or 0.0)) * 2.0)
                elif kind == 'centroid':
                    d = float(r.get('centroid_distance', 1.0) or 1.0)
                    lw = 1.0 + 2.0 / max(1.0, 1.0 + d)
                ax.plot([Y[sid][0], Y[nid][0]], [Y[sid][1], Y[nid][1]], color=color, linestyle=ls, linewidth=lw, alpha=0.68)

        for nid in node_ids:
            info = summary_by_id.get(int(nid), {})
            t = str(info.get('top_collection_type', 'Unknown') or 'Unknown')
            support = float(info.get('support_docs', 0.0) or 0.0)
            size = 85.0 + 25.0 * math.sqrt(max(0.0, support))
            if int(nid) == int(sid):
                size *= 1.45
            ax.scatter([Y[nid][0]], [Y[nid][1]], s=size, color=color_map.get(t, '0.6'), edgecolors=('red' if int(nid) == int(sid) else 'black'), linewidths=(2.0 if int(nid) == int(sid) else 0.6), zorder=5)
            ax.text(Y[nid][0], Y[nid][1], f'S{nid}', ha='center', va='center', fontsize=8, zorder=6)

        title_lines = [f'Shape-neighbor graph for S{sid}', 'blue=centroid, orange=co-occurrence, green=collection-profile']
        ax.set_title('\n'.join(title_lines), fontsize=10)
        ax.set_xlabel('shape-centroid PCA 1 / local layout')
        ax.set_ylabel('shape-centroid PCA 2 / local layout')
        ax.axhline(0, color='0.90', linewidth=0.6)
        ax.axvline(0, color='0.90', linewidth=0.6)
        ax.legend(loc='best', fontsize=7)
        ax.set_aspect('equal', adjustable='datalim')
        fig.tight_layout()
        self._draw_figure('shapeneighbors', self.shapeneighbor_frame, fig)
        self.status_var.set(f'Rendered shape-neighbor graph for S{sid} with {len(node_ids)} nodes.')

    def populate_collection_table(self) -> None:
        for item in self.collection_tree.get_children():
            self.collection_tree.delete(item)
        if not isinstance(self.experiment, dict):
            return
        for i, r in enumerate(self.experiment.get('collection_summary', [])):
            vals = []
            for c in self.collection_tree['columns']:
                v = r.get(c, '')
                vals.append(_fmt_float(v, 4) if isinstance(v, (float, np.floating)) else str(v))
            self.collection_tree.insert('', 'end', iid=str(i), values=vals)

    def populate_dominant_distribution_table(self) -> None:
        if not hasattr(self, 'dominant_dist_tree'):
            return
        for item in self.dominant_dist_tree.get_children():
            self.dominant_dist_tree.delete(item)
        if not isinstance(self.experiment, dict):
            return
        self._ensure_collection_profile_analysis()
        rows = list((self.experiment.get('dominant_shape_distribution') or {}).get('rows', []))
        # Keep the table inspectable: show shapes that are high by either mean
        # membership or dominant-document fraction, plus every non-zero dominant count.
        rows = [
            r for r in rows
            if int(r.get('dominant_doc_count', 0) or 0) > 0
            or int(r.get('rank_by_mean_membership', 9999) or 9999) <= 12
            or int(r.get('rank_by_dominant_fraction', 9999) or 9999) <= 12
        ]
        rows.sort(key=lambda r: (str(r.get('collection_type', '')), int(r.get('rank_by_dominant_fraction', 9999) or 9999), int(r.get('rank_by_mean_membership', 9999) or 9999), int(r.get('shape_id', 0) or 0)))
        cols = list(self.dominant_dist_tree['columns'])
        for i, r in enumerate(rows[:3000]):
            vals = []
            for c in cols:
                v = r.get(c, '')
                vals.append(_fmt_float(v, 5) if isinstance(v, (float, np.floating)) else str(v))
            self.dominant_dist_tree.insert('', 'end', iid=str(i), values=vals)

    def populate_collection_similarity_table(self) -> None:
        if not hasattr(self, 'collection_similarity_tree'):
            return
        for item in self.collection_similarity_tree.get_children():
            self.collection_similarity_tree.delete(item)
        if not isinstance(self.experiment, dict):
            return
        self._ensure_collection_profile_analysis()
        rows = list((self.experiment.get('collection_profile_similarity') or {}).get('pair_rows', []))
        rows.sort(key=lambda r: (float(r.get('cosine_similarity', 0.0) or 0.0), float(r.get('jensen_shannon_similarity', 0.0) or 0.0)), reverse=True)
        cols = list(self.collection_similarity_tree['columns'])
        for i, r in enumerate(rows):
            vals = []
            for c in cols:
                v = r.get(c, '')
                vals.append(_fmt_float(v, 5) if isinstance(v, (float, np.floating)) else str(v))
            self.collection_similarity_tree.insert('', 'end', iid=str(i), values=vals)

    def update_doc_selector(self) -> None:
        docs = list(self.experiment.get('doc_ids', [])) if isinstance(self.experiment, dict) else []
        self.doc_combo.configure(values=docs)
        if docs and not self.selected_doc_var.get():
            self.selected_doc_var.set(docs[0])

    def _clear_canvas(self, key: str, frame: ttk.Frame) -> None:
        c = self.canvas.get(key)
        if c is not None:
            try: c.get_tk_widget().destroy()
            except Exception: pass
            self.canvas[key] = None
        tb = self.toolbar.get(key)
        if tb is not None:
            try: tb.destroy()
            except Exception: pass
            self.toolbar[key] = None

    def _draw_figure(self, key: str, frame: ttk.Frame, fig: Figure) -> None:
        self._clear_canvas(key, frame)
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew')
        toolbar = NavigationToolbar2Tk(canvas, frame, pack_toolbar=False)
        toolbar.update(); toolbar.grid(row=1, column=0, sticky='ew')
        self.canvas[key] = canvas; self.toolbar[key] = toolbar

    def _membership_heatmap_row_order(self, M: np.ndarray, docs: Sequence[str], coll: Sequence[str], mode: str) -> List[int]:
        """Return document row order for the membership heatmap."""
        X = np.asarray(M, dtype=float)
        n = X.shape[0] if X.ndim == 2 else 0
        docs_l = [str(d) for d in docs]
        coll_l = [str(c) if str(c).strip() else 'Unknown' for c in coll]
        if len(docs_l) < n:
            docs_l += [str(i) for i in range(len(docs_l), n)]
        if len(coll_l) < n:
            coll_l += ['Unknown'] * (n - len(coll_l))
        dominant = np.argmax(X, axis=1) if X.ndim == 2 and X.shape[1] else np.zeros(n, dtype=int)
        mode = str(mode or 'collection_dominant_shape')
        if mode == 'doc_id':
            return sorted(range(n), key=lambda i: docs_l[i])
        if mode == 'collection_doc_id':
            return sorted(range(n), key=lambda i: (coll_l[i], docs_l[i]))
        if mode == 'dominant_shape':
            return sorted(range(n), key=lambda i: (int(dominant[i]), coll_l[i], docs_l[i]))
        if mode in {'membership_pca', 'collection_membership_pca'}:
            Y = _arr_pca_2d(X)
            if Y.shape[0] != n:
                Y = np.column_stack([np.arange(n, dtype=float), np.zeros(n, dtype=float)])
            if mode == 'collection_membership_pca':
                return sorted(range(n), key=lambda i: (coll_l[i], float(Y[i, 0]), float(Y[i, 1]), docs_l[i]))
            return sorted(range(n), key=lambda i: (float(Y[i, 0]), float(Y[i, 1]), coll_l[i], docs_l[i]))
        if mode == 'random_within_collection':
            rng = np.random.default_rng(0)
            jitter = {i: float(rng.random()) for i in range(n)}
            return sorted(range(n), key=lambda i: (coll_l[i], jitter[i]))
        # Default: tuned for the mixed-collection experiment.
        return sorted(range(n), key=lambda i: (coll_l[i], int(dominant[i]), docs_l[i]))

    def render_heatmap(self) -> None:
        if not isinstance(self.experiment, dict):
            messagebox.showinfo('No experiment', 'Build or load an arrangement experiment first.')
            return
        M = np.asarray(self.experiment.get('doc_shape_membership', []), dtype=float)
        docs = [str(d) for d in self.experiment.get('doc_ids', [])]
        coll = [str(c) for c in self.experiment.get('collection_type', [])]
        if M.ndim != 2 or M.size == 0:
            return
        row_mode = str(self._var('heatmap_row_order', 'collection_dominant_shape').get() or 'collection_dominant_shape')
        row_order = self._membership_heatmap_row_order(M, docs, coll, row_mode)
        # Columns remain sorted by total membership so the most-used shape bins stay leftmost.
        col_order = list(np.argsort(M.sum(axis=0))[::-1])
        X = M[np.ix_(row_order, col_order)]

        # Build collection-block metadata in the same row order used by the heatmap.
        ordered_labels = [coll[i] if i < len(coll) and str(coll[i]).strip() else 'Unknown' for i in row_order]
        unique_labels: List[str] = []
        label_to_code: Dict[str, int] = {}
        for lab in ordered_labels:
            if lab not in label_to_code:
                label_to_code[lab] = len(unique_labels)
                unique_labels.append(lab)
        block_codes = np.asarray([[label_to_code.get(lab, 0)] for lab in ordered_labels], dtype=float)
        cmap_base = cm.get_cmap('tab20', max(1, len(unique_labels)))
        block_cmap = mcolors.ListedColormap([cmap_base(i) for i in range(max(1, len(unique_labels)))])

        # Group boundaries for collection labels.
        blocks: List[Tuple[str, int, int]] = []
        if ordered_labels:
            start_idx = 0
            current = ordered_labels[0]
            for pos, lab in enumerate(ordered_labels[1:], start=1):
                if lab != current:
                    blocks.append((current, start_idx, pos - 1))
                    start_idx = pos
                    current = lab
            blocks.append((current, start_idx, len(ordered_labels) - 1))

        fig = Figure(figsize=(11.5, 7.2), dpi=100)
        gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 10.0], wspace=0.02)
        ax_block = fig.add_subplot(gs[0, 0])
        ax = fig.add_subplot(gs[0, 1], sharey=ax_block)

        # Left stacked-bar collection label block.
        ax_block.imshow(block_codes, aspect='auto', interpolation='nearest', cmap=block_cmap)
        ax_block.set_title('collection', fontsize=9)
        ax_block.set_xticks([])
        ax_block.set_yticks([])
        ax_block.set_ylabel('documents grouped by collection label')
        for lab, a, b in blocks:
            mid = (a + b) / 2.0
            height = b - a + 1
            fs = 7 if height >= 4 else 6
            ax_block.text(0, mid, f'{lab}\n(n={height})', ha='center', va='center', fontsize=fs, color='black')
            # Boundary lines in both the collection block and heatmap.
            ax_block.axhline(b + 0.5, color='white', linewidth=1.1)
            ax.axhline(b + 0.5, color='white', linewidth=0.9, alpha=0.9)

        im = ax.imshow(X, aspect='auto', interpolation='nearest', cmap='viridis')
        ax.set_title(f'Document × morphism-shape category membership · row order: {row_mode}')
        ax.set_xlabel('shape category (sorted by total membership)')
        ax.set_ylabel('')
        if len(row_order) <= 80:
            ax.set_yticks(range(len(row_order)))
            ax.set_yticklabels([docs[i] if i < len(docs) else str(i) for i in row_order], fontsize=6)
        else:
            ax.set_yticks([])
        if len(col_order) <= 80:
            ax.set_xticks(range(len(col_order)))
            ax.set_xticklabels([f'S{j}' for j in col_order], rotation=90, fontsize=6)
        else:
            ax.set_xticks([])
        fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label='membership weight')

        def _cell_readout(x: float, y: float) -> str:
            try:
                ix = int(round(float(x)))
                iy = int(round(float(y)))
                if 0 <= iy < len(row_order) and 0 <= ix < len(col_order):
                    doc_idx = int(row_order[iy])
                    shape_id = int(col_order[ix])
                    doc_id = docs[doc_idx] if doc_idx < len(docs) else str(doc_idx)
                    collection = coll[doc_idx] if doc_idx < len(coll) else 'Unknown'
                    value = float(M[doc_idx, shape_id])
                    return (
                        f'x={ix}, y={iy} | doc={doc_id} | collection={collection} | '
                        f'shape=S{shape_id} | membership={value:.6g}'
                    )
            except Exception:
                pass
            return f'x={x:.3f}, y={y:.3f}'

        ax.format_coord = _cell_readout

        fig.tight_layout()
        self._draw_figure('heatmap', self.heatmap_frame, fig)
        canvas = self.canvas.get('heatmap')
        if canvas is not None:
            def _on_motion(event: Any) -> None:
                if event.inaxes is ax and event.xdata is not None and event.ydata is not None:
                    self.status_var.set(_cell_readout(event.xdata, event.ydata))
            canvas.mpl_connect('motion_notify_event', _on_motion)
        self.status_var.set(
            f'Heatmap rendered: {len(row_order):,} documents, {len(col_order):,} shape categories, '
            f'{len(unique_labels):,} collection labels; row order={row_mode}.'
        )

    def render_dominant_bars(self) -> None:
        if not isinstance(self.experiment, dict):
            messagebox.showinfo('No experiment', 'Build or load an arrangement experiment first.')
            return
        self._ensure_collection_profile_analysis()
        dom = self.experiment.get('dominant_shape_distribution') or {}
        types = list(dom.get('collection_types', []))
        frac = np.asarray(dom.get('dominant_fraction_matrix', []), dtype=float)
        counts = np.asarray(dom.get('dominant_count_matrix', []), dtype=float)
        n_docs = np.asarray(dom.get('n_docs_by_collection', []), dtype=int)
        if frac.ndim != 2 or frac.size == 0 or not types:
            messagebox.showinfo('No dominant distribution', 'The experiment does not contain dominant-shape distribution data.')
            return
        # Show the most common dominant shapes globally; aggregate the remainder.
        max_shapes = min(14, frac.shape[1])
        global_counts = counts.sum(axis=0) if counts.size else frac.sum(axis=0)
        top_shapes = list(np.argsort(global_counts)[::-1][:max_shapes])
        other_shapes = [s for s in range(frac.shape[1]) if s not in set(top_shapes)]

        fig = Figure(figsize=(8.8, max(4.8, 0.48 * len(types) + 2.0)), dpi=100)
        ax = fig.add_subplot(111)
        y = np.arange(len(types))
        left = np.zeros(len(types), dtype=float)
        for sid in top_shapes:
            vals = frac[:, int(sid)]
            if np.all(vals <= 0):
                continue
            ax.barh(y, vals, left=left, label=f'S{int(sid)}')
            left = left + vals
        if other_shapes:
            other = frac[:, other_shapes].sum(axis=1)
            if np.any(other > 0):
                ax.barh(y, other, left=left, label='Other')
        labels = [f'{t} (n={int(n_docs[i]) if i < n_docs.shape[0] else 0})' for i, t in enumerate(types)]
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlim(0.0, max(1.0, float(np.nanmax(frac.sum(axis=1))) if frac.size else 1.0))
        ax.set_xlabel('fraction of documents where shape is the dominant membership category')
        ax.set_title('Dominant morphism-shape distribution by collection type')
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize=7, title='shape')
        fig.tight_layout()
        self._draw_figure('bars', self.bars_frame, fig)
        self.populate_dominant_distribution_table()
        try:
            self.nb.select(self.collection_tab)
            self.collection_right_nb.select(0)
        except Exception:
            pass
        self.status_var.set('Rendered dominant-shape stacked bars and updated the distribution table.')

    def render_collection_similarity_matrix(self) -> None:
        if not isinstance(self.experiment, dict):
            messagebox.showinfo('No experiment', 'Build or load an arrangement experiment first.')
            return
        self._ensure_collection_profile_analysis()
        sim = self.experiment.get('collection_profile_similarity') or {}
        types = list(sim.get('collection_types', []))
        cos = np.asarray(sim.get('cosine_matrix', []), dtype=float)
        js = np.asarray(sim.get('jensen_shannon_matrix', []), dtype=float)
        if cos.ndim != 2 or js.ndim != 2 or cos.size == 0 or js.size == 0 or not types:
            messagebox.showinfo('No similarity matrix', 'The experiment does not contain collection-profile similarity data.')
            return
        fig = Figure(figsize=(10.5, max(4.8, 0.36 * len(types) + 2.0)), dpi=100)
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        for ax, mat, title in ((ax1, cos, 'Cosine similarity'), (ax2, js, 'Jensen-Shannon similarity')):
            im = ax.imshow(mat, vmin=0.0, vmax=1.0, cmap='viridis', interpolation='nearest')
            ax.set_title(title)
            ax.set_xticks(range(len(types)))
            ax.set_yticks(range(len(types)))
            ax.set_xticklabels(types, rotation=90, fontsize=7)
            ax.set_yticklabels(types, fontsize=7)
            for i in range(len(types)):
                for j in range(len(types)):
                    if len(types) <= 12:
                        ax.text(j, i, f'{mat[i, j]:.2f}', ha='center', va='center', fontsize=6, color='white' if mat[i, j] < 0.55 else 'black')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle('Collection-profile similarity over average shape-membership vectors', fontsize=10)
        fig.tight_layout()
        self._draw_figure('collection_similarity', self.collection_similarity_frame, fig)
        self.populate_collection_similarity_table()
        try:
            self.nb.select(self.collection_tab)
            self.collection_right_nb.select(1)
        except Exception:
            pass
        self.status_var.set('Rendered collection-profile similarity matrices and updated the pairwise table.')


    def populate_roc_table(self) -> None:
        if not hasattr(self, 'roc_tree'):
            return
        for item in self.roc_tree.get_children():
            self.roc_tree.delete(item)
        if not isinstance(self.experiment, dict):
            return
        self._ensure_collection_roc_analysis()
        roc = self.experiment.get('collection_roc') or {}
        rows = list(roc.get('rows', []))
        # Put one-vs-rest rows first, ordered by score mode and AUC, then pairwise summary rows.
        def _sort_key(r: Dict[str, Any]) -> Tuple[str, str, float, str]:
            auc = float(r.get('auc', float('nan')))
            auc_key = -auc if np.isfinite(auc) else 999.0
            return (str(r.get('roc_type', '')), str(r.get('score_mode', '')), auc_key, str(r.get('collection_type', '')))
        rows.sort(key=_sort_key)
        cols = list(self.roc_tree['columns'])
        for i, r in enumerate(rows):
            vals = []
            for c in cols:
                v = r.get(c, '')
                if isinstance(v, (float, np.floating)):
                    vals.append(_fmt_float(v, 5, blank_for_nan=False))
                else:
                    vals.append(str(v))
            self.roc_tree.insert('', 'end', iid=str(i), values=vals)

    def render_collection_roc(self) -> None:
        if not isinstance(self.experiment, dict):
            messagebox.showinfo('No experiment', 'Build or load an arrangement experiment first.')
            return
        self._ensure_collection_roc_analysis()
        roc = self.experiment.get('collection_roc') or {}
        one = roc.get('one_vs_rest') or {}
        pair = roc.get('pairwise') or {}
        if not one and not pair:
            messagebox.showinfo('No ROC data', 'The experiment does not contain collection ROC data.')
            return

        # Panel 1: existing shape-membership ROC curves.
        fig = Figure(figsize=(14.2, 5.2), dpi=100)
        ax_cos = fig.add_subplot(131)
        ax_js = fig.add_subplot(132)
        ax_pair = fig.add_subplot(133)
        axes = [ax_cos, ax_js, ax_pair]
        for ax in axes:
            ax.plot([0, 1], [0, 1], color='0.75', linestyle='--', linewidth=1.0)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel('False positive rate')
            ax.set_ylabel('True positive rate')
            ax.grid(True, alpha=0.25)

        def _plot_one_vs_rest(ax: Any, mode: str, title: str) -> None:
            block = one.get(mode) or {}
            curves = block.get('curves') or {}
            types = sorted(curves.keys())
            cmap = cm.get_cmap('tab20', max(1, len(types)))
            plotted = 0
            for idx, ctype in enumerate(types):
                curve = curves.get(ctype) or {}
                fpr = np.asarray(curve.get('fpr', []), dtype=float)
                tpr = np.asarray(curve.get('tpr', []), dtype=float)
                auc = float(curve.get('auc', float('nan')))
                if fpr.size < 2 or tpr.size < 2 or not np.isfinite(auc):
                    continue
                n_pos = int(curve.get('n_pos', 0) or 0)
                ax.plot(fpr, tpr, color=cmap(idx), linewidth=1.7, label=f'{ctype} AUC={auc:.3f} n+={n_pos}')
                plotted += 1
            macro = block.get('macro_auc', float('nan'))
            wmacro = block.get('weighted_macro_auc', float('nan'))
            subtitle = ''
            if np.isfinite(float(macro)):
                subtitle = f'\nmacro={float(macro):.3f}; weighted={float(wmacro):.3f}'
            ax.set_title(title + subtitle, fontsize=9)
            if plotted:
                ax.legend(loc='lower right', fontsize=6)
            else:
                ax.text(0.5, 0.5, 'No valid one-vs-rest curves', ha='center', va='center', transform=ax.transAxes)

        _plot_one_vs_rest(ax_cos, 'shape_membership_cosine', 'One-vs-rest ROC\nscore = shape-membership cosine')
        _plot_one_vs_rest(ax_js, 'shape_membership_jensen_shannon', 'One-vs-rest ROC\nscore = shape-membership JS')

        pair_curves = pair.get('curves') or {}
        pair_specs = [
            ('shape_membership_cosine', '#1f77b4', 'Shape cosine'),
            ('shape_membership_jensen_shannon', '#2ca02c', 'Shape JS'),
        ]
        plotted_pair = 0
        for mode, color, label in pair_specs:
            curve = pair_curves.get(mode) or {}
            fpr = np.asarray(curve.get('fpr', []), dtype=float)
            tpr = np.asarray(curve.get('tpr', []), dtype=float)
            auc = float(curve.get('auc', float('nan')))
            if fpr.size >= 2 and tpr.size >= 2 and np.isfinite(auc):
                ax_pair.plot(fpr, tpr, color=color, linewidth=2.0, label=f'{label} AUC={auc:.3f}')
                plotted_pair += 1
        sampling = pair.get('sampling', '')
        n_pairs = int(pair.get('n_pairs_evaluated', 0) or 0)
        ax_pair.set_title(f'Pairwise ROC\nsame collection vs cross collection\n{sampling}; pairs={n_pairs:,}', fontsize=9)
        if plotted_pair:
            ax_pair.legend(loc='lower right', fontsize=7)
        else:
            ax_pair.text(0.5, 0.5, 'No valid pairwise curves', ha='center', va='center', transform=ax_pair.transAxes)
        fig.suptitle('Collection separability from morphism-shape membership profiles', fontsize=11)
        fig.tight_layout()
        self._draw_figure('roc', self.roc_frame, fig)

        # Panel 2: one-vs-rest AUC baseline comparison across representation domains.
        baseline_modes = [
            ('raw_sbert_cosine', 'Raw SBERT'),
            ('manifold_residual_cosine', 'Residual doc'),
            ('shape_membership_cosine', 'Shape cosine'),
            ('shape_membership_jensen_shannon', 'Shape JS'),
        ]
        coll_types = list(roc.get('collection_types', []))
        if not coll_types:
            coll_types = sorted({str(r.get('collection_type')) for r in roc.get('rows', []) if r.get('roc_type') == 'one_vs_rest'})
        auc_by_mode: Dict[str, Dict[str, float]] = {m: {} for m, _ in baseline_modes}
        for r in roc.get('rows', []):
            if r.get('roc_type') != 'one_vs_rest':
                continue
            mode = str(r.get('score_mode', ''))
            ctype = str(r.get('collection_type', ''))
            try:
                auc_by_mode.setdefault(mode, {})[ctype] = float(r.get('auc', float('nan')))
            except Exception:
                pass
        fig2 = Figure(figsize=(14.2, max(5.0, 0.32 * len(coll_types) + 2.2)), dpi=100)
        ax_bar = fig2.add_subplot(121)
        ax_heat = fig2.add_subplot(122)
        x = np.arange(len(coll_types), dtype=float)
        width = 0.18
        offsets = np.linspace(-1.5 * width, 1.5 * width, len(baseline_modes))
        cmap = cm.get_cmap('tab10', len(baseline_modes))
        heat = np.full((len(baseline_modes), len(coll_types)), np.nan, dtype=float)
        for mi, (mode, label) in enumerate(baseline_modes):
            vals = np.asarray([auc_by_mode.get(mode, {}).get(c, np.nan) for c in coll_types], dtype=float)
            heat[mi, :] = vals
            if np.isfinite(vals).any():
                ax_bar.bar(x + offsets[mi], np.nan_to_num(vals, nan=0.0), width=width, label=label, color=cmap(mi))
        ax_bar.axhline(0.5, color='0.55', linestyle='--', linewidth=1.0)
        ax_bar.set_ylim(0.0, 1.02)
        ax_bar.set_ylabel('one-vs-rest AUC')
        ax_bar.set_title('Collection ROC baseline comparison')
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(coll_types, rotation=90, fontsize=7)
        ax_bar.legend(loc='lower right', fontsize=7)
        im = ax_heat.imshow(heat, vmin=0.0, vmax=1.0, cmap='viridis', interpolation='nearest', aspect='auto')
        ax_heat.set_title('AUC heatmap by representation')
        ax_heat.set_yticks(range(len(baseline_modes)))
        ax_heat.set_yticklabels([label for _mode, label in baseline_modes], fontsize=8)
        ax_heat.set_xticks(range(len(coll_types)))
        ax_heat.set_xticklabels(coll_types, rotation=90, fontsize=7)
        for mi in range(heat.shape[0]):
            for ci in range(heat.shape[1]):
                if np.isfinite(heat[mi, ci]) and len(coll_types) <= 16:
                    ax_heat.text(ci, mi, f'{heat[mi, ci]:.2f}', ha='center', va='center', fontsize=6, color='white' if heat[mi, ci] < 0.55 else 'black')
        fig2.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
        fig2.tight_layout()
        self._draw_figure('roc_baseline', self.roc_baseline_frame, fig2)

        # Panel 3: pairwise ROC baselines.
        fig3 = Figure(figsize=(8.5, 6.4), dpi=100)
        ax = fig3.add_subplot(111)
        ax.plot([0, 1], [0, 1], color='0.75', linestyle='--', linewidth=1.0)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel('False positive rate')
        ax.set_ylabel('True positive rate')
        ax.set_title(f'Pairwise same-collection vs cross-collection ROC\n{sampling}; pairs={n_pairs:,}')
        ax.grid(True, alpha=0.25)
        pair_modes = [
            ('raw_sbert_cosine', 'Raw SBERT'),
            ('manifold_residual_cosine', 'Residual doc'),
            ('shape_membership_cosine', 'Shape cosine'),
            ('shape_membership_jensen_shannon', 'Shape JS'),
        ]
        cmap2 = cm.get_cmap('tab10', len(pair_modes))
        plotted = 0
        for mi, (mode, label) in enumerate(pair_modes):
            curve = pair_curves.get(mode) or {}
            fpr = np.asarray(curve.get('fpr', []), dtype=float)
            tpr = np.asarray(curve.get('tpr', []), dtype=float)
            auc = float(curve.get('auc', float('nan')))
            if fpr.size >= 2 and tpr.size >= 2 and np.isfinite(auc):
                ax.plot(fpr, tpr, linewidth=2.0, color=cmap2(mi), label=f'{label} AUC={auc:.3f}')
                plotted += 1
        if plotted:
            ax.legend(loc='lower right', fontsize=8)
        else:
            ax.text(0.5, 0.5, 'No valid pairwise baseline curves', ha='center', va='center', transform=ax.transAxes)
        fig3.tight_layout()
        self._draw_figure('roc_pairwise_baseline', self.roc_pairwise_baseline_frame, fig3)

        # Panel 4: similarity-matrix / pairwise score-vector correlations.
        corr = roc.get('similarity_matrix_correlations') or {}
        reps = list(corr.get('representations', []))
        pear = np.asarray(corr.get('pearson_matrix', []), dtype=float)
        spear = np.asarray(corr.get('spearman_matrix', []), dtype=float)
        fig4 = Figure(figsize=(11.0, max(4.8, 0.55 * len(reps) + 2.0)), dpi=100)
        ax1 = fig4.add_subplot(121)
        ax2 = fig4.add_subplot(122)
        if pear.ndim == 2 and spear.ndim == 2 and pear.size and spear.size and reps:
            labels = [r.replace('_', '\n') for r in reps]
            for axm, mat, title in ((ax1, pear, 'Pearson correlation'), (ax2, spear, 'Spearman rank correlation')):
                imc = axm.imshow(mat, vmin=-1.0, vmax=1.0, cmap='coolwarm', interpolation='nearest')
                axm.set_title(title)
                axm.set_xticks(range(len(reps)))
                axm.set_yticks(range(len(reps)))
                axm.set_xticklabels(labels, rotation=90, fontsize=7)
                axm.set_yticklabels(labels, fontsize=7)
                for i in range(len(reps)):
                    for j in range(len(reps)):
                        if len(reps) <= 8 and np.isfinite(mat[i, j]):
                            axm.text(j, i, f'{mat[i, j]:.2f}', ha='center', va='center', fontsize=7, color='white' if abs(mat[i, j]) > 0.55 else 'black')
                fig4.colorbar(imc, ax=axm, fraction=0.046, pad=0.04)
            fig4.suptitle(f'Representation similarity-matrix correlations; pairs={int(corr.get("n_pairs_used", 0) or 0):,}', fontsize=10)
        else:
            ax1.text(0.5, 0.5, 'No representation-correlation data available', ha='center', va='center', transform=ax1.transAxes)
            ax2.axis('off')
        fig4.tight_layout()
        self._draw_figure('roc_corr', self.roc_corr_frame, fig4)

        self.populate_roc_table()
        try:
            self.nb.select(self.roc_tab)
        except Exception:
            pass
        self.status_var.set('Rendered Collection ROC baselines, pairwise baselines, and similarity-correlation panels.')

    def populate_similar_docs(self) -> None:
        for item in self.similar_tree.get_children():
            self.similar_tree.delete(item)
        if not isinstance(self.experiment, dict):
            return
        docs = list(self.experiment.get('doc_ids', [])); coll = list(self.experiment.get('collection_type', []))
        M = np.asarray(self.experiment.get('doc_shape_membership', []), dtype=float)
        S = np.asarray((self.experiment.get('doc_similarity') or {}).get('matrix', []), dtype=float)
        d = self.selected_doc_var.get()
        if d not in docs or S.ndim != 2:
            return
        i = docs.index(d)
        vals = S[i].copy(); vals[i] = -np.inf
        order = np.argsort(vals)[::-1][:50]
        for rank, j in enumerate(order, start=1):
            if not np.isfinite(vals[j]):
                continue
            shared = np.argsort(np.minimum(M[i], M[j]))[::-1][:5] if M.ndim == 2 and M.size else []
            diff = np.argsort(np.abs(M[i] - M[j]))[::-1][:5] if M.ndim == 2 and M.size else []
            row = {
                'rank': rank,
                'doc_id': docs[int(j)],
                'collection_type': coll[int(j)] if int(j) < len(coll) else '',
                'similarity': float(vals[j]),
                'same_collection_type': str((coll[int(j)] if int(j) < len(coll) else '') == (coll[i] if i < len(coll) else '')),
                'top_shared_shapes': '; '.join(f'S{int(s)}' for s in shared),
                'top_differing_shapes': '; '.join(f'S{int(s)}' for s in diff),
            }
            vals_out = [row.get(c, '') if c != 'similarity' else _fmt_float(row.get(c), 4) for c in self.similar_tree['columns']]
            self.similar_tree.insert('', 'end', iid=str(rank), values=vals_out)

    def render_doc_graph(self) -> None:
        if not isinstance(self.experiment, dict):
            messagebox.showinfo('No experiment', 'Build or load an arrangement experiment first.')
            return
        M = np.asarray(self.experiment.get('doc_shape_membership', []), dtype=float)
        docs = list(self.experiment.get('doc_ids', [])); coll = list(self.experiment.get('collection_type', []))
        edges = list((self.experiment.get('doc_similarity') or {}).get('edges', []))
        if M.ndim != 2 or M.shape[0] == 0:
            return
        Y = _arr_pca_2d(M)
        types = sorted(set(coll))
        color_map = {t: cm.tab20(i % 20) for i, t in enumerate(types)}
        fig = Figure(figsize=(10, 8), dpi=100)
        ax = fig.add_subplot(111)
        max_edges = 1000
        for a, b, w in edges[:max_edges]:
            ax.plot([Y[a, 0], Y[b, 0]], [Y[a, 1], Y[b, 1]], color='0.78', linewidth=0.4 + 1.8 * float(w), alpha=0.5)
        for t in types:
            idx = [i for i, c in enumerate(coll) if c == t]
            if not idx: continue
            ax.scatter(Y[idx, 0], Y[idx, 1], s=36, label=t, color=color_map[t], edgecolors='black', linewidths=0.3)
        if len(docs) <= 120:
            for i, d in enumerate(docs):
                ax.text(Y[i, 0], Y[i, 1], str(d), fontsize=6)
        ax.set_title('Document graph by shared morphism-shape membership')
        ax.set_xlabel('membership PCA 1'); ax.set_ylabel('membership PCA 2')
        ax.legend(loc='best', fontsize=7)
        fig.tight_layout()
        self._draw_figure('docgraph', self.docgraph_frame, fig)

    def render_shape_graph(self) -> None:
        if not isinstance(self.experiment, dict):
            messagebox.showinfo('No experiment', 'Build or load an arrangement experiment first.')
            return
        co = self.experiment.get('shape_cooccurrence') or {}
        W = np.asarray(co.get('matrix', []), dtype=float)
        edges = list(co.get('edges', []))
        M = np.asarray(self.experiment.get('doc_shape_membership', []), dtype=float)
        summaries = list(self.experiment.get('shape_summary', []))
        if W.ndim != 2 or W.shape[0] == 0:
            return
        # Layout from shape-membership columns; fallback to circle.
        Y = _arr_pca_2d(M.T) if M.ndim == 2 and M.shape[1] == W.shape[0] else np.zeros((W.shape[0], 2), dtype=float)
        if np.allclose(Y, 0):
            ang = np.linspace(0, 2 * math.pi, W.shape[0], endpoint=False)
            Y = np.column_stack([np.cos(ang), np.sin(ang)])
        support = np.asarray([s.get('support_docs', 0) for s in summaries], dtype=float) if summaries else np.ones(W.shape[0])
        entropy = np.asarray([s.get('collection_type_entropy', 0.0) for s in summaries], dtype=float) if summaries else np.zeros(W.shape[0])
        fig = Figure(figsize=(10, 8), dpi=100)
        ax = fig.add_subplot(111)
        for a, b, w, cnt in edges[:1000]:
            ax.plot([Y[a, 0], Y[b, 0]], [Y[a, 1], Y[b, 1]], color='0.70', linewidth=0.4 + 2.0 * min(1.0, float(w)), alpha=0.45)
        sizes = 40.0 + 18.0 * np.sqrt(np.maximum(0.0, support))
        sc = ax.scatter(Y[:, 0], Y[:, 1], s=sizes, c=entropy, cmap='viridis', edgecolors='black', linewidths=0.4)
        for i in range(W.shape[0]):
            ax.text(Y[i, 0], Y[i, 1], f'S{i}', fontsize=7, ha='center', va='center')
        fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02, label='collection-type entropy')
        ax.set_title('Shape-category co-occurrence graph')
        ax.set_xlabel('shape membership PCA 1'); ax.set_ylabel('shape membership PCA 2')
        fig.tight_layout()
        self._draw_figure('shapegraph', self.shapegraph_frame, fig)


# -----------------------------------------------------------------------------
# Main application shell
# -----------------------------------------------------------------------------

class MorphismAnalysisApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Morphism Analysis and Evidence Platform")
        self.geometry("1500x920")
        self.minsize(1100, 720)
        self.model = MorphismComparisonModel()
        self._loading_thread: Optional[threading.Thread] = None
        self.status_var = tk.StringVar(value="Open a morphism_comparison.pkl file to begin.")
        self._build_menu()
        self._build_body()

    def _build_menu(self) -> None:
        menubar = tk.Menu(self)
        file_menu = tk.Menu(menubar, tearoff=False)
        file_menu.add_command(label="Open morphism_comparison.pkl...", command=self.open_comparison)
        file_menu.add_separator()
        file_menu.add_command(label="Open companion document_delta_dict.pkl...", command=self.open_document_delta)
        file_menu.add_command(label="Open companion segments_by_doc.pkl...", command=self.open_segments)
        file_menu.add_separator()
        file_menu.add_command(label="Export current query rows...", command=lambda: self.query_tab.export_rows())
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.destroy)
        menubar.add_cascade(label="File", menu=file_menu)

        view_menu = tk.Menu(menubar, tearoff=False)
        view_menu.add_command(label="Refresh all views", command=self.refresh_all)
        view_menu.add_command(label="Render current graph view", command=lambda: self.plot_tab.render())
        view_menu.add_command(label="Render selected edge match 3D", command=lambda: self.edge3d_tab.render_async())
        menubar.add_cascade(label="View", menu=view_menu)

        help_menu = tk.Menu(menubar, tearoff=False)
        help_menu.add_command(label="About", command=self.show_about)
        menubar.add_cascade(label="Help", menu=help_menu)
        self.config(menu=menubar)

    def _build_body(self) -> None:
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.notebook = ttk.Notebook(self)
        self.notebook.grid(row=0, column=0, sticky="nsew")
        self.schema_tab = SchemaInspector(self.notebook, self.model)
        self.query_tab = MatchQueryWorkbench(
            self.notebook,
            self.model,
            on_select_match=self._selected_match_changed,
            on_open_evidence=self._open_match_in_evidence,
            on_open_3d=self._open_match_in_3d,
        )
        self.plot_tab = PlotCacheView(self.notebook, self.model)
        self.evidence_tab = EvidenceBrowser(self.notebook, self.model)
        self.edge3d_tab = EdgeMatch3DView(self.notebook, self.model)
        self.arrangement_tab = ArrangementExperimentView(self.notebook, self.model)
        self.notebook.add(self.schema_tab, text="1. Schema inspector")
        self.notebook.add(self.query_tab, text="2. Match query workbench")
        self.notebook.add(self.plot_tab, text="3. Plot-cache graph views")
        self.notebook.add(self.evidence_tab, text="4. Candidate evidence browser")
        self.notebook.add(self.edge3d_tab, text="5. Edge match 3D")
        self.notebook.add(self.arrangement_tab, text="6. Arrangement experiment")
        status = ttk.Label(self, textvariable=self.status_var, anchor="w")
        status.grid(row=1, column=0, sticky="ew", padx=6, pady=3)

    def show_about(self) -> None:
        messagebox.showinfo(
            "About",
            "Standalone Morphism Analysis and Evidence Documentation Platform\n\n"
            "Loads compact/enriched morphism_comparison.pkl files and provides:\n"
            "1. Schema inspector\n2. Query workbench\n3. Plot-cache graph views\n4. Evidence browser\n"
            "5. Selected edge-match 3D visualization with on-demand segment re-embedding\n"
            "6. Arrangement experiment over morphism-shape category membership\n\n"
            "Only open pickle files from trusted sources.",
        )

    def _install_model(self, model: MorphismComparisonModel) -> None:
        """Install a freshly loaded model and repoint all tab controllers."""
        self.model = model
        self.schema_tab.model = model
        self.query_tab.model = model
        self.plot_tab.model = model
        self.evidence_tab.model = model
        self.edge3d_tab.model = model
        self.arrangement_tab.model = model

    def _after_comparison_loaded(self, model: MorphismComparisonModel, elapsed: float) -> None:
        self._install_model(model)
        self.status_var.set(
            f"Loaded {os.path.basename(model.path or '')}: {model.n_docs:,} docs; "
            f"{model.n_edges:,} edges; {model.n_matches:,} matches. "
            f"pickle load: {elapsed:.2f}s. Schema refreshed; run queries/candidates when needed."
        )
        # Keep open cheap: refresh only schema and plot-cache metadata.  Query
        # tables, candidate tables, and graph rendering are intentionally manual
        # because they can scan/sort/insert thousands of rows.
        self.refresh_all()
        try:
            self.query_tab.clear_table("File loaded. Click Run query to populate the table.")
            self.evidence_tab.clear_candidates("File loaded. Click Refresh candidates to populate candidate evidence rows.")
            self.edge3d_tab.clear_view()
            self.edge3d_tab.status_var.set("File loaded. Run a query, select a row, then click Open selected in 3D.")
            self.arrangement_tab.clear_for_new_document_delta()
        except Exception:
            self.query_tab.status_var.set("File loaded. Click Run query to populate the table.")
            self.evidence_tab.status_var.set("File loaded. Click Refresh candidates to populate candidate evidence rows.")
            self.edge3d_tab.status_var.set("File loaded. Run a query, select a row, then click Open selected in 3D.")
            self.arrangement_tab.clear_for_new_document_delta()

    def open_comparison(self) -> None:
        path = filedialog.askopenfilename(
            title="Open morphism_comparison.pkl",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")],
        )
        if not path:
            return
        if self._loading_thread is not None and self._loading_thread.is_alive():
            messagebox.showinfo("Load in progress", "A comparison file is already loading.")
            return
        old_model = self.model
        self.status_var.set(f"Loading {path} ...")
        self.update_idletasks()

        def worker() -> None:
            t0 = time.perf_counter()
            try:
                model = MorphismComparisonModel()
                # Preserve already loaded companion artifacts across comparison-file swaps.
                model.document_delta_dict = old_model.document_delta_dict
                model.segments_by_doc = old_model.segments_by_doc
                model._doc_key_by_str = dict(old_model._doc_key_by_str)
                model._seg_key_by_str = dict(old_model._seg_key_by_str)
                model.load_comparison(path)
                elapsed = time.perf_counter() - t0
                self.after(0, lambda: self._after_comparison_loaded(model, elapsed))
            except Exception as ex:
                err = str(ex)
                tb = traceback.format_exc(limit=6)
                self.after(0, lambda err=err, tb=tb: (
                    self.status_var.set("Load failed."),
                    messagebox.showerror("Open failed", f"{err}\n\n{tb}")
                ))

        self._loading_thread = threading.Thread(target=worker, daemon=True)
        self._loading_thread.start()
    def open_segments(self) -> None:
        path = filedialog.askopenfilename(
            title="Open segments_by_doc.pkl",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            self.model.load_segments(path)
            self.status_var.set(f"Loaded companion segments_by_doc: {os.path.basename(path)}")
            if self.evidence_tab.selected_match_row is not None:
                self.evidence_tab._render_evidence()
            if self.edge3d_tab.selected_match_row is not None:
                self.edge3d_tab.status_var.set("segments_by_doc loaded. Click Render to rebuild the selected 3D view with point clouds.")
        except Exception as ex:
            messagebox.showerror("Open segments failed", f"{ex}\n\n{traceback.format_exc(limit=4)}")

    def open_document_delta(self) -> None:
        path = filedialog.askopenfilename(
            title="Open document_delta_dict.pkl",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            self.model.load_document_delta_dict(path)
            self.status_var.set(f"Loaded companion document_delta_dict: {os.path.basename(path)}")
            if self.evidence_tab.selected_match_row is not None:
                self.evidence_tab._render_evidence()
            if self.edge3d_tab.selected_match_row is not None:
                self.edge3d_tab.status_var.set("document_delta_dict loaded. Click Render to rebuild the selected 3D view.")
            self.arrangement_tab.clear_for_new_document_delta()
        except Exception as ex:
            messagebox.showerror("Open document delta failed", f"{ex}\n\n{traceback.format_exc(limit=4)}")

    def refresh_all(self) -> None:
        self.schema_tab.refresh()
        self.plot_tab.refresh()
        try:
            self.arrangement_tab.refresh_all_views()
        except Exception:
            pass

    def _selected_match_changed(self, match_row: int) -> None:
        """Keep evidence and 3D tabs synchronized with the query selection without switching tabs."""
        try:
            self.evidence_tab.select_match(match_row)
        except Exception:
            pass
        try:
            self.edge3d_tab.set_match(match_row, render=False)
        except Exception:
            pass

    def _open_match_in_evidence(self, match_row: int) -> None:
        try:
            self.evidence_tab.select_match(match_row)
            self.edge3d_tab.set_match(match_row, render=False)
            self.notebook.select(self.evidence_tab)
        except Exception:
            pass

    def _open_match_in_3d(self, match_row: int) -> None:
        try:
            self.evidence_tab.select_match(match_row)
        except Exception:
            pass
        try:
            self.edge3d_tab.set_match(match_row, render=False)
            self.notebook.select(self.edge3d_tab)
        except Exception:
            pass


def main() -> None:
    app = MorphismAnalysisApp()
    if len(sys.argv) > 1:
        candidate = sys.argv[1]
        if os.path.exists(candidate):
            try:
                t0 = time.perf_counter()
                model = MorphismComparisonModel()
                model.load_comparison(candidate)
                app._after_comparison_loaded(model, time.perf_counter() - t0)
            except Exception as ex:
                messagebox.showerror("Open failed", f"{ex}\n\n{traceback.format_exc(limit=6)}")
    app.mainloop()


if __name__ == "__main__":
    main()
