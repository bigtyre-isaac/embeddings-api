import logging
import os
import time
import hashlib
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from fastapi.middleware.gzip import GZipMiddleware

# --- Logging / model init ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
logger = logging.getLogger("embeddings-api")

MODEL_NAME = os.getenv("MODEL_NAME", "all-MiniLM-L6-v2")
CACHE_DIR = Path(os.getenv("CACHE_DIR", "/cache"))  # mounted volume recommended
CACHE_DIR.mkdir(parents=True, exist_ok=True)

logger.info(f"Loading embedding model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)

app = FastAPI(title="Embeddings API", version="1.3.0")
app.add_middleware(GZipMiddleware, minimum_size=512)  # compress responses

# --- Middleware: request logs ---
@app.middleware("http")
async def request_logger(request: Request, call_next):
    start = time.time()
    logger.info(f">>> {request.method} {request.url.path}")
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("Unhandled error")
        raise
    dur = int((time.time() - start) * 1000)
    logger.info(f"<<< {request.method} {request.url.path} {response.status_code} ({dur} ms)")
    return response

# --- Health ---
@app.get("/healthz")
async def healthz():
    _ = model.get_sentence_embedding_dimension()
    return {"status": "ok", "model": MODEL_NAME}

# --- Helpers (normalized everywhere) ---
def _l2_normalize_np(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n[n == 0] = 1.0
    return v / n

def _hash_text(text: str) -> str:
    # Include model name to invalidate cache when model changes
    h = hashlib.sha256()
    h.update(MODEL_NAME.encode("utf-8"))
    h.update(b"\x00")
    # Normalise whitespace; case-sensitive to preserve semantics like SKUs if needed
    norm = " ".join(text.split())
    h.update(norm.encode("utf-8"))
    return h.hexdigest()

def _cache_path(text: str) -> Path:
    return CACHE_DIR / f"{_hash_text(text)}.npy"  # float32, normalized

def _load_cached(text: str) -> Optional[np.ndarray]:
    p = _cache_path(text)
    #logger.info(f"Checking for cached result at {p} for {text}")
    if not p.exists():
        #logger.info(f"Cached result does not exist for {text}")
        return None
    try:
        #logger.info(f"Cached result found for {text}. Loading...");

        # mmap for zero-copy reads
        arr = np.load(p, mmap_mode="r").astype(np.float32)
        # safety: re-normalize (cheap) in case of legacy entries
        return _l2_normalize_np(arr)
    except Exception:
        logger.warning(f"Failed to read cache file {p}, ignoring.")
        return None

def _save_cache(text: str, vec: np.ndarray) -> None:
    p = _cache_path(text) 
    p.parent.mkdir(parents=True, exist_ok=True)

    try:
        np.save(p, vec.astype(np.float32))   # overwrite directly
    except Exception:
        logger.warning(f"Failed to write cache file {p}", exc_info=True)

def _embed_texts_normalized(texts: List[str]) -> List[np.ndarray]:
    """Batch-embed texts with sentence-transformers, normalized output."""
    # sentence-transformers does batching internally; adjust if needed: encode(..., batch_size=32)
    vecs = model.encode(texts, normalize_embeddings=True)
    return [np.asarray(v, dtype=np.float32) for v in vecs]

# --- Schemas ---
class EmbeddingRequest(BaseModel):
    text: str

class EmbeddingResponse(BaseModel):
    text: str
    embedding: List[float]
    model: str
    dim: int

class CompareTextRequest(BaseModel):
    text: str
    candidates: List[str]
    top: Optional[int] = None          # replaces top_k
    include_debug: bool = False

# Result item (unchanged)
class CompareResult(BaseModel):
    index: int
    score: float
    text: Optional[str] = None

# Response (no topk; scores is a sorted dict)
class CompareTextResponse(BaseModel):
    metric: str
    model: str
    scores: Dict[str, float] = Field(default_factory=dict)  # sorted desc by score
    best_match: Optional[CompareResult] = None
    debug: Optional[Dict[str, int]] = None  # {"cache_hits": n, "cache_misses": m}

# --- Endpoints ---
@app.post("/api/embeddings/generate", response_model=EmbeddingResponse)
async def generate_embedding(req: EmbeddingRequest):
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Field 'text' is required and cannot be empty.")
    try:
        vec = model.encode(text, normalize_embeddings=True).tolist()
    except Exception as ex:
        logger.exception("Embedding generation failed")
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {type(ex).__name__}")
    return EmbeddingResponse(text=req.text, embedding=vec, model=MODEL_NAME, dim=len(vec))

@app.post("/api/similarity/compare-text", response_model=CompareTextResponse)
async def similarity_compare_text(req: CompareTextRequest):
    q_text = (req.text or "").strip()
    if not q_text:
        raise HTTPException(status_code=400, detail="Field 'text' is required and cannot be empty.")
    c_texts = [(" ".join(t.split())).strip() for t in req.candidates]
    if not c_texts or any(t == "" for t in c_texts):
        raise HTTPException(status_code=400, detail="All 'candidates' must be non-empty strings.")

    # 1) Query embedding (cache → compute if miss)
    q_vec = _load_cached(q_text)
    cache_hits = 0
    cache_misses = 0
    if q_vec is None:
        cache_misses += 1
        q_vec = _embed_texts_normalized([q_text])[0]
        _save_cache(q_text, q_vec)
    else:
        cache_hits += 1

    # 2) Candidate embeddings with dedupe + batch
    unique_texts: Dict[str, int] = {}
    for t in c_texts:
        unique_texts[t] = unique_texts.get(t, 0) + 1

    cand_vecs: Dict[str, np.ndarray] = {}
    to_compute: List[str] = []
    for t in unique_texts.keys():
        v = _load_cached(t)
        if v is None:
            to_compute.append(t)
        else:
            cand_vecs[t] = v
            cache_hits += 1

    if to_compute:
        new_vecs = _embed_texts_normalized(to_compute)
        for t, v in zip(to_compute, new_vecs):
            cand_vecs[t] = v
            _save_cache(t, v)
        cache_misses += len(to_compute)

    # 3) Reconstruct candidate matrix in original order and score
    C = np.stack([cand_vecs[t] for t in c_texts], axis=0)  # (N, D)
    q = q_vec  # (D,)
    scores = (C @ q).astype(np.float32)  # cosine == dot (unit vectors)

    # Best match (from full set)
    best_idx = int(np.argmax(scores))
    best_match = CompareResult(
        index=best_idx,
        score=float(scores[best_idx]),
        text=c_texts[best_idx]
    )

    # Build sorted { text: score } dictionary (desc). Collapse duplicates (keep max).
    score_map: Dict[str, float] = {}
    for t, s in zip(c_texts, scores):
        fs = float(s)
        if t not in score_map or fs > score_map[t]:
            score_map[t] = fs

    sorted_items = sorted(score_map.items(), key=lambda kv: kv[1], reverse=True)

    # Apply 'top' limit if provided
    if req.top and req.top > 0:
        k = min(req.top, len(sorted_items))
        sorted_items = sorted_items[:k]

    scores_dict = dict(sorted_items)  # preserves sorted order in Python 3.7+

    return CompareTextResponse(
        metric="cosine",
        model=MODEL_NAME,
        scores=scores_dict,
        best_match=best_match,
        debug={"cache_hits": cache_hits, "cache_misses": cache_misses} if req.include_debug else None
    )


@app.get("/")
async def root():
    return {
        "name": "Embeddings API",
        "model": MODEL_NAME,
        "normalized_everywhere": True,
        "cache_dir": str(CACHE_DIR),
        "endpoints": {
            "health": "/healthz",
            "generate": "POST /api/embeddings/generate",
            "compare_text": "POST /api/similarity/compare-text"
        }
    }
