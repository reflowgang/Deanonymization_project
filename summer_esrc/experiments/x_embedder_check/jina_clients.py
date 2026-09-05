from __future__ import annotations
import json
import os
import urllib.error
import urllib.request
from typing import Sequence
import numpy as np
from openai import OpenAI
from esrc.generate import get_client
JINA_EMBED_MODEL = 'jina-embeddings-v3'
JINA_RERANK_MODEL = 'jina-reranker-v2-base-multilingual'

def embed_texts_jina(texts: Sequence[str], *, model: str=JINA_EMBED_MODEL, batch_size: int=64, client: OpenAI | None=None) -> np.ndarray:
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    client = client or get_client(timeout=300.0)
    out: list[list[float]] = []
    items = list(texts)
    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        resp = client.embeddings.create(model=model, input=batch)
        out.extend((row.embedding for row in resp.data))
    return np.asarray(out, dtype=np.float32)

def _truncate_for_rerank(query: str, documents: Sequence[str], *, max_query_chars: int=600, max_doc_chars: int=180) -> tuple[str, list[str]]:
    q = query.strip()[:max_query_chars]
    docs = [d.strip()[:max_doc_chars] for d in documents]
    return (q, docs)

def rerank_documents(query: str, documents: Sequence[str], *, model: str=JINA_RERANK_MODEL, base_url: str | None=None, api_key: str | None=None, timeout: float=120.0) -> list[tuple[int, float]]:
    if not documents:
        return []
    resolved_base = (base_url or os.getenv('VLLM_BASE_URL') or '').rstrip('/')
    if not resolved_base:
        raise RuntimeError('VLLM_BASE_URL is not set')
    resolved_key = api_key or os.getenv('VLLM_API_KEY') or 'EMPTY'
    (q, docs) = _truncate_for_rerank(query, documents)
    payload = {'model': model, 'query': q, 'documents': docs}
    req = urllib.request.Request(f'{resolved_base}/rerank', data=json.dumps(payload).encode('utf-8'), headers={'Authorization': f'Bearer {resolved_key}', 'Content-Type': 'application/json'}, method='POST')
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode('utf-8'))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode('utf-8', errors='replace')
        raise RuntimeError(f'rerank HTTP {exc.code}: {detail[:500]}') from exc
    results = body.get('results') or []
    ranked: list[tuple[int, float]] = []
    for row in results:
        ranked.append((int(row['index']), float(row['relevance_score'])))
    ranked.sort(key=lambda x: -x[1])
    return ranked
