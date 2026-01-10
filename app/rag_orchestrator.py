from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from app.embeddings import create_embedding
from app.settings import get_settings
from app.vector_store import VectorStore


def _truncate(text: str, max_chars: int) -> str:
    t = (text or "").strip()
    return t if len(t) <= max_chars else t[: max(0, max_chars - 3)] + "..."


def retrieve_support_docs(
    queries: Iterable[str],
    *,
    per_query_k: Optional[int] = None,
    top_k: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieve top-k support snippets from Chroma for the provided queries.
    Returns a list of unique citations sorted by distance ascending.
    """
    settings = get_settings()
    vs = VectorStore(
        persist_directory=settings.chroma_db_path,
        collection_name=settings.chroma_collection_name,
    )
    per_q = per_query_k or settings.retrieval_per_query_k
    cap = top_k or settings.retrieval_top_k

    aggregated: List[Tuple[str, float, str, Dict[str, Any]]] = []

    for q in queries:
        q = (q or "").strip()
        if not q:
            continue
        q_emb = create_embedding(q, model_name=settings.embedding_model_name)
        res = vs.search(query_embedding=q_emb, n_results=per_q)
        ids = res.get("ids", [])
        dists = res.get("distances", [])
        docs = res.get("documents", [])
        metas = res.get("metadatas", [])
        for i in range(min(len(ids), len(dists), len(docs), len(metas))):
            aggregated.append((ids[i], dists[i], docs[i], metas[i] or {}))

    # Deduplicate by id, keep best (lowest distance)
    best_by_id: Dict[str, Tuple[str, float, str, Dict[str, Any]]] = {}
    for cid, dist, doc, meta in aggregated:
        if cid not in best_by_id or dist < best_by_id[cid][1]:
            best_by_id[cid] = (cid, dist, doc, meta)

    unique = list(best_by_id.values())
    unique.sort(key=lambda x: x[1])
    unique = unique[:cap]

    citations: List[Dict[str, Any]] = []
    for cid, dist, doc, meta in unique:
        citations.append(
            {
                "id": cid,
                "file": meta.get("source_file"),
                "page": meta.get("page_number"),
                "distance": float(dist) if dist is not None else None,
                "snippet": _truncate(doc, 800),
            }
        )
    return citations


def _load_prompt_template() -> str:
    # Prefer external template if exists, else built-in fallback
    template_path = Path(__file__).parent / "prompts" / "rag_answering.md"
    if template_path.exists():
        return template_path.read_text(encoding="utf-8")
    # Fallback minimal template
    return (
        "You are a senior service technician. Use ONLY the provided citations to answer.\n"
        "If a detail is not supported by citations, explicitly say it is not confirmed.\n"
        "Respond in the requested language. Provide a concise, actionable plan.\n"
    )


def compose_grounded_prompt(
    *,
    transcript: str,
    analysis: Dict[str, Any],
    clarifying_questions: List[str],
    citations: List[Dict[str, Any]],
    language: str,
) -> str:
    tmpl = _load_prompt_template()
    appliance = analysis.get("appliance_type")
    brand = analysis.get("brand_or_model")
    issue = analysis.get("issue_summary")

    citations_block_lines: List[str] = []
    for c in citations:
        src = c.get("file") or "unknown.pdf"
        page = c.get("page")
        dist = c.get("distance")
        snip = c.get("snippet") or ""
        citations_block_lines.append(
            f"- [{src} p.{page}] (distance={dist}): {snip}"
        )
    citations_block = "\n".join(citations_block_lines)

    clarifying_block = "\n".join(f"- {q}" for q in clarifying_questions)

    prompt = (
        f"{tmpl}\n\n"
        f"Language: {language}\n"
        f"Appliance: {appliance}\n"
        f"Brand/Model: {brand}\n"
        f"Issue summary: {issue}\n\n"
        "Transcript (verbatim, may be noisy):\n"
        f"{_truncate(transcript, 4000)}\n\n"
        "ClarifyingQuestions (ask only if needed and not already answered):\n"
        f"{clarifying_block}\n\n"
        "Citations:\n"
        f"{citations_block}\n\n"
        "Instructions:\n"
        "- Base every technical claim on the citations. If unclear, state uncertainty.\n"
        "- Provide a safe, step-by-step troubleshooting plan tailored to this brand/model.\n"
        "- Include required tools/parts from citations when available.\n"
        "- Keep it concise and practical.\n"
    )
    return prompt


def answer_with_gemini(
    *,
    api_key: str,
    model: str,
    prompt: str,
) -> Dict[str, Any]:
    try:
        from google import genai  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise RuntimeError('Missing dependency "google-genai". Install it with: pip install google-genai') from e

    client = genai.Client(api_key=api_key)
    resp = client.models.generate_content(model=model, contents=[prompt])
    text = (resp.text or "").strip()
    return {"text": text}


