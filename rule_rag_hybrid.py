"""Rule-guided retrieval and grounded LLM selection for nested MLB JSON QA.

The rule solver supplies a strong baseline and a small set of relevant JSON
leaves. Sparse and dense retrieval reorder those leaves. The language model
selects a path; Python copies the original scalar value unchanged.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from rule_based_solution import Solver, scalar_text


@dataclass(frozen=True)
class Evidence:
    path: str
    value: str
    source: str
    rule_score: float
    retrieval_score: float = 0.0


def _scale(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    spread = values.max() - values.min()
    return (values - values.min()) / spread if spread > 1e-12 else np.ones_like(values)


def _search_text(path: str) -> str:
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", path)
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def retrieve_evidence(question: str, candidates, encoder=None, top_k: int = 16) -> list[Evidence]:
    """Hybrid sparse/dense reranking of exact JSON leaves from the rule solver."""
    unique = {}
    for candidate in candidates:
        evidence = Evidence(
            path="/".join(candidate.path),
            value=scalar_text(candidate.value),
            source=candidate.source,
            rule_score=float(candidate.score),
        )
        unique.setdefault((evidence.path, evidence.value), evidence)
    rows = list(unique.values())
    if not rows:
        return []

    documents = [_search_text(row.source + " " + row.path) for row in rows]
    query = _search_text(question)
    word = TfidfVectorizer(ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b")
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
    word_matrix, char_matrix = word.fit_transform(documents), char.fit_transform(documents)
    word_score = (word_matrix @ word.transform([query]).T).toarray().ravel()
    char_score = (char_matrix @ char.transform([query]).T).toarray().ravel()
    sparse = 0.55 * _scale(word_score) + 0.45 * _scale(char_score)
    rule = _scale(np.array([row.rule_score for row in rows]))
    if encoder is not None:
        vectors = encoder.encode(
            documents, batch_size=64, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        )
        qvector = encoder.encode(
            "Represent this sentence for searching relevant database fields: " + query,
            normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False,
        )
        dense = _scale(vectors @ qvector)
        scores = 0.35 * sparse + 0.35 * dense + 0.30 * rule
    else:
        scores = 0.60 * sparse + 0.40 * rule

    order = np.argsort(-scores)
    # Keep a few high-scoring rule paths even when their sparse wording differs.
    chosen = list(dict.fromkeys([0, *order[:top_k].tolist()]))[:top_k]
    return [Evidence(**{**rows[i].__dict__, "retrieval_score": float(scores[i])}) for i in chosen]


def build_messages(question: str, evidence: list[Evidence]):
    lines = [
        f"C{i}: source={row.source} | path={row.path} | value={row.value}"
        for i, row in enumerate(evidence)
    ]
    instruction = (
        "You are selecting an exact scalar field from a baseball JSON database. "
        "Match every explicit qualifier in the question: entity, year, season type, "
        "opponent, month, game, role, and statistic. Return only one candidate "
        "label such as C3. Do not calculate, rewrite, or invent a value.\n\n"
        f"Question: {question}\n\nCandidates:\n" + "\n".join(lines)
    )
    return [{"role": "user", "content": [{"type": "text", "text": instruction}]}]


def choose_with_gemma(question: str, evidence: list[Evidence], model, processor):
    """Fail loudly on invalid output; never silently turn a failed LLM into top-1."""
    import torch

    if not evidence:
        raise ValueError("No retrieved evidence")
    inputs = processor.apply_chat_template(
        build_messages(question, evidence), add_generation_prompt=True,
        tokenize=True, return_dict=True, return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    prompt_length = inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    tokens = generated[0][prompt_length:]
    raw = processor.decode(tokens, skip_special_tokens=True).strip()
    labels = re.findall(r"\bC(\d+)\b", raw)
    if len(labels) != 1 or int(labels[0]) >= len(evidence):
        debug = processor.decode(tokens, skip_special_tokens=False)
        raise RuntimeError(f"Gemma did not select exactly one valid path: {debug!r}")
    index = int(labels[0])
    return evidence[index], raw


def passes_evidence_gate(
    chosen: Evidence,
    evidence: list[Evidence],
    *,
    rule_tolerance: float = 5.0,
    retrieval_tolerance: float = 0.10,
) -> bool:
    """Accept a model choice only when its evidence is close to both strongest signals."""
    if chosen not in evidence or not evidence:
        return False
    return (
        chosen.rule_score >= max(row.rule_score for row in evidence) - rule_tolerance
        and chosen.retrieval_score
        >= max(row.retrieval_score for row in evidence) - retrieval_tolerance
    )
