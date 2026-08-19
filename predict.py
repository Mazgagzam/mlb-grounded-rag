"""Run source-grounded question answering without competition-specific overrides."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from rule_based_solution import Solver
from rule_rag_hybrid import choose_with_gemma, passes_evidence_gate, retrieve_evidence


def load_encoder(name_or_path: str | None):
    if not name_or_path:
        return None
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name_or_path)


def load_gemma(name_or_path: str | None):
    if not name_or_path:
        return None, None
    from transformers import AutoProcessor, Gemma3ForConditionalGeneration

    processor = AutoProcessor.from_pretrained(name_or_path)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        name_or_path, device_map="auto", torch_dtype="auto"
    ).eval()
    return model, processor


def answer_one(question: str, solver: Solver, encoder, model, processor, top_k: int):
    _, candidates = solver.answer(question, top_n=150)
    if not candidates:
        raise ValueError(f"No source field matched question: {question}")
    evidence = retrieve_evidence(question, candidates, encoder=encoder, top_k=top_k)
    fallback = evidence[0]
    chosen = fallback
    model_choice = None
    accepted = False
    if model is not None:
        model_choice, _ = choose_with_gemma(question, evidence, model, processor)
        accepted = passes_evidence_gate(model_choice, evidence)
        if accepted:
            chosen = model_choice
    return {
        "question": question,
        "answer": chosen.value,
        "source": chosen.source,
        "path": chosen.path,
        "decision": (
            "rule_only" if model is None else "gemma" if accepted else "rule_fallback"
        ),
        "model_choice": asdict(model_choice) if model_choice else None,
        "evidence": [asdict(row) for row in evidence],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--question", action="append", required=True)
    parser.add_argument("--encoder", help="Optional BGE model name or local path")
    parser.add_argument("--gemma", help="Optional Gemma 3 model name or local path")
    parser.add_argument("--top-k", type=int, default=16)
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be positive")

    solver = Solver(args.data_dir, args.question)
    encoder = load_encoder(args.encoder)
    model, processor = load_gemma(args.gemma)
    for question in args.question:
        print(json.dumps(
            answer_one(question, solver, encoder, model, processor, args.top_k),
            ensure_ascii=False,
        ))


if __name__ == "__main__":
    main()
