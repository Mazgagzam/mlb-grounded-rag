"""Evaluate candidate and retrieval recall on independently labeled JSON paths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from predict import load_encoder
from rule_based_solution import Solver, scalar_text
from rule_rag_hybrid import retrieve_evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--encoder", help="Optional BGE model name or local path")
    parser.add_argument("--top-k", type=int, default=16)
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be positive")

    cases = [
        json.loads(line) for line in args.labels.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not cases:
        raise ValueError("No labeled cases")
    for case in cases:
        if not all(key in case for key in ("question", "gold_path", "answer")):
            raise ValueError("Each case needs question, gold_path, and answer")
    solver = Solver(args.data_dir, [case["question"] for case in cases])
    encoder = load_encoder(args.encoder)
    candidate_hits = retrieval_hits = answer_hits = 0
    for case in cases:
        answer, candidates = solver.answer(case["question"], top_n=150)
        gold_path = case["gold_path"]
        candidate_paths = {"/".join(candidate.path) for candidate in candidates}
        evidence = retrieve_evidence(
            case["question"], candidates, encoder=encoder, top_k=args.top_k
        )
        retrieved_paths = {row.path for row in evidence}
        candidate_hits += gold_path in candidate_paths
        retrieval_hits += gold_path in retrieved_paths
        answer_hits += scalar_text(answer) == str(case["answer"])
        print(json.dumps({
            "question": case["question"],
            "candidate_hit": gold_path in candidate_paths,
            "retrieval_hit": gold_path in retrieved_paths,
            "rule_answer_correct": scalar_text(answer) == str(case["answer"]),
        }))
    n = len(cases)
    print(json.dumps({
        "count": n,
        "candidate_recall_at_150": candidate_hits / n,
        f"retrieval_recall_at_{args.top_k}": retrieval_hits / n,
        "rule_answer_exact_match": answer_hits / n,
    }, indent=2))


if __name__ == "__main__":
    main()
