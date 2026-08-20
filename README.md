# Query-driven RAG for structured baseball statistics

This project answers natural-language questions over nested baseball JSON. It combines a baseball-aware path resolver, sparse and optional dense retrieval, and an optional Gemma 3 path selector. The answer is copied from the selected JSON field, and the output includes its source path.

The code takes a new question at runtime. It does not look up a question ID or store answers from the competition leaderboard.

## How it works

1. **Candidate generation.** A deterministic resolver identifies the player or team and extracts constraints such as season, opponent, venue, role, and statistic. It ranks up to 150 matching scalar JSON fields.
2. **Retrieval.** Word and character TF-IDF, optional BGE embeddings, and the resolver score rerank those fields. Up to 16 source paths are sent to the selector.
3. **Grounded selection.** Gemma 3 chooses a candidate ID, not a numeric answer. Python copies the chosen value and JSON path from the source data.
4. **Evidence gate.** A model choice must stay close to the strongest rule and retrieval scores. Otherwise the deterministic answer is returned.

The resolver is specific to the baseball schema used in the [Who's the Best Pitcher? v2 competition](https://www.kaggle.com/competitions/who-s-the-best-pitcher-v2/overview); it is not a general JSON question-answering engine.

## Quick start

Requires Python 3.10 or newer.

    python -m venv .venv
    pip install -r requirements.txt
    python predict.py --data-dir examples --question "How many walks did the Harbor City Herons have in the 2025 regular season?"
    python evaluate_retrieval.py --data-dir examples --labels examples/labels.jsonl

The example data is synthetic and includes two questions. The CLI prints the exact value, source path, decision type, and retrieved evidence. Without --gemma, it returns the rule-based answer while still showing retrieval results.

For the first example question, the answer is 430 and its source path is data/2025/REG/pitching/overall/bb. Changing the question to preseason selects a different branch.

For the full pipeline, install the optional model dependencies and provide locally available model paths:

    pip install -r requirements-llm.txt
    python predict.py --data-dir /path/to/competition-data --encoder /path/to/bge-large-en-v1.5 --gemma /path/to/gemma-3-4b-it --question "Your question here"

The data directory must contain players.json, teams.json, and league.json. The competition data and model weights are not included in this repository. Running Gemma requires substantially more memory than the synthetic, retrieval-only example.

## Evaluation status

- The two synthetic examples are smoke checks: both gold paths were retrieved and both rule answers matched. They do **not** estimate competition accuracy.
- In a targeted 10-question Kaggle audit, Gemma proposed three answer changes that missed explicit qualifiers. The evidence gate rejected all three. The guarded run produced no answer changes relative to the existing baseline.
- A separate competition submission scored **0.90625 private / 0.95652 public**, but it contained 38 leaderboard-calibrated corrections. Those corrections are intentionally absent here, and those scores must not be attributed to this RAG pipeline.

The next evaluation should use independently labeled questions that were not selected from leaderboard feedback. evaluate_retrieval.py reports candidate recall@150, retrieval recall@k, and deterministic answer exact match for such labels. A full evaluation should also report final model-selected answer accuracy and source-path validity.

## Repository contents

| File | Purpose |
| --- | --- |
| rule_based_solution.py | Schema-aware entity resolution and candidate JSON path generation |
| rule_rag_hybrid.py | Sparse/dense reranking, Gemma path selection, and evidence gate |
| predict.py | Query-driven CLI with optional BGE and Gemma models |
| evaluate_retrieval.py | Evaluation against independently labeled source paths |
| examples/ | Small fictional JSON fixture and two labeled smoke cases |

This repository is a cleaned portfolio version of the Kaggle experiment. It omits competition data, model weights, private notebook metadata, local paths, and answer overrides.
