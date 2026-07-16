"""Export LegalBench queries in the format expected by the LLM reformulation step."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


DATA_DIR = Path("data/processed/legalbench/rag_mini")
QUERIES_PATH = DATA_DIR / "queries.jsonl"

OUT_DIR = DATA_DIR / "reformulations/input"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_JSONL = OUT_DIR / "legalbench_mini_queries_for_reformulation.jsonl"
OUT_CSV = OUT_DIR / "legalbench_mini_queries_for_reformulation.csv"
OUT_SUMMARY = OUT_DIR / "legalbench_reformulation_input_summary.csv"
OUT_PROMPT = OUT_DIR / "legalbench_reformulation_prompt_template.md"


def read_jsonl(path: Path) -> list[dict]:
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                rows.append(json.loads(line))

    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def count_words(text: str) -> int:
    return len((text or "").split())


def build_prompt_template() -> str:
    """
    Prompt template destiné à générer les trois champs utilisés
    dans nos expériences.
    """
    return """# LegalBench-RAG mini reformulation prompt template

You are helping prepare query reformulations for a legal information retrieval experiment.

Given one legal retrieval query, produce three alternative retrieval-oriented query views.

Important constraints:
- Keep the reformulations in English.
- Do not answer the legal question.
- Do not invent facts.
- Do not cite sources.
- Do not include any gold answer or document-specific hidden information.
- Preserve the legal meaning of the original query.
- Return valid JSON only.

Definitions:

1. legal_rewrite
A clearer legal rewriting of the original query.
It should preserve the same information need, but make the legal issue easier to retrieve.

2. keyword_expansion
A comma-separated list or short phrase containing useful retrieval keywords.
It may include legal terms, synonyms, document cues, clause names, and related concepts.
It should not become a full answer.

3. hyde_style
A short hypothetical passage describing what a relevant answer-bearing passage might discuss.
It should sound like a generic relevant legal passage, not like a known gold answer.

Input:
{
  "query_id": "<QUERY_ID>",
  "task": "<TASK>",
  "original_text": "<ORIGINAL_QUERY>"
}

Output JSON schema:
{
  "query_id": "<QUERY_ID>",
  "task": "<TASK>",
  "legal_rewrite": "...",
  "keyword_expansion": "...",
  "hyde_style": "..."
}
"""


def main() -> None:
    if not QUERIES_PATH.exists():
        raise FileNotFoundError(
            f"Missing queries file: {QUERIES_PATH}\n"
            "Run script 03_prepare_legalbench_rag_mini.py first."
        )

    print("=" * 80)
    print("Preparing LegalBench-RAG mini queries for LLM reformulation")
    print("=" * 80)

    print(f"Loading queries from: {QUERIES_PATH}")
    queries = read_jsonl(QUERIES_PATH)

    output_rows = []

    for row in queries:
        query_id = str(row["query_id"])
        task = str(row.get("task", "unknown"))
        original_index = row.get("original_index")
        text = str(row.get("text", "")).strip()

        if not text:
            continue

        output_rows.append(
            {
                "query_id": query_id,
                "task": task,
                "original_index": original_index,
                "language": "en",
                "dataset": "legalbench_rag_mini",
                "prompt_version": "legalbench_v1",
                "original_text": text,
                "question": text,
                "num_chars": len(text),
                "num_words": count_words(text),
            }
        )

    if not output_rows:
        raise ValueError("No valid queries found.")

    write_jsonl(OUT_JSONL, output_rows)
    pd.DataFrame(output_rows).to_csv(OUT_CSV, index=False)

    summary_df = (
        pd.DataFrame(output_rows)
        .groupby("task")
        .agg(
            num_queries=("query_id", "count"),
            avg_chars=("num_chars", "mean"),
            avg_words=("num_words", "mean"),
            min_words=("num_words", "min"),
            max_words=("num_words", "max"),
        )
        .reset_index()
    )

    summary_df.to_csv(OUT_SUMMARY, index=False)

    OUT_PROMPT.write_text(build_prompt_template(), encoding="utf-8")

    print("\nSaved files:")
    print(OUT_JSONL)
    print(OUT_CSV)
    print(OUT_SUMMARY)
    print(OUT_PROMPT)

    print("\nSummary by task:")
    print(summary_df.to_string(index=False))

    print("\nFirst prepared query:")
    print(json.dumps(output_rows[0], indent=2, ensure_ascii=False))

    print("\nNext step:")
    print("  Generate GPT and DeepSeek reformulations using this JSONL input.")
    print("")
    print("Expected future raw outputs:")
    print("  data/processed/legalbench/rag_mini/reformulations/raw/gpt_mini.jsonl")
    print("  data/processed/legalbench/rag_mini/reformulations/raw/deepseek_mini.jsonl")


if __name__ == "__main__":
    main()