"""Generate LegalBench reformulations with the DeepSeek API."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm


INPUT_PATH = Path(
    "data/processed/legalbench/rag_mini/reformulations/input/"
    "legalbench_mini_queries_for_reformulation.jsonl"
)

RAW_OUT_DIR = Path("data/processed/legalbench/rag_mini/reformulations/raw")
RAW_OUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_BASE_URL = "https://api.deepseek.com"

REQUIRED_GENERATED_FIELDS = [
    "legal_rewrite",
    "keyword_expansion",
    "hyde_style",
]

ASSERTIVE_HYDE_PATTERNS = [
    "the agreement expressly",
    "the agreement explicitly",
    "the agreement includes",
    "the agreement contains",
    "the agreement provides",
    "the document expressly",
    "the document explicitly",
    "the document includes",
    "the document contains",
    "the document provides",
    "the clause states",
    "the clause provides",
    "shall not",
    "shall be",
]


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at line {line_number} in {path}: {exc}"
                ) from exc
    return rows


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_done_query_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            query_id = row.get("query_id")
            if query_id:
                done.add(str(query_id))
    return done


def build_system_prompt() -> str:
    return (
        "You are helping prepare query reformulations for a legal information "
        "retrieval experiment. Produce retrieval-oriented reformulations in "
        "English. Do not answer the legal question. Do not invent facts. "
        "Do not cite sources. Do not use hidden gold answers. Do not state "
        "that a document contains a clause unless this was already stated in "
        "the query. For hyde_style, write only a cautious hypothetical "
        "description of what a relevant passage may discuss. Return valid "
        "JSON only."
    )


def build_user_prompt(row: dict) -> str:
    query_id = row["query_id"]
    task = row.get("task", "unknown")
    original_text = row.get("original_text") or row.get("question") or row.get("text")

    return f"""
Given one LegalBench-RAG retrieval query, produce three alternative retrieval-oriented query views.

Definitions:

1. legal_rewrite
A clearer legal rewriting of the original query.
It should preserve the same information need, but make the legal issue easier to retrieve.
It must not answer the question.

2. keyword_expansion
A comma-separated list or short phrase containing useful retrieval keywords.
It may include legal terms, synonyms, document cues, party names, clause names, and related concepts.
It should not become a full answer.

3. hyde_style
A short hypothetical passage describing what a relevant answer-bearing passage may discuss.
It must be cautious and non-committal.
Use wording such as:
- "A relevant passage may discuss whether..."
- "A relevant clause might address..."
- "A responsive section may concern..."
Do not say that the document actually contains the answer.
Do not use phrases such as:
- "the agreement explicitly states"
- "the agreement provides"
- "the document includes"
- "the clause states"
unless those words are already in the original query.
It should sound like a generic retrieval target, not like a known gold answer.

Important constraints:
- Keep everything in English.
- Do not answer the legal question.
- Do not invent facts.
- Do not cite sources.
- Preserve the legal meaning of the original query.
- In hyde_style, use cautious language; do not assert the answer.
- Return valid JSON only.

Input:
{{
  "query_id": {json.dumps(query_id, ensure_ascii=False)},
  "task": {json.dumps(task, ensure_ascii=False)},
  "original_text": {json.dumps(original_text, ensure_ascii=False)}
}}

Return exactly this JSON schema:
{{
  "query_id": {json.dumps(query_id, ensure_ascii=False)},
  "task": {json.dumps(task, ensure_ascii=False)},
  "legal_rewrite": "...",
  "keyword_expansion": "...",
  "hyde_style": "..."
}}
""".strip()


def extract_json_object(text: str) -> dict:
    text = text.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON object found in response: {text[:500]}")
    candidate = text[start : end + 1]
    parsed = json.loads(candidate)
    if not isinstance(parsed, dict):
        raise ValueError("Parsed JSON is not an object.")
    return parsed


def detect_assertive_hyde_terms(hyde_style: str) -> list[str]:
    text = (hyde_style or "").lower()
    return [pattern for pattern in ASSERTIVE_HYDE_PATTERNS if pattern in text]


def validate_reformulation(
    *,
    parsed: dict,
    source_row: dict,
    model: str,
    prompt_version: str,
) -> dict:
    query_id = str(source_row["query_id"])
    task = str(source_row.get("task", "unknown"))
    original_text = (
        source_row.get("original_text")
        or source_row.get("question")
        or source_row.get("text")
        or ""
    )

    parsed["query_id"] = query_id
    parsed["task"] = task

    missing = []
    for field in REQUIRED_GENERATED_FIELDS:
        value = parsed.get(field)
        if not isinstance(value, str) or not value.strip():
            missing.append(field)
            parsed[field] = ""
        else:
            parsed[field] = value.strip()

    hyde_style = parsed.get("hyde_style", "")
    hyde_style_warning_terms = detect_assertive_hyde_terms(hyde_style)
    validation_status = "valid" if not missing else "invalid"

    return {
        "query_id": query_id,
        "task": task,
        "original_text": original_text,
        "question": original_text,
        "legal_rewrite": parsed.get("legal_rewrite", ""),
        "keyword_expansion": parsed.get("keyword_expansion", ""),
        "hyde_style": hyde_style,
        "generator": "deepseek",
        "generator_model": model,
        "prompt_version": prompt_version,
        "validation_status": validation_status,
        "missing_fields": missing,
        "hyde_style_warning_terms": hyde_style_warning_terms,
    }


def call_deepseek(
    *,
    client: OpenAI,
    model: str,
    source_row: dict,
    prompt_version: str,
    temperature: float,
    max_tokens: int,
    max_retries: int,
    sleep_seconds: float,
) -> dict:
    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(source_row)
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            if content is None:
                raise ValueError("Empty response content.")
            parsed = extract_json_object(content)
            normalized = validate_reformulation(
                parsed=parsed,
                source_row=source_row,
                model=model,
                prompt_version=prompt_version,
            )
            normalized["attempts"] = attempt
            return normalized
        except Exception as exc:
            last_error = exc
            if attempt < max_retries:
                wait = sleep_seconds * attempt
                print(
                    f"\nRetry {attempt}/{max_retries} failed for "
                    f"{source_row.get('query_id')}: {exc}. Waiting {wait:.1f}s..."
                )
                time.sleep(wait)

    query_id = str(source_row.get("query_id"))
    task = str(source_row.get("task", "unknown"))
    original_text = (
        source_row.get("original_text")
        or source_row.get("question")
        or source_row.get("text")
        or ""
    )
    return {
        "query_id": query_id,
        "task": task,
        "original_text": original_text,
        "question": original_text,
        "legal_rewrite": "",
        "keyword_expansion": "",
        "hyde_style": "",
        "generator": "deepseek",
        "generator_model": model,
        "prompt_version": prompt_version,
        "validation_status": "api_error",
        "missing_fields": REQUIRED_GENERATED_FIELDS,
        "hyde_style_warning_terms": [],
        "error": str(last_error),
        "attempts": max_retries,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate LegalBench-RAG mini reformulations with DeepSeek API."
    )
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--output", type=str, default="deepseek_mini_pilot20_v2.jsonl")
    parser.add_argument("--limit", type=int, default=20, help="Use 0 for all queries.")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL)
    parser.add_argument("--prompt-version", type=str, default="legalbench_v2_deepseek")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=700)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--sleep-seconds", type=float, default=1.0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError(
            "Missing DEEPSEEK_API_KEY environment variable.\n"
            "PowerShell example:\n"
            '$env:DEEPSEEK_API_KEY="your_api_key_here"'
        )

    if not args.input.exists():
        raise FileNotFoundError(
            f"Missing input file: {args.input}\n"
            "Run script 07_prepare_queries_for_reformulation.py first."
        )

    output_path = RAW_OUT_DIR / args.output

    print("=" * 80)
    print("Generating DeepSeek reformulations — LegalBench-RAG mini")
    print("=" * 80)
    print(f"Input: {args.input}")
    print(f"Output: {output_path}")
    print(f"Model: {args.model}")
    print(f"Base URL: {args.base_url}")
    print(f"Prompt version: {args.prompt_version}")
    print(f"Limit: {args.limit}")
    print(f"Offset: {args.offset}")
    print(f"Resume: {args.resume}")

    rows = read_jsonl(args.input)
    if args.offset < 0:
        raise ValueError("--offset cannot be negative.")
    rows = rows[args.offset :]
    if args.limit > 0:
        rows = rows[: args.limit]

    done_query_ids = load_done_query_ids(output_path) if args.resume else set()
    if done_query_ids:
        print(f"Already completed query_ids in output: {len(done_query_ids)}")

    client = OpenAI(api_key=api_key, base_url=args.base_url)

    num_written = 0
    num_skipped = 0
    num_invalid = 0
    num_errors = 0
    num_hyde_warnings = 0

    for row in tqdm(rows, desc="DeepSeek reformulations"):
        query_id = str(row["query_id"])
        if query_id in done_query_ids:
            num_skipped += 1
            continue

        result = call_deepseek(
            client=client,
            model=args.model,
            source_row=row,
            prompt_version=args.prompt_version,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            max_retries=args.max_retries,
            sleep_seconds=args.sleep_seconds,
        )
        append_jsonl(output_path, result)
        num_written += 1

        if result["validation_status"] != "valid":
            num_invalid += 1
        if result["validation_status"] == "api_error":
            num_errors += 1
        if result.get("hyde_style_warning_terms"):
            num_hyde_warnings += 1

        time.sleep(0.2)

    print("\n" + "=" * 80)
    print("DeepSeek generation summary")
    print("=" * 80)
    print(f"Rows selected: {len(rows)}")
    print(f"Rows written: {num_written}")
    print(f"Rows skipped because already done: {num_skipped}")
    print(f"Invalid rows: {num_invalid}")
    print(f"API error rows: {num_errors}")
    print(f"Rows with hyde_style warning terms: {num_hyde_warnings}")
    print(f"Output file: {output_path}")

    print("\nPilot command:")
    print(
        "python scripts/legalbench/08_generate_deepseek_reformulations_api.py "
        "--limit 20 --output deepseek_mini_pilot20_v2.jsonl"
    )

    print("\nFull command after validation:")
    print(
        "python scripts/legalbench/08_generate_deepseek_reformulations_api.py "
        "--limit 0 --output deepseek_mini.jsonl "
        "--prompt-version legalbench_v2_deepseek --resume"
    )


if __name__ == "__main__":
    main()
