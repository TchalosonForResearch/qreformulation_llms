"""Generate LegalBench-RAG mini reformulations with the OpenAI Responses API."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import openai
from openai import OpenAI
from tqdm import tqdm


INPUT_PATH = Path(
    "data/processed/legalbench/rag_mini/reformulations/input/"
    "legalbench_mini_queries_for_reformulation.jsonl"
)
RAW_OUT_DIR = Path("data/processed/legalbench/rag_mini/reformulations/raw")

DEFAULT_MODEL = "gpt-5.5"
DEFAULT_PROMPT_VERSION = "legalbench_v3_gpt55_api"

REQUIRED_GENERATED_FIELDS = (
    "legal_rewrite",
    "keyword_expansion",
    "hyde_style",
)

ASSERTIVE_HYDE_PATTERNS = (
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
)

RESPONSE_FORMAT = {
    "type": "json_schema",
    "name": "legalbench_query_reformulations",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "query_id": {"type": "string"},
            "task": {"type": "string"},
            "legal_rewrite": {"type": "string"},
            "keyword_expansion": {"type": "string"},
            "hyde_style": {"type": "string"},
        },
        "required": [
            "query_id",
            "task",
            "legal_rewrite",
            "keyword_expansion",
            "hyde_style",
        ],
        "additionalProperties": False,
    },
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at line {line_number} in {path}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object at line {line_number} in {path}.")
            rows.append(row)
    return rows


def write_jsonl_atomic(
    path: Path,
    row_map: dict[str, dict[str, Any]],
    query_order: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")

    ordered_ids = [query_id for query_id in query_order if query_id in row_map]
    ordered_set = set(ordered_ids)
    ordered_ids.extend(sorted(query_id for query_id in row_map if query_id not in ordered_set))

    with temporary_path.open("w", encoding="utf-8") as stream:
        for query_id in ordered_ids:
            stream.write(json.dumps(row_map[query_id], ensure_ascii=False) + "\n")

    temporary_path.replace(path)


def load_existing_rows(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}

    rows_by_id: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        query_id = row.get("query_id")
        if query_id is not None:
            rows_by_id[str(query_id)] = row
    return rows_by_id


def resolve_output_path(output: Path) -> Path:
    if output.is_absolute() or output.parent != Path("."):
        return output
    return RAW_OUT_DIR / output


def source_text(row: dict[str, Any]) -> str:
    value = row.get("original_text") or row.get("question") or row.get("text") or ""
    return str(value).strip()


def build_system_prompt() -> str:
    return (
        "You are helping prepare query reformulations for a legal information "
        "retrieval experiment. Produce retrieval-oriented reformulations in "
        "English. Do not answer the legal question. Do not invent facts. "
        "Do not cite sources. Do not use hidden gold answers. Do not state "
        "that a document contains a clause unless this was already stated in "
        "the query. For hyde_style, write only a cautious hypothetical "
        "description of what a relevant passage may discuss. Return only the "
        "requested JSON object."
    )


def build_user_prompt(row: dict[str, Any]) -> str:
    query_id = str(row["query_id"])
    task = str(row.get("task", "unknown"))
    original_text = source_text(row)

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


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError(f"No JSON object found in response: {text[:500]}")
        parsed = json.loads(text[start : end + 1])

    if not isinstance(parsed, dict):
        raise ValueError("The model response is not a JSON object.")
    return parsed


def detect_assertive_hyde_terms(hyde_style: str) -> list[str]:
    lowered = hyde_style.lower()
    return [pattern for pattern in ASSERTIVE_HYDE_PATTERNS if pattern in lowered]


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def usage_to_dict(usage: Any) -> dict[str, Any] | None:
    if usage is None:
        return None
    if hasattr(usage, "to_dict"):
        return usage.to_dict()
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    return None


def normalize_response(
    *,
    parsed: dict[str, Any],
    source_row: dict[str, Any],
    model: str,
    prompt_version: str,
    response: Any,
    attempt: int,
) -> dict[str, Any]:
    query_id = str(source_row["query_id"])
    task = str(source_row.get("task", "unknown"))
    original_text = source_text(source_row)

    missing_fields: list[str] = []
    generated: dict[str, str] = {}
    for field in REQUIRED_GENERATED_FIELDS:
        value = parsed.get(field)
        if not isinstance(value, str) or not value.strip():
            missing_fields.append(field)
            generated[field] = ""
        else:
            generated[field] = value.strip()

    hyde_warnings = detect_assertive_hyde_terms(generated["hyde_style"])

    return {
        "query_id": query_id,
        "task": task,
        "original_text": original_text,
        "question": original_text,
        "legal_rewrite": generated["legal_rewrite"],
        "keyword_expansion": generated["keyword_expansion"],
        "hyde_style": generated["hyde_style"],
        "generator": "gpt",
        "provider": "openai",
        "generator_model": model,
        "api_endpoint": "responses",
        "prompt_version": prompt_version,
        "validation_status": "valid" if not missing_fields else "invalid",
        "missing_fields": missing_fields,
        "hyde_style_warning_terms": hyde_warnings,
        "attempts": attempt,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "response_id": getattr(response, "id", None),
        "request_id": getattr(response, "_request_id", None),
        "usage": usage_to_dict(getattr(response, "usage", None)),
        "resolved_model": getattr(response, "model", None),
        "openai_sdk_version": getattr(openai, "__version__", None),
    }


def call_openai(
    *,
    client: OpenAI,
    model: str,
    source_row: dict[str, Any],
    prompt_version: str,
    reasoning_effort: str,
    verbosity: str,
    max_output_tokens: int,
    max_retries: int,
    sleep_seconds: float,
) -> dict[str, Any]:
    last_error: Exception | None = None

    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(source_row)

    for attempt in range(1, max_retries + 1):
        try:
            request: dict[str, Any] = {
                "model": model,
                "instructions": system_prompt,
                "input": user_prompt,
                "max_output_tokens": max_output_tokens,
                "text": {
                    "verbosity": verbosity,
                    "format": RESPONSE_FORMAT,
                },
                "metadata": {
                    "dataset": "legalbench_rag_mini",
                    "query_id": str(source_row["query_id"]),
                    "prompt_version": prompt_version,
                },
            }
            if reasoning_effort != "default":
                request["reasoning"] = {"effort": reasoning_effort}

            response = client.responses.create(**request)
            content = getattr(response, "output_text", None)
            if not isinstance(content, str) or not content.strip():
                raise ValueError("The API returned no output_text.")

            parsed = extract_json_object(content)
            normalized = normalize_response(
                parsed=parsed,
                source_row=source_row,
                model=model,
                prompt_version=prompt_version,
                response=response,
                attempt=attempt,
            )
            normalized["reasoning_effort"] = reasoning_effort
            normalized["verbosity"] = verbosity
            normalized["max_output_tokens"] = max_output_tokens
            normalized["system_prompt_sha256"] = sha256_text(system_prompt)
            normalized["user_prompt_sha256"] = sha256_text(user_prompt)
            return normalized
        except Exception as exc:
            last_error = exc
            if attempt < max_retries:
                wait = sleep_seconds * (2 ** (attempt - 1)) + random.uniform(0.0, 0.5)
                print(
                    f"\nAttempt {attempt}/{max_retries} failed for "
                    f"{source_row.get('query_id')}: {exc}. Retrying in {wait:.1f}s."
                )
                time.sleep(wait)

    query_id = str(source_row.get("query_id"))
    task = str(source_row.get("task", "unknown"))
    original_text = source_text(source_row)
    return {
        "query_id": query_id,
        "task": task,
        "original_text": original_text,
        "question": original_text,
        "legal_rewrite": "",
        "keyword_expansion": "",
        "hyde_style": "",
        "generator": "gpt",
        "provider": "openai",
        "generator_model": model,
        "api_endpoint": "responses",
        "prompt_version": prompt_version,
        "validation_status": "api_error",
        "missing_fields": list(REQUIRED_GENERATED_FIELDS),
        "hyde_style_warning_terms": [],
        "error": str(last_error),
        "attempts": max_retries,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "reasoning_effort": reasoning_effort,
        "verbosity": verbosity,
        "max_output_tokens": max_output_tokens,
        "system_prompt_sha256": sha256_text(system_prompt),
        "user_prompt_sha256": sha256_text(user_prompt),
        "openai_sdk_version": getattr(openai, "__version__", None),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate LegalBench-RAG mini reformulations with GPT-5.5."
    )
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--output", type=Path, default=Path("gpt55_mini_pilot20.jsonl"))
    parser.add_argument("--limit", type=int, default=20, help="Use 0 for all queries.")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--base-url", type=str, default="")
    parser.add_argument("--prompt-version", type=str, default=DEFAULT_PROMPT_VERSION)
    parser.add_argument(
        "--reasoning-effort",
        choices=("default", "none", "minimal", "low", "medium", "high"),
        default="minimal",
    )
    parser.add_argument("--verbosity", choices=("low", "medium", "high"), default="low")
    parser.add_argument("--max-output-tokens", type=int, default=1200)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--sleep-seconds", type=float, default=2.0)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--inter-request-delay", type=float, default=0.25)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.offset < 0:
        raise ValueError("--offset cannot be negative.")
    if args.limit < 0:
        raise ValueError("--limit cannot be negative.")
    if args.max_output_tokens < 200:
        raise ValueError("--max-output-tokens should be at least 200.")
    if not args.input.exists():
        raise FileNotFoundError(
            f"Missing input file: {args.input}\n"
            "Run scripts/legalbench/07_prepare_queries_for_reformulation.py first."
        )

    output_path = resolve_output_path(args.output)
    rows = read_jsonl(args.input)
    query_order = [str(row["query_id"]) for row in rows]
    selected_rows = rows[args.offset :]
    if args.limit > 0:
        selected_rows = selected_rows[: args.limit]

    if args.dry_run:
        if not selected_rows:
            raise ValueError("No query selected for the dry run.")
        print(build_system_prompt())
        print("\n" + "=" * 80 + "\n")
        print(build_user_prompt(selected_rows[0]))
        print("\nOutput path:", output_path)
        return

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError(
            "Missing OPENAI_API_KEY environment variable.\n"
            "PowerShell: $env:OPENAI_API_KEY=\"your_api_key_here\"\n"
            "bash/zsh: export OPENAI_API_KEY=\"your_api_key_here\""
        )

    if output_path.exists() and not args.resume and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}\n"
            "Use --resume to continue or --overwrite to replace it."
        )

    existing_rows = {} if args.overwrite else load_existing_rows(output_path)
    valid_query_ids = {
        query_id
        for query_id, row in existing_rows.items()
        if row.get("validation_status") == "valid"
    }

    client_kwargs: dict[str, Any] = {
        "api_key": api_key,
        "timeout": args.request_timeout,
        "max_retries": 0,
    }
    if args.base_url.strip():
        client_kwargs["base_url"] = args.base_url.strip()
    client = OpenAI(**client_kwargs)

    print("=" * 80)
    print("Generating GPT reformulations — LegalBench-RAG mini")
    print("=" * 80)
    print(f"Input: {args.input}")
    print(f"Output: {output_path}")
    print(f"Model: {args.model}")
    print(f"Prompt version: {args.prompt_version}")
    print(f"Reasoning effort: {args.reasoning_effort}")
    print(f"Selected rows: {len(selected_rows)}")
    print(f"Resume: {args.resume}")

    written = 0
    skipped = 0

    for row in tqdm(selected_rows, desc="GPT-5.5 reformulations"):
        query_id = str(row["query_id"])
        if args.resume and query_id in valid_query_ids:
            skipped += 1
            continue

        result = call_openai(
            client=client,
            model=args.model,
            source_row=row,
            prompt_version=args.prompt_version,
            reasoning_effort=args.reasoning_effort,
            verbosity=args.verbosity,
            max_output_tokens=args.max_output_tokens,
            max_retries=args.max_retries,
            sleep_seconds=args.sleep_seconds,
        )
        existing_rows[query_id] = result
        write_jsonl_atomic(output_path, existing_rows, query_order)
        written += 1

        if args.inter_request_delay > 0:
            time.sleep(args.inter_request_delay)

    selected_ids = {str(row["query_id"]) for row in selected_rows}
    selected_outputs = [existing_rows[qid] for qid in selected_ids if qid in existing_rows]
    invalid = sum(row.get("validation_status") != "valid" for row in selected_outputs)
    api_errors = sum(row.get("validation_status") == "api_error" for row in selected_outputs)
    hyde_warnings = sum(bool(row.get("hyde_style_warning_terms")) for row in selected_outputs)

    print("\n" + "=" * 80)
    print("GPT generation summary")
    print("=" * 80)
    print(f"Rows selected: {len(selected_rows)}")
    print(f"Rows written or replaced: {written}")
    print(f"Valid rows skipped during resume: {skipped}")
    print(f"Invalid rows in selected output: {invalid}")
    print(f"API error rows in selected output: {api_errors}")
    print(f"Rows with HyDE warning terms: {hyde_warnings}")
    print(f"Output file: {output_path}")


if __name__ == "__main__":
    main()
