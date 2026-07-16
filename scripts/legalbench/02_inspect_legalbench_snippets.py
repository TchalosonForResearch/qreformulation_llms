"""Inspect LegalBench-RAG reference snippets and source-file resolution."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


RAW_DIR = Path("data/raw/legalbench")
BENCHMARKS_DIR = RAW_DIR / "benchmarks"
CORPUS_DIR = RAW_DIR / "corpus"

TASKS = [
    "contractnli",
    "cuad",
    "maud",
    "privacy_qa",
]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_json_items(data: Any) -> list:
    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        candidate_keys = [
            "tests",
            "test_cases",
            "examples",
            "data",
            "queries",
            "items",
        ]

        for key in candidate_keys:
            if key in data and isinstance(data[key], list):
                return data[key]

        for value in data.values():
            if isinstance(value, list):
                return value

    return []


def compact_value(value: Any, max_chars: int = 500) -> Any:
    if isinstance(value, str):
        text = value.replace("\n", "\\n")
        if len(text) > max_chars:
            return text[:max_chars] + "..."
        return text

    if isinstance(value, list):
        return [compact_value(v, max_chars=max_chars) for v in value[:5]]

    if isinstance(value, dict):
        return {
            k: compact_value(v, max_chars=max_chars)
            for k, v in value.items()
        }

    return value


def collect_snippet_key_stats(items: list) -> Counter:
    key_counter = Counter()

    for item in items:
        snippets = item.get("snippets", []) if isinstance(item, dict) else []

        for snippet in snippets:
            if isinstance(snippet, dict):
                key_counter.update(snippet.keys())
            else:
                key_counter.update([type(snippet).__name__])

    return key_counter


def find_possible_file_values(snippet: dict) -> list[str]:
    """
    Essaie de repérer les champs qui ressemblent à un nom de fichier.
    """
    candidates = []

    for key, value in snippet.items():
        if not isinstance(value, str):
            continue

        lower_key = key.lower()
        lower_value = value.lower()

        if (
            "file" in lower_key
            or "path" in lower_key
            or lower_value.endswith(".txt")
            or ".txt" in lower_value
            or ".pdf" in lower_value
        ):
            candidates.append(f"{key}={value}")

    return candidates


def check_file_candidate_exists(task: str, value: str) -> bool:
    """
    Vérifie approximativement si une valeur string correspond à un fichier corpus.
    """
    task_dir = CORPUS_DIR / task

    if not task_dir.exists():
        return False

    direct_path = task_dir / value

    if direct_path.exists():
        return True

    # Recherche par nom exact dans le sous-dossier.
    for path in task_dir.rglob("*"):
        if path.is_file() and path.name == value:
            return True

    return False


def inspect_task(task: str, max_examples: int = 3) -> None:
    benchmark_path = BENCHMARKS_DIR / f"{task}.json"

    print("\n" + "=" * 100)
    print(f"TASK: {task}")
    print("=" * 100)

    if not benchmark_path.exists():
        print(f"Missing benchmark file: {benchmark_path}")
        return

    data = load_json(benchmark_path)
    items = get_json_items(data)

    print(f"Benchmark file: {benchmark_path}")
    print(f"Detected items: {len(items)}")

    key_counter = collect_snippet_key_stats(items)

    print("\nSnippet key frequencies:")
    for key, count in key_counter.most_common():
        print(f"  {key}: {count}")

    snippet_count_distribution = Counter()

    for item in items:
        snippets = item.get("snippets", []) if isinstance(item, dict) else []
        snippet_count_distribution[len(snippets)] += 1

    print("\nNumber of snippets per query distribution:")
    for n_snippets, count in sorted(snippet_count_distribution.items()):
        print(f"  {n_snippets} snippets: {count} queries")

    print("\nFull sample examples:")

    shown = 0

    for item_index, item in enumerate(items):
        if shown >= max_examples:
            break

        if not isinstance(item, dict):
            continue

        snippets = item.get("snippets", [])

        if not snippets:
            continue

        print("\n" + "-" * 100)
        print(f"Example index: {item_index}")
        print("Query:")
        print(item.get("query", "")[:1000])

        print(f"\nNumber of snippets: {len(snippets)}")

        for snippet_index, snippet in enumerate(snippets[:3]):
            print("\n" + f"Snippet {snippet_index}:")

            if isinstance(snippet, dict):
                print("Keys:", list(snippet.keys()))

                print(json.dumps(
                    compact_value(snippet, max_chars=800),
                    indent=2,
                    ensure_ascii=False,
                ))

                possible_files = find_possible_file_values(snippet)

                if possible_files:
                    print("\nPossible file/path fields:")
                    for candidate in possible_files:
                        print(f"  {candidate}")
                else:
                    print("\nNo obvious file/path field detected.")
            else:
                print(type(snippet).__name__)
                print(repr(snippet)[:1000])

        shown += 1


def main() -> None:
    print("Inspecting LegalBench-RAG snippet schema")

    for task in TASKS:
        inspect_task(task, max_examples=3)

    print("\n" + "=" * 100)
    print("Snippet inspection completed.")
    print("=" * 100)
    print(
        "\nSend me the output for one or two examples per task, especially:\n"
        "  - Snippet key frequencies\n"
        "  - Keys inside Snippet 0\n"
        "  - Any fields that look like file/path/start/end/text\n"
    )


if __name__ == "__main__":
    main()