"""Validate the raw LegalBench-RAG files before preprocessing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


RAW_DIR = Path("data/raw/legalbench")

CORPUS_DIR = RAW_DIR / "corpus"
BENCHMARKS_DIR = RAW_DIR / "benchmarks"

TASKS = [
    "contractnli",
    "cuad",
    "maud",
    "privacy_qa",
]


def load_json(path: Path) -> Any:
    """
    Charge un fichier JSON.
    """
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def count_files_recursive(path: Path) -> int:
    """
    Compte tous les fichiers dans un dossier, récursivement.
    """
    return sum(1 for item in path.rglob("*") if item.is_file())


def get_json_items(data: Any) -> list:
    """
    Essaie de retrouver la liste des exemples dans un JSON.

    Les datasets peuvent avoir plusieurs formats :
      - une liste directement ;
      - un dictionnaire avec une clé contenant une liste ;
      - un dictionnaire plus complexe.

    On reste volontairement souple ici.
    """
    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        # Cas fréquent : {"tests": [...]} ou {"examples": [...]}.
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

        # Sinon, on cherche la première valeur qui est une liste.
        for value in data.values():
            if isinstance(value, list):
                return value

    return []


def preview_json_structure(data: Any, max_depth: int = 2, indent: int = 0) -> None:
    """
    Affiche un aperçu simple de la structure JSON.
    """
    prefix = " " * indent

    if max_depth < 0:
        print(prefix + "...")
        return

    if isinstance(data, dict):
        print(prefix + f"dict with {len(data)} keys")
        for key, value in list(data.items())[:10]:
            value_type = type(value).__name__
            print(prefix + f"- {key!r}: {value_type}")

            if isinstance(value, (dict, list)):
                preview_json_structure(value, max_depth=max_depth - 1, indent=indent + 4)

    elif isinstance(data, list):
        print(prefix + f"list with {len(data)} items")
        if data:
            print(prefix + "first item:")
            preview_json_structure(data[0], max_depth=max_depth - 1, indent=indent + 4)

    else:
        print(prefix + f"{type(data).__name__}: {str(data)[:120]}")


def print_sample_item(item: Any) -> None:
    """
    Affiche un exemple de test case de manière lisible.
    """
    print("\nSample item preview:")

    if isinstance(item, dict):
        print("Keys:")
        print(list(item.keys()))

        print("\nCompact sample:")
        compact = {}

        for key, value in item.items():
            if isinstance(value, str):
                compact[key] = value[:300]
            elif isinstance(value, list):
                compact[key] = f"list[{len(value)}]"
            elif isinstance(value, dict):
                compact[key] = f"dict[{len(value)} keys]"
            else:
                compact[key] = value

        print(json.dumps(compact, indent=2, ensure_ascii=False))

    else:
        print(repr(item)[:1000])


def check_required_paths() -> None:
    """
    Vérifie l'existence des dossiers et fichiers principaux.
    """
    print("=" * 80)
    print("Checking LegalBench-RAG directory structure")
    print("=" * 80)

    print(f"Expected raw directory: {RAW_DIR}")

    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"Missing directory: {RAW_DIR}\n\n"
            "Place LegalBench-RAG data under:\n"
            "  data/raw/legalbench/\n\n"
            "Expected structure:\n"
            "  data/raw/legalbench/corpus/\n"
            "  data/raw/legalbench/benchmarks/"
        )

    if not CORPUS_DIR.exists():
        raise FileNotFoundError(f"Missing corpus directory: {CORPUS_DIR}")

    if not BENCHMARKS_DIR.exists():
        raise FileNotFoundError(f"Missing benchmarks directory: {BENCHMARKS_DIR}")

    print("OK: raw directory exists")
    print("OK: corpus directory exists")
    print("OK: benchmarks directory exists")


def inspect_corpus() -> None:
    """
    Inspecte les sous-dossiers corpus.
    """
    print("\n" + "=" * 80)
    print("Inspecting corpus")
    print("=" * 80)

    for task in TASKS:
        task_dir = CORPUS_DIR / task

        if not task_dir.exists():
            print(f"WARNING: missing corpus subdirectory: {task_dir}")
            continue

        num_files = count_files_recursive(task_dir)

        print(f"\nTask: {task}")
        print(f"Corpus directory: {task_dir}")
        print(f"Number of corpus files: {num_files}")

        sample_files = [p for p in task_dir.rglob("*") if p.is_file()][:5]

        if sample_files:
            print("Sample files:")
            for path in sample_files:
                print(f"  - {path.relative_to(RAW_DIR)}")
        else:
            print("No files found in this corpus directory.")


def inspect_benchmarks() -> None:
    """
    Inspecte les fichiers benchmarks JSON.
    """
    print("\n" + "=" * 80)
    print("Inspecting benchmarks")
    print("=" * 80)

    total_items = 0

    for task in TASKS:
        benchmark_path = BENCHMARKS_DIR / f"{task}.json"

        print("\n" + "-" * 80)
        print(f"Task: {task}")
        print(f"Benchmark file: {benchmark_path}")

        if not benchmark_path.exists():
            print(f"WARNING: missing benchmark file: {benchmark_path}")
            continue

        data = load_json(benchmark_path)
        items = get_json_items(data)

        print(f"Top-level JSON type: {type(data).__name__}")
        print(f"Detected test items: {len(items)}")

        total_items += len(items)

        print("\nJSON structure preview:")
        preview_json_structure(data, max_depth=2)

        if items:
            print_sample_item(items[0])
        else:
            print("WARNING: could not detect test items in this benchmark.")

    print("\n" + "=" * 80)
    print(f"Total detected benchmark items: {total_items}")


def main() -> None:
    check_required_paths()
    inspect_corpus()
    inspect_benchmarks()

    print("\n" + "=" * 80)
    print("LegalBench-RAG data check completed.")
    print("=" * 80)
    print(
        "\nIf everything looks correct, the next script will be:\n"
        "  scripts/legalbench/02_inspect_legalbench_snippets.py\n\n"
        "Then run 03_prepare_legalbench_rag_mini.py to create:\n"
        "  corpus.jsonl\n"
        "  queries.jsonl\n"
        "  qrels.tsv\n"
    )


if __name__ == "__main__":
    main()