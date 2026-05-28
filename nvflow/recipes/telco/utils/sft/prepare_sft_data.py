# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Normalize telco task JSONL data for SFT workflows."""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from nvflow.utils import setup_logger

logger = setup_logger(__name__)

DEFAULT_SOURCE_KEYS = ("problem", "source", "input", "text")
DEFAULT_TARGET_KEYS = ("generation", "target", "output", "answer", "nl")
DEFAULT_METADATA_KEYS = ("problem_id", "submission_id")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in {path}:{line_no}: {e}") from e
    return records


def _split_keys(values: str | Sequence[str] | None) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]

    keys: list[str] = []
    seen = set()
    for raw_value in values:
        for key in str(raw_value).split(","):
            key = key.strip()
            if key and key not in seen:
                keys.append(key)
                seen.add(key)
    return keys


def _resolve_keys(
    explicit_keys: str | Sequence[str] | None,
    single_key: str | None,
    fallback_keys: Sequence[str],
) -> list[str]:
    keys = _split_keys(explicit_keys)
    if keys:
        return keys
    return _split_keys([single_key or "", *fallback_keys])


def _stable_uuid(record: dict[str, Any], split: str, problem: str, generation: str) -> str:
    if record.get("uuid"):
        return str(record["uuid"])

    parts = [split]
    for key in ("problem_id", "submission_id"):
        if record.get(key):
            parts.append(str(record[key]))

    if len(parts) > 1:
        return "-".join(parts)

    digest = hashlib.sha256(f"{problem}\n\n{generation}".encode("utf-8")).hexdigest()[:16]
    return f"{split}-{digest}"


def _first_present(record: dict[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        value = record.get(key)
        if value is not None and value != "":
            return value
    return None


def normalize_record(
    record: dict[str, Any],
    split: str,
    source_keys: Sequence[str],
    target_keys: Sequence[str],
    task_name: str,
    metadata_keys: Sequence[str] = DEFAULT_METADATA_KEYS,
) -> dict[str, Any]:
    """Convert one raw task record to the SFT ``problem``/``generation`` schema."""
    problem = _first_present(record, source_keys)
    generation = _first_present(record, target_keys)

    if problem is None:
        expected = "/".join(source_keys)
        raise ValueError(f"Missing source text: expected one of {expected}")
    if generation is None:
        expected = "/".join(target_keys)
        raise ValueError(f"Missing target text: expected one of {expected}")

    normalized = {
        "uuid": _stable_uuid(record, split, str(problem), str(generation)),
        "problem": str(problem),
        "generation": str(generation),
        "context": record.get("context", ""),
        "question_type": task_name,
        "task": task_name,
        "split": split,
    }

    for key in metadata_keys:
        if key in record:
            normalized[key] = record[key]

    return normalized


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_chunks(output_dir: Path, records: list[dict[str, Any]], num_chunks: int) -> None:
    chunks_dir = output_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    # Remove old chunk files without touching unrelated outputs.
    for old_chunk in chunks_dir.glob("chunk_*.jsonl"):
        old_chunk.unlink()

    num_chunks = max(1, min(num_chunks, max(1, len(records))))
    chunk_size = (len(records) + num_chunks - 1) // num_chunks

    for chunk_idx in range(num_chunks):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, len(records))
        if start >= end:
            break
        _write_jsonl(chunks_dir / f"chunk_{chunk_idx}.jsonl", records[start:end])


def prepare_sft_data(
    train_file: Path,
    output_dir: Path,
    val_file: Path | None = None,
    test_file: Path | None = None,
    num_chunks: int = 1,
    source_key: str | None = None,
    target_key: str | None = None,
    source_keys: Sequence[str] | None = None,
    target_keys: Sequence[str] | None = None,
    task_name: str = "sft",
    metadata_keys: Sequence[str] = DEFAULT_METADATA_KEYS,
) -> dict[str, int]:
    """Normalize train/val/test files and write SFT-ready artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_source_keys = _resolve_keys(source_keys, source_key, DEFAULT_SOURCE_KEYS)
    resolved_target_keys = _resolve_keys(target_keys, target_key, DEFAULT_TARGET_KEYS)
    resolved_metadata_keys = _split_keys(metadata_keys)

    train = [
        normalize_record(
            record,
            "train",
            resolved_source_keys,
            resolved_target_keys,
            task_name,
            resolved_metadata_keys,
        )
        for record in _read_jsonl(train_file)
    ]
    _write_chunks(output_dir, train, num_chunks)
    _write_jsonl(output_dir / "train.jsonl", train)

    counts = {"train": len(train), "val": 0, "test": 0}

    if val_file:
        val = [
            normalize_record(
                record,
                "val",
                resolved_source_keys,
                resolved_target_keys,
                task_name,
                resolved_metadata_keys,
            )
            for record in _read_jsonl(val_file)
        ]
        _write_jsonl(output_dir / "val.jsonl", val)
        counts["val"] = len(val)

    if test_file:
        test = [
            normalize_record(
                record,
                "test",
                resolved_source_keys,
                resolved_target_keys,
                task_name,
                resolved_metadata_keys,
            )
            for record in _read_jsonl(test_file)
        ]
        _write_jsonl(output_dir / "test.jsonl", test)
        counts["test"] = len(test)

    _write_jsonl(output_dir / "stats.jsonl", [counts])
    return counts


def main(
    argv: Sequence[str] | None = None,
    *,
    description: str = "Prepare telco task data for SFT",
    default_source_key: str | None = None,
    default_target_key: str | None = None,
    default_task_name: str = "sft",
    default_metadata_keys: Sequence[str] = DEFAULT_METADATA_KEYS,
) -> int:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--train_file", required=True, type=Path)
    parser.add_argument("--val_file", type=Path)
    parser.add_argument("--test_file", type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--source_key", default=default_source_key)
    parser.add_argument("--target_key", default=default_target_key)
    parser.add_argument("--source_keys", nargs="*")
    parser.add_argument("--target_keys", nargs="*")
    parser.add_argument("--task_name", default=default_task_name)
    parser.add_argument("--metadata_keys", nargs="*", default=list(default_metadata_keys))

    args = parser.parse_args(argv)
    counts = prepare_sft_data(
        train_file=args.train_file,
        val_file=args.val_file,
        test_file=args.test_file,
        output_dir=args.output_dir,
        num_chunks=args.num_chunks,
        source_key=args.source_key,
        target_key=args.target_key,
        source_keys=args.source_keys,
        target_keys=args.target_keys,
        task_name=args.task_name,
        metadata_keys=args.metadata_keys,
    )

    logger.info("Prepared telco SFT data")
    logger.info(f"Train: {counts['train']:,}")
    logger.info(f"Val:   {counts['val']:,}")
    logger.info(f"Test:  {counts['test']:,}")
    logger.info(f"Output: {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
