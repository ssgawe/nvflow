#!/usr/bin/env python3
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
"""Download and prepare SECQUE benchmark from HuggingFace.

This script downloads the SECQUE dataset and converts it to nemo-skills format.
"""

import argparse
import json
import sys
import uuid
from pathlib import Path

from nvflow.utils import setup_logger

# Initialize logger
logger = setup_logger(__name__)

try:
    from datasets import load_dataset
except ImportError:
    logger.error("'datasets' library not found. Install with: pip install datasets")
    sys.exit(1)


def save_data(split="train", output_format="minimal", output_dir=None):
    """Download and format SECQUE from HuggingFace.

    Args:
        split: Dataset split to download (default: "train" - SECQUE only has train split on HuggingFace)
        output_format: Output format ("minimal" or "full")
            - "minimal": Only required fields for evaluation (problem, expected_answer, context, etc.)
            - "full": All metadata including page numbers, accession numbers, SEC filing details
        output_dir: Directory to save the dataset (default: same directory as prepare.py)

    Note: Output is saved as "eval.jsonl" since SECQUE is an evaluation-only benchmark.
    """
    if output_dir is None:
        data_dir = Path(__file__).absolute().parent
    else:
        data_dir = Path(output_dir)

    data_dir.mkdir(exist_ok=True, parents=True)
    # Save as eval.jsonl since SECQUE is for evaluation only (not training)
    output_file = data_dir / "eval.jsonl"

    logger.info("Downloading SECQUE dataset from HuggingFace...")
    logger.info(f"Split: {split}")

    try:
        # Load from HuggingFace
        dataset = load_dataset("nogabenyoash/SecQue", split=split)
        logger.info(f"Downloaded {len(dataset)} samples")
    except Exception as e:
        logger.error(f"Error downloading dataset: {e}")
        sys.exit(1)

    # Convert to nemo-skills format
    data = []
    question_type_counts = {}

    for item in dataset:
        # Track question type distribution
        qtype = item.get("question_type", "unknown")
        question_type_counts[qtype] = question_type_counts.get(qtype, 0) + 1

        # Core fields for nemo-skills compatibility
        entry = {
            "uuid": str(uuid.uuid4()),  # Unique identifier for debugging and tracking
            "problem": item["Question"],
            "expected_answer": item["ground_truth_answer"],
        }

        # Add SECQUE-specific metadata
        if output_format == "full":
            entry.update(
                {
                    "qid": item["QID"],
                    "question_type": qtype,
                    "page_number": item.get("page_number", ""),
                    "accession_number": item.get("accession_number", ""),
                    "item": item.get("item", ""),
                    # Include context with markdown formatting for better readability
                    "context_markdown": item.get("context_markdown_with_headers", ""),
                    # Alternative: plain context without headers
                    # "context": item.get("context_markdown_without_headers", ""),
                }
            )
        else:
            # Minimal format: only required fields for evaluation
            entry.update(
                {
                    "qid": item["QID"],
                    "question_type": qtype,
                    "context": item.get("context_markdown_with_headers", ""),
                }
            )

        data.append(entry)

    # Save as JSONL
    with open(output_file, "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    logger.info(f"Successfully saved {len(data)} samples to {output_file}")
    logger.info("Question type distribution:")
    for qtype, count in sorted(question_type_counts.items()):
        logger.info(f"  - {qtype}: {count}")

    # Save statistics
    stats_file = data_dir / "eval_stats.json"
    stats = {
        "total_samples": len(data),
        "question_types": question_type_counts,
        "split": split,
    }
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Dataset statistics saved to {stats_file}")

    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and prepare SECQUE benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python prepare.py
  python prepare.py --split train --format full
  python prepare.py --output_dir /path/to/custom/dir
  python prepare.py --format full --output_dir ./my_datasets
        """,
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train"],
        help="Dataset split (SECQUE only has 'train' split with 565 samples)",
    )
    parser.add_argument(
        "--format",
        default="minimal",
        choices=["minimal", "full"],
        help="Output format: 'minimal' (required fields only) or 'full' (all metadata)",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory to save eval.jsonl and eval_stats.json (default: same directory as prepare.py)",
    )

    args = parser.parse_args()

    try:
        save_data(split=args.split, output_format=args.format, output_dir=args.output_dir)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)
