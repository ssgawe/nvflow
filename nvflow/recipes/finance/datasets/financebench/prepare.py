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
"""Download and prepare FinanceBench benchmark from HuggingFace.

This script downloads the FinanceBench dataset and converts it to nemo-skills format.

FinanceBench is a benchmark for evaluating LLMs on open book financial question answering.
It contains 150 annotated examples about publicly traded companies.

Reference:
    - Paper: https://arxiv.org/abs/2311.11944
    - Dataset: https://huggingface.co/datasets/PatronusAI/financebench
    - GitHub: https://github.com/patronus-ai/financebench
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
    """Download and format FinanceBench from HuggingFace.

    Args:
        split: Dataset split to download (default: "train" - FinanceBench only has train split)
        output_format: Output format ("minimal" or "full")
            - "minimal": Only required fields for evaluation (problem, expected_answer, context, etc.)
            - "full": All metadata including justification, evidence, company info
        output_dir: Directory to save the dataset (default: same directory as prepare.py)

    Note: Output is saved as "eval.jsonl" since FinanceBench is an evaluation benchmark.
    """
    if output_dir is None:
        data_dir = Path(__file__).absolute().parent
    else:
        data_dir = Path(output_dir)

    data_dir.mkdir(exist_ok=True, parents=True)
    # Save as eval.jsonl since FinanceBench is for evaluation
    output_file = data_dir / "eval.jsonl"

    logger.info("Downloading FinanceBench dataset from HuggingFace...")
    logger.info(f"Split: {split}")

    try:
        # Load from HuggingFace
        dataset = load_dataset("PatronusAI/financebench", split=split)
        logger.info(f"Downloaded {len(dataset)} samples")
    except Exception as e:
        logger.error(f"Error downloading dataset: {e}")
        sys.exit(1)

    # Convert to nemo-skills format
    data = []
    question_type_counts = {}
    reasoning_type_counts = {}

    for item in dataset:
        # Track question type distribution
        qtype = item.get("question_type") or "unknown"
        question_type_counts[qtype] = question_type_counts.get(qtype, 0) + 1

        # Track reasoning type distribution
        rtype = item.get("question_reasoning") or "unknown"
        reasoning_type_counts[rtype] = reasoning_type_counts.get(rtype, 0) + 1

        # Extract evidence/context from the evidence list
        # FinanceBench evidence is a list of dicts
        # Use 'evidence_text_full_page' (full page text) instead of 'evidence_text' (human annotated excerpt)
        evidence_list = item.get("evidence", [])
        if evidence_list:
            # Combine all evidence texts into context
            context_parts = []
            for i, ev in enumerate(evidence_list, 1):
                if isinstance(ev, dict) and "evidence_text_full_page" in ev:
                    context_parts.append(f"[Evidence {i}]\n{ev['evidence_text_full_page']}")
                elif isinstance(ev, dict) and "evidence_text" in ev:
                    # Fallback to evidence_text if full_page not available
                    context_parts.append(f"[Evidence {i}]\n{ev['evidence_text']}")
                elif isinstance(ev, str):
                    context_parts.append(f"[Evidence {i}]\n{ev}")
            context = "\n\n".join(context_parts)
        else:
            context = ""

        # Core fields for nemo-skills compatibility
        # Using same structure as SECQUE for consistency
        entry = {
            "uuid": str(uuid.uuid4()),  # Unique identifier for debugging and tracking
            "problem": item["question"],
            "expected_answer": item["answer"],
            "context": context,
        }

        # Add FinanceBench-specific metadata
        if output_format == "full":
            entry.update(
                {
                    "financebench_id": item.get("financebench_id", ""),
                    "company": item.get("company", ""),
                    "doc_name": item.get("doc_name", ""),
                    "question_type": qtype,
                    "question_reasoning": rtype,
                    "justification": item.get("justification", ""),
                    "domain_question_num": item.get("domain_question_num", ""),
                }
            )
        else:
            # Minimal format: only required fields for evaluation
            entry.update(
                {
                    "financebench_id": item.get("financebench_id", ""),
                    "question_type": qtype,
                    "question_reasoning": rtype,
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
    logger.info("Question reasoning distribution:")
    for rtype, count in sorted(reasoning_type_counts.items()):
        logger.info(f"  - {rtype}: {count}")

    # Save statistics
    stats_file = data_dir / "eval_stats.json"
    stats = {
        "total_samples": len(data),
        "question_types": question_type_counts,
        "question_reasoning": reasoning_type_counts,
        "split": split,
    }
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Dataset statistics saved to {stats_file}")

    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and prepare FinanceBench benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python prepare.py
  python prepare.py --split train --format full
  python prepare.py --output_dir /path/to/custom/dir
  python prepare.py --format full --output_dir ./my_datasets

Reference:
  - Paper: https://arxiv.org/abs/2311.11944
  - Dataset: https://huggingface.co/datasets/PatronusAI/financebench
  - GitHub: https://github.com/patronus-ai/financebench
        """,
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train"],
        help="Dataset split (FinanceBench only has 'train' split with 150 samples)",
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
