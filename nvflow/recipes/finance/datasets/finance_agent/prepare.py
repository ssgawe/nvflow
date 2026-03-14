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
"""Download and prepare the vals-ai/finance-agent benchmark.

Downloads public.csv from GitHub and converts to nemo-skills eval.jsonl format.
Each row becomes:
  {
    "problem": "<Question>",
    "expected_answer": "<Answer>",
    "question_type": "<Type>",
    "expert_time_mins": <int>,
    "rubric": "<JSON string>"
  }

The rubric field is preserved for potential future rubric-based judging.
"""

import argparse
import csv
import io
import json
import sys
import urllib.request
from pathlib import Path

from nvflow.utils import setup_logger

LOG = setup_logger(__name__)

GITHUB_RAW_BASE = "https://raw.githubusercontent.com/vals-ai/finance-agent/main/data"


def download_text(url: str) -> str:
    """Download a text file from a URL."""
    LOG.info("Downloading %s", url)
    req = urllib.request.Request(url, headers={"User-Agent": "nvflow/prepare"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read().decode("utf-8")


def save_data(output_dir: str | None = None, source: str = "github"):
    """Download and convert finance-agent benchmark data to eval.jsonl.

    Args:
        output_dir: Where to write eval.jsonl. Defaults to this script's directory.
        source: "github" to download from vals-ai/finance-agent, or a local
                path to a directory containing public.csv.
    """
    if output_dir is None:
        data_dir = Path(__file__).absolute().parent
    else:
        data_dir = Path(output_dir)
    data_dir.mkdir(exist_ok=True, parents=True)

    output_file = data_dir / "eval.jsonl"

    # --- Load CSV data ---------------------------------------------------
    if source == "github":
        csv_text = download_text(f"{GITHUB_RAW_BASE}/public.csv")
        # Also save raw CSV for reference
        (data_dir / "public.csv").write_text(csv_text, encoding="utf-8")
    else:
        csv_path = Path(source) / "public.csv"
        if not csv_path.exists():
            LOG.error("public.csv not found at %s", csv_path)
            sys.exit(1)
        csv_text = csv_path.read_text(encoding="utf-8")

    reader = csv.DictReader(io.StringIO(csv_text))

    # --- Convert to eval.jsonl -------------------------------------------
    data = []
    type_counts: dict[str, int] = {}

    for row in reader:
        question = row["Question"].strip()
        answer = row["Answer"].strip()
        qtype = row.get("Question Type", "unknown").strip()
        expert_time = row.get("Expert time (mins)", "0").strip()
        rubric = row.get("Rubric", "").strip()

        type_counts[qtype] = type_counts.get(qtype, 0) + 1

        entry = {
            "problem": question,
            "expected_answer": answer,
            "question_type": qtype,
            "expert_time_mins": int(expert_time) if expert_time.isdigit() else 0,
            "rubric": rubric,
        }
        data.append(entry)

    with open(output_file, "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    LOG.info("Saved %d samples to %s", len(data), output_file)
    LOG.info("Question type distribution:")
    for qtype, count in sorted(type_counts.items()):
        LOG.info("  %s: %d", qtype, count)

    # Save stats
    stats = {
        "total_samples": len(data),
        "question_types": type_counts,
        "source": source,
    }
    stats_file = data_dir / "eval_stats.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and prepare vals-ai/finance-agent benchmark"
    )
    parser.add_argument(
        "--source",
        default="github",
        help=(
            "'github' to download from vals-ai/finance-agent repo, "
            "or a local directory path containing public.csv"
        ),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory (default: same as this script)",
    )
    args = parser.parse_args()

    save_data(output_dir=args.output_dir, source=args.source)
