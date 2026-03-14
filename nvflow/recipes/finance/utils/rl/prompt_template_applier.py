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
"""Apply a prompt template to SDG data and extract the expected answer.

Reads chunked JSONL from data_transformation, applies a YAML prompt template
to create a ``prompt`` field (merging instruction + context + question),
and extracts the concise answer after a configurable prefix (e.g. "Answer:")
from the ``generation`` field.

Input schema (6-field data_transformation output)::

    {"uuid": "...", "problem": "...", "context": "...",
     "reasoning_content": "...", "generation": "...", "question_type": "..."}

Output schema (8 fields -- adds ``prompt`` and ``expected_answer``)::

    prompt          -> NEW: formatted prompt (instruction + context + question)
    expected_answer -> NEW: extracted answer (after prefix) or full generation if no prefix
    problem         -> unchanged (raw question)
    context         -> "" (absorbed into prompt)
    generation      -> unchanged (original full model output from SDG)

Usage::

    python -m nvflow.recipes.finance.utils.rl.prompt_template_applier \\
        <input_dir> <output_dir> --prompt_template <template.yaml> \\
        [--answer_prefix "Answer:"]
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

from nvflow.utils import setup_logger

logger = setup_logger(__name__)


def load_prompt_template(template_path: str) -> str:
    """Load the user prompt template from a YAML file.

    Expects a YAML file with a ``user`` key containing a format string
    with ``{context}`` and ``{problem}`` placeholders.
    """
    with open(template_path) as f:
        config = yaml.safe_load(f)

    template = config.get("user")
    if not template:
        logger.error("Prompt template YAML must have a 'user' key: %s", template_path)
        sys.exit(1)

    if "{context}" not in template or "{problem}" not in template:
        logger.warning(
            "Template may be missing {context} or {problem} placeholders: %s",
            template_path,
        )

    return template


def apply_template(record: dict, template: str) -> dict:
    """Apply the prompt template to a single record.

    Formats the template with ``context`` and ``problem`` from the record,
    stores the result in a new ``prompt`` field, keeps ``problem`` unchanged
    (raw question), and clears ``context``.
    """
    context = record.get("context", "")
    problem = record.get("problem", "")

    formatted_prompt = template.format(context=context, problem=problem)

    result = dict(record)
    result["prompt"] = formatted_prompt
    result["context"] = ""
    return result


def extract_answer(generation: str, prefix: str) -> str:
    """Extract the answer after the last occurrence of *prefix*.

    Case-insensitive.  Falls back to the full text when the prefix is
    absent or nothing follows it.
    """
    idx = generation.lower().rfind(prefix.lower())
    if idx >= 0:
        answer = generation[idx + len(prefix) :].strip()
        if answer:
            return answer
    return generation


def process_file(
    input_path: Path,
    output_path: Path,
    template: str,
    answer_prefix: str | None,
) -> tuple[int, int, list[dict]]:
    """Process a single JSONL file.

    Returns (processed_count, extracted_count, error_rows).
    """
    processed = 0
    extracted = 0
    errors: list[dict] = []

    with open(input_path) as fin, open(output_path, "w") as fout:
        for line_num, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning("Skipping malformed JSON at %s:%d: %s", input_path, line_num, e)
                errors.append({"file": str(input_path), "line": line_num, "error": str(e)})
                continue

            result = apply_template(record, template)

            generation = result.get("generation", "")
            if answer_prefix:
                answer = extract_answer(generation, answer_prefix)
                result["expected_answer"] = answer
                if answer != generation:
                    extracted += 1
            else:
                result["expected_answer"] = generation

            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            processed += 1

    return processed, extracted, errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply prompt template and extract expected answer"
    )
    parser.add_argument("input_dir", help="Directory with JSONL chunk files")
    parser.add_argument("output_dir", help="Output directory for processed chunks")
    parser.add_argument("--prompt_template", required=True, help="Path to prompt template YAML")
    parser.add_argument(
        "--answer_prefix",
        default=None,
        help='Prefix to extract answer after (e.g. "Answer:"). If not set, generation is kept as-is.',
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("APPLY PROMPT TEMPLATE")
    logger.info("=" * 70)
    logger.info("Input:    %s", input_dir)
    logger.info("Output:   %s", output_dir)
    logger.info("Template: %s", args.prompt_template)
    logger.info("Answer prefix: %s", args.answer_prefix or "(none -- keep full generation)")

    template = load_prompt_template(args.prompt_template)
    logger.info("Template loaded (%d chars)", len(template))

    jsonl_files = sorted(input_dir.glob("*.jsonl"))
    if not jsonl_files:
        logger.error("No .jsonl files found in %s", input_dir)
        return 1

    logger.info("Found %d JSONL file(s)", len(jsonl_files))

    total_processed = 0
    total_extracted = 0
    all_errors: list[dict] = []

    for fpath in jsonl_files:
        out_path = output_dir / fpath.name
        processed, extracted, errors = process_file(fpath, out_path, template, args.answer_prefix)
        total_processed += processed
        total_extracted += extracted
        all_errors.extend(errors)
        logger.info("  %s: %d records processed", fpath.name, processed)

    if all_errors:
        errors_path = output_dir / "errors.jsonl"
        with open(errors_path, "w") as ef:
            for err in all_errors:
                ef.write(json.dumps(err) + "\n")
        logger.warning("Errors: %d -> %s", len(all_errors), errors_path)

    logger.info("")
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info("Total processed: %d", total_processed)
    if args.answer_prefix:
        logger.info(
            "Answer extracted: %d (%.1f%%)",
            total_extracted,
            total_extracted / total_processed * 100 if total_processed else 0,
        )
        logger.info(
            "Kept full text:   %d (%.1f%%)",
            total_processed - total_extracted,
            (total_processed - total_extracted) / total_processed * 100 if total_processed else 0,
        )
    logger.info("Errors:           %d", len(all_errors))
    logger.info("Output:           %s", output_dir)
    logger.info("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
