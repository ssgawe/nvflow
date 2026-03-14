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
"""Transform raw SDG dataset to standard training format (shared: SFT + GRPO)."""

from typing import Any

from nvflow.core import BaseStage, StageRegistry, console


@StageRegistry.register(recipe="finance", workflow="sft", stage="data_transformation")
@StageRegistry.register(recipe="finance", workflow="grpo", stage="data_transformation")
class DataTransformationStage(BaseStage):
    """Transform raw SDG dataset to standard training format (shared: SFT + GRPO).

    Normalises field names to a model-agnostic schema:
      problem, context, reasoning_content, generation, uuid, question_type

    Generates uuid: Deterministic hash based on problem + generation content.
    Computes length statistics for key fields.
    """

    def execute(
        self,
        config: dict[str, Any],
        cluster: str,
        expname: str,
        run_after: list[str] | None = None,
    ) -> None:
        """Execute data transformation."""
        from nemo_skills.pipeline.cli import run_cmd, wrap_arguments

        input_files = config["input_files"]
        if isinstance(input_files, str):
            input_files = [input_files]

        output_file = config["output_file"]

        # Source format - how the SDG data is structured
        # Options: "separated", "think_tags", "inline"
        source_format = config.get("source_format", "separated")

        # Reasoning mode - how to format generation for training
        # Options: "thinking", "natural", "none"
        reasoning_mode = config.get("reasoning_mode", "none")

        num_chunks = config.get("num_chunks", 1)
        output_dir = config.get("output_dir", "/tmp")

        console.status("Transforming dataset to training format")
        console.detail("Input files", f"{len(input_files)} file(s)")
        for idx, f in enumerate(input_files, 1):
            console.detail(f"  File {idx}", f)
        console.detail("Output", f"{output_dir}/chunks/ ({num_chunks} chunks)")
        console.detail("Source format", source_format)
        console.detail("Reasoning mode", reasoning_mode)
        console.blank()

        # Build the transformation command with multiple input files
        input_files_str = " ".join(f"'{f}'" for f in input_files)
        cmd = (
            f"python -m nvflow.recipes.finance.utils.shared.dataset_transformer "
            f"    {input_files_str} "
            f"    --output_file '{output_file}'"
        )

        cmd += f" --source_format {source_format}"
        cmd += f" --reasoning_mode {reasoning_mode}"

        # Add chunking options if enabled
        if num_chunks > 1:
            cmd += f" --num_chunks {num_chunks}"

        # Add filtering options if enabled
        filter_outliers = config.get("filter_outliers", False)
        if filter_outliers:
            cmd += " --filter_outliers"
            filter_config = config.get("filter_config", {})

            context_min = filter_config.get("context_min_percentile", 1.0)
            context_max = filter_config.get("context_max_percentile", 99.0)
            reasoning_min = filter_config.get("reasoning_min_percentile", 1.0)
            reasoning_max = filter_config.get("reasoning_max_percentile", 99.0)

            cmd += f" --context_min_percentile {context_min}"
            cmd += f" --context_max_percentile {context_max}"
            cmd += f" --reasoning_min_percentile {reasoning_min}"
            cmd += f" --reasoning_max_percentile {reasoning_max}"

        # Submit transformation job
        run_cmd(
            ctx=wrap_arguments(cmd),
            cluster=cluster,
            log_dir=f"{config.get('output_dir', '/tmp')}/logs",
            expname=expname,
            run_after=run_after,
        )

        # Output always goes to chunks/ directory (consistent structure regardless of num_chunks)
        console.success(
            f"Data transformation job submitted → {output_dir}/chunks/ ({num_chunks} chunks)"
        )

    def validate_config(self, config: dict[str, Any]) -> None:
        """Validate that required configuration fields are present."""
        for field in ("input_files", "output_file"):
            if field not in config:
                raise ValueError(f"'{field}' is required in data_transformation config")

        if config.get("source_format") == "inline" and config.get("reasoning_mode") == "thinking":
            raise ValueError(
                "Invalid combination: source_format='inline' + reasoning_mode='thinking'. "
                "Use reasoning_mode='natural' or 'none' for inline source format."
            )
