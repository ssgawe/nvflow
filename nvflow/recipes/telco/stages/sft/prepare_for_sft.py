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
"""Prepare telco data for supervised fine-tuning."""

import re
from typing import Any

from nvflow.core import BaseStage, StageRegistry, console


@StageRegistry.register(recipe="telco", workflow="sft", stage="prepare_for_sft")
class PrepareForSFTStage(BaseStage):
    """Apply telco prompt templates and tokenizer formatting for SFT."""

    workflow = "sft"

    def execute(
        self,
        config: dict[str, Any],
        cluster: str,
        expname: str,
        run_after: list[str] | None = None,
    ) -> None:
        from nemo_skills.pipeline.cli import run_cmd, wrap_arguments

        input_dir = config["input_dir"]
        output_dir = config["output_dir"]
        prepare_data_kwargs = config.get("prepare_data_kwargs", {})
        prepare_data_ctx_args = prepare_data_kwargs.get("ctx_args", "")
        add_token_counts = config.get("add_token_counts", False)
        val_input_file = config.get("val_input_file")
        val_output_file = f"{output_dir}/val_result.jsonl" if val_input_file else None

        output_file = f"{output_dir}/final_result.jsonl"
        temp_dir = f"{output_dir}/temp_chunks"
        tokenizer_path = self._extract_tokenizer_path(prepare_data_ctx_args)

        console.status("Preparing telco data for SFT training")
        console.detail("Input directory", input_dir)
        console.detail("Output file", output_file)
        if val_input_file:
            console.detail("Val input file", val_input_file)
            console.detail("Val output file", val_output_file)
        console.detail("Add token counts", "Yes" if add_token_counts else "No")
        console.blank()

        cmd = (
            'CONFIG_DIR=$(python -c "'
            "from importlib.resources import files; "
            "print(files('nvflow.recipes.telco.stages.sft').joinpath('config'))"
            '") && '
            f"mkdir -p {temp_dir} && "
            f"for file in {input_dir}/*.jsonl; do "
            f'  if [ -f "$file" ]; then '
            f'    filename=$(basename "$file"); '
            f'    echo "Processing $filename..."; '
            f"    python -m nemo_skills.training.prepare_data "
            f'      --config-path "$CONFIG_DIR" '
            f"      --config-name data_prep.yaml "
            f'      ++input_files="$file" '
            f'      ++output_path="{temp_dir}/$filename" '
            f"      {prepare_data_ctx_args} || exit 1; "
            f"  fi; "
            f"done && "
            f"mkdir -p {output_dir} && "
            f"cat {temp_dir}/*.jsonl > {output_file} && "
            f"rm -rf {temp_dir}"
        )

        if add_token_counts and tokenizer_path:
            cmd += (
                f" && python -m nvflow.recipes.telco.utils.sft.add_token_counts "
                f"    '{output_file}' "
                f"    --output_file '{output_file}' "
                f"    --tokenizer_path '{tokenizer_path}'"
            )

        if val_input_file and val_output_file:
            cmd += (
                f" && python -m nemo_skills.training.prepare_data "
                f'  --config-path "$CONFIG_DIR" '
                f"  --config-name data_prep.yaml "
                f'  ++input_files="{val_input_file}" '
                f'  ++output_path="{val_output_file}" '
                f"  {prepare_data_ctx_args}"
            )

        run_cmd(
            ctx=wrap_arguments(cmd),
            cluster=cluster,
            log_dir=f"{output_dir}/logs",
            expname=expname,
            run_after=run_after,
        )

        console.success(f"Telco SFT data preparation job submitted -> {output_file}")

    def _extract_tokenizer_path(self, ctx_args: str) -> str | None:
        match = re.search(r"\+\+tokenizer=(['\"]?)([^'\"\s]+)\1", ctx_args)
        return match.group(2) if match else None

    def validate_config(self, config: dict[str, Any]) -> None:
        for field in ("input_dir", "output_dir"):
            if field not in config:
                raise ValueError(f"'{field}' is required in prepare_for_sft config")

        if config.get("add_token_counts", False):
            ctx_args = config.get("prepare_data_kwargs", {}).get("ctx_args", "")
            if "++tokenizer=" not in ctx_args:
                console.warning(
                    "add_token_counts is enabled but no tokenizer is set in ctx_args."
                )

