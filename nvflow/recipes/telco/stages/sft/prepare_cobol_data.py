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
"""Prepare COBOL-to-text data for telco SFT workflows."""

from typing import Any

from nvflow.core import BaseStage, StageRegistry, console


@StageRegistry.register(recipe="telco", workflow="sft", stage="prepare_cobol_data")
class PrepareCobolDataStage(BaseStage):
    """Normalize raw COBOL JSONL files into the SFT ``problem``/``generation`` schema."""

    workflow = "sft"

    def execute(
        self,
        config: dict[str, Any],
        cluster: str,
        expname: str,
        run_after: list[str] | None = None,
    ) -> None:
        from nemo_skills.pipeline.cli import run_cmd, wrap_arguments

        train_file = config["train_file"]
        val_file = config.get("val_file")
        test_file = config.get("test_file")
        output_dir = config["output_dir"]
        num_chunks = config.get("num_chunks", 1)
        source_key = config.get("source_key", "cobol")
        target_key = config.get("target_key", "description")
        task_name = config.get("task_name", "cobol_to_text")

        console.status("Preparing COBOL-to-text data")
        console.detail("Train file", train_file)
        if val_file:
            console.detail("Val file", val_file)
        if test_file:
            console.detail("Test file", test_file)
        console.detail("Output directory", output_dir)
        console.detail("Chunks", str(num_chunks))
        console.blank()

        cmd = (
            "python -m nvflow.recipes.telco.utils.sft.prepare_cobol_data "
            f"--train_file '{train_file}' "
            f"--output_dir '{output_dir}' "
            f"--num_chunks {num_chunks} "
            f"--source_key '{source_key}' "
            f"--target_key '{target_key}' "
            f"--task_name '{task_name}'"
        )
        if val_file:
            cmd += f" --val_file '{val_file}'"
        if test_file:
            cmd += f" --test_file '{test_file}'"

        run_cmd(
            ctx=wrap_arguments(cmd),
            cluster=cluster,
            log_dir=f"{output_dir}/logs",
            expname=expname,
            run_after=run_after,
            **config.get("stage_kwargs", {}),
        )

        console.success(f"COBOL data preparation job submitted -> {output_dir}")

    def validate_config(self, config: dict[str, Any]) -> None:
        for field in ("train_file", "output_dir"):
            if not config.get(field):
                raise ValueError(f"'{field}' is required in prepare_cobol_data config")

