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
"""Convert Q&A data to NeMo-Gym Responses API format for GRPO training."""

from typing import Any

from nvflow.core import BaseStage, StageRegistry, console


@StageRegistry.register(recipe="finance", workflow="grpo", stage="convert_to_responses_api")
class ConvertToResponsesAPIStage(BaseStage):
    """Lossless conversion to NeMo-Gym ``responses_create_params`` format.

    Runs ``responses_api_converter.py`` inside a Slurm container (CPU-only).
    Expects apply_prompt_template output (JSONL with ``prompt`` field).
    """

    def execute(
        self,
        config: dict[str, Any],
        cluster: str,
        expname: str,
        run_after: list[str] | None = None,
    ) -> None:
        """Submit the data conversion Slurm job."""
        from nemo_skills.pipeline.cli import run_cmd, wrap_arguments

        input_path = config["input_path"]
        output_dir = config["output_dir"]
        container = config["container"]
        output_file = f"{output_dir}/final_result.jsonl"

        console.status("Converting data to NeMo-Gym Responses API format")
        console.detail("Input", input_path)
        console.detail("Output", output_file)
        console.blank()

        cmd = (
            f"python -m nvflow.recipes.finance.utils.rl.responses_api_converter "
            f"    '{input_path}' '{output_file}'"
        )

        run_cmd(
            ctx=wrap_arguments(cmd),
            cluster=cluster,
            container=container,
            num_gpus=0,
            log_dir=f"{output_dir}/logs",
            expname=expname,
            run_after=run_after,
        )

        console.success(f"Conversion job submitted → {output_file}")

    def validate_config(self, config: dict[str, Any]) -> None:
        """Check that all required fields are present."""
        for field in ("input_path", "output_dir", "container"):
            if not config.get(field):
                raise ValueError(f"'{field}' is required in convert_to_responses_api config")
