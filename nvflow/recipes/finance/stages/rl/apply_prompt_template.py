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
"""Apply prompt template to data_transformation output for GRPO training.

Formats the ``problem`` field using a YAML prompt template (merging
instruction + context + question) and optionally extracts the concise
answer after a configurable prefix (e.g. "Answer:") from ``generation``.

This ensures the model receives the same structured prompt it was
SFT-trained on, and the judge evaluates against a clean expected answer.
"""

from typing import Any

from nvflow.core import BaseStage, StageRegistry, console


@StageRegistry.register(recipe="finance", workflow="grpo", stage="apply_prompt_template")
class ApplyPromptTemplateStage(BaseStage):
    """Apply prompt template and extract expected answer.

    Runs ``prompt_template_applier.py`` inside a Slurm container (CPU-only).
    Reads chunked JSONL from data_transformation, writes processed chunks
    to the output directory.
    """

    def execute(
        self,
        config: dict[str, Any],
        cluster: str,
        expname: str,
        run_after: list[str] | None = None,
    ) -> None:
        """Submit the prompt template application Slurm job."""
        from nemo_skills.pipeline.cli import run_cmd, wrap_arguments

        input_dir = config["input_dir"]
        output_dir = config["output_dir"]
        prompt_template = config["prompt_template"]
        answer_prefix = config.get("answer_prefix")

        console.status("Applying prompt template and extracting expected answer")
        console.detail("Input", input_dir)
        console.detail("Output", output_dir)
        console.detail("Template", prompt_template)
        console.detail("Answer prefix", answer_prefix or "(none -- keep full generation)")
        console.blank()

        cmd = (
            f"python -m nvflow.recipes.finance.utils.rl.prompt_template_applier "
            f"    '{input_dir}' '{output_dir}' "
            f"    --prompt_template '{prompt_template}'"
        )
        if answer_prefix:
            cmd += f" --answer_prefix '{answer_prefix}'"

        run_cmd(
            ctx=wrap_arguments(cmd),
            cluster=cluster,
            num_gpus=0,
            log_dir=f"{output_dir}/logs",
            expname=expname,
            run_after=run_after,
        )

        console.success(f"Prompt template job submitted → {output_dir}/")

    def validate_config(self, config: dict[str, Any]) -> None:
        """Check that all required fields are present."""
        for field in ("input_dir", "output_dir", "prompt_template"):
            if not config.get(field):
                raise ValueError(f"'{field}' is required in apply_prompt_template config")
