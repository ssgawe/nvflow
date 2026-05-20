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
"""Group telco SFT data by sequence length."""

from typing import Any

from nvflow.core import BaseStage, StageRegistry, console


@StageRegistry.register(recipe="telco", workflow="sft", stage="sequence_length_grouping")
class SequenceLengthGroupingStage(BaseStage):
    """Bucket telco training examples by total token length."""

    workflow = "sft"

    def execute(
        self,
        config: dict[str, Any],
        cluster: str,
        expname: str,
        run_after: list[str] | None = None,
    ) -> None:
        from nemo_skills.pipeline.cli import run_cmd, wrap_arguments

        input_file = config["input_file"]
        output_dir = config["output_dir"]
        tokenizer_path = config.get("tokenizer_path")
        bucket_sizes = config.get("bucket_sizes", [1024, 2048, 4096])

        console.status("Grouping telco data by sequence length")
        console.detail("Input file", input_file)
        console.detail("Output directory", output_dir)
        console.detail("Tokenizer", tokenizer_path or "Not needed")
        console.detail("Bucket sizes", str(bucket_sizes))
        console.blank()

        bucket_sizes_str = " ".join(str(b) for b in bucket_sizes)
        cmd = (
            f"python -m nvflow.recipes.telco.utils.sft.sequence_batcher "
            f"    '{input_file}' "
            f"    --output_dir '{output_dir}' "
            f"    --to_bucket "
            f"    --bucket_sizes {bucket_sizes_str}"
        )
        if tokenizer_path:
            cmd += f" --tokenizer_path '{tokenizer_path}'"

        run_cmd(
            ctx=wrap_arguments(cmd),
            cluster=cluster,
            log_dir=f"{output_dir}/logs",
            expname=expname,
            run_after=run_after,
        )

        console.success(f"Telco sequence grouping job submitted -> {output_dir}")

    def validate_config(self, config: dict[str, Any]) -> None:
        for field in ("input_file", "output_dir"):
            if field not in config:
                raise ValueError(f"'{field}' is required in sequence_length_grouping config")

