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
"""Prepare generic telco task data for SFT workflows."""

import shlex
from typing import Any

from nvflow.core import BaseStage, StageRegistry, console


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        values = [value]
    else:
        values = list(value)

    result = []
    for item in values:
        for key in str(item).split(","):
            key = key.strip()
            if key:
                result.append(key)
    return result


def _append_list_args(cmd: list[str], flag: str, values: list[str]) -> None:
    if values:
        cmd.append(flag)
        cmd.extend(values)


@StageRegistry.register(recipe="telco", workflow="sft", stage="prepare_sft_data")
class PrepareSFTDataStage(BaseStage):
    """Normalize raw telco task JSONL files into the SFT schema."""

    workflow = "sft"
    module_name = "nvflow.recipes.telco.utils.sft.prepare_sft_data"
    default_task_name = "sft"

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
        source_key = config.get("source_key")
        target_key = config.get("target_key")
        source_keys = _as_list(config.get("source_keys"))
        target_keys = _as_list(config.get("target_keys"))
        metadata_keys = _as_list(config.get("metadata_keys"))
        task_name = config.get("task_name", self.default_task_name)

        console.status(f"Preparing {task_name} data for SFT")
        console.detail("Train file", train_file)
        if val_file:
            console.detail("Val file", val_file)
        if test_file:
            console.detail("Test file", test_file)
        console.detail("Output directory", output_dir)
        console.detail("Chunks", str(num_chunks))
        if source_keys:
            console.detail("Source keys", ", ".join(source_keys))
        elif source_key:
            console.detail("Source key", source_key)
        if target_keys:
            console.detail("Target keys", ", ".join(target_keys))
        elif target_key:
            console.detail("Target key", target_key)
        console.blank()

        cmd = [
            "python",
            "-m",
            self.module_name,
            "--train_file",
            train_file,
            "--output_dir",
            output_dir,
            "--num_chunks",
            str(num_chunks),
            "--task_name",
            task_name,
        ]
        if val_file:
            cmd.extend(["--val_file", val_file])
        if test_file:
            cmd.extend(["--test_file", test_file])
        if source_key:
            cmd.extend(["--source_key", source_key])
        if target_key:
            cmd.extend(["--target_key", target_key])
        _append_list_args(cmd, "--source_keys", source_keys)
        _append_list_args(cmd, "--target_keys", target_keys)
        _append_list_args(cmd, "--metadata_keys", metadata_keys)

        run_cmd(
            ctx=wrap_arguments(" ".join(shlex.quote(part) for part in cmd)),
            cluster=cluster,
            log_dir=f"{output_dir}/logs",
            expname=expname,
            run_after=run_after,
            **config.get("stage_kwargs", {}),
        )

        console.success(f"SFT data preparation job submitted -> {output_dir}")

    def validate_config(self, config: dict[str, Any]) -> None:
        for field in ("train_file", "output_dir"):
            if not config.get(field):
                raise ValueError(f"'{field}' is required in prepare_sft_data config")

        if not (config.get("source_key") or config.get("source_keys")):
            console.warning(
                "prepare_sft_data has no source_key/source_keys; generic fallbacks will be used."
            )
        if not (config.get("target_key") or config.get("target_keys")):
            console.warning(
                "prepare_sft_data has no target_key/target_keys; generic fallbacks will be used."
            )
