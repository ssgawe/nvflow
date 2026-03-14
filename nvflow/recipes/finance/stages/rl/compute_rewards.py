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
"""Compute rewards stage (finance recipe).

Thin wrapper around :func:`nvflow.lib.rl.verify` that registers the
stage with the finance/grpo workflow and supplies recipe-specific
utility module names.
"""

from typing import Any

from nvflow.core import BaseStage, StageRegistry

_UTILS = "nvflow.recipes.finance.utils.rl"


@StageRegistry.register(recipe="finance", workflow="grpo", stage="compute_rewards")
class ComputeRewardsStage(BaseStage):
    """Re-compute rewards on existing rollouts using a different judge.

    Delegates all orchestration to :func:`nvflow.lib.rl.verify`.
    This stage only handles registration, config validation, and
    passing recipe-specific module names.
    """

    def execute(
        self,
        config: dict[str, Any],
        cluster: str,
        expname: str,
        run_after: list[str] | None = None,
    ) -> None:
        from nvflow.lib.rl.verify import verify

        verify(
            config,
            cluster,
            expname,
            run_after,
            analyze_module=f"{_UTILS}.analyze_rollouts",
            aggregate_module=f"{_UTILS}.aggregate_seeds",
            filter_module=f"{_UTILS}.filter_training_data",
        )

    def validate_config(self, config: dict[str, Any]) -> None:
        from nvflow.lib.rl.helpers import determine_judge_mode, validate_judge_config

        for field in ("output_dir", "gym_path", "container"):
            if not config.get(field):
                raise ValueError(f"'{field}' is required in compute_rewards config")

        rcfg = config.get("rejudge") or {}
        for field in ("input_dir", "environment_name", "nemo_gym_config_paths"):
            if not rcfg.get(field):
                raise ValueError(f"'rejudge.{field}' is required in compute_rewards config")

        determine_judge_mode(rcfg, allow_policy_as_judge=False)
        validate_judge_config(rcfg)
