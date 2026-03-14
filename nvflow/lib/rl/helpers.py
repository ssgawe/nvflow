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
"""Shared helpers for RL rollout and reward orchestration.

Recipe-agnostic utilities used by ``nvflow.lib.rl.rollout`` and
``nvflow.lib.rl.verify``.

General utilities:
  - ``resolve_host_path``: maps /workspace/ container paths to host paths.
  - ``build_config_paths_str``: assembles NeMo-Gym config_paths with overlay.

vLLM server configuration:
  - ``build_vllm_server_args``: converts vLLM YAML config to ``--key value``
    CLI args suitable for ``nemo_skills.pipeline.utils.scripts.ServerScript(server_args=...)``.
  - ``compute_num_gpus``: auto-compute Slurm GPU request from per-endpoint num_gpus.

Judge configuration:
  - ``determine_judge_mode``, ``validate_judge_config``: mode detection & validation.
  - ``build_judge_ng_run_overrides``: ng_run CLI overrides for the judge
    (for collect_rollouts / compute_rewards -- shell command strings).
  - ``build_judge_nemo_gym_config``: dict fragment for NeMo-Gym
    ``initial_global_config_dict`` (for GRPO training -- in-memory config).
  - ``log_judge_details``: console output for judge config.

Shell / script templates:
  - ``SHELL_WAIT_FOR_SERVER``: reusable bash function for health-check polling.
"""

from pathlib import Path
from typing import Any

# ============================================================================
# General utilities
# ============================================================================


def resolve_host_path(container_path: str) -> Path:
    """Map a /workspace/ container path to the host filesystem.

    The Lustre mount maps the nvflow project root to /workspace inside the
    container (see cluster_configs/my_cluster.yaml mounts).  On the submission host
    /workspace doesn't exist, so we replace the prefix with "./" which
    resolves to the same Lustre directory.

    IMPORTANT: Assumes ``uv run nflow ...`` is executed from the nvflow
    project root.  ``uv run`` enforces this by default.
    """
    if container_path.startswith("/workspace/") and not Path("/workspace").exists():
        return Path(container_path.replace("/workspace/", "./"))
    return Path(container_path)


def build_config_paths_str(config: dict[str, Any]) -> str:
    """Build the comma-separated NeMo-Gym config_paths string.

    Starts from ``nemo_gym_config_paths`` and appends the agent config
    overlay from ``prepare_data_dir`` if set.
    """
    config_paths = list(config["nemo_gym_config_paths"])
    prepare_data_dir = config.get("prepare_data_dir")
    if prepare_data_dir:
        config_paths.append(f"{prepare_data_dir}/agent_config_overlay.yaml")
    return ",".join(config_paths)


# ============================================================================
# Judge configuration
# ============================================================================

JUDGE_NON_VLLM_KEYS = frozenset(
    {
        "num_gpus",
        "server_nodes",
        "model_path",
        "base_url",
        "openai_base_url",
        "openai_model",
        "openai_api_key",
        "uses_reasoning_parser",
        "tensor_parallel_size",
        "trust_remote_code",
    }
)
"""Keys in ``judge_vllm`` config that are NOT vLLM server CLI flags."""


def _judge_cfg(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("judge_vllm") or {}


def _policy_cfg(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("policy_vllm") or {}


def compute_num_gpus(config: dict[str, Any], *, has_policy: bool = True) -> int:
    """Compute total Slurm GPU request from per-endpoint num_gpus.

    Args:
        config: Stage config with ``policy_vllm`` and/or ``judge_vllm`` blocks.
        has_policy: If True (collect_rollouts), include policy GPUs.
            If False (compute_rewards), only count judge GPUs.
    """
    pcfg = _policy_cfg(config)
    jcfg = _judge_cfg(config)
    policy_gpus = pcfg.get("num_gpus", 0) if has_policy else 0
    judge_gpus = jcfg.get("num_gpus", 0)
    return policy_gpus + judge_gpus


def determine_judge_mode(
    config: dict[str, Any],
    *,
    allow_policy_as_judge: bool = True,
) -> str:
    """Return 'local_vllm', 'external_vllm', 'openai', or 'policy_as_judge'.

    Reads from ``config.judge_vllm`` sub-config.

    - ``base_url`` set → ``external_vllm`` (pre-launched vLLM server)
    - ``model_path`` set (no base_url) → ``local_vllm`` (launch server in job)
    - ``openai_base_url`` set → ``openai`` (OpenAI-compatible API)
    - None of the above → ``policy_as_judge`` (if allowed)
    """
    jcfg = _judge_cfg(config)
    if jcfg.get("base_url"):
        return "external_vllm"
    if jcfg.get("model_path"):
        return "local_vllm"
    if jcfg.get("openai_base_url"):
        return "openai"
    if allow_policy_as_judge:
        return "policy_as_judge"
    raise ValueError(
        "A judge configuration is required. "
        "Set judge_vllm.model_path (local vLLM), "
        "judge_vllm.base_url (external vLLM), or "
        "judge_vllm.openai_base_url (OpenAI API)."
    )


def validate_judge_config(config: dict[str, Any]) -> None:
    """Validate judge-related config fields (call after determine_judge_mode)."""
    jcfg = _judge_cfg(config)
    if jcfg.get("openai_base_url") and not jcfg.get("openai_model"):
        raise ValueError(
            "'judge_vllm.openai_model' is required when 'judge_vllm.openai_base_url' is set"
        )


def log_judge_details(console_obj: Any, config: dict[str, Any], judge_mode: str) -> None:
    jcfg = _judge_cfg(config)
    console_obj.detail("Judge mode", judge_mode)
    if judge_mode == "local_vllm":
        console_obj.detail("Judge model", jcfg["model_path"])
        console_obj.detail("Judge GPUs", str(jcfg.get("num_gpus", 0)))
    elif judge_mode == "external_vllm":
        console_obj.detail("Judge endpoint", jcfg["base_url"])
    elif judge_mode == "openai":
        console_obj.detail("Judge endpoint", jcfg["openai_base_url"])
        console_obj.detail("Judge model", jcfg.get("openai_model", "(default)"))
        if jcfg.get("openai_api_key"):
            console_obj.detail("Judge API key", "from workflow YAML (explicit override)")
        else:
            console_obj.detail("Judge API key", "$OPENAI_API_KEY from container env")


def build_vllm_server_args(overrides: dict[str, Any]) -> str:
    """Convert a dict of vLLM overrides into CLI flags.

    Every key in *overrides* is emitted as a flag -- the caller is responsible
    for stripping non-vLLM keys before calling this function.

    Underscores are converted to hyphens (``max_model_len`` → ``--max-model-len``).
    Boolean ``True`` emits a bare flag; ``False`` suppresses it.
    """
    parts: list[str] = []
    for key, value in overrides.items():
        cli_key = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                parts.append(cli_key)
        else:
            parts.append(f"{cli_key} {value}")
    return " ".join(parts)


def build_judge_ng_run_overrides(
    config: dict[str, Any],
    judge_mode: str,
    *,
    judge_url_var: str = "$JUDGE_VLLM_URL",
) -> str:
    """Build ng_run CLI overrides that configure the judge model server.

    Reads from ``config["judge_vllm"]`` and ``config["environment_name"]``.
    For policy_as_judge: returns empty string (judge uses policy_model).

    Args:
        judge_url_var: URL or shell variable for the judge vLLM endpoint.
    """
    jcfg = _judge_cfg(config)
    env_name = config["environment_name"]
    judge_server_override = (
        f'    "+{env_name}.resources_servers.{env_name}.judge_model_server.name=judge_model" \\\n'
    )

    if judge_mode in ("local_vllm", "external_vllm"):
        judge_model = jcfg.get("model_path", "")
        uses_reasoning_parser = str(jcfg.get("uses_reasoning_parser", False)).lower()
        if judge_mode == "external_vllm":
            judge_url_var = jcfg["base_url"]
        return (
            '    "+judge_model.responses_api_models.vllm_model.entrypoint=app.py" \\\n'
            f'    "+judge_model.responses_api_models.vllm_model.base_url={judge_url_var}" \\\n'
            '    "+judge_model.responses_api_models.vllm_model.api_key=EMPTY" \\\n'
            f'    "+judge_model.responses_api_models.vllm_model.model={judge_model}" \\\n'
            '    "+judge_model.responses_api_models.vllm_model.return_token_id_information=false" \\\n'
            f'    "+judge_model.responses_api_models.vllm_model.uses_reasoning_parser={uses_reasoning_parser}" \\\n'
            + judge_server_override
        )

    if judge_mode == "openai":
        base_url = jcfg["openai_base_url"]
        model = jcfg["openai_model"]
        api_key = jcfg.get("openai_api_key", "$OPENAI_API_KEY")
        lines = [
            '    "+judge_model.responses_api_models.openai_model.entrypoint=app.py" \\\n',
            f'    "+judge_model.responses_api_models.openai_model.openai_base_url={base_url}" \\\n',
            f'    "+judge_model.responses_api_models.openai_model.openai_api_key={api_key}" \\\n',
            f'    "+judge_model.responses_api_models.openai_model.openai_model={model}" \\\n',
            judge_server_override,
        ]
        return "".join(lines)

    return ""


def build_judge_nemo_gym_config(
    config: dict[str, Any],
    judge_mode: str,
    *,
    environment_name: str = "equivalence_llm_judge",
    judge_url_var: str = "",
) -> dict[str, Any]:
    """Build a dict fragment to merge into NeMo-Gym ``initial_global_config_dict``.

    This is the in-memory equivalent of :func:`build_judge_ng_run_overrides`
    (which produces CLI strings for ``ng_run``).  Used by the GRPO training
    stage where the config is passed as a Python dict, not shell arguments.

    Returns an empty dict for ``policy_as_judge`` mode (no-op).

    For ``local_vllm`` / ``external_vllm`` / ``openai`` modes, returns a dict
    with two top-level keys:

    - ``judge_model``: a new ``responses_api_models`` entry (vllm_model or
      openai_model adapter) that NeMo-Gym's ``RunHelper`` will launch.
    - ``<environment_name>``: override of ``judge_model_server.name`` to
      route judge requests to the new ``judge_model`` server.

    Args:
        config: Stage config containing ``judge_vllm`` sub-config.
        judge_mode: One of the modes returned by :func:`determine_judge_mode`.
        environment_name: NeMo-Gym resource server name (top-level config key).
        judge_url_var: For ``local_vllm`` mode, the URL (or shell variable)
            where the judge vLLM engine will be reachable.  Ignored for other
            modes.
    """
    if judge_mode == "policy_as_judge":
        return {}

    jcfg = _judge_cfg(config)

    judge_server_name_override = {
        environment_name: {
            "resources_servers": {
                environment_name: {
                    "judge_model_server": {"name": "judge_model"},
                },
            },
        },
    }

    if judge_mode in ("local_vllm", "external_vllm"):
        judge_model = jcfg.get("model_path", "")
        uses_reasoning_parser = jcfg.get("uses_reasoning_parser", False)
        base_url = jcfg["base_url"] if judge_mode == "external_vllm" else judge_url_var
        return {
            "judge_model": {
                "responses_api_models": {
                    "vllm_model": {
                        "entrypoint": "app.py",
                        "base_url": base_url,
                        "api_key": "EMPTY",
                        "model": judge_model,
                        "return_token_id_information": False,
                        "uses_reasoning_parser": uses_reasoning_parser,
                    },
                },
            },
            **judge_server_name_override,
        }

    if judge_mode == "openai":
        return {
            "judge_model": {
                "responses_api_models": {
                    "openai_model": {
                        "entrypoint": "app.py",
                        "openai_base_url": jcfg["openai_base_url"],
                        "openai_api_key": jcfg.get("openai_api_key", ""),
                        "openai_model": jcfg["openai_model"],
                    },
                },
            },
            **judge_server_name_override,
        }

    return {}


# ============================================================================
# Shell script templates
# ============================================================================

# Reusable bash snippets concatenated into inline command strings built
# by _build_client_cmd() and similar builders.

SHELL_FIND_FREE_PORT = """\
find_free_port() {
    python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()"
}
"""

SHELL_WAIT_FOR_SERVER = """\
wait_for_server() {
    local url="$1" name="$2" pid="$3" max_attempts="$4" log="$5"
    echo "  Waiting for $name at $url ..."
    for i in $(seq 1 $max_attempts); do
        if curl -s -m 5 "$url" > /dev/null 2>&1; then
            echo "  $name ready after $((i * 5))s"
            return 0
        fi
        if ! kill -0 $pid 2>/dev/null; then
            echo "ERROR: $name died. Check $log"
            exit 1
        fi
        sleep 5
    done
    echo "ERROR: $name did not start within $((max_attempts * 5))s"
    exit 1
}
"""
