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
"""Collect rollouts against a NeMo-Gym environment.

Provides :func:`rollout` -- the RL equivalent of nemo-skills'
``generate()``.  Handles Slurm orchestration (single or heterogeneous
jobs for dual-server setups), vLLM lifecycle, chunking, seeding,
merge, and cross-seed aggregation.

URL resolution uses the same lazy ``set_inline(callable)`` pattern as
nemo-skills' ``GenerationClientScript`` so that ``hostname_ref()``
resolves correctly after the Pipeline assigns het-group indices.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nemo_skills.pipeline.utils.scripts import BaseJobScript, ServerScript

from nvflow.core import console

from .helpers import (
    SHELL_FIND_FREE_PORT,
    SHELL_WAIT_FOR_SERVER,
    build_config_paths_str,
    build_judge_ng_run_overrides,
    build_vllm_server_args,
    compute_num_gpus,
    determine_judge_mode,
    log_judge_details,
    resolve_host_path,
)


@dataclass(kw_only=True)
class BashScript(BaseJobScript):
    """A ``BaseJobScript`` that runs a bash command string.

    Module-level class so Fiddle/nemo-run can serialize it by import path.
    """

    cmd: str = ""

    def __post_init__(self):
        self.set_inline(self.cmd)
        super().__post_init__()


@dataclass(kw_only=True)
class RolloutClientScript(BaseJobScript):
    """Client script for NeMo-Gym rollout collection with lazy URL resolution.

    Uses the same lazy ``set_inline(callable)`` pattern as nemo-skills'
    ``GenerationClientScript``.  The callable is evaluated by the Pipeline
    **after** ``het_group_index`` has been assigned to all scripts, so
    ``hostname_ref()`` returns the correct Slurm shell variable for
    cross-node communication in heterogeneous jobs.
    """

    policy_server: ServerScript | None = None
    judge_server: ServerScript | None = None
    policy_base_url: str = ""
    config: dict | None = field(default=None, repr=False)
    judge_mode: str = ""
    judge_ng_run_overrides: str = ""

    output_dir: str = ""
    gym_path: str = ""
    model_path: str = ""
    agent_name: str = ""
    input_data: str = ""
    output_file: str = ""
    done_file: str = ""
    config_paths: str = ""
    num_parallel: int = 4
    job_label: str = ""
    max_num_samples: int = 0

    log_prefix: str = field(default="main", init=False)

    def __post_init__(self):
        def build_cmd() -> str:
            if self.policy_server is not None:
                policy_url = (
                    f"http://{self.policy_server.hostname_ref()}:{self.policy_server.port}/v1"
                )
            else:
                policy_url = self.policy_base_url

            if self.judge_server is not None:
                judge_url = f"http://{self.judge_server.hostname_ref()}:{self.judge_server.port}/v1"
                judge_overrides = build_judge_ng_run_overrides(
                    self.config, self.judge_mode, judge_url_var=judge_url
                )
            else:
                judge_url = ""
                judge_overrides = self.judge_ng_run_overrides

            return _build_client_cmd(
                output_dir=self.output_dir,
                gym_path=self.gym_path,
                model_path=self.model_path,
                agent_name=self.agent_name,
                input_data=self.input_data,
                output_file=self.output_file,
                done_file=self.done_file,
                config_paths=self.config_paths,
                num_parallel=self.num_parallel,
                job_label=self.job_label,
                policy_vllm_url=policy_url,
                judge_vllm_url=judge_url,
                judge_ng_run_overrides=judge_overrides,
                max_num_samples=self.max_num_samples,
            )

        self.set_inline(build_cmd)
        super().__post_init__()


# ---------------------------------------------------------------------------
# Script helpers
# ---------------------------------------------------------------------------


# Keys stripped before building vLLM server_args:
#  - Orchestration keys consumed by our pipeline (num_gpus, model_path, etc.)
#  - NeMo-Gym keys consumed by build_judge_ng_run_overrides (uses_reasoning_parser)
#  - Keys already emitted by nemo-skills' serve_vllm.py (tensor_parallel_size,
#    trust_remote_code) -- passing them again causes duplicate-flag warnings.
_NON_VLLM_KEYS = frozenset(
    {
        "num_gpus",
        "base_url",
        "model_path",
        "server_nodes",
        "openai_base_url",
        "openai_model",
        "openai_api_key",
        "tensor_parallel_size",
        "trust_remote_code",
        "uses_reasoning_parser",
    }
)


def _make_server_script(
    vllm_cfg: dict[str, Any],
    cluster_config: dict,
) -> ServerScript:
    if "num_gpus" not in vllm_cfg:
        raise ValueError("vLLM config must specify 'num_gpus'")
    vllm_overrides = {k: v for k, v in vllm_cfg.items() if k not in _NON_VLLM_KEYS}
    return ServerScript(
        server_type="vllm",
        model_path=vllm_cfg["model_path"],
        cluster_config=cluster_config,
        num_gpus=vllm_cfg["num_gpus"],
        num_nodes=vllm_cfg.get("server_nodes", 1),
        server_args=build_vllm_server_args(vllm_overrides),
    )


def _make_bash_script(
    bash_cmd: str,
    *,
    installation_command: str | None = None,
) -> BashScript:
    return BashScript(cmd=bash_cmd, installation_command=installation_command)


# ---------------------------------------------------------------------------
# Filename conventions
# ---------------------------------------------------------------------------


def _output_filename(seed: int, chunk_id: int) -> str:
    return f"output-rs{seed}_chunk_{chunk_id}.jsonl"


def _merged_filename(seed: int) -> str:
    return f"output-rs{seed}.jsonl"


# ---------------------------------------------------------------------------
# Input splitting
# ---------------------------------------------------------------------------


def _split_input(
    input_path: Path,
    chunks_dir: Path,
    num_chunks: int,
    max_num_samples: int = 0,
) -> tuple[int, int]:
    """Split a JSONL file into *num_chunks* roughly equal chunk files.

    Returns ``(total, used)`` or ``(-1, -1)`` if chunks already exist (resume).
    """
    chunk_files = [chunks_dir / f"input_chunk_{i}.jsonl" for i in range(num_chunks)]

    if all(f.exists() for f in chunk_files):
        return -1, -1

    for f in chunk_files:
        f.unlink(missing_ok=True)

    chunks_dir.mkdir(parents=True, exist_ok=True)
    with open(input_path) as f:
        lines = f.readlines()

    total = len(lines)
    if 0 < max_num_samples < total:
        lines = lines[:max_num_samples]
    used = len(lines)

    per_chunk = max(1, (used + num_chunks - 1) // num_chunks)

    for i, chunk_file in enumerate(chunk_files):
        start = i * per_chunk
        end = min(start + per_chunk, used)
        with open(chunk_file, "w") as out:
            out.writelines(lines[start:end])

    return total, used


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------


def _get_remaining_jobs(
    host_dir: Path,
    seeds: list[int],
    chunk_ids: list[int],
    rerun_done: bool,
) -> list[tuple[int, int]]:
    """Return ``(seed, chunk)`` pairs that still need to run."""
    if rerun_done:
        for s in seeds:
            for c in chunk_ids:
                fname = _output_filename(s, c)
                (host_dir / f"{fname}.done").unlink(missing_ok=True)
                (host_dir / f"{fname}-async").unlink(missing_ok=True)
                (host_dir / fname).unlink(missing_ok=True)
        return [(s, c) for s in seeds for c in chunk_ids]
    return [
        (s, c)
        for s in seeds
        for c in chunk_ids
        if not (host_dir / f"{_output_filename(s, c)}.done").exists()
    ]


# ---------------------------------------------------------------------------
# Inline command builders
# ---------------------------------------------------------------------------


def _build_vllm_wait_snippet(policy_url: str, judge_url: str = "") -> str:
    """Return bash snippet that polls vLLM servers until ready.

    Reuses the ``wait_for_server`` bash function (from SHELL_WAIT_FOR_SERVER)
    which is always emitted earlier in the client command.
    Uses ``$$`` (current shell PID) as a dummy -- vLLM runs in a separate
    het-group so we cannot check its PID, but ``kill -0 $$`` always succeeds,
    effectively skipping the "process died" early-exit while keeping curl polling.
    """
    snippet = f'wait_for_server "{policy_url}/models" "Policy vLLM" $$ 120 /dev/null\n'
    if judge_url:
        snippet += f'wait_for_server "{judge_url}/models" "Judge vLLM" $$ 120 /dev/null\n'
    return snippet


def _build_client_cmd(
    *,
    output_dir: str,
    gym_path: str,
    model_path: str,
    agent_name: str,
    input_data: str,
    output_file: str,
    done_file: str,
    config_paths: str,
    num_parallel: int,
    job_label: str,
    policy_vllm_url: str,
    judge_vllm_url: str = "",
    judge_ng_run_overrides: str,
    max_num_samples: int = 0,
) -> str:
    wait_for_vllm = _build_vllm_wait_snippet(policy_vllm_url, judge_vllm_url)

    return (
        "set -e\n"
        "\n"
        f'OUTPUT_DIR="{output_dir}"\n'
        f'GYM_PATH="{gym_path}"\n'
        f'MODEL_PATH="{model_path}"\n'
        f'AGENT_NAME="{agent_name}"\n'
        f'INPUT_DATA="{input_data}"\n'
        f'OUTPUT_FILE="{output_file}"\n'
        f'DONE_FILE="{done_file}"\n'
        f'CONFIG_PATHS="{config_paths}"\n'
        f'NUM_PARALLEL="{num_parallel}"\n'
        f'JOB_LABEL="{job_label}"\n'
        f'VLLM_URL="{policy_vllm_url}"\n'
        f'JUDGE_URL="{judge_vllm_url}"\n'
        "\n"
        'mkdir -p "$OUTPUT_DIR/logs"\n'
        "\n" + SHELL_FIND_FREE_PORT + "\n"
        "HEAD_SERVER_PORT=$(find_free_port)\n"
        "\n"
        'NG_RUN_PID=""\n'
        "\n"
        "cleanup() {\n"
        '    echo ""\n'
        '    echo "[Cleanup] Shutting down NeMo-Gym servers ..."\n'
        '    [ -n "$NG_RUN_PID" ] && kill $NG_RUN_PID 2>/dev/null && wait $NG_RUN_PID 2>/dev/null || true\n'
        "}\n"
        "trap cleanup EXIT\n"
        "\n" + SHELL_WAIT_FOR_SERVER + "\n"
        'echo "============================================================"\n'
        'echo "Rollout Collection  [$JOB_LABEL]"\n'
        'echo "============================================================"\n'
        'echo "Model:       $MODEL_PATH"\n'
        'echo "Agent:       $AGENT_NAME"\n'
        'echo "Input data:  $INPUT_DATA"\n'
        'echo "Output file: $OUTPUT_FILE"\n'
        'echo "Policy URL:  $VLLM_URL"\n'
        '[ -n "$JUDGE_URL" ] && echo "Judge URL:   $JUDGE_URL"\n'
        'echo "============================================================"\n'
        "\n"
        'echo ""\n'
        'echo "[Step 1/3] Waiting for vLLM servers ..."\n' + wait_for_vllm + "\n"
        'ASYNC_FILE="$OUTPUT_FILE-async"\n'
        'REMAINING_INPUT="$OUTPUT_DIR/remaining_input_$JOB_LABEL.jsonl"\n'
        "\n"
        f'if ! PYTHONPATH=/workspace python3 -m nvflow.lib.rl.resume_filter "$ASYNC_FILE" "$INPUT_DATA" "$REMAINING_INPUT" {max_num_samples}; then\n'
        '    echo "ERROR: resume_filter failed" >&2\n'
        "    exit 1\n"
        "fi\n"
        "\n"
        'if [ -f "$ASYNC_FILE" ] && [ ! -s "$REMAINING_INPUT" ]; then\n'
        '    echo "All rows already completed in -async -- finalizing."\n'
        '    mv "$ASYNC_FILE" "$OUTPUT_FILE"\n'
        '    touch "$DONE_FILE"\n'
        '    echo "Done [$JOB_LABEL]."\n'
        "    exit 0\n"
        "fi\n"
        "\n"
        'cd "$GYM_PATH"\n'
        "source .venv/bin/activate\n"
        "\n"
        'echo ""\n'
        'echo "[Step 2/3] Starting NeMo-Gym servers ..."\n'
        'ng_run "+config_paths=[$CONFIG_PATHS]" \\\n'
        '    "+policy_model.responses_api_models.vllm_model.base_url=$VLLM_URL" \\\n'
        '    "+policy_model.responses_api_models.vllm_model.api_key=EMPTY" \\\n'
        '    "+policy_model.responses_api_models.vllm_model.model=$MODEL_PATH" \\\n'
        '    "+head_server.host=127.0.0.1" \\\n'
        '    "+head_server.port=$HEAD_SERVER_PORT" \\\n'
        f"{judge_ng_run_overrides}"
        '    > "$OUTPUT_DIR/logs/ng_run_$JOB_LABEL.log" 2>&1 &\n'
        "NG_RUN_PID=$!\n"
        "\n"
        'wait_for_server "http://127.0.0.1:$HEAD_SERVER_PORT/" "NeMo-Gym" $NG_RUN_PID 60 "$OUTPUT_DIR/logs/ng_run_$JOB_LABEL.log"\n'
        "\n"
        'echo ""\n'
        'echo "[Step 3/3] Collecting rollouts ..."\n'
        "ng_collect_rollouts \\\n"
        "    +agent_name=$AGENT_NAME \\\n"
        "    +input_jsonl_fpath=$REMAINING_INPUT \\\n"
        "    +output_jsonl_fpath=$ASYNC_FILE \\\n"
        "    +num_repeats=1 \\\n"
        "    +num_samples_in_parallel=$NUM_PARALLEL \\\n"
        "    +head_server.host=127.0.0.1 \\\n"
        "    +head_server.port=$HEAD_SERVER_PORT\n"
        "\n"
        'mv "$ASYNC_FILE" "$OUTPUT_FILE"\n'
        'rm -f "$REMAINING_INPUT"\n'
        'touch "$DONE_FILE"\n'
        'echo "Done [$JOB_LABEL]. Cleanup via trap."\n'
    )


def _build_merge_cmd(
    *,
    gym_path: str,
    merged_file: str,
    analysis_dir: str,
    seed_label: str,
    num_chunks: int,
    chunk_file_pattern: str,
    merged_done_file: str,
    analyze_module: str,
    enrich_module: str,
    input_data: str,
) -> str:
    return (
        "set -e\n"
        "\n"
        f'MERGED_FILE="{merged_file}"\n'
        f'ANALYSIS_DIR="{analysis_dir}"\n'
        f'SEED_LABEL="{seed_label}"\n'
        f"NUM_CHUNKS={num_chunks}\n"
        f'INPUT_DATA="{input_data}"\n'
        "\n"
        'echo "============================================================"\n'
        'echo "Merge Rollout Chunks  [$SEED_LABEL]"\n'
        'echo "============================================================"\n'
        "\n"
        '> "$MERGED_FILE"\n'
        "for i in $(seq 0 $((NUM_CHUNKS - 1))); do\n"
        f'    CHUNK_FILE="{chunk_file_pattern}"\n'
        '    if [ ! -f "$CHUNK_FILE" ]; then\n'
        '        echo "WARNING: Missing chunk file: $CHUNK_FILE"\n'
        "        continue\n"
        "    fi\n"
        '    LINES=$(wc -l < "$CHUNK_FILE")\n'
        '    echo "  Chunk $i: $LINES lines"\n'
        '    cat "$CHUNK_FILE" >> "$MERGED_FILE"\n'
        "done\n"
        'TOTAL=$(wc -l < "$MERGED_FILE")\n'
        'echo "  Merged total: $TOTAL lines"\n'
        "\n"
        "for i in $(seq 0 $((NUM_CHUNKS - 1))); do\n"
        f'    CHUNK_FILE="{chunk_file_pattern}"\n'
        '    rm -f "$CHUNK_FILE"\n'
        "done\n"
        "\n"
        'echo ""\n'
        'echo "[Step 2/3] Enriching rollouts with input metadata ..."\n'
        f"PYTHONPATH=/workspace python3 -m {enrich_module} \\\n"
        '    "$INPUT_DATA" \\\n'
        '    "$MERGED_FILE"\n'
        "\n"
        'echo ""\n'
        'echo "[Step 3/3] Analyzing rollouts ..."\n'
        f"PYTHONPATH=/workspace python3 -m {analyze_module} \\\n"
        '    "$MERGED_FILE" \\\n'
        f'    "$ANALYSIS_DIR"\n'
        "\n"
        f'touch "{merged_done_file}"\n'
        'echo "Done [$SEED_LABEL]."\n'
        'echo ""\n'
        'echo "To browse rollouts interactively (requires Gym venv):"\n'
        f'echo "  cd {gym_path} && source .venv/bin/activate"\n'
        'echo "  ng_viewer +jsonl_fpath=$MERGED_FILE"\n'
    )


def _build_aggregate_cmd(
    *,
    rollout_dir: str,
    aggregate_module: str,
) -> str:
    return (
        "set -e\n"
        'echo "Cross-Seed Aggregation (pass@k)"\n'
        f"PYTHONPATH=/workspace python3 -m {aggregate_module} \\\n"
        f'    "{rollout_dir}" \\\n'
        f'    "{rollout_dir}/aggregate"\n'
        f'echo "Done. Results in {rollout_dir}/aggregate/"\n'
    )


def _build_filter_cmd(
    *,
    output_dir: str,
    difficulty_dir: str,
    filter_module: str,
    train_data: str,
    validation_data: str,
    min_pass_rate: float = 0.0,
    max_pass_rate: float = 1.0,
) -> str:
    cmd = (
        "set -e\n"
        'echo "Filter Training Data (reward-profile difficulty)"\n'
        f"PYTHONPATH=/workspace python3 -m {filter_module} \\\n"
        f'    "{train_data}" \\\n'
        f'    "{difficulty_dir}/aggregate/difficulty.jsonl" \\\n'
        f'    "{output_dir}" \\\n'
        f"    --min-pass-rate {min_pass_rate} \\\n"
        f"    --max-pass-rate {max_pass_rate}"
    )
    if validation_data:
        cmd += f' \\\n    --validation-data "{validation_data}"'
    cmd += "\n"
    cmd += f'echo "Done. Filtered data in {output_dir}/"\n'
    return cmd


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def rollout(
    config: dict[str, Any],
    cluster: str,
    expname: str,
    run_after: list[str] | None = None,
    *,
    analyze_module: str = "",
    enrich_module: str = "",
    aggregate_module: str = "",
    filter_module: str = "",
) -> None:
    """Collect rollouts via NeMo-Gym, orchestrated through the nemo-skills Pipeline."""
    import nemo_skills.pipeline.utils as pipeline_utils
    from nemo_skills.pipeline.utils.declarative import (
        Command,
        CommandGroup,
        HardwareConfig,
        Pipeline,
    )

    output_dir = config["output_dir"]
    gym_path = config["gym_path"]
    client_container = config["container"]
    server_container = "vllm"
    installation_command = config.get("installation_command")

    rcfg = config["rollout"]
    input_data = rcfg["input_data"]
    agent_name = rcfg["agent_name"]
    num_gpus = compute_num_gpus(rcfg, has_policy=True)
    num_parallel = rcfg.get("num_samples_in_parallel", 4)
    num_chunks = rcfg.get("num_chunks", 1)
    num_random_seeds = rcfg.get("num_random_seeds", 1)
    starting_seed = rcfg.get("starting_seed", 0)
    rerun_done = rcfg.get("rerun_done", False)
    max_num_samples = rcfg.get("max_num_samples") or 0

    pcfg = rcfg.get("policy_vllm") or {}
    jcfg = rcfg.get("judge_vllm") or {}
    model_path = pcfg["model_path"]

    config_paths_str = build_config_paths_str(rcfg)
    judge_mode = determine_judge_mode(rcfg)
    judge_ng_run_overrides = build_judge_ng_run_overrides(rcfg, judge_mode)

    need_policy_server = not pcfg.get("base_url")
    need_judge_server = judge_mode == "local_vllm" and bool(jcfg.get("model_path"))

    cluster_config = pipeline_utils.get_cluster_config(cluster)

    host_dir = resolve_host_path(output_dir)
    host_dir.mkdir(parents=True, exist_ok=True)

    rollout_dir = f"{output_dir}/rollout"
    host_rollout_dir = host_dir / "rollout"
    host_rollout_dir.mkdir(parents=True, exist_ok=True)

    seeds = list(range(starting_seed, starting_seed + num_random_seeds))
    chunk_ids = list(range(num_chunks))

    # -- Prepare input (split only when num_chunks > 1) ------------------
    if num_chunks == 1:
        chunk_input_map = {0: input_data}
    else:
        host_input = resolve_host_path(input_data)
        if not host_input.exists():
            raise FileNotFoundError(f"Input file not found: {host_input}")
        total, used = _split_input(
            host_input, host_rollout_dir / "chunks", num_chunks, max_num_samples
        )
        chunk_input_map = {i: f"{rollout_dir}/chunks/input_chunk_{i}.jsonl" for i in chunk_ids}
        if total > 0:
            detail = f"{used} lines -> {num_chunks} chunks"
            if max_num_samples and used < total:
                detail += f" (truncated from {total}, max_num_samples={max_num_samples})"
            console.detail("Input", detail)

    # -- Resume: find remaining (seed, chunk) pairs ---------------------
    remaining = _get_remaining_jobs(host_rollout_dir, seeds, chunk_ids, rerun_done)
    skipped = len(seeds) * len(chunk_ids) - len(remaining)

    console.status("Collecting rollouts (ng_collect_rollouts)")
    console.detail("Model", model_path)
    console.detail("Agent", agent_name)
    if pcfg.get("base_url"):
        console.detail("Policy vLLM", f"external ({pcfg['base_url']})")
    else:
        console.detail("Policy vLLM", f"local (GPUs={pcfg.get('num_gpus', 0)})")
    log_judge_details(console, rcfg, judge_mode)
    if need_policy_server and need_judge_server:
        console.detail(
            "Slurm GPUs/job",
            f"{num_gpus} (policy={pcfg.get('num_gpus', 0)} + judge={jcfg.get('num_gpus', 0)}, het-group)",
        )
    else:
        console.detail("Slurm GPUs/job", str(num_gpus))
    console.detail(
        "Jobs",
        f"{len(remaining)} to submit, {skipped} done | {num_chunks} chunks x {num_random_seeds} seeds",
    )
    console.detail("Output", output_dir)
    console.blank()

    filter_cfg = config.get("filter") or {}

    if not remaining and not filter_cfg:
        console.success("All rollout jobs already complete (use rerun_done to force).")
        return

    # -- Build Pipeline jobs ---------------------------------------------
    jobs: list[dict] = []
    chunk_job_specs: dict[int, list[dict]] = {}

    for seed, chunk_id in remaining:
        job_lbl = f"rs{seed}_chunk{chunk_id}"
        out_filename = _output_filename(seed, chunk_id)

        policy_script = _make_server_script(pcfg, cluster_config) if need_policy_server else None
        judge_script = _make_server_script(jcfg, cluster_config) if need_judge_server else None

        client_cmd = RolloutClientScript(
            policy_server=policy_script,
            judge_server=judge_script,
            policy_base_url=pcfg.get("base_url", ""),
            config=rcfg,
            judge_mode=judge_mode,
            judge_ng_run_overrides=judge_ng_run_overrides,
            output_dir=rollout_dir,
            gym_path=gym_path,
            model_path=model_path,
            agent_name=agent_name,
            input_data=chunk_input_map[chunk_id],
            output_file=f"{rollout_dir}/{out_filename}",
            done_file=f"{rollout_dir}/{out_filename}.done",
            config_paths=config_paths_str,
            num_parallel=num_parallel,
            job_label=job_lbl,
            max_num_samples=max_num_samples if num_chunks == 1 else 0,
            installation_command=installation_command,
        )

        # Dual local servers -> het-group per server for dedicated GPUs.
        if judge_script is not None and policy_script is not None:
            policy_nodes = max(1, policy_script.num_nodes)
            judge_nodes = max(1, judge_script.num_nodes)
            primary_group = CommandGroup(
                commands=[
                    Command(
                        script=policy_script, container=server_container, name=f"{job_lbl}_policy"
                    ),
                    Command(script=client_cmd, container=client_container, name=job_lbl),
                ],
                hardware=HardwareConfig(
                    num_gpus=pcfg.get("num_gpus", 0),
                    num_nodes=policy_nodes,
                ),
                name=job_lbl,
                log_dir=f"{output_dir}/logs",
            )
            judge_group = CommandGroup(
                commands=[
                    Command(
                        script=judge_script, container=server_container, name=f"{job_lbl}_judge"
                    ),
                ],
                hardware=HardwareConfig(
                    num_gpus=jcfg.get("num_gpus", 0),
                    num_nodes=judge_nodes,
                ),
                name=f"{job_lbl}_judge",
                log_dir=f"{output_dir}/logs",
            )
            job_spec = {
                "name": f"{expname}-rs{seed}-chunk{chunk_id}",
                "groups": [primary_group, judge_group],
                "dependencies": run_after if run_after else None,
            }
        else:
            components: list[Command] = []
            max_nodes = 1
            if policy_script is not None:
                components.append(
                    Command(
                        script=policy_script, container=server_container, name=f"{job_lbl}_policy"
                    )
                )
                max_nodes = max(max_nodes, policy_script.num_nodes)
            if judge_script is not None:
                components.append(
                    Command(
                        script=judge_script, container=server_container, name=f"{job_lbl}_judge"
                    )
                )
                max_nodes = max(max_nodes, judge_script.num_nodes)
            components.append(Command(script=client_cmd, container=client_container, name=job_lbl))
            cmd_group = CommandGroup(
                commands=components,
                hardware=HardwareConfig(
                    num_gpus=num_gpus,
                    num_nodes=max_nodes,
                ),
                name=job_lbl,
                log_dir=f"{output_dir}/logs",
            )
            job_spec = {
                "name": f"{expname}-rs{seed}-chunk{chunk_id}",
                "group": cmd_group,
                "dependencies": run_after if run_after else None,
            }
        jobs.append(job_spec)
        chunk_job_specs.setdefault(seed, []).append(job_spec)

    # -- Merge jobs (one per seed, depends on that seed's chunks) --------
    merge_job_specs: list[dict] = []
    for seed in seeds:
        seed_label = f"rs{seed}"
        merged_filename = _merged_filename(seed)

        if (host_rollout_dir / f"{merged_filename}.done").exists() and not rerun_done:
            continue

        chunk_pattern = f"{rollout_dir}/output-rs{seed}_chunk_$i.jsonl"

        merge_cmd_str = _build_merge_cmd(
            gym_path=gym_path,
            merged_file=f"{rollout_dir}/{merged_filename}",
            analysis_dir=f"{rollout_dir}/analysis_{seed_label}",
            seed_label=seed_label,
            num_chunks=num_chunks,
            chunk_file_pattern=chunk_pattern,
            merged_done_file=f"{rollout_dir}/{merged_filename}.done",
            analyze_module=analyze_module,
            enrich_module=enrich_module,
            input_data=input_data,
        )

        merge_cmd = Command(
            script=_make_bash_script(merge_cmd_str),
            container=client_container,
            name=f"merge-{seed_label}",
        )
        merge_group = CommandGroup(
            commands=[merge_cmd],
            hardware=HardwareConfig(num_gpus=0),
            name=f"merge-{seed_label}",
            log_dir=f"{output_dir}/logs",
        )

        seed_deps = chunk_job_specs.get(seed, [])
        merge_job_spec = {
            "name": f"{expname}-merge-{seed_label}",
            "group": merge_group,
            "dependencies": seed_deps if seed_deps else None,
        }
        jobs.append(merge_job_spec)
        merge_job_specs.append(merge_job_spec)

    # -- Cross-seed aggregation job (pass@k) ----------------------------
    # Always run aggregate when filter is requested (needs difficulty.jsonl),
    # or when there are multiple seeds for cross-seed metrics.
    run_aggregate = aggregate_module and (num_random_seeds > 1 or filter_module)
    agg_job_spec: dict | None = None

    if run_aggregate:
        agg_cmd_str = _build_aggregate_cmd(
            rollout_dir=rollout_dir,
            aggregate_module=aggregate_module,
        )

        agg_cmd = Command(
            script=_make_bash_script(agg_cmd_str),
            container=client_container,
            name="aggregate",
        )
        agg_group = CommandGroup(
            commands=[agg_cmd],
            hardware=HardwareConfig(num_gpus=0),
            name="aggregate",
            log_dir=f"{output_dir}/logs",
        )
        agg_job_spec = {
            "name": f"{expname}-aggregate",
            "group": agg_group,
            "dependencies": merge_job_specs or None,
        }
        jobs.append(agg_job_spec)

    # -- Filter job (CPU, depends on aggregate) --------------------------
    if filter_module and filter_cfg:
        filter_cmd_str = _build_filter_cmd(
            output_dir=output_dir,
            difficulty_dir=rollout_dir,
            filter_module=filter_module,
            train_data=filter_cfg["input_data"],
            validation_data=filter_cfg.get("validation_data", ""),
            min_pass_rate=filter_cfg.get("min_pass_rate", 0.0),
            max_pass_rate=filter_cfg.get("max_pass_rate", 1.0),
        )

        filter_cmd = Command(
            script=_make_bash_script(filter_cmd_str),
            container=client_container,
            name="filter",
        )
        filter_group = CommandGroup(
            commands=[filter_cmd],
            hardware=HardwareConfig(num_gpus=0),
            name="filter",
            log_dir=f"{output_dir}/logs",
        )
        filter_deps = [agg_job_spec] if agg_job_spec else (merge_job_specs or None)
        jobs.append(
            {
                "name": f"{expname}-filter",
                "group": filter_group,
                "dependencies": filter_deps,
            }
        )

    # -- Submit via Pipeline ---------------------------------------------
    if not jobs:
        console.success("All rollout jobs already complete (use rerun_done to force).")
        return

    pipeline = Pipeline(
        name=expname,
        cluster_config=cluster_config,
        jobs=jobs,
    )
    pipeline.run()

    console.success(f"{len(jobs)} job(s) submitted -> {output_dir}/")
