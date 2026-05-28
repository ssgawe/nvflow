# Telco Recipe User Guide

This guide explains how to run, inspect, and extend the telco recipe. The
current primary workflow is supervised fine-tuning (SFT) for COBOL-to-text:
given COBOL source code, train a model to produce a concise natural-language
description of what the program does.

The recipe is designed so task-specific data and prompt choices are separate
from model-specific training choices. Use that split for new telco SFT jobs
instead of copying a full workflow file.

## Recipe Layout

| Path | Purpose |
| --- | --- |
| `README.md` | Short recipe summary and primary run commands. |
| `USER_GUIDE.md` | This operational guide. |
| `datasets/` | Small checked-in COBOL train, validation, and test JSONL files. |
| `prompts/` | Prompt templates used by `nemo_skills.training.prepare_data`. |
| `stages/sft/` | NVFlow stage implementations for the SFT pipeline. |
| `utils/sft/` | Local JSONL preparation and sequence bucketing utilities. |
| `workflows/sft/telco_sft_base.yaml` | Generic telco SFT pipeline base. |
| `workflows/sft/tasks/` | Task-specific data schema, prompt, and task settings. |
| `workflows/sft/model_configs/` | Model-specific tokenizer, checkpoint, backend, and NeMo-RL overrides. |
| `workflows/sft/*_cobol.yaml` | Top-level runnable workflow configs. |

Prefer these current composable workflow entrypoints:

```bash
nvflow/recipes/telco/workflows/sft/qwen3_30b_a3b_cobol.yaml
nvflow/recipes/telco/workflows/sft/nemotron_30b_cobol.yaml
```

Older configs remain in `workflows/sft/` for compatibility and historical
experiments. New SFT work should use `telco_sft_base.yaml`, `tasks/`, and
`model_configs/`.

## Prerequisites

Run commands from the repository root.

The workflow configs assume:

- The `nflow` CLI can be run with `uv run nflow ...`.
- The configured cluster name exists in the NeMo-Skills cluster config. The
  checked-in telco configs default to `cluster: my_cluster`; change this to the
  cluster name you actually use.
- Model checkpoints/tokenizers are available at the paths referenced by the
  model config, for example `/hf_models/Qwen/Qwen3-30B-A3B-Instruct-2507`.
- The run environment can submit the NeMo-Skills jobs used by `run_cmd` and
  `sft_nemo_rl`.
- `base_output_dir` points to storage visible to the submitted jobs. The current
  configs use `/workspace/outputs/...`.

## Quick Start

Validate the Qwen3 COBOL config:

```bash
uv run nflow validate --config nvflow/recipes/telco/workflows/sft/qwen3_30b_a3b_cobol.yaml
```

List the stages that config will run:

```bash
uv run nflow list-stages --config nvflow/recipes/telco/workflows/sft/qwen3_30b_a3b_cobol.yaml
```

Run the full Qwen3 COBOL workflow:

```bash
uv run nflow run-all --config nvflow/recipes/telco/workflows/sft/qwen3_30b_a3b_cobol.yaml
```

Run the full Nemotron COBOL workflow:

```bash
uv run nflow run-all --config nvflow/recipes/telco/workflows/sft/nemotron_30b_cobol.yaml
```

Run one or more specific stages:

```bash
uv run nflow run prepare_sft_data --config nvflow/recipes/telco/workflows/sft/qwen3_30b_a3b_cobol.yaml
uv run nflow run prepare_sft_data prepare_for_sft --config nvflow/recipes/telco/workflows/sft/qwen3_30b_a3b_cobol.yaml
```

When running a downstream stage by itself, make sure its upstream artifacts
already exist. Stage dependencies are passed to the job submitter as `run_after`
job names; selecting only a downstream stage does not regenerate missing input
files.

## Pipeline Stages

The generic telco SFT pipeline is defined in
`workflows/sft/telco_sft_base.yaml`:

```yaml
pipeline_stages:
  - prepare_sft_data
  - prepare_for_sft
  - sequence_length_grouping
  - training
```

### 1. `prepare_sft_data`

Normalizes task-specific raw JSONL records into the SFT schema.

Key config fields:

- `train_file`, `val_file`, `test_file`: raw JSONL inputs.
- `output_dir`: normalized output directory.
- `num_chunks`: number of train chunks to write.
- `source_keys`: raw fields to try, in priority order, for the prompt input.
- `target_keys`: raw fields to try, in priority order, for the target answer.
- `metadata_keys`: raw metadata fields to preserve when present.
- `task_name`: value written to `question_type` and `task`.

For COBOL, the task config uses:

```yaml
source_keys:
  - cobol
  - problem
target_keys:
  - description
  - generation
  - nl
task_name: cobol_to_text
```

Outputs:

- `train.jsonl`
- `val.jsonl`
- `test.jsonl`
- `stats.jsonl`
- `chunks/chunk_0.jsonl`, `chunks/chunk_1.jsonl`, and so on

You can also run the utility directly for local data checks:

```bash
uv run python -m nvflow.recipes.telco.utils.sft.prepare_sft_data \
  --train_file nvflow/recipes/telco/datasets/telco_sft_train.jsonl \
  --val_file nvflow/recipes/telco/datasets/telco_sft_val.jsonl \
  --output_dir /tmp/telco-prep-check \
  --source_keys cobol problem \
  --target_keys description generation nl \
  --task_name cobol_to_text
```

### 2. `prepare_for_sft`

Applies the prompt template and tokenizer/chat formatting needed by NeMo-Skills.
It processes every JSONL chunk from `prepare_sft_data`.

Key config fields:

- `input_dir`: normalized train chunks.
- `output_dir`: formatted SFT output directory.
- `val_input_file`: normalized validation JSONL.
- `prepare_data_kwargs.ctx_args`: tokenizer and prompt arguments passed through
  to `nemo_skills.training.prepare_data`.
- `add_token_counts`: optional token count enrichment.

For COBOL, the prompt is:

```yaml
nvflow/recipes/telco/prompts/cobol_to_text_template.yaml
```

Outputs:

- `final_result.jsonl`: formatted training data.
- `val_result.jsonl`: formatted validation data.

### 3. `sequence_length_grouping`

Buckets formatted training examples by total token length. This is a data
quality and planning step: it helps you inspect whether the chosen sequence
length is reasonable before or during training.

Key config fields:

- `input_file`: usually `step-1-prepare-for-sft/final_result.jsonl`.
- `output_dir`: bucket output directory.
- `bucket_sizes`: sequence length buckets, currently `1024`, `2048`, and `4096`.
- `tokenizer_path`: model tokenizer used if records do not already contain
  `total_token_length`.

Outputs are named from the input file stem, for example:

- `final_result_bucket_1024.jsonl`
- `final_result_bucket_2048.jsonl`
- `final_result_bucket_4096.jsonl`
- `final_result_bucket_overflow.jsonl`

### 4. `training`

Submits a NeMo-RL SFT job through NeMo-Skills.

Key config fields:

- `training_data`: formatted train JSONL.
- `validation_data`: formatted validation JSONL.
- `output_dir`: base training output directory.
- `model_name`: model identifier used in run naming and NeMo-RL config.
- `hf_checkpoint_path`: checkpoint path passed to `sft_nemo_rl`.
- `backend`: currently `fsdp` for the primary 30B configs.
- `preset`: NeMo-RL preset name, currently `sft-base`.
- `overrides`: direct NeMo-RL overrides for `sft`, `checkpointing`, `data`,
  `policy`, and related sections.
- `stage_kwargs`: optional submit-time settings such as
  `installation_command`.

The training stage creates a run-specific subdirectory under `output_dir`. The
run name includes the model, node count, parallelism, and sequence length, for
example:

```text
model-qwen3-30b-a3b-instruct-2507-1n-tp1-pp1-cp1-seq4k
```

The stage writes run metadata before submission and NeMo-RL writes logs and
checkpoints under the run output directory.

## Data Schema

Raw records can use task-specific field names. For the built-in COBOL task,
records look like:

```json
{
  "problem_id": "p02547",
  "submission_id": "s973069594",
  "cobol": "IDENTIFICATION DIVISION...",
  "description": "Problem statement or natural-language description..."
}
```

`prepare_sft_data` converts that record to the normalized SFT shape:

```json
{
  "uuid": "train-p02547-s973069594",
  "problem": "IDENTIFICATION DIVISION...",
  "generation": "Problem statement or natural-language description...",
  "context": "",
  "question_type": "cobol_to_text",
  "task": "cobol_to_text",
  "split": "train",
  "problem_id": "p02547",
  "submission_id": "s973069594"
}
```

The normalizer chooses the first non-empty field from `source_keys` for
`problem` and the first non-empty field from `target_keys` for `generation`.
Use that priority order to support multiple source data variants without
rewriting the preparation code.

## Config Composition

The runner supports `_base_` as either a string or a list. With a list, bases
are merged in order, then the child config is applied. Later values override
earlier values.

The current primary COBOL entrypoints follow this pattern:

```yaml
_base_:
  - tasks/cobol_to_text.yaml
  - model_configs/qwen3_30b_a3b_fsdp_lora.yaml

base_output_dir: /workspace/outputs/telco/workflow-sft-cobol/qwen3-30b-a3b
```

The task layer provides data, prompt, source/target keys, and the task name.
The model layer provides tokenizer paths, model checkpoint paths, training
backend, and NeMo-RL overrides. The top-level run config should usually only
choose the task, choose the model, and set `base_output_dir`.

## Adding a New Telco SFT Task

Create a prompt template in `nvflow/recipes/telco/prompts/`. The template should
refer to `{problem}` because the normalization stage writes the selected source
field to `problem`.

Create `nvflow/recipes/telco/workflows/sft/tasks/<task>.yaml`:

```yaml
_base_: ../telco_sft_base.yaml

workflow:
  description: "Ticket-to-resolution supervised fine-tuning workflow"

base_output_dir: /workspace/outputs/telco/workflow-sft-ticket/base

data:
  raw_train_file: /workspace/data/ticket_train.jsonl
  raw_val_file: /workspace/data/ticket_val.jsonl
  raw_test_file: /workspace/data/ticket_test.jsonl
  prompt_config: nvflow/recipes/telco/prompts/ticket_resolution_template.yaml
  task_name: ticket_resolution
  source_keys:
    - ticket_text
    - problem
  target_keys:
    - resolution
    - answer
    - generation
  metadata_keys:
    - ticket_id
    - customer_id

stages:
  prepare_sft_data:
    num_chunks: 8
```

Then create a runnable workflow that composes the task with a model:

```yaml
_base_:
  - tasks/ticket_resolution.yaml
  - model_configs/qwen3_30b_a3b_fsdp_lora.yaml

base_output_dir: /workspace/outputs/telco/workflow-sft-ticket/qwen3-30b-a3b
```

Validate it before submitting:

```bash
uv run nflow validate --config nvflow/recipes/telco/workflows/sft/<your-run>.yaml
```

## Adding a New Model

Create a model layer in `nvflow/recipes/telco/workflows/sft/model_configs/`.
Use `model_configs/`, not `models/`; the repository ignores directories named
`models`.

At minimum, set:

- `stages.prepare_for_sft.prepare_data_kwargs.ctx_args` with tokenizer and
  prompt config arguments.
- `stages.sequence_length_grouping.tokenizer_path`.
- `stages.training.model_name`.
- `stages.training.hf_checkpoint_path`.
- `stages.training.backend`.
- `stages.training.preset`.
- `stages.training.overrides`.

Use a top-level runnable config to compose the new model layer with an existing
task:

```yaml
_base_:
  - tasks/cobol_to_text.yaml
  - model_configs/<new-model>.yaml

base_output_dir: /workspace/outputs/telco/workflow-sft-cobol/<new-model-run>
```

If a model needs an operational patch or setup command, put it in
`stages.training.stage_kwargs.installation_command` in the model layer so the
requirement stays explicit and model-scoped.

## Comparing Qwen and Nemotron Runs

The current Qwen and Nemotron COBOL configs use the same task config. That means
they share:

- raw train, validation, and test data
- prompt template
- normalized schema
- bucket sizes
- SFT schedule
- LoRA and FSDP shape, unless overridden in the model layer

Their outputs are separated by `base_output_dir`:

```text
/workspace/outputs/telco/workflow-sft-cobol/qwen3-30b-a3b
/workspace/outputs/telco/workflow-sft-cobol/nemotron-3-nano-30b
```

Compare validation loss, run metadata, logs, checkpoints, and any post-training
evaluation under those directories.

## Operational Notes

- `run-all` submits every configured stage. The stages also declare
  dependencies so submitted jobs can wait for upstream stage jobs.
- `run <stage>` submits only the selected stage or stages. Use this for
  rerunning a failed step after confirming its inputs exist.
- Keep task decisions in `tasks/` and model decisions in `model_configs/`.
  This keeps new telco jobs reviewable and avoids accidental prompt/model
  coupling.
- Do not put generated model checkpoints under the recipe source tree. Use
  workflow output directories or a shared model/cache location.
- For local preparation debugging, prefer the direct
  `python -m nvflow.recipes.telco.utils.sft.prepare_sft_data` utility command
  before submitting cluster jobs.

## Troubleshooting

`nflow validate` says a stage is missing:

Check that the stage name exists in `pipeline_stages`, has a matching entry in
`stages`, and is registered under `telco.sft`.

`prepare_sft_data` reports missing source or target text:

Update `data.source_keys` or `data.target_keys` in the task config so they match
the raw JSONL schema. The keys are tried in order.

`sequence_length_grouping` reports missing `total_token_length`:

Make sure the model layer sets `stages.sequence_length_grouping.tokenizer_path`
to a tokenizer path available in the run environment.

Training cannot find the model or tokenizer:

Check `model_name`, `hf_checkpoint_path`, tokenizer paths in
`prepare_data_kwargs.ctx_args`, and the mounts/cache layout visible to the
submitted job.

A new model config does not show up in `git status`:

Make sure it is under `workflows/sft/model_configs/`. Paths named `models/` are
ignored by the repository.
