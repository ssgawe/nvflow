# Telco Recipe

This recipe contains telco code-to-text workflows.  The initial supported task is
COBOL-to-text supervised fine-tuning: given COBOL source code, train a model to
produce a concise natural-language description of the program's behavior.

For operational details, extension patterns, and troubleshooting, see
[`USER_GUIDE.md`](USER_GUIDE.md).

## Supported Workflows

Run Qwen3-30B-A3B:

```bash
uv run nflow run-all --config nvflow/recipes/telco/workflows/sft/qwen3_30b_a3b_cobol.yaml
```

Run Nemotron-3-Nano-30B:

```bash
uv run nflow run-all --config nvflow/recipes/telco/workflows/sft/nemotron_30b_cobol.yaml
```

Both configs compose the same COBOL task layer with different model layers, so
they use the same data, prompt, validation file, sequence length, and training
schedule. Compare the resulting training metadata, logs, checkpoints, and
validation metrics under:

- `/workspace/outputs/telco/workflow-sft-cobol/qwen3-30b-a3b`
- `/workspace/outputs/telco/workflow-sft-cobol/nemotron-3-nano-30b`

## Generalizing SFT Jobs

Telco SFT configs are split into reusable layers:

- `workflows/sft/telco_sft_base.yaml`: generic prepare, prompt formatting,
  sequence grouping, and training stages.
- `workflows/sft/tasks/*.yaml`: task data paths, prompt, raw schema keys, and
  task name.
- `workflows/sft/model_configs/*.yaml`: tokenizer, model checkpoint, backend, and
  model-specific NeMo-RL overrides.
- top-level run configs such as `qwen3_30b_a3b_cobol.yaml`: compose one task
  layer and one model layer with a list-valued `_base_`.

To add another telco SFT task, create a task YAML that sets
`data.raw_train_file`, `data.raw_val_file`, `data.prompt_config`,
`data.task_name`, `data.source_keys`, and `data.target_keys`. To add another
model, create a model YAML with the tokenizer/model paths and training
overrides, then compose the two:

```yaml
_base_:
  - tasks/<task>.yaml
  - model_configs/<model>.yaml

base_output_dir: /workspace/outputs/telco/<run-name>
```

## Data Schema

Raw records may use either:

- `cobol` + `description`
- `cobol` + `nl`
- already-normalized `problem` + `generation`

`prepare_sft_data` normalizes task-specific source and target keys to the SFT
schema:

- `problem`: COBOL source
- `generation`: target natural-language description
- `question_type`: `cobol_to_text`

The older `prepare_cobol_data` path remains available for configs that still
reference it directly.
