# Telco Recipe

This recipe contains telco code-to-text workflows.  The initial supported task is
COBOL-to-text supervised fine-tuning: given COBOL source code, train a model to
produce a concise natural-language description of the program's behavior.

## Supported Workflows

Run Qwen3-30B-A3B:

```bash
uv run nflow run-all --config nvflow/recipes/telco/workflows/sft/qwen3_30b_a3b_cobol.yaml
```

Run Nemotron-3-Nano-30B:

```bash
uv run nflow run-all --config nvflow/recipes/telco/workflows/sft/nemotron_30b_cobol.yaml
```

Both configs inherit `workflows/sft/cobol_base.yaml`, so they use the same data,
prompt, validation file, sequence length, and training schedule.  Compare the
resulting training metadata, logs, checkpoints, and validation metrics under:

- `/workspace/outputs/telco/workflow-sft-cobol/qwen3-30b-a3b`
- `/workspace/outputs/telco/workflow-sft-cobol/nemotron-3-nano-30b`

## Data Schema

Raw records may use either:

- `cobol` + `description`
- `cobol` + `nl`
- already-normalized `problem` + `generation`

`prepare_cobol_data` normalizes those variants to the SFT schema:

- `problem`: COBOL source
- `generation`: target natural-language description
- `question_type`: `cobol_to_text`
