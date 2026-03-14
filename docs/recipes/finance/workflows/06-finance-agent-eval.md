# Finance Agent Benchmark

## Overview

The **finance_agent** benchmark ([vals-ai/finance-agent](https://github.com/vals-ai/finance-agent)) is now integrated into the main [evaluation workflow](05-eval.md). It is defined as a benchmark entry in `workflows/eval/base.yaml` alongside SEC-QUE and FinanceBench.

> **Note:** The standalone `finance_agent_eval.yaml` workflow has been removed. All finance_agent evaluation now runs through the unified eval configs in `workflows/eval/`.

## What is finance_agent?

- **50 public questions** from vals-ai/finance-agent
- **Multi-turn**: Model can take up to 50 turns (tool calls + reasoning)
- **Tools**: Web search (Tavily), SEC EDGAR lookup, HTML parsing
- **Judge**: GPT-5 mini with strict finance-domain prompts (`sec_judge_strict.yaml`)

## Configuration

The finance_agent benchmark is configured in `workflows/eval/base.yaml` under the `benchmarks` section:

```yaml
benchmarks:
  finance_agent:
    seeds: 5
    judge: *judge_finance_strict
    installation_command: "pip install -q ..."
    extra_args: >-
      ++max_turns=50
      ++inference.tokens_to_generate=32000
      ++inference.temperature=0.0
      ++max_concurrent_requests=1
```

Any model YAML that inherits from `base.yaml` will automatically include finance_agent in its evaluation benchmarks.

## Usage

Run finance_agent evaluation as part of any eval context:

```bash
# Evaluate baselines on all benchmarks (including finance_agent)
uv run nflow run-all --config nvflow/recipes/finance/workflows/eval/baselines.yaml

# SFT training + checkpoint eval (includes finance_agent)
uv run nflow run-all --config nvflow/recipes/finance/workflows/sft/qwen3_14b.yaml
```

## Output Structure

Outputs appear under the model's eval-results directory:

```
outputs/finance/sap-500/workflow-1-baseline-eval/
└── baselines/
    └── gpt-oss-120b/
        └── eval-results/
            └── finance_agent/
                ├── metrics.json      # Aggregated metrics
                └── output*.jsonl     # Predictions per seed
```

## Related

- **[Evaluation Workflow (05-eval)](05-eval.md)** – Full eval documentation, including all benchmarks
- **[Eval Stages Reference](../stages/eval.md)** – Technical stage documentation
