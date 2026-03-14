# Workflow 6: GRPO Reinforcement Learning

## Purpose

Further improve fine-tuned models using Group Relative Policy Optimization (GRPO) with LLM-as-judge reward signals from NeMo-Gym.

> **Note:** GRPO training builds on an SFT checkpoint (or base model). For best results, run [SFT](04-sft.md) first.

## Quick Navigation

- [Prerequisites](#prerequisites)
- [Workflow Overview](#workflow-overview)
- [Usage](#usage)
- [Data & Results](#data--results)
- [Customization](#customization)
- [Additional Resources](#additional-resources)

---

## Prerequisites

- ✅ Base model or SFT checkpoint accessible on cluster (Qwen3-4B for demo)
- ✅ NeMo-RL container with NeMo-Gym (`nemo-rl` container)
- ✅ GPU resources (8 GPUs / 1 node for demo)

## Workflow Overview

### Stages

```
┌──────────────────────────────┐
│ 0. data_transformation       │  SDG cleanup → model-agnostic schema (CPU)
└───────────┬──────────────────┘
            │
            ▼
┌──────────────────────────────┐
│ 1. apply_prompt_template     │  Apply prompt template + extract answer (CPU)
└───────────┬──────────────────┘
            │
            ▼
┌──────────────────────────────┐
│ 2. convert_to_responses_api  │  Convert to NeMo-Gym Responses API format (CPU)
└───────────┬──────────────────┘
            │
            ▼
┌──────────────────────────────┐
│ 3. train_validation_split    │  Split into train/val sets (CPU)
└───────────┬──────────────────┘
            │
            ▼
┌──────────────────────────────┐
│ 4. prepare_data              │  Add agent_ref routing fields (CPU)
└───────────┬──────────────────┘
            │
            ▼
┌──────────────────────────────┐
│ 5. collect_rollouts          │  Rollout collection + reward profiling + filter (GPU)
└───────────┬──────────────────┘
            │
            ▼
┌──────────────────────────────┐
│ 6. compute_rewards           │  [Optional] Re-judge with different model (GPU/CPU)
└───────────┬──────────────────┘
            │
            ▼
┌──────────────────────────────┐
│ 7. training                  │  GRPO training with NeMo-Gym environment (GPU)
└───────────┬──────────────────┘
            │
            ▼
┌──────────────────────────────┐
│ 8. eval                      │  Evaluate checkpoints on finance benchmarks (GPU)
└──────────────────────────────┘
```

**9 Stages (8 active + 1 optional):**
1. **data_transformation** (Step 0): Normalize raw SDG data to model-agnostic schema (shared with SFT pipeline)
2. **apply_prompt_template** (Step 1): Format the problem field using a prompt template and extract the concise expected answer
3. **convert_to_responses_api** (Step 2): Convert prompted data to NeMo-Gym Responses API format (lossless)
4. **train_validation_split** (Step 3): Split data into train/val sets with stratified sampling (shared with SFT pipeline)
5. **prepare_data** (Step 4): Run `ng_prepare_data` to stamp JSONL records with agent routing fields for NeMo-Gym
6. **collect_rollouts** (Step 5): Collect model rollouts with reward scoring — includes enrichment (restore metadata), analysis (reward distribution, difficulty), and filtering
7. **compute_rewards** (Step 6, optional): Re-judge existing rollouts with a different/stronger judge model without re-generating responses
8. **training** (Step 7): GRPO training using NeMo-RL with online NeMo-Gym environment rewards
9. **eval** (Step 8): Evaluate GRPO checkpoints on finance benchmarks

**See [technical reference](../stages/grpo.md) for detailed stage documentation.**

### Model Configurations

| Config | Model | GPUs | Status |
|--------|-------|------|--------|
| `grpo/qwen3_4b.yaml` | Qwen3-4B | 8 (1 node) | Demo |

> **Note:** Additional model configs (Qwen3-14B, Nemotron) will be added as GRPO training is validated at scale.

## Usage

### Run Complete Workflow (Demo)

```bash
# Qwen3-4B (demo — skips compute_rewards)
uv run nflow run-all --config nvflow/recipes/finance/workflows/grpo/qwen3_4b.yaml
```

### Stage-by-Stage Execution

```bash
CONFIG=nvflow/recipes/finance/workflows/grpo/qwen3_4b.yaml

# Step 0: SDG cleanup → model-agnostic schema (CPU)
uv run nflow run data_transformation --config $CONFIG

# Step 1: Apply prompt template + extract expected answer (CPU)
uv run nflow run apply_prompt_template --config $CONFIG

# Step 2: Convert to NeMo-Gym Responses API format (CPU)
uv run nflow run convert_to_responses_api --config $CONFIG

# Step 3: Split into train/val sets (CPU)
uv run nflow run train_validation_split --config $CONFIG

# Step 4: Prepare data — add agent routing fields (CPU)
uv run nflow run prepare_data --config $CONFIG

# Step 5: Collect rollouts (inference + reward scoring + filter) (GPU)
uv run nflow run collect_rollouts --config $CONFIG

# Step 7: GRPO training (GPU)
uv run nflow run training --config $CONFIG

# Step 8: Evaluate checkpoints (GPU)
uv run nflow run eval --config $CONFIG
```

### Optional: Re-judge Rollouts

To re-judge rollouts with a different judge model, enable `compute_rewards` in `pipeline_stages` and configure the judge:

```yaml
# In your model config
pipeline_stages:
  - data_transformation
  - apply_prompt_template
  - convert_to_responses_api
  - train_validation_split
  - prepare_data
  - collect_rollouts
  - compute_rewards          # Uncomment to enable
  - training
  - eval

stages:
  compute_rewards:
    rejudge:
      judge_vllm:
        num_gpus: 0
        openai_base_url: "https://api.openai.com/v1"
        openai_model: "gpt-4o"
```

## Data & Results

### Output Structure

```
outputs/finance/demo/workflow-5-grpo/qwen3_4b/
├── step-0-data-transformation/
│   ├── final_result.jsonl               # Normalized SDG data (model-agnostic schema)
│   ├── chunks/                          # Chunked input for parallel processing
│   └── logs/
├── step-1-apply-prompt-template/
│   ├── *.jsonl                          # Prompted data with extracted answers
│   └── logs/
├── step-2-convert-to-responses-api/
│   ├── final_result.jsonl               # Data in Responses API format
│   └── logs/
├── step-3-train-validation-split/
│   ├── train.jsonl                      # Training split
│   ├── val.jsonl                        # Validation split
│   └── logs/
├── step-4-prepare-data/
│   ├── train.jsonl                      # Training data with agent_ref
│   ├── validation.jsonl                 # Validation data with agent_ref
│   └── agent_config_overlay.yaml        # Auto-generated agent config
├── step-5-collect-rollouts/
│   ├── rollouts-rs0.jsonl               # Merged rollouts (enriched)
│   ├── train.jsonl                      # Filtered training data (from filter sub-job)
│   ├── validation.jsonl                 # Filtered validation data
│   ├── analysis_rs0/
│   │   ├── summary.txt                  # Reward distribution report
│   │   ├── correct.jsonl                # Samples with reward == 1.0
│   │   ├── incorrect.jsonl              # Samples with reward == 0.0
│   │   └── difficulty.jsonl             # Per-question pass rates (if repeats)
│   ├── scripts/                         # Generated Slurm scripts
│   └── logs/                            # vLLM and ng_run logs
├── step-6-compute-rewards/              # (only if compute_rewards enabled)
│   ├── train.jsonl                      # Re-judged + filtered training data
│   ├── validation.jsonl                 # Re-judged + filtered validation data
│   └── ...
├── step-7-training/
│   └── grpo-qwen3-4b-1n-tp2-cp1-seq32k/
│       ├── checkpoints/                 # GRPO model checkpoints
│       └── training-logs/
└── step-8-eval/
    └── ...                              # Benchmark evaluation results
```

### Expected Results (Demo)

| Stage | Output | Notes |
|-------|--------|-------|
| data_transformation | `final_result.jsonl` — normalized SDG schema | CPU-only, ~1 min |
| apply_prompt_template | Prompted JSONL with extracted answers | CPU-only, ~1 min |
| convert_to_responses_api | `final_result.jsonl` in Responses API format | CPU-only, ~1 min |
| train_validation_split | `train.jsonl` + `val.jsonl` | CPU-only, ~1 min |
| prepare_data | `train.jsonl` + `validation.jsonl` with agent_ref fields | CPU-only, ~1 min |
| collect_rollouts | `rollouts-rs0.jsonl` + analysis + filtered train/val | GPU inference, ~10 min |
| training | GRPO checkpoint | GPU training, ~20 min |
| eval | Benchmark scores | GPU inference, ~10 min |

### Validation

```bash
OUTPUT_DIR="outputs/finance/demo/workflow-5-grpo/qwen3_4b"

# Check normalized SDG data (Step 0)
head -1 $OUTPUT_DIR/step-0-data-transformation/final_result.jsonl | jq 'keys'

# Check prompted data (Step 1)
head -1 $OUTPUT_DIR/step-1-apply-prompt-template/*.jsonl | jq '.problem' | head -c 200

# Check Responses API conversion (Step 2)
head -1 $OUTPUT_DIR/step-2-convert-to-responses-api/final_result.jsonl | jq 'keys'

# Check train/val split (Step 3)
wc -l $OUTPUT_DIR/step-3-train-validation-split/train.jsonl $OUTPUT_DIR/step-3-train-validation-split/val.jsonl

# Check prepared data with agent_ref (Step 4)
head -1 $OUTPUT_DIR/step-4-prepare-data/train.jsonl | jq '.agent_ref'

# Check rollout analysis (Step 5)
cat $OUTPUT_DIR/step-5-collect-rollouts/analysis_rs0/summary.txt

# Check training checkpoint (Step 7)
ls $OUTPUT_DIR/step-7-training/grpo-*/checkpoints/

# Check eval results (Step 8)
ls $OUTPUT_DIR/step-8-eval/
```

## Customization

### Judge Configuration

Three judge modes for `collect_rollouts`:

```yaml
stages:
  collect_rollouts:
    # Option A: Local vLLM judge (needs extra GPUs)
    judge_model_path: /hf_models/Qwen/Qwen3-30B-A3B-Instruct-2507
    judge_tensor_parallel_size: 4
    num_gpus: 8   # Must cover policy TP + judge TP

    # Option B: OpenAI API judge (no local GPU for judge)
    # judge_openai_base_url: "https://api.openai.com/v1"
    # judge_openai_model: "gpt-4o"

    # Option C: Policy-as-judge (default, testing only)
    # Neither set — judge reuses the policy model
```

### Scaling for Production

For large-scale rollout collection (300K+ samples):

```yaml
stages:
  collect_rollouts:
    num_chunks: 16              # Split input into 16 parallel Slurm jobs
    num_random_seeds: 5         # 5 independent runs for diversity
    num_repeats: 5              # 5 repeats per sample (for pass_rate analysis)
    num_samples_in_parallel: 8  # Concurrent requests per job
    rerun_done: false           # Resume from .done files
```

### External vLLM Server (SDG Mode)

Use a pre-launched vLLM server (any version) instead of the self-contained launch:

```yaml
stages:
  collect_rollouts:
    vllm_base_url: "http://<host>:<port>/v1"
```

### Training Parameters

```yaml
stages:
  training:
    num_nodes: 4                # Scale up for production
    dependent_jobs: 3           # Multi-job chaining for long runs

    overrides:
      grpo:
        num_prompts_per_step: 64
        num_generations_per_prompt: 16
        max_num_steps: 1000
      policy:
        train_global_batch_size: 1024
```

### Megatron Backend

```yaml
stages:
  training:
    backend: megatron

    overrides:
      policy:
        megatron_cfg:
          tensor_model_parallel_size: 4
          pipeline_model_parallel_size: 1
          context_parallel_size: 1
          activation_checkpointing: true
          converter_type: "Qwen2ForCausalLM"
```

## Additional Resources

### Monitoring

```bash
# Check Slurm jobs
squeue --me

# View rollout collection logs
tail -f $OUTPUT_DIR/step-5-collect-rollouts/logs/ng_run_rs0_chunk0.log

# View training logs
tail -f $OUTPUT_DIR/step-7-training/grpo-*/training-logs/*.log
```

### Troubleshooting

**vLLM server fails to start:**
- Check GPU availability: `sinfo -p interactive`
- Review vLLM logs: `cat $OUTPUT_DIR/step-5-collect-rollouts/logs/vllm_server_rs0_chunk0.log`
- Ensure `tensor_parallel_size` doesn't exceed available GPUs

**All rewards are 0.0 or 1.0:**
- Check the rollout analysis: `cat $OUTPUT_DIR/step-5-collect-rollouts/analysis_rs0/summary.txt`
- Policy-as-judge produces circular evaluation — use a separate judge for meaningful rewards
- Review judge logs: `cat $OUTPUT_DIR/step-5-collect-rollouts/logs/ng_run_rs0_chunk0.log`

**Rollouts missing metadata (uuid, question):**
- The enrichment step automatically restores fields dropped by NeMo-Gym environments
- Check enrichment output in the merge job logs

### Next Steps

After GRPO training:

- **[Evaluate GRPO Model](05-eval.md)** - Compare GRPO checkpoint with SFT baseline
- **Scale Up** - Increase data, nodes, and training steps for production
- **Judge Iteration** - Try different judge models via `compute_rewards`

### Technical Details

This workflow uses **NeMo-RL** for GRPO training and **NeMo-Gym** for environment-based reward computation. The `equivalence_llm_judge` environment provides semantic equivalence scoring via an LLM judge.

For comprehensive stage-by-stage documentation:
- **[GRPO Stages Reference](../stages/grpo.md)**
