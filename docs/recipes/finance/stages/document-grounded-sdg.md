# Document-Grounded SDG Stages Reference

Technical reference for all 7 stages in the document-grounded-sdg workflow.

## Quick Navigation

- [dg_sdg_preprocess](#dg_sdg_preprocess)
- [generate_verified_qa](#generate_verified_qa)
- [genselect_answers](#genselect_answers)
- [evaluate_answers](#evaluate_answers)
- [aggregate_answers](#aggregate_answers)
- [difficulty_estimation](#difficulty_estimation)
- [dgsdg_post_process](#dgsdg_post_process)

---

## dg_sdg_preprocess

**File:** `nvflow/recipes/finance/stages/sdg/dg_sdg_preprocess.py`
**Registry:** `recipe="finance"`, `workflow="document_grounded_sdg"`, `stage="dg_sdg_preprocess"`

### Purpose

Converts raw SEC 10-K and 10-Q HTML filings into structured JSONL data for question/answer generation. This preprocessing stage performs three main operations:

1. **Chunk HTML files**: Split large documents into token-limited chunks with overlap
2. **Generate CSV lists**: Create file manifests tracking all chunks
3. **Create JSONL data**: Sample chunks following SecQue benchmark distribution

### Inputs

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `input_dir` | path | Raw SEC filings directory (10-K and 10-Q HTML files) | Required |
| `output_dir` | path | Preprocessed data output directory | Required |
| `distribution_dir` | path | Directory with distribution CSVs (SecQue benchmark) | Required |
| `max_tokens` | int | Maximum tokens per chunk | 2000 |
| `overlap_tokens` | int | Overlap tokens between chunks for context coverage | 100 |
| `total_samples` | int | Total samples to generate following distribution | 150000 |
| `max_skip_count` | int | Stop sampling after this many skips (non-repeatable) | 20000 |
| `seed` | int | Random seed for reproducibility | 42 |
| `preprocess_kwargs` | dict | Additional CPU job settings (partition, etc.) | `{}` |

### Expected Input Structure

Your SEC filings should follow this structure (created by Workflow 1):

```
${input_dir}/
├── {TICKER}/                      # e.g., AAPL, MSFT, TSLA
│   ├── 10-K/
│   │   └── {YEAR}/                # e.g., 2019, 2020, 2021
│   │       └── {FILING_ID}/       # SEC Filing ID
│   │           ├── 1.html         # Item sections
│   │           ├── 1A.html
│   │           ├── 7.html
│   │           └── exhibits/
│   │               └── EX-*.html  # Exhibits
│   └── 10-Q/
│       └── {YEAR}/
│           └── {FILING_ID}/...
└── {TICKER_2}/...
```

### Outputs

```
${output_dir}/
├── chunks/                        # Chunked HTML files (3 formats)
│   ├── markdown/                  # Markdown-converted chunks
│   ├── clean_html/                # Cleaned HTML chunks
│   └── original_html/             # Original HTML chunks
├── csv/                           # File lists for tracking
│   ├── 10k_1company.csv
│   ├── 10k_2company.csv
│   ├── 10q_1company.csv
│   └── 10q_2company.csv
└── jsonl/                         # Sampled training data ← Used by next stage
    └── training_data.jsonl
```

### Configuration Example

```yaml
dg_sdg_preprocess:
  input_dir: ${filings_dir}/data
  output_dir: ${base_data_dir}/step-0-preprocess
  distribution_dir: /workspace/nvflow/recipes/finance/workflows/sdg/dg_sdg_distribution
  max_tokens: 3000
  overlap_tokens: 500
  total_samples: 150000
  max_skip_count: 20000
  seed: 42

```

### Resources

- **Runtime:** ~2-4 hours for full S&P 500 dataset
- **Compute:** CPU only (no GPUs required)
- **Output Size:** ~10GB JSONL data for 150K samples

### Notes

- Distribution CSVs define sampling proportions matching SecQue benchmark
- Chunking with overlap ensures context continuity across boundaries
- `max_skip_count` prevents infinite loops when distribution can't be satisfied

---

## generate_verified_qa

**File:** `nvflow/recipes/finance/stages/sdg/document_grounded_question_answer_generation_pipeline.py`
**Registry:** `recipe="finance"`, `workflow="document_grounded_sdg"`, `stage="generate_verified_qa"`

### Purpose

Combined stage that generates questions from SEC filing documents, verifies their quality, and generates answers. Executes 6 internal sub-steps.

### Internal Sub-Steps

1. **Preprocess Documents** (CPU): Preprocess sampled data for question generation
2. **Generate Questions** (GPU): Create questions from documents
3. **Preprocess Questions** (CPU): Prepare for verification
4. **Verify Questions** (GPU): Verify quality with 5 random seeds
5. **Preprocess Verified** (CPU): Filter by threshold, prepare for answers
6. **Generate Answers** (GPU): Generate answers with 5 random seeds

### Inputs

| Parameter | Type | Description |
|-----------|------|-------------|
| `input_folder` | path | Preprocessed JSONL data directory from `dg_sdg_preprocess` (`${base_data_dir}/step-0-preprocess/jsonl/`) |
| `output_dir` | path | Base output directory for all sub-steps |
| `question_preprocess_kwargs` | dict | CPU job settings for preprocessing |
| `question_generation_kwargs` | dict | GPU settings for question generation |
| `question_verify_kwargs` | dict | GPU settings for verification (5 seeds) |
| `answer_preprocess_kwargs` | dict | CPU settings, includes `threshold` |
| `answer_generation_kwargs` | dict | GPU settings for answer generation (5 seeds) |

### Outputs

```
${output_dir}/
├── question_pipeline/
│   ├── generate_input.jsonl          # Preprocessed documents
│   ├── generated/                    # Generated questions
│   │   ├── seed_0.jsonl
│   │   └── ...
│   ├── verify_input.jsonl            # Questions to verify
│   └── verified/                     # Verified questions
│       ├── seed_0.jsonl
│       └── ...
└── answer_pipeline/
    ├── answer_input.jsonl            # Verified questions
    └── generated/                    # Generated answers ← Output
        ├── seed_0.jsonl
        └── ...
```

### Configuration Example

```yaml
generate_verified_qa:
  input_folder: ${base_data_dir}/step-0-preprocess/jsonl
  output_dir: ${base_data_dir}/step-1-qa-pipeline

  question_generation_kwargs:
    args:
      model: /models/gpt-oss-120b
      server_gpus: 8
      num_chunks: 5
      num_random_seeds: 1
    ctx_args: >-
      ++prompt_config=nvflow/recipes/finance/prompts/document_grounded_generate_questions.yaml
      ++inference.temperature=0.9

  question_verify_kwargs:
    args:
      model: /models/Qwen3-235B
      server_gpus: 8
      num_chunks: 5
      num_random_seeds: 5

  answer_generation_kwargs:
    args:
      model: /models/gpt-oss-120b
      server_gpus: 8
      num_chunks: 5
      num_random_seeds: 5
```

### Resources

- **Total Runtime:** ~4-8 hours for full pipeline
- **GPUs:** 40 for question generation, 200 for question verification and answer generation
- **Models:** GPT-OSS-120B (questions, answers), Qwen3-235B (verification)

---

## genselect_answers

**File:** `nvflow/recipes/finance/stages/sdg/genselect_answers.py`
**Registry:** `recipe="finance"`, `workflow="document_grounded_sdg"`, `stage="genselect_answers"`

### Purpose

Select best answer from multiple candidates (same as template-based, but for document-grounded data).

### Inputs

| Parameter | Type | Description |
|-----------|------|-------------|
| `input_dir` | path | Answer candidates from generate_verified_qa |
| `output_file` | path | Selected answers output file |
| `prompt_config` | path | GenSelect prompt |

### Outputs

JSONL file with selected best answers.

### Configuration Example

```yaml
  genselect_answers:
    input_dir: ${base_data_dir}/step-1-qa-pipeline/answer_pipeline/generated
    output_file: ${base_data_dir}/step-2-genselect/selected_answers.jsonl
    prompt_config: nvflow/recipes/finance/prompts/genselect_answers.yaml
    inline_args: "++inference.tokens_to_generate=16384"
    dependencies: [generate_verified_qa]
    stage_kwargs:
      model: /models/Qwen3-235B-A22B-Instruct-2507
      server_type: vllm
      server_gpus: 8
      server_nodes: 1
      num_chunks: 15
      partition: batch
```

### Resources

- **GPUs:** 120
- **Model:** Qwen3-235B
- **Runtime:** 2-4 hours

---

## evaluate_answers

**File:** `nvflow/recipes/finance/stages/sdg/evaluate_answers.py`
**Registry:** `recipe="finance"`, `workflow="document_grounded_sdg"`, `stage="evaluate_answers"`

### Purpose

Evaluate answer quality using a large model judge. Runs 5 random seeds for robustness.

### Inputs

| Parameter | Type | Description |
|-----------|------|-------------|
| `input_file` | path | Selected answers from genselect_answers |
| `output_dir` | path | Directory for evaluation results |
| `prompt_config` | path | Evaluation prompt |

### Outputs

```
${output_dir}/
├── seed_0.jsonl
├── seed_1.jsonl
├── seed_2.jsonl
├── seed_3.jsonl
└── seed_4.jsonl
```

Each file contains evaluation scores:
```json
{
  "question": "...",
  "generation": "...",
  "evaluate_generation": "Score: 4.5/5\nReasoning: ...",
  "evaluation_score": 4.5
}
```

### Configuration Example

```yaml
  evaluate_answers:
    input_file: ${base_data_dir}/step-2-genselect/selected_answers.jsonl
    output_dir: ${base_data_dir}/step-3-evaluate
    prompt_config: nvflow/recipes/finance/prompts/evaluate_answers.yaml
    inline_args: "++generation_key=evaluate_generation ++inference.top_p=0.9 ++inference.temperature=0.8"
    dependencies: [genselect_answers]
    stage_kwargs:
      model: /models/Qwen3-235B-A22B-Instruct-2507
      server_type: vllm
      server_gpus: 8
      server_nodes: 1
      num_chunks: 5
      num_random_seeds: 5
      partition: batch
```

### Resources

- **GPUs:** 200
- **Model:** Qwen3-235B
- **Runtime:** 8 hours

---

## aggregate_answers

**File:** `nvflow/recipes/finance/stages/sdg/aggregate_answers.py`
**Registry:** `recipe="finance"`, `workflow="document_grounded_sdg"`, `stage="aggregate_answers"`

### Purpose

Aggregate evaluation results from 5 random seeds into final scores.

### Inputs

| Parameter | Type | Description |
|-----------|------|-------------|
| `input_dir` | path | Evaluation results from evaluate_answers |
| `output_file` | path | Aggregated results output |

### Outputs

JSONL file with aggregated scores:
```json
{
  "question": "...",
  "generation": "...",
  "evaluation_scores": [4.5, 4.8, 4.3, 4.7, 4.6],
  "mean_score": 4.58,
  "std_score": 0.18
}
```

### Resources

- **Compute:** CPU only
- **Runtime:** 5-10 min

---

## difficulty_estimation

**File:** `nvflow/recipes/finance/stages/sdg/difficulty_estimation.py`
**Registry:** `recipe="finance"`, `workflow="document_grounded_sdg"`, `stage="difficulty_estimation"`

### Purpose

Estimate question difficulty by testing if a small model can answer correctly. Questions the small model fails are considered harder.

### Two-Step Process

1. **Small Model Answering**: Qwen3-4B attempts to answer (5 seeds)
2. **Large Model Judging**: GPT-OSS-120B judges if small model succeeded

### Inputs

| Parameter | Type | Description |
|-----------|------|-------------|
| `input_file` | path | Aggregated answers |
| `output_file` | path | Answers with difficulty scores |
| `work_dir` | path | Working directory for intermediate files |
| `num_random_seeds` | int | Random seeds for small model (default: 5) |
| `answer_model_kwargs` | dict | Settings for small model (Qwen3-4B) |
| `judge_model_kwargs` | dict | Settings for judge model (GPT-OSS-120B) |

### Outputs

JSONL file with difficulty scores:
```json
{
  "question": "...",
  "generation": "...",
  "difficulty_score": 0,  # 0 = hard (small model failed)
  "small_model_correct": false,
  "small_model_attempts": 5,
  "small_model_successes": 0
}
```

**Difficulty Score:**
- `0`: Hard (small model failed all attempts)
- `1-4`: Medium (small model succeeded on some attempts)
- `5`: Easy (small model succeeded on all attempts)

### Configuration Example

```yaml
  difficulty_estimation:
    input_file: ${base_data_dir}/step-4-aggregate/aggregated_answers.jsonl
    output_file: ${base_data_dir}/step-5-difficulty-data/answers_with_difficulty.jsonl
    work_dir: ${base_data_dir}/step-5-difficulty
    num_random_seeds: 5
    dependencies: [aggregate_answers]


    # Small model for answering (Qwen3-4B)
    answer_model_kwargs:
      args:
        model: /models/qwen34b
        server_type: vllm
        server_gpus: 8
        server_nodes: 1
        num_chunks: 5
        partition: batch
      ctx_args: >-
        ++prompt_config=nvflow/recipes/finance/prompts/secque_template.yaml
        ++inference.temperature=0.7

    answer_prompt_config: nvflow/recipes/finance/prompts/secque_template.yaml

    # Large model for judging (GPT-OSS120)
    judge_model_kwargs:
      args:
        model: /models/gpt-oss-120b
        server_type: vllm
        server_gpus: 8
        server_nodes: 1
        num_chunks: 10
        partition: batch
      ctx_args: >-
        ++inference.temperature=0.1

    judge_prompt_config: nvflow/recipes/finance/prompts/judge_difficulty.yaml
```

### Resources

- **GPUs:** 200 (small model), 400 (judge model)
- **Runtime:** 4-8 hours

---

## dgsdg_post_process

**File:** `nvflow/recipes/finance/stages/sdg/document_grounded_data.py`
**Registry:** `recipe="finance"`, `workflow="document_grounded_sdg"`, `stage="dgsdg_post_process"`

### Purpose

Clean data and create difficulty-stratified training datasets.

### Inputs

| Parameter | Type | Description |
|-----------|------|-------------|
| `input_file` | path | Answers with difficulty from difficulty_estimation |
| `output_dir` | path | Output directory for final datasets |
| `seed` | int | Random seed for reproducibility (default: 42) |

### Outputs

```
${output_dir}/
├── full_data.jsonl                # All cleaned records
├── final_result.jsonl          # difficulty_score in [1,2,3,4], filtered
└── hard_rl_data.jsonl             # difficulty_score = 0 (hardest)
```

**final_result.jsonl** - Medium difficulty training data:
- Medium difficulty questions
- Filtered by filing type and quality
- Ready for training

**hard_rl_data.jsonl** - Hard difficulty training data:
- Hardest questions (small model failed)
- High-quality answers
- Suitable for advanced training or challenging evaluation

### Resources

- **Compute:** CPU only
- **Runtime:** 5-10 min

---

## Pipeline Summary

| Stage | Purpose | Compute | Runtime |
|-------|---------|---------|---------|
| generate_verified_qa | Generate & verify Q&A | GPU | 6-8h |
| genselect_answers | Select best answers | GPU | 1-2h |
| evaluate_answers | Evaluate quality | GPU | 2-3h |
| aggregate_answers | Aggregate scores | CPU | 10m |
| difficulty_estimation | Estimate difficulty | GPU | 2-3h |
| dgsdg_post_process | Create final datasets | CPU | 10m |

**Total:** ~10-12 hours for full production run

See [Document-Grounded SDG Workflow](../workflows/03-document-grounded-sdg.md) for usage examples and configuration details.
