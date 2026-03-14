# Workflow 3: Document-Grounded SDG

## Purpose

Generate high-quality financial Q&A pairs directly from SEC filing documents with built-in verification, evaluation, and difficulty estimation.

> **Note:** This workflow generates ~800K Q&A pairs. SFT integration is currently in progress. For production SFT pipeline, see [Template-Based SDG](02-template-based-sdg.md).

## Prerequisites

- ✅ SEC filings downloaded ([Workflow 1](01-download-sec.md))
- Will be preprocessed in Stage 0 (dg_sdg_preprocess)

## Key Differences from Template-Based

| Aspect | Template-Based | Document-Grounded |
|--------|----------------|-------------------|
| **Question Source** | Seed questions | Generated from documents |
| **Verification** | None | Built-in verification step |
| **Quality Control** | GenSelect + Filter | GenSelect + Evaluation + Aggregation |
| **Difficulty** | Not estimated | Estimated via small model testing |
| **Output** | Single dataset | Stratified by difficulty (medium/hard) |

## Pipeline Flow
```
┌─────────────────────────┐
│ 0. dg_sdg_preprocess    │  Preprocessing: SEC HTML → Chunked JSONL
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ 1. generate_verified_qa │  Q&A Generation: Questions + Answers
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ 2. genselect_answers    │  Selection: Best answer from candidates
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ 3. evaluate_answers     │  Evaluation: Quality scoring (5 seeds)
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ 4. aggregate_answers    │  Aggregation: Combine evaluation results
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ 5. difficulty_estimation│  Difficulty: Small model testing
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ 6. dgsdg_post_process   │  Output: Stratified training datasets
└─────────────────────────┘
```

## 7 Stages (Overview)

0. **dg_sdg_preprocess**: Preprocess SEC filings (chunk HTML → create JSONL data following SecQue distribution)
1. **generate_verified_qa**: Generate questions from documents, verify them, generate answers (6 internal sub-steps)
2. **genselect_answers**: Select best answer from multiple candidates
3. **evaluate_answers**: Evaluate answer quality (5 random seeds for robustness)
4. **aggregate_answers**: Aggregate evaluation results
5. **difficulty_estimation**: Estimate difficulty using small model
6. **dgsdg_post_process**: Clean and create difficulty-stratified datasets

**See [technical reference](../stages/document-grounded-sdg.md) for detailed stage documentation.**

## Configuration

**File:** `workflows/sdg/document-grounded-sdg.yaml`

- Production-ready configuration
- Uses large models (GPT-OSS-120B, Qwen3-235B)
- Configured for `/workspace/outputs/finance/sap-500/workflow-2-download-sec/` input

## Usage

### Run Complete Workflow

```bash
uv run nflow run-all --config nvflow/recipes/finance/workflows/sdg/document-grounded-sdg.yaml
```

### Run Individual Stages

```bash
# Stage 0: Preprocess SEC filings
uv run nflow run dg_sdg_preprocess --config nvflow/recipes/finance/workflows/sdg/document-grounded-sdg.yaml

# Stage 1: Generate verified Q&A
uv run nflow run generate_verified_qa --config nvflow/recipes/finance/workflows/sdg/document-grounded-sdg.yaml

# Stage 2: Select best answers
uv run nflow run genselect_answers --config nvflow/recipes/finance/workflows/sdg/document-grounded-sdg.yaml

# Stage 3: Evaluate answers
uv run nflow run evaluate_answers --config nvflow/recipes/finance/workflows/sdg/document-grounded-sdg.yaml

# Stage 4: Aggregate results
uv run nflow run aggregate_answers --config nvflow/recipes/finance/workflows/sdg/document-grounded-sdg.yaml

# Stage 5: Estimate difficulty
uv run nflow run difficulty_estimation --config nvflow/recipes/finance/workflows/sdg/document-grounded-sdg.yaml

# Stage 6: Post process
uv run nflow run dgsdg_post_process --config nvflow/recipes/finance/workflows/sdg/document-grounded-sdg.yaml
```

## Output Structure

```
${base_data_dir}/
├── step-0-preprocess/
│   ├── chunks/                          # Chunked HTML files (markdown, clean HTML, original)
│   ├── csv_lists/                       # CSV manifests of chunks
│   └── jsonl/
│       ├── 10-k-data.jsonl              # Sampled 10-K data
│       └── 10-q-data.jsonl              # Sampled 10-Q data
├── step-1-qa-pipeline/
│   ├── question_pipeline/
│   │   ├── generated/                   # Generated questions
│   │   └── verified/                    # Verified questions
│   └── answer_pipeline/
│       └── generated/                   # Generated answers
├── step-2-genselect/
│   └── selected_answers.jsonl
├── step-3-evaluate/
│   └── evaluation results (5 seeds)
├── step-4-aggregate/
│   └── aggregated_answers.jsonl
├── step-5-difficulty/
│   └── difficulty scoring results
└── step-6-post-process/
    ├── full_data.jsonl                  # All cleaned records
    ├── final_result.jsonl            # Medium difficulty (for SFT)
    └── hard_rl_data.jsonl               # Hard difficulty training data (difficulty_score=0)
```

## Expected Results

**Production (S&P 500):**

| Metric | Value |
|--------|-------|
| Questions Generated | ~2M+ |
| Verified Questions | ~1.6M |
| Final Q&A Pairs | ~800K |
| Medium Difficulty | ~100K |
| Hard Difficulty | ~400K |
| Time | ~30 hours, affected by resources used |

## Output Format

### Final Training Data

**final_result.jsonl** - For supervised fine-tuning:
```json
{
  "question": "Based on the risk factors, what are Tesla's main supply chain concerns?",
  "context": "...SEC filing excerpt...",
  "generation": "<reasoning>...\n<answer>...</answer>",
  "difficulty_score": 2,
  "evaluation_score": 4.5
}
```

**hard_rl_data.jsonl** - Hard difficulty training data:
```json
{
  "question": "How does NVIDIA's revenue recognition differ for bundled products?",
  "context": "...complex accounting excerpt...",
  "generation": "<reasoning>...\n<answer>...</answer>",
  "difficulty_score": 0,
  "evaluation_score": 4.8
}
```

## Validation

```bash
# Check all outputs exist (from nvflow directory)
BASE_DIR="outputs/finance/sap-500/workflow-3-document-grounded-sdg"

# Stage outputs
ls $BASE_DIR/step-1-qa-pipeline/answer_pipeline/generated/
ls $BASE_DIR/step-2-genselect/selected_answers.jsonl
ls $BASE_DIR/step-4-aggregate/aggregated_answers.jsonl

# Final datasets
ls $BASE_DIR/step-6-post-process/

# Count Q&A by difficulty
echo "Medium difficulty:"
wc -l $BASE_DIR/step-6-post-process/final_result.jsonl

echo "Hard difficulty:"
wc -l $BASE_DIR/step-6-post-process/hard_rl_data.jsonl

# Inspect samples
head -n 3 $BASE_DIR/step-6-post-process/final_result.jsonl | jq .
```

## Stage 0: dg_sdg_preprocess Details

Converts raw SEC 10-K and 10-Q HTML filings into structured JSONL data for downstream processing.

### Steps

1. **Chunk HTML files**: Split large HTML documents into token-limited chunks with overlap
2. **Generate CSV lists**: Create file manifests tracking all chunks
3. **Create JSONL data**: Sample chunks following SecQue benchmark distribution

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `input_dir` | Raw SEC filings directory (10-K and 10-Q HTML files) | `${filings_dir}/data` |
| `output_dir` | Preprocessed data output directory | `${base_data_dir}/step-0-preprocess` |
| `distribution_dir` | Directory with distribution CSVs (SecQue benchmark) | `/workspace/nvflow/recipes/finance/workflows/sdg/dg_sdg_distribution` |
| `max_tokens` | Maximum tokens per chunk | 3000 |
| `overlap_tokens` | Overlap tokens between chunks for context coverage | 500 |
| `total_samples` | Total samples to generate following distribution | 150000 |
| `max_skip_count` | Stop sampling after this many skips (non-repeatable) | 20000 |
| `seed` | Random seed for reproducibility | 42 |

### Input Structure

Your SEC filings should follow this structure:

```
${filings_dir}/data/
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

This structure is created automatically by the SEC download workflow ([Workflow 1](01-download-sec.md)).

## Stage 1: generate_verified_qa Details

This stage performs 6 internal sub-steps:

1. **Preprocess Documents** (CPU): Prepare SEC filings for question generation
2. **Generate Questions** (GPU): Create questions from documents using GPT-OSS-120B
3. **Preprocess Questions** (CPU): Prepare for verification
4. **Verify Questions** (GPU): Verify quality using Qwen3-235B (5 seeds)
5. **Preprocess Verified** (CPU): Filter by threshold, prepare for answers
6. **Generate Answers** (GPU): Create answers using GPT-OSS-120B (5 seeds)

See [technical reference](../stages/document-grounded-sdg.md#generate_verified_qa) for details.

## Customization

### Adjust Resources

Edit `workflows/sdg/document-grounded-sdg.yaml` to control parallelization and GPU allocation:
**Example: Increase parallelization For GPU Stages**
```yaml
num_chunks: 10  # Change from 1 → 10 to run 10 jobs in parallel
```
> **Note:** Total GPU usage = `num_chunks × server_gpus`. Ensure cluster has enough resources.

### Change Models

```yaml
stages:
  generate_verified_qa:
    question_generation_kwargs:
      args:
        model: /path/to/your/model
        server_gpus: 8
```

### Modify Prompts

Edit prompts in `nvflow/recipes/finance/prompts/`:
- `document_grounded_generate_questions.yaml` - Question generation
- `document_grounded_verify_questions.yaml` - Question verification
- `generate_answers.yaml` - Answer generation
- `evaluate_answers.yaml` - Answer evaluation
- `judge_difficulty.yaml` - Difficulty judging

## Common Issues

### "Input folder empty"

**Solution:** Ensure SEC filings downloaded:
```bash
# From nvflow directory
ls outputs/finance/sap-500/workflow-2-download-sec/step-0-download/data/
# Should have company directories
```

### Low verification rate

**Solution:**
- Check verification threshold (default: 1.0 means all 5 seeds must verify)
- Lower threshold to 0.6 (3 out of 5 seeds)
- Review question generation prompt

### Difficulty estimation takes too long

**Solution:**
- Reduce `num_random_seeds` for answer generation
- Use fewer `num_chunks` for parallelization
- Use smaller judge model

## Combining with Template-Based

You can combine both SDG approaches:

```bash
# Merge datasets (from nvflow directory)
cat outputs/finance/sap-500/workflow-3-template-based-sdg/step-5-filter-answers/final_result.jsonl \
    outputs/finance/sap-500/workflow-3-document-grounded-sdg/step-6-post-process/final_result.jsonl \
    > combined_training_data.jsonl

# Use combined data for SFT
# Update SFT workflow to point to combined_training_data.jsonl
```

## Next Steps

After completing document-grounded SDG:

- **[SFT Training](04-sft.md)** - Train on stratified datasets
- **[Evaluation](05-eval.md)** - Test model performance
- Combine with template-based data for more diversity

## Technical Details

For comprehensive stage-by-stage documentation:
- **[Document-Grounded SDG Stages Reference](../stages/document-grounded-sdg.md)**

## Models Used

| Model | Usage | Size |
|-------|-------|------|
| GPT-OSS-120B | Question generation, answer generation | 120B |
| Qwen3-235B-A22B | Question verification, answer selection, evaluation | 235B |
| Qwen3-4B | Difficulty estimation (small model baseline) | 4B |

All models are configurable in the workflow YAML.
