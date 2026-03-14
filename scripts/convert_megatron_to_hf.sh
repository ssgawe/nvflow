#!/bin/bash
# ============================================================================
# Convert Megatron Checkpoint to HuggingFace Format
# ============================================================================
# This script manually converts a Megatron checkpoint to HuggingFace format
# using ns run_cmd. It bypasses the hf_overrides argument that causes failures
# in nemo-rl 0.7.1 containers.
#
# Usage:
#   ./scripts/convert_megatron_to_hf.sh [STEP_NUMBER]
#
# Examples:
#   ./scripts/convert_megatron_to_hf.sh          # Convert latest checkpoint (step_606)
#   ./scripts/convert_megatron_to_hf.sh 500      # Convert step_500 checkpoint
#
# ============================================================================

set -euo pipefail

# ============================================================================
# Configuration - Modify these for your run
# ============================================================================

# Model configuration
MODEL_NAME="Qwen/Qwen3-14B"
TOKENIZER_PATH="Qwen/Qwen3-14B"

# Paths (using /workspace which maps to the project root in container)
RUN_NAME="model-qwen3-14b-32n-tp4-pp1-cp16-seq64k"
BASE_DIR="/workspace/outputs/finance/sft/step-5-sft/${RUN_NAME}"
CHECKPOINT_DIR="${BASE_DIR}/checkpoints"

# Default to step_606 (final checkpoint) if not specified
STEP_NUMBER="${1:-606}"
INPUT_PATH="${CHECKPOINT_DIR}/step_${STEP_NUMBER}/policy/weights/iter_0000000"

# Output directories (organized under manual_conversion/)
OUTPUT_PATH="${BASE_DIR}/manual_conversion/hf/step_${STEP_NUMBER}"
LOG_DIR="${BASE_DIR}/manual_conversion/logs/step_${STEP_NUMBER}"

# Cluster configuration
CLUSTER="nrt"
NUM_GPUS=8
CONTAINER="nemo-rl"
EXPNAME="convert-${RUN_NAME}-step${STEP_NUMBER}"

# ============================================================================
# Script Logic
# ============================================================================

echo "============================================"
echo "Megatron to HuggingFace Conversion"
echo "============================================"
echo ""
echo "Configuration:"
echo "  Model:       ${MODEL_NAME}"
echo "  Step:        ${STEP_NUMBER}"
echo "  Input:       ${INPUT_PATH}"
echo "  Output:      ${OUTPUT_PATH}"
echo "  Cluster:     ${CLUSTER}"
echo "  GPUs:        ${NUM_GPUS}"
echo "  Experiment:  ${EXPNAME}"
echo "  Log Dir:     ${LOG_DIR}"
echo ""

# Change to project root
cd "$(dirname "$0")/.."

# Build the Python conversion command
# Note: We explicitly DO NOT pass hf_overrides to avoid the version mismatch error
# in the container's megatron-bridge library
# Using escaped quotes to preserve them through shell expansion
PYTHON_CODE="from nemo_rl.models.megatron.community_import import export_model_from_megatron; export_model_from_megatron(hf_model_name=\\\"${MODEL_NAME}\\\", input_path=\\\"${INPUT_PATH}\\\", output_path=\\\"${OUTPUT_PATH}\\\", hf_tokenizer_path=\\\"${TOKENIZER_PATH}\\\", overwrite=True)"

echo "Submitting conversion job..."
echo ""

# Build the full command matching the original job setup:
# - Set UV_PROJECT to /opt/NeMo-RL (where nemo-rl source is)
# - Use uv run --extra mcore to get megatron dependencies
# - Run python with our conversion code
FULL_CMD="export UV_PROJECT=/opt/NeMo-RL && uv run --extra mcore python -c \"${PYTHON_CODE}\""

# Run the conversion via ns run_cmd
uv run ns run_cmd \
    --cluster "${CLUSTER}" \
    --num_gpus "${NUM_GPUS}" \
    --container "${CONTAINER}" \
    --expname "${EXPNAME}" \
    --log_dir "${LOG_DIR}" \
    --command "${FULL_CMD}"

echo ""
echo "============================================"
echo "Job submitted!"
echo "============================================"
echo ""
echo "Monitor logs at:"
echo "  ${LOG_DIR/\/workspace\//./}/"
echo ""
echo "Or check nemo-run job status:"
echo "  ls -la outputs/jobs/nemo-run/${EXPNAME}/"
echo ""
echo "Output HF model will be saved to:"
echo "  ${OUTPUT_PATH/\/workspace\//./}/"
echo ""
