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
"""SECQUE: A Benchmark for Evaluating Real-World Financial Analysis Capabilities.

Reference: https://arxiv.org/pdf/2504.04596
Dataset: https://huggingface.co/datasets/nogabenyoash/SecQue
"""

# Dataset group (following nemo-skills pattern)
DATASET_GROUP = "finance"

# Dataset split to use for evaluation
EVAL_SPLIT = "eval"

# Custom 3-tier metrics for finance benchmarks
# Parses Rating: [[X]] format (0=incorrect, 1=partial, 2=correct)
# Reports: judge_correct (strict), judge_lenient (lenient), judge_partial
METRICS_TYPE = "nvflow.recipes.finance.datasets.finance_metrics::FinanceMetrics"

# Code execution sandbox not required
REQUIRES_SANDBOX = False
