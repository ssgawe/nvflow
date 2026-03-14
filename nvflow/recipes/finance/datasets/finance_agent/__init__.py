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
"""vals-ai/finance-agent benchmark.

A multi-turn agentic financial QA benchmark requiring tool use
(web search, SEC EDGAR, HTML parsing) to answer 50 public questions.

Reference: https://github.com/vals-ai/finance-agent
Data:      https://github.com/vals-ai/finance-agent/tree/main/data
"""

DATASET_GROUP = "finance"

EVAL_SPLIT = "eval"

METRICS_TYPE = "nvflow.recipes.finance.datasets.finance_metrics::FinanceMetrics"

REQUIRES_SANDBOX = False

GENERATION_MODULE = "nvflow.recipes.finance.datasets.finance_agent.agent_gen"

GENERATION_ARGS = "++prompt_format=openai"
