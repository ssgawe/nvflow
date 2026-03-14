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
"""Finance-agent generation module (nvflow integration adapter).

Thin adapter that wires the finance-agent Agent class into nemo-skills'
GenerationTask framework.  The core agent loop lives in agent.py (copied
from finance-agent/src/agent.py with minimal adaptations).

This file is the nvflow counterpart of finance-agent/src/run_agent.py:
it handles LLM creation, tool instantiation, Hydra config, and result
formatting -- but delegates the actual agent loop to Agent.run().
"""

from __future__ import annotations

import asyncio
import logging
import sys
from dataclasses import asdict, field
from pathlib import Path
from typing import Any

import hydra
import yaml
from nemo_skills.inference.generate import (
    GenerationTask,
    GenerationTaskConfig,
    InferenceConfig,
)
from nemo_skills.inference.model import server_params
from nemo_skills.utils import (
    get_help_message,
    get_logger_name,
    nested_dataclass,
    setup_logging,
)

from .agent import Agent

LOG = logging.getLogger(get_logger_name(__file__))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_TURNS_DEFAULT = 50
_DEFAULT_MAX_TOKENS = 32_768


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@nested_dataclass(kw_only=True)
class FinanceAgentConfig(GenerationTaskConfig):
    """Finance-agent generation config."""

    inference: InferenceConfig = field(default_factory=InferenceConfig)
    server: dict = field(default_factory=dict)

    max_turns: int = MAX_TURNS_DEFAULT
    # Comma-separated tool names to enable, or None = all. Options: web_search, retrieve_information, parse_html_page, edgar_search
    enabled_tools: str | None = None
    # Path to YAML file with agent instructions prompt (must contain a 'user' key).
    # Consistent with prompt_config used in other eval workflows.
    agent_prompt_config: str | None = None

    def _post_init_validate_params(self):
        if self.prompt_format not in ["ns", "openai"]:
            raise ValueError(f"prompt_format must be 'ns' or 'openai', got '{self.prompt_format}'")
        if self.prompt_format == "openai":
            assert self.prompt_config is None, "prompt_config not supported for openai format"
        for param, default_value in self._get_disallowed_params():
            if getattr(self, param) != default_value:
                raise ValueError(f"{param} must be {default_value}")

    def _get_disallowed_params(self):
        return [("prompt_config", None)]


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_finance_agent_config", node=FinanceAgentConfig)


# ---------------------------------------------------------------------------
# Generation task
# ---------------------------------------------------------------------------


class FinanceAgentGenerationTask(GenerationTask):
    """Multi-turn finance-agent evaluation following the BFCL pattern."""

    def __init__(self, cfg: FinanceAgentConfig):
        from omegaconf import DictConfig, open_dict

        if isinstance(cfg, DictConfig):
            with open_dict(cfg):
                if cfg.chat_template_kwargs:
                    cfg.inference.extra_body.chat_template_kwargs = cfg.chat_template_kwargs
                    cfg.chat_template_kwargs = None

        elif cfg.chat_template_kwargs:
            eb = cfg.inference.extra_body
            if isinstance(eb, DictConfig):
                with open_dict(eb):
                    eb.chat_template_kwargs = cfg.chat_template_kwargs
            else:
                eb["chat_template_kwargs"] = (
                    dict(cfg.chat_template_kwargs)
                    if hasattr(cfg.chat_template_kwargs, "items")
                    else cfg.chat_template_kwargs
                )
                cfg.inference.extra_body = eb
            cfg.chat_template_kwargs = None

        super().__init__(cfg)

        self._model_name = self.llm.model_name_or_path
        base_url = self.llm.base_url
        api_key = self.llm.litellm_kwargs.get("api_key", "EMPTY")

        from omegaconf import DictConfig, OmegaConf

        if isinstance(cfg.inference, DictConfig):
            inf = OmegaConf.to_container(cfg.inference, resolve=True)
        else:
            inf = asdict(cfg.inference)

        extra_body: dict[str, Any] = dict(inf.get("extra_body", {}) or {})
        top_k = inf.get("top_k", -1)
        if top_k > 0:
            extra_body["top_k"] = top_k
        min_p = inf.get("min_p", 0.0)
        if min_p > 0:
            extra_body["min_p"] = min_p
        rep_penalty = inf.get("repetition_penalty", 1.0)
        if rep_penalty != 1.0:
            extra_body["repetition_penalty"] = rep_penalty

        from model_library.base import DelegateConfig, LLMConfig
        from model_library.providers.openai import OpenAIModel
        from pydantic import SecretStr

        llm_config = LLMConfig(
            max_tokens=inf.get("tokens_to_generate") or _DEFAULT_MAX_TOKENS,
            temperature=inf.get("temperature", 0.0),
            reasoning=True,
            supports_tools=True,
        )
        delegate_config = DelegateConfig(
            base_url=base_url,
            api_key=SecretStr(api_key),
        )
        self._ml_llm = OpenAIModel(
            model_name=self._model_name,
            provider="vllm",
            config=llm_config,
            use_completions=True,
            delegate_config=delegate_config,
        )
        self._query_kwargs: dict[str, Any] = {}
        if extra_body:
            self._query_kwargs["extra_body"] = extra_body

        if self.cfg.agent_prompt_config:
            prompt_path = Path(self.cfg.agent_prompt_config)
            if not prompt_path.is_file():
                raise FileNotFoundError(f"agent_prompt_config not found: {prompt_path}")
            prompt_data = yaml.safe_load(prompt_path.read_text())
            self._instructions_prompt = prompt_data["user"]
            LOG.info("Loaded agent instructions from %s", prompt_path)
        else:
            raise ValueError(
                "agent_prompt_config is required. "
                "Set ++agent_prompt_config=/path/to/finance_agent_instructions.yaml"
            )

        from .tools import (
            VALID_TOOLS,
            EDGARSearch,
            ParseHtmlPage,
            RetrieveInformation,
            SubmitFinalResult,
            TavilyWebSearch,
        )

        all_tools = {
            "web_search": TavilyWebSearch(),
            "retrieve_information": RetrieveInformation(),
            "parse_html_page": ParseHtmlPage(),
            "edgar_search": EDGARSearch(),
            "submit_final_result": SubmitFinalResult(),
        }
        # Filter by enabled_tools if set (__ or ; or comma-separated string)
        if self.cfg.enabled_tools:
            raw = str(self.cfg.enabled_tools)
            names = [
                n.strip() for n in raw.replace("__", ",").replace(";", ",").split(",") if n.strip()
            ]
            invalid = [n for n in names if n not in VALID_TOOLS and n != "submit_final_result"]
            if invalid:
                raise ValueError(
                    f"enabled_tools: invalid {invalid}. Valid options: {VALID_TOOLS}, submit_final_result (always included)"
                )
            self._tools = {
                k: v for k, v in all_tools.items() if k in names or k == "submit_final_result"
            }
        else:
            self._tools = all_tools

        self._agent_semaphore = asyncio.Semaphore(max(self.cfg.max_concurrent_requests, 1))

    def setup_prompt(self):
        return None

    def log_example_prompt(self, data):
        return

    # -- main entry point ---------------------------------------------------

    async def process_single_datapoint(self, data_point, all_data):
        """Run the full finance-agent loop for one question.

        Creates an Agent instance, delegates to Agent.run(), and converts
        the result into the dict format expected by nemo-skills.
        """
        question = data_point["problem"]

        agent = Agent(
            tools=self._tools,
            llm=self._ml_llm,
            max_turns=self.cfg.max_turns,
            instructions_prompt=self._instructions_prompt,
            query_kwargs=self._query_kwargs,
        )

        async with self._agent_semaphore:
            final_answer, metadata = await agent.run(question)

        return {
            "generation": final_answer,
            "final_answer": final_answer,
            "num_generated_tokens": metadata.total_tokens.out_tokens or 0,
            "num_turns": len(metadata.turns),
        }


GENERATION_TASK_CLASS = FinanceAgentGenerationTask


# ---------------------------------------------------------------------------
# Hydra entry point (for standalone usage / debugging)
# ---------------------------------------------------------------------------


@hydra.main(version_base=None, config_name="base_finance_agent_config")
def finance_agent_generation(cfg: FinanceAgentConfig):
    cfg = FinanceAgentConfig(_init_nested=True, **cfg)
    LOG.info("Config: %s", cfg)
    FinanceAgentGenerationTask(cfg).generate()


HELP_MESSAGE = get_help_message(
    FinanceAgentConfig,
    server_params=server_params(),
)

if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print(HELP_MESSAGE)
    else:
        setup_logging()
        finance_agent_generation()
