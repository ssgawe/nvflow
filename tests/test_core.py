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
"""Tests for core functionality."""

import json

import pytest

from nvflow.core import BaseStage, StageRegistry, WorkflowRunner
from nvflow.recipes.telco.utils.sft.prepare_sft_data import normalize_record, prepare_sft_data


def test_stage_registry_hierarchical():
    """Test hierarchical stage registration and retrieval."""

    # Clear registry for testing
    StageRegistry.clear()

    # Register a test stage
    @StageRegistry.register(recipe="test", workflow="example", stage="stage1")
    class TestStage(BaseStage):
        workflow = "example"

        def execute(self, config, cluster, expname, run_after=None):
            pass

    # Check stage is registered
    assert StageRegistry.has("test", "example", "stage1")

    # Retrieve stage
    stage_class = StageRegistry.get("test", "example", "stage1")
    assert stage_class == TestStage

    # List stages
    stages = StageRegistry.list_stages("test", "example")
    assert "stage1" in stages

    # List all stages
    all_stages = StageRegistry.list_all_stages()
    assert ("test", "example", "stage1") in all_stages


def test_stage_registry_multiple_levels():
    """Test registry with multiple recipes and workflows."""

    # Clear registry
    StageRegistry.clear()

    # Register stages in different recipes/workflows
    @StageRegistry.register(recipe="finance", workflow="training", stage="sft")
    class FinanceSFTStage(BaseStage):
        def execute(self, config, cluster, expname, run_after=None):
            pass

    @StageRegistry.register(recipe="finance", workflow="sdg", stage="generate")
    class FinanceSDGStage(BaseStage):
        def execute(self, config, cluster, expname, run_after=None):
            pass

    @StageRegistry.register(recipe="retail", workflow="training", stage="sft")
    class RetailSFTStage(BaseStage):
        def execute(self, config, cluster, expname, run_after=None):
            pass

    # Check recipes
    recipes = StageRegistry.list_recipes()
    assert "finance" in recipes
    assert "retail" in recipes

    # Check workflows
    finance_workflows = StageRegistry.list_workflows("finance")
    assert "training" in finance_workflows
    assert "sdg" in finance_workflows

    # Check stages
    assert StageRegistry.get("finance", "training", "sft") == FinanceSFTStage
    assert StageRegistry.get("finance", "sdg", "generate") == FinanceSDGStage
    assert StageRegistry.get("retail", "training", "sft") == RetailSFTStage

    # Verify same stage name can exist in different workflows
    assert StageRegistry.has("finance", "training", "sft")
    assert StageRegistry.has("retail", "training", "sft")


def test_stage_registry_error():
    """Test stage registry error handling."""

    # Clear registry
    StageRegistry.clear()

    # Try to get non-existent recipe
    with pytest.raises(KeyError):
        StageRegistry.get("nonexistent", "workflow", "stage")

    # Try to get non-existent workflow
    @StageRegistry.register(recipe="test", workflow="example", stage="stage1")
    class TestStage(BaseStage):
        def execute(self, config, cluster, expname, run_after=None):
            pass

    with pytest.raises(KeyError):
        StageRegistry.get("test", "nonexistent", "stage")

    # Try to get non-existent stage
    with pytest.raises(KeyError):
        StageRegistry.get("test", "example", "nonexistent")


def test_stage_registry_duplicate():
    """Test that duplicate registration raises error."""

    # Clear registry
    StageRegistry.clear()

    # Register first stage
    @StageRegistry.register(recipe="test", workflow="example", stage="duplicate")
    class TestStage1(BaseStage):
        def execute(self, config, cluster, expname, run_after=None):
            pass

    # Try to register with same path
    with pytest.raises(ValueError):

        @StageRegistry.register(recipe="test", workflow="example", stage="duplicate")
        class TestStage2(BaseStage):
            def execute(self, config, cluster, expname, run_after=None):
                pass


def test_stage_validation():
    """Test stage config validation."""

    @StageRegistry.register(recipe="test", workflow="example", stage="validation")
    class ValidationStage(BaseStage):
        workflow = "example"

        def execute(self, config, cluster, expname, run_after=None):
            pass

        def validate_config(self, config):
            if "required_field" not in config:
                raise ValueError("required_field is required")

    stage = ValidationStage()

    # Valid config
    stage.validate_config({"required_field": "value"})

    # Invalid config
    with pytest.raises(ValueError):
        stage.validate_config({})


def test_workflow_runner_merges_multiple_base_configs(tmp_path):
    """Test workflow config composition with scalar and list base layers."""
    base_config = tmp_path / "base.yaml"
    base_config.write_text(
        """
recipe: telco
workflow:
  name: sft
  type: training
cluster: test_cluster
base_output_dir: /tmp/base-output
data:
  task_name: base_task
  source_keys:
    - source
  target_keys:
    - target
pipeline_stages:
  - prepare_sft_data
  - training
stages:
  prepare_sft_data:
    task_name: ${data.task_name}
    source_keys: ${data.source_keys}
    target_keys: ${data.target_keys}
  training:
    output_dir: ${base_output_dir}/training
""",
        encoding="utf-8",
    )

    task_config = tmp_path / "task.yaml"
    task_config.write_text(
        """
_base_: base.yaml
data:
  task_name: ticket_summary
  source_keys:
    - ticket_text
  target_keys:
    - summary
""",
        encoding="utf-8",
    )

    model_config = tmp_path / "model.yaml"
    model_config.write_text(
        """
stages:
  training:
    model_name: Example/Model
    backend: fsdp
    num_gpus: 8
""",
        encoding="utf-8",
    )

    child_config = tmp_path / "workflow.yaml"
    child_config.write_text(
        """
_base_:
  - task.yaml
  - model.yaml
base_output_dir: /tmp/run-output
data:
  task_name: ticket_resolution
stages:
  training:
    num_gpus: 4
""",
        encoding="utf-8",
    )

    runner = WorkflowRunner(str(child_config))

    assert runner.config["data"]["task_name"] == "ticket_resolution"
    assert runner.config["stages"]["prepare_sft_data"]["task_name"] == "ticket_resolution"
    assert runner.config["stages"]["prepare_sft_data"]["source_keys"] == ["ticket_text"]
    assert runner.config["stages"]["training"]["model_name"] == "Example/Model"
    assert runner.config["stages"]["training"]["backend"] == "fsdp"
    assert runner.config["stages"]["training"]["num_gpus"] == 4
    assert runner.config["stages"]["training"]["output_dir"] == "/tmp/run-output/training"


def test_normalize_record_uses_configured_schema():
    """Test generic SFT normalization with task-specific field names."""
    normalized = normalize_record(
        {
            "ticket_text": "Base station alarm A123 is firing.",
            "summary": "Investigate alarm A123.",
            "ticket_id": "T-1",
        },
        split="train",
        source_keys=["ticket_text", "problem"],
        target_keys=["summary", "generation"],
        task_name="ticket_summary",
        metadata_keys=["ticket_id"],
    )

    assert normalized["problem"] == "Base station alarm A123 is firing."
    assert normalized["generation"] == "Investigate alarm A123."
    assert normalized["question_type"] == "ticket_summary"
    assert normalized["task"] == "ticket_summary"
    assert normalized["split"] == "train"
    assert normalized["ticket_id"] == "T-1"
    assert normalized["uuid"].startswith("train-")


def test_prepare_sft_data_writes_normalized_outputs(tmp_path):
    """Test generic SFT preparation writes train, val, chunks, and stats."""
    train_file = tmp_path / "train.jsonl"
    val_file = tmp_path / "val.jsonl"
    output_dir = tmp_path / "out"

    train_records = [
        {"ticket_text": "Alarm A", "summary": "Handle A", "ticket_id": "T-1"},
        {"ticket_text": "Alarm B", "summary": "Handle B", "ticket_id": "T-2"},
    ]
    val_records = [{"ticket_text": "Alarm C", "summary": "Handle C", "ticket_id": "T-3"}]

    train_file.write_text(
        "".join(json.dumps(record) + "\n" for record in train_records),
        encoding="utf-8",
    )
    val_file.write_text(
        "".join(json.dumps(record) + "\n" for record in val_records),
        encoding="utf-8",
    )

    counts = prepare_sft_data(
        train_file=train_file,
        val_file=val_file,
        output_dir=output_dir,
        num_chunks=2,
        source_keys=["ticket_text"],
        target_keys=["summary"],
        task_name="ticket_summary",
        metadata_keys=["ticket_id"],
    )

    assert counts == {"train": 2, "val": 1, "test": 0}
    assert (output_dir / "chunks" / "chunk_0.jsonl").exists()
    assert (output_dir / "chunks" / "chunk_1.jsonl").exists()

    train_output = [
        json.loads(line) for line in (output_dir / "train.jsonl").read_text().splitlines()
    ]
    val_output = [
        json.loads(line) for line in (output_dir / "val.jsonl").read_text().splitlines()
    ]
    stats_output = [
        json.loads(line) for line in (output_dir / "stats.jsonl").read_text().splitlines()
    ]

    assert train_output[0]["problem"] == "Alarm A"
    assert train_output[0]["generation"] == "Handle A"
    assert train_output[0]["task"] == "ticket_summary"
    assert val_output[0]["split"] == "val"
    assert stats_output == [counts]
