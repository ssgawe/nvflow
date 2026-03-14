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

import pytest

from nvflow.core import BaseStage, StageRegistry


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
