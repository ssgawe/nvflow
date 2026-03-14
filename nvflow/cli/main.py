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
"""Main CLI entry point for NVFlow.

Usage:
    nflow list-stages [--config CONFIG] [--recipe RECIPE] [--workflow WORKFLOW]
    nflow stage-info STAGE_PATH [--recipe RECIPE] [--workflow WORKFLOW]
    nflow run STAGE [STAGE...] --config CONFIG
    nflow run-all --config CONFIG
    nflow validate --config CONFIG
    nflow version

Examples:
    # List all stages
    nflow list-stages

    # List stages for a specific recipe
    nflow list-stages --recipe finance

    # List stages for a specific workflow
    nflow list-stages --recipe finance --workflow training_sft

    # Get stage info (full path)
    nflow stage-info finance.training_sft.sft

    # Get stage info (with options)
    nflow stage-info sft --recipe finance --workflow training_sft

    # Run workflow
    nflow run-all --config nvflow/recipes/finance/workflows/training_sft.yaml
"""

import sys
from collections import defaultdict
from pathlib import Path
from typing import Annotated

import typer
from omegaconf import OmegaConf
from rich import print
from rich.table import Table

# Auto-discover all recipes and stages (must be after core imports)
import nvflow.recipes.finance  # noqa: E402, F401 - import for side-effect
from nvflow import __version__
from nvflow.core import BaseStage, StageRegistry, WorkflowRunner

app = typer.Typer(
    name="nflow",
    help="NVFlow - Workflow orchestration for end-to-end model training",
    add_completion=False,
)


def _get_stage_order(recipe_name: str, workflow_name: str) -> list[str] | None:
    """Get pipeline_stages order from workflow config file.

    Args:
        recipe_name: Recipe name (e.g., "finance")
        workflow_name: Workflow name (e.g., "training_sft")

    Returns:
        List of stage names in pipeline order, or None if not found
    """
    try:
        # Look for workflow config file
        workflow_dir = Path(__file__).parent.parent / "recipes" / recipe_name / "workflows"
        if not workflow_dir.exists():
            return None

        # Try to find matching workflow config (search subdirectories too)
        for config_file in workflow_dir.glob("**/*.yaml"):
            try:
                cfg = OmegaConf.load(config_file)
                cfg_workflow_name = cfg.get("workflow", {}).get("name")
                if cfg_workflow_name == workflow_name:
                    return cfg.get("pipeline_stages", [])
            except Exception:
                continue
        return None
    except Exception:
        return None


def _order_stages(stages: list[str], pipeline_order: list[str] | None) -> list[str]:
    """Order stages based on pipeline config, fallback to alphabetical.

    Args:
        stages: List of stage names to order
        pipeline_order: Desired order from config, or None

    Returns:
        Ordered list of stage names
    """
    if not pipeline_order:
        return sorted(stages)

    # Maintain pipeline order, append remaining stages alphabetically
    ordered = []
    remaining = set(stages)

    for stage in pipeline_order:
        if stage in remaining:
            ordered.append(stage)
            remaining.remove(stage)

    # Add any stages not in config (e.g., newly registered)
    ordered.extend(sorted(remaining))

    return ordered


@app.command(name="list-stages")
def list_stages(
    config: str | None = typer.Option(
        None, "--config", "-c", help="Optional: Show only stages defined in this config file"
    ),
    recipe: str | None = typer.Option(
        None, "--recipe", "-r", help="Optional: Filter by recipe (finance, retail, healthcare)"
    ),
    workflow: str | None = typer.Option(
        None, "--workflow", "-w", help="Optional: Filter by workflow (training_sft, sdg_basic)"
    ),
):
    """List all available stages or stages in a workflow config."""

    if config:
        # List stages from config file
        try:
            runner = WorkflowRunner(config)
            stages = runner.config["pipeline_stages"]

            print(f"\n[bold]Stages in {config}:[/bold]")
            print(f"Recipe: {runner.recipe}")
            print(f"Workflow: {runner.workflow_name} ({runner.workflow_type})\n")

            table = Table(show_header=True, header_style="bold cyan")
            table.add_column("#", style="dim", width=4)
            table.add_column("Stage", style="green")
            table.add_column("Dependencies", style="yellow")
            table.add_column("Full Path", style="dim")

            for idx, stage_name in enumerate(stages, 1):
                stage_config = runner.config["stages"].get(stage_name, {})
                deps = ", ".join(stage_config.get("dependencies", [])) or "None"
                full_path = f"{runner.recipe}.{runner.workflow_name}.{stage_name}"
                table.add_row(str(idx), stage_name, deps, full_path)

            print(table)
            print(f"\nTotal: {len(stages)} stages")

        except Exception as e:
            print(f"[red]Error loading config:[/red] {e}")
            sys.exit(1)
    else:
        # List all registered stages from hierarchical registry
        all_stages = StageRegistry.list_all_stages()

        # Filter by recipe and/or workflow if specified
        if recipe and workflow:
            try:
                stages = StageRegistry.list_stages(recipe, workflow)
                filtered = [(recipe, workflow, s) for s in stages]
                print(f"\n[bold]Available stages in {recipe}.{workflow}:[/bold]\n")
            except KeyError as e:
                print(f"[red]Error:[/red] {e}")
                sys.exit(1)
        elif recipe:
            try:
                workflows = StageRegistry.list_workflows(recipe)
                filtered = []
                for wf in workflows:
                    stages = StageRegistry.list_stages(recipe, wf)
                    filtered.extend([(recipe, wf, s) for s in stages])
                print(f"\n[bold]Available stages in recipe '{recipe}':[/bold]\n")
            except KeyError as e:
                print(f"[red]Error:[/red] {e}")
                sys.exit(1)
        elif workflow:
            # Search all recipes for this workflow
            filtered = [(r, wf, s) for r, wf, s in all_stages if wf == workflow]
            if filtered:
                print(f"\n[bold]Available stages in workflow '{workflow}' (all recipes):[/bold]\n")
            else:
                print(f"[yellow]No stages found for workflow '{workflow}'[/yellow]")
                return
        else:
            filtered = all_stages
            print("\n[bold]All available stages:[/bold]\n")

        if not filtered:
            print("[yellow]No stages found.[/yellow]")
            if recipe:
                print(f"\nNo stages registered for recipe: {recipe}")
            if workflow:
                print(f"\nNo stages registered for workflow: {workflow}")
            return

        # Group stages by recipe and workflow
        by_recipe: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
        for recipe_name, workflow_name, stage_name in filtered:
            by_recipe[recipe_name][workflow_name].append(stage_name)

        # Display stages organized by recipe and workflow
        for recipe_name in sorted(by_recipe.keys()):
            print(f"[bold cyan]{recipe_name}:[/bold cyan]")
            # Use workflow order from recipe config (falls back to alphabetical)
            workflows = StageRegistry.list_workflows(recipe_name)
            for workflow_name in [w for w in workflows if w in by_recipe[recipe_name]]:
                print(f"  [cyan]{workflow_name}:[/cyan]")

                stages = by_recipe[recipe_name][workflow_name]
                pipeline_order = _get_stage_order(recipe_name, workflow_name)
                ordered_stages = _order_stages(stages, pipeline_order)

                for stage_name in ordered_stages:
                    print(f"    • {stage_name}")
            print()

        total = len(filtered)
        recipes = len(by_recipe)
        workflows = sum(len(wfs) for wfs in by_recipe.values())
        print(f"Total: {total} stage(s) across {recipes} recipe(s) and {workflows} workflow(s)")


@app.command()
def run(
    stages: Annotated[
        list[str],
        typer.Argument(help="Stage name(s) to run (e.g., download generate_qa)"),
    ],
    config: Annotated[
        str, typer.Option("--config", "-c", help="Path to workflow configuration file")
    ],
):
    """Run one or more specific stages."""

    try:
        runner = WorkflowRunner(config)
        runner.run(stages=stages)
    except Exception as e:
        print(f"[red]Error:[/red] {e}")
        sys.exit(1)


@app.command(name="run-all")
def run_all(
    config: str = typer.Option(..., "--config", "-c", help="Path to workflow configuration file"),
):
    """Run all stages defined in the workflow config."""

    try:
        runner = WorkflowRunner(config)
        runner.run()  # No stages argument = run all
    except Exception as e:
        print(f"[red]Error:[/red] {e}")
        sys.exit(1)


@app.command()
def validate(
    config: str = typer.Option(..., "--config", "-c", help="Path to workflow configuration file"),
):
    """Validate a workflow configuration file."""

    try:
        runner = WorkflowRunner(config)
        runner.validate_config()
        print("\n[green]✓ Configuration is valid[/green]")
        print(f"\nRecipe: {runner.recipe}")
        print(f"Workflow: {runner.workflow_name}")
        print(f"Type: {runner.workflow_type}")
        print(f"Cluster: {runner.cluster}")
        print(f"Stages: {len(runner.config['pipeline_stages'])}")
    except Exception as e:
        print(f"[red]✗ Configuration is invalid:[/red] {e}")
        sys.exit(1)


@app.command(name="stage-info")
def stage_info(
    stage_path: str = typer.Argument(
        ...,
        help="Stage path (e.g., finance.training_sft.sft or just sft with --recipe and --workflow)",
    ),
    recipe: str | None = typer.Option(None, "--recipe", "-r", help="Recipe name (if not in path)"),
    workflow: str | None = typer.Option(
        None, "--workflow", "-w", help="Workflow name (if not in path)"
    ),
):
    """Show detailed information about a specific stage.

    Usage:
        nflow stage-info finance.training_sft.sft
        nflow stage-info sft --recipe finance --workflow training_sft
    """

    try:
        # Parse stage path
        if "." in stage_path:
            # Full path provided: recipe.workflow.stage
            parts = stage_path.split(".")
            if len(parts) == 3:
                recipe_name, workflow_name, stage_name = parts
            else:
                print(
                    f"[red]Invalid stage path:[/red] {stage_path}\n"
                    "Expected format: recipe.workflow.stage (e.g., finance.training_sft.sft)"
                )
                sys.exit(1)
        else:
            # Short name provided, need recipe and workflow options
            if not recipe or not workflow:
                print(
                    "[red]Error:[/red] When using short stage name, "
                    "you must provide --recipe and --workflow options"
                )
                print(
                    f"\nExample: nflow stage-info {stage_path} --recipe finance --workflow training_sft"
                )
                sys.exit(1)
            recipe_name = recipe
            workflow_name = workflow
            stage_name = stage_path

        # Get stage class from hierarchical registry
        stage_class = StageRegistry.get(recipe_name, workflow_name, stage_name)

        # Create instance to get methods
        stage = stage_class()

        # Display information
        full_path = f"{recipe_name}.{workflow_name}.{stage_name}"
        print(f"\n[bold cyan]Stage: {full_path}[/bold cyan]\n")

        print(f"[bold]Recipe:[/bold] {recipe_name}")
        print(f"[bold]Workflow:[/bold] {workflow_name}")
        print(f"[bold]Stage:[/bold] {stage_name}")
        print(f"[bold]Class Name:[/bold] {stage_class.__name__}")

        # Show docstring
        if stage_class.__doc__:
            print("\n[bold]Documentation:[/bold]")
            print(stage_class.__doc__)

        # Show module path
        print("\n[bold]Source File:[/bold]")
        import inspect

        source_file = inspect.getfile(stage_class)
        print(f"  {source_file}")

        # Show available methods
        print("\n[bold]Methods:[/bold]")
        print("  • execute() - Main execution logic")
        if (
            hasattr(stage, "validate_config")
            and stage.validate_config.__func__ != BaseStage.validate_config
        ):
            print("  • validate_config() - Custom validation implemented")

    except KeyError as e:
        print(f"[red]Error:[/red] {e}")
        print("\n[yellow]Use 'nflow list-stages' to see all available stages[/yellow]")
        sys.exit(1)


@app.command()
def version():
    """Show NVFlow version."""
    print(f"NVFlow version {__version__}")


def main():
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
