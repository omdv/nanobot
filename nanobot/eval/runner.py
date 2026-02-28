#!/usr/bin/env python3
"""
Evaluation harness for testing agent behavior across different models.

Usage:
    # Run with models from file (default: models.txt)
    python -m nanobot.eval.runner --models-file
    python -m nanobot.eval.runner -f custom_models.txt

    # Run with specific models
    python -m nanobot.eval.runner -m anthropic/claude-sonnet-4,openai/gpt-4o

    # Run specific test cases
    python -m nanobot.eval.runner -f --cases greeting,file_write

    # Keep workspace for debugging
    python -m nanobot.eval.runner -f --keep-workspace
"""

import argparse
import asyncio
import json
import shutil
import statistics
import tempfile
import time
from datetime import datetime
from pathlib import Path

import yaml

from nanobot.config.loader import load_config
from nanobot.providers.litellm_provider import LiteLLMProvider
from nanobot.agent.loop import AgentLoop
from nanobot.bus.queue import MessageBus
from nanobot.session.manager import SessionManager
from nanobot.cron.service import CronService


CASES_FILE = Path(__file__).parent / "cases.yaml"
RESULTS_DIR = Path(__file__).parent / "results"
DEFAULT_MODELS_FILE = Path(__file__).parent / "models.txt"


def load_models_from_file(file_path: Path) -> list[str]:
    """Load model list from a text file (one per line, # for comments)."""
    models = []
    with open(file_path) as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if line and not line.startswith("#"):
                models.append(line)
    return models


def verify_file_check(workspace: Path, verify_spec: dict | None) -> tuple[bool, str]:
    """Verify a file exists and optionally contains expected content."""
    if not verify_spec or "path" not in verify_spec:
        return False, "Invalid verify_file spec"

    file_path = workspace / verify_spec["path"]
    if not file_path.exists():
        return False, f"File not found: {verify_spec['path']}"

    if "contains" in verify_spec and verify_spec["contains"]:
        try:
            content = file_path.read_text(encoding="utf-8")
            if verify_spec["contains"].lower() not in content.lower():
                return False, f"File missing expected content: {verify_spec['contains']}"
        except Exception as e:
            return False, f"Error reading file: {e}"

    return True, "OK"


def verify_cron_check(workspace: Path, verify_spec: dict | None) -> tuple[bool, str]:
    """Verify a cron job was created with matching criteria."""
    if not verify_spec:
        return False, "Invalid verify_cron spec"

    cron_store = workspace / "cron" / "jobs.json"
    if not cron_store.exists():
        return False, "Cron store not found"

    try:
        with open(cron_store) as f:
            data = json.load(f)

        jobs = data.get("jobs") or []
        if not jobs:
            return False, "No cron jobs found"

        if "name_contains" in verify_spec and verify_spec["name_contains"]:
            search = verify_spec["name_contains"].lower()
            found = any(search in (j.get("name") or "").lower() for j in jobs)
            if not found:
                return False, f"No job with name containing: {verify_spec['name_contains']}"

        return True, "OK"
    except Exception as e:
        return False, f"Error reading cron store: {e}"


def load_cases(filter_ids: list[str] | None = None) -> list[dict]:
    """Load test cases from YAML file."""
    with open(CASES_FILE) as f:
        data = yaml.safe_load(f)

    tasks = data.get("tasks", [])
    if filter_ids:
        tasks = [t for t in tasks if t["id"] in filter_ids]
    return tasks


def make_provider(config, model: str) -> LiteLLMProvider:
    """Create a provider for a specific model."""
    p = config.get_provider(model)
    if not p or not p.api_key:
        raise ValueError(f"No API key configured for model: {model}")

    return LiteLLMProvider(
        api_key=p.api_key,
        api_base=config.get_api_base(model),
        default_model=model,
        extra_headers=p.extra_headers,
        provider_name=config.get_provider_name(model),
    )


def make_agent(config, provider: LiteLLMProvider, model: str, workspace: Path) -> AgentLoop:
    """Create an agent loop for evaluation with isolated workspace."""
    bus = MessageBus()

    # Create isolated session manager
    session_manager = SessionManager(workspace)
    session_manager.sessions_dir = workspace / "sessions"
    session_manager.sessions_dir.mkdir(parents=True, exist_ok=True)

    # Create isolated cron service
    cron_store_path = workspace / "cron" / "jobs.json"
    cron_store_path.parent.mkdir(parents=True, exist_ok=True)
    cron_service = CronService(cron_store_path)

    return AgentLoop(
        bus=bus,
        provider=provider,
        workspace=workspace,
        model=model,
        max_iterations=config.agents.defaults.max_tool_iterations,
        memory_window=config.agents.defaults.memory_window,
        brave_api_key=config.tools.web.search.api_key or None,
        exec_config=config.tools.exec,
        restrict_to_workspace=config.tools.restrict_to_workspace,
        thinking_config=config.agents.defaults.thinking,
        session_manager=session_manager,
        cron_service=cron_service,
    )


async def run_task(agent: AgentLoop, task: dict, session_key: str, workspace: Path) -> dict:
    """Run a single task and capture results."""
    task_id = task["id"]
    prompt = task["prompt"]

    start_time = time.perf_counter()

    try:
        response = await agent.process_direct(prompt, session_key=session_key)
        response = response or ""  # Handle None response
        error = None
    except Exception as e:
        response = ""
        error = str(e)

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    # Get session stats
    session = agent.sessions.get_or_create(session_key)
    usage = session.metadata.get("usage", {}) or {}

    # Get tools used from last assistant message
    tools_used = []
    if session.messages:
        last_msg = session.messages[-1]
        if last_msg.get("role") == "assistant":
            tools_used = last_msg.get("tools_used") or []  # Handle None value

    result = {
        "task_id": task_id,
        "prompt": prompt.strip(),
        "response": response,
        "error": error,
        "elapsed_ms": round(elapsed_ms, 2),
        "tools_used": tools_used,
        "session_messages": len(session.messages),
        "usage_snapshot": {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
            "cost": round(usage.get("cost", 0.0), 6),
        },
    }

    # Check expectations
    if "expect_contains" in task and task["expect_contains"]:
        result["expect_contains"] = task["expect_contains"]
        result["pass_contains"] = task["expect_contains"].lower() in response.lower()

    if "expect_tools" in task and task["expect_tools"]:
        result["expect_tools"] = task["expect_tools"]
        result["pass_tools"] = all(t in tools_used for t in task["expect_tools"])

    # File verification
    if "verify_file" in task:
        passed, msg = verify_file_check(workspace, task["verify_file"])
        result["verify_file"] = task["verify_file"]
        result["pass_file"] = passed
        result["file_check_msg"] = msg

    # Cron verification
    if "verify_cron" in task:
        passed, msg = verify_cron_check(workspace, task["verify_cron"])
        result["verify_cron"] = task["verify_cron"]
        result["pass_cron"] = passed
        result["cron_check_msg"] = msg

    return result


async def evaluate_model(
    config, model: str, tasks: list[dict], base_workspace: Path, verbose: bool = True
) -> dict:
    """Run all tasks for a single model with isolated workspace."""
    if verbose:
        print(f"\n{'='*60}")
        print(f"Evaluating: {model}")
        print(f"{'='*60}")

    # Each model gets its own workspace subdirectory for full isolation
    model_safe = model.replace("/", "_").replace(":", "_")
    workspace = base_workspace / model_safe
    workspace.mkdir(parents=True, exist_ok=True)

    provider = make_provider(config, model)
    agent = make_agent(config, provider, model, workspace)

    # Session key (workspace is already isolated, so simple key is fine)
    session_key = "eval"

    results = []
    for task in tasks:
        if verbose:
            print(f"\n[{task['id']}] {task.get('description', task['prompt'][:50])}")

        result = await run_task(agent, task, session_key, workspace)
        results.append(result)

        if verbose:
            status_parts = []

            # Check for errors first
            if result.get("error"):
                status_parts.append("ERROR")
            else:
                status_parts.append("OK")

            # Response content check
            if "pass_contains" in result:
                status_parts.append(f"content:{'PASS' if result['pass_contains'] else 'FAIL'}")

            # Tools check
            if "pass_tools" in result:
                status_parts.append(f"tools:{'PASS' if result['pass_tools'] else 'FAIL'}")

            # File verification
            if "pass_file" in result:
                status_parts.append(f"file:{'PASS' if result['pass_file'] else 'FAIL'}")

            # Cron verification
            if "pass_cron" in result:
                status_parts.append(f"cron:{'PASS' if result['pass_cron'] else 'FAIL'}")

            status = " | ".join(status_parts)
            print(f"  -> {status} ({result['elapsed_ms']:.0f}ms, {result['usage_snapshot']['total_tokens']} tokens)")

            if result.get("error"):
                print(f"  ERROR: {result['error']}")
            if result.get("file_check_msg") and not result.get("pass_file"):
                print(f"  FILE: {result['file_check_msg']}")
            if result.get("cron_check_msg") and not result.get("pass_cron"):
                print(f"  CRON: {result['cron_check_msg']}")

    # Final session stats
    session = agent.sessions.get_or_create(session_key)
    final_usage = session.metadata.get("usage", {})

    # Count passes and failures
    def task_passed(r: dict) -> bool:
        if r.get("error"):
            return False
        if "pass_contains" in r and not r["pass_contains"]:
            return False
        if "pass_tools" in r and not r["pass_tools"]:
            return False
        if "pass_file" in r and not r["pass_file"]:
            return False
        if "pass_cron" in r and not r["pass_cron"]:
            return False
        return True

    passed = sum(1 for r in results if task_passed(r))
    failed = len(results) - passed

    # Calculate timing statistics
    times = [r["elapsed_ms"] for r in results]
    total_ms = sum(times)
    mean_ms = statistics.mean(times) if times else 0
    min_ms = min(times) if times else 0
    max_ms = max(times) if times else 0
    stddev_ms = statistics.stdev(times) if len(times) > 1 else 0

    return {
        "model": model,
        "session_key": session_key,
        "workspace": str(workspace),
        "timestamp": datetime.now().isoformat(),
        "tasks": results,
        "summary": {
            "total_tasks": len(results),
            "passed": passed,
            "failed": failed,
            "errors": sum(1 for r in results if r.get("error")),
            "total_tokens": final_usage.get("total_tokens", 0),
            "total_cost": round(final_usage.get("cost", 0.0), 6),
            "timing": {
                "total_ms": round(total_ms, 2),
                "mean_ms": round(mean_ms, 2),
                "min_ms": round(min_ms, 2),
                "max_ms": round(max_ms, 2),
                "stddev_ms": round(stddev_ms, 2),
            },
        },
    }


def save_results(results: list[dict], run_id: str):
    """Save results to JSON files."""
    run_dir = RESULTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save per-model results
    for r in results:
        model_safe = r["model"].replace("/", "_").replace(":", "_")
        with open(run_dir / f"{model_safe}.json", "w") as f:
            json.dump(r, f, indent=2)

    # Generate summary markdown
    summary_lines = [
        f"# Evaluation Run: {run_id}",
        f"\nGenerated: {datetime.now().isoformat()}",
        "",
        "## Summary",
        "",
        "| Model | Passed | Tokens | Cost | Total | Mean |",
        "|-------|--------|--------|------|-------|------|",
    ]

    for r in results:
        s = r["summary"]
        t = s["timing"]
        summary_lines.append(
            f"| {r['model']} | {s['passed']}/{s['total_tasks']} | "
            f"{s['total_tokens']:,} | ${s['total_cost']:.4f} | "
            f"{t['total_ms']/1000:.1f}s | {t['mean_ms']/1000:.1f}s |"
        )

    summary_lines.extend([
        "",
        "## Task Results",
        "",
    ])

    # Task comparison table
    if results:
        task_ids = [t["task_id"] for t in results[0]["tasks"]]
        header = "| Task | " + " | ".join(r["model"].split("/")[-1] for r in results) + " |"
        separator = "|------|" + "|".join(["------"] * len(results)) + "|"
        summary_lines.extend([header, separator])

        for task_id in task_ids:
            row = f"| {task_id} |"
            for r in results:
                task = next((t for t in r["tasks"] if t["task_id"] == task_id), None)
                if task:
                    # Determine if any check failed
                    failed = False
                    if task.get("error"):
                        failed = True
                    if "pass_contains" in task and not task["pass_contains"]:
                        failed = True
                    if "pass_tools" in task and not task["pass_tools"]:
                        failed = True
                    if "pass_file" in task and not task["pass_file"]:
                        failed = True
                    if "pass_cron" in task and not task["pass_cron"]:
                        failed = True

                    if task.get("error"):
                        status = "ERROR"
                    elif failed:
                        status = "FAIL"
                    else:
                        status = f"{task['elapsed_ms']:.0f}ms"
                    row += f" {status} |"
                else:
                    row += " - |"
            summary_lines.append(row)

    with open(run_dir / "summary.md", "w") as f:
        f.write("\n".join(summary_lines))

    print(f"\nResults saved to: {run_dir}")


async def main():
    parser = argparse.ArgumentParser(
        description="Evaluate agent across models",
        epilog="Examples:\n"
               "  python -m nanobot.eval.runner --models-file\n"
               "  python -m nanobot.eval.runner -m anthropic/claude-sonnet-4,openai/gpt-4o\n"
               "  python -m nanobot.eval.runner -f custom_models.txt --cases greeting,file_write",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--models", "-m",
        help="Comma-separated list of models to evaluate"
    )
    parser.add_argument(
        "--models-file", "-f",
        nargs="?",
        const=str(DEFAULT_MODELS_FILE),
        help=f"Load models from file (default: {DEFAULT_MODELS_FILE.name})"
    )
    parser.add_argument(
        "--cases", "-c",
        help="Comma-separated list of case IDs to run (default: all)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose output"
    )
    parser.add_argument(
        "--keep-workspace",
        action="store_true",
        help="Keep temporary workspace after run (for inspection)"
    )
    args = parser.parse_args()

    # Load models from file or command line
    if args.models_file:
        models_path = Path(args.models_file)
        if not models_path.exists():
            print(f"Models file not found: {models_path}")
            return
        models = load_models_from_file(models_path)
        print(f"Loaded {len(models)} models from {models_path}")
    elif args.models:
        models = [m.strip() for m in args.models.split(",")]
    else:
        print("Error: Either --models or --models-file is required")
        parser.print_help()
        return

    if not models:
        print("No models to evaluate!")
        return

    case_filter = [c.strip() for c in args.cases.split(",")] if args.cases else None

    config = load_config()
    tasks = load_cases(case_filter)

    if not tasks:
        print("No tasks to run!")
        return

    # Create isolated workspace for this eval run
    workspace = Path(tempfile.mkdtemp(prefix="nanobot-eval-"))
    print(f"Using isolated workspace: {workspace}")
    print(f"Running {len(tasks)} tasks across {len(models)} models")

    results = []
    for model in models:
        try:
            result = await evaluate_model(
                config, model, tasks, workspace, verbose=not args.quiet
            )
            results.append(result)
        except Exception as e:
            print(f"ERROR evaluating {model}: {e}")
            results.append({
                "model": model,
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "tasks": [],
                "summary": {"total_tasks": 0, "errors": 1, "total_tokens": 0, "total_cost": 0, "total_elapsed_ms": 0},
            })

    # Save results
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_results(results, run_id)

    # Print final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    for r in results:
        s = r["summary"]
        t = s["timing"]
        print(f"{r['model']}: {s['passed']}/{s['total_tasks']} passed, "
              f"{s['total_tokens']:,} tokens, ${s['total_cost']:.4f}, "
              f"mean {t['mean_ms']/1000:.1f}s (±{t['stddev_ms']/1000:.2f}s)")

    # Cleanup workspace
    if args.keep_workspace:
        print(f"\nWorkspace kept at: {workspace}")
    else:
        shutil.rmtree(workspace)
        print("\nWorkspace cleaned up")


if __name__ == "__main__":
    asyncio.run(main())
