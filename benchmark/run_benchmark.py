#!/usr/bin/env python3
"""
SGLang vs vLLM A/B Benchmark Harness for DeepSeek-V2-Lite-Chat.

Launches inference servers, runs identical workloads, collects per-token
telemetry, and outputs a side-by-side comparison report.

Usage:
    # Full comparison (both frameworks, all scenarios)
    python run_benchmark.py --model-path /workspace/models/DeepSeek-V2-Lite-Chat

    # Single framework
    python run_benchmark.py --model-path /workspace/models/DeepSeek-V2-Lite-Chat --framework sglang

    # Single scenario
    python run_benchmark.py --model-path /workspace/models/DeepSeek-V2-Lite-Chat --scenario prefix_heavy

    # Cross-validate against upstream bench_serving.py
    python run_benchmark.py --model-path /workspace/models/DeepSeek-V2-Lite-Chat --validate
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import platform
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from openai import AsyncOpenAI

log = logging.getLogger("benchmark")
log.setLevel(logging.INFO)

_console_handler = logging.StreamHandler()
_console_handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
)
log.addHandler(_console_handler)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_PORT = 30000
HEALTH_TIMEOUT_S = 600
HEALTH_POLL_INTERVAL_S = 2
WARMUP_REQUESTS = 10
REQUEST_TIMEOUT_S = 120
SERVER_SHUTDOWN_WAIT_S = 10
GPU_RELEASE_POLL_S = 5
GPU_RELEASE_TIMEOUT_S = 60
GPU_MEMORY_THRESHOLD_MB = 1000
CONCURRENCY_LEVELS = [8, 16, 32, 64]

SCENARIO_ORDER = ["prefix_heavy", "multi_turn", "high_concurrency", "long_context"]

# Venv Python paths for each framework (RunPod layout from setup.sh).
# Each framework's server launches with its own venv's Python so both
# can coexist even though they have conflicting dependencies.
VENV_PYTHON = {
    "sglang": "/workspace/sglang-env/bin/python",
    "vllm": "/workspace/vllm-env/bin/python",
}


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------


@dataclass
class PromptRequest:
    """A single benchmark request loaded from prompts.json."""

    request_id: int
    messages: list[dict[str, str]]
    max_tokens: int


@dataclass
class LoadedScenario:
    """A scenario with its pre-generated prompts and config."""

    name: str
    description: str
    concurrency: int
    requests: list[PromptRequest]


def load_prompts(prompts_path: Path) -> dict:
    """Load the full prompts.json file."""
    with open(prompts_path) as f:
        return json.load(f)


def load_scenario(prompts_data: dict, scenario_name: str) -> LoadedScenario:
    """Load a single scenario from the prompts data."""
    if scenario_name not in prompts_data:
        available = [k for k in prompts_data if k != "warmup"]
        raise KeyError(
            f"Scenario '{scenario_name}' not found in prompts.json. "
            f"Available: {available}"
        )

    raw = prompts_data[scenario_name]
    requests = [
        PromptRequest(
            request_id=r["request_id"],
            messages=r["messages"],
            max_tokens=r["max_tokens"],
        )
        for r in raw["requests"]
    ]

    return LoadedScenario(
        name=scenario_name,
        description=raw.get("description", ""),
        concurrency=raw.get("concurrency", 1),
        requests=requests,
    )


def load_warmup(prompts_data: dict) -> list[PromptRequest]:
    """Load warm-up requests from prompts data."""
    if "warmup" not in prompts_data:
        raise KeyError("'warmup' key not found in prompts.json")

    raw = prompts_data["warmup"]
    return [
        PromptRequest(
            request_id=r["request_id"],
            messages=r["messages"],
            max_tokens=r["max_tokens"],
        )
        for r in raw["requests"]
    ]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class RequestResult:
    """Timing data for a single request."""

    request_id: int
    ttft_ms: float
    tpot_ms: float
    e2e_latency_ms: float
    tokens_generated: int
    error: Optional[str] = None


@dataclass
class ScenarioResult:
    """Aggregated results for one scenario on one framework."""

    framework: str
    scenario: str
    requests: list[RequestResult] = field(default_factory=list)
    total_tokens: int = 0
    wall_clock_s: float = 0.0

    ttft_p50_ms: float = 0.0
    ttft_p90_ms: float = 0.0
    ttft_p99_ms: float = 0.0
    tpot_p50_ms: float = 0.0
    tpot_p90_ms: float = 0.0
    tpot_p99_ms: float = 0.0
    throughput_tok_s: float = 0.0
    e2e_p50_ms: float = 0.0
    e2e_p90_ms: float = 0.0
    error_count: int = 0


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------


def _python_for(framework: str) -> str:
    """Return the Python executable for a framework's venv.

    Uses the venv path from VENV_PYTHON if it exists on disk,
    otherwise falls back to sys.executable (e.g. local testing).
    """
    venv_path = VENV_PYTHON.get(framework)
    if venv_path and Path(venv_path).exists():
        return venv_path
    return sys.executable


def build_server_cmd(framework: str, model_path: str, port: int) -> list[str]:
    python = _python_for(framework)
    if framework == "sglang":
        return [
            python, "-m", "sglang.launch_server",
            "--model-path", model_path,
            "--host", "127.0.0.1",
            "--port", str(port),
            "--mem-fraction-static", "0.8",
        ]
    elif framework == "vllm":
        return [
            python, "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_path,
            "--host", "127.0.0.1",
            "--port", str(port),
            "--trust-remote-code",
            "--gpu-memory-utilization", "0.8",
        ]
    else:
        raise ValueError(f"Unknown framework: {framework}")


def launch_server(
    framework: str, model_path: str, port: int, log_dir: Path | None = None,
) -> tuple[subprocess.Popen, Path | None]:
    cmd = build_server_cmd(framework, model_path, port)
    log.info("Launching %s server: %s", framework, " ".join(cmd))

    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        server_log = log_dir / f"{framework}_{ts}.log"
        log.info("Server log: %s", server_log)
        log_f = open(server_log, "w")
        proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)
        return proc, server_log

    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE), None


async def wait_for_health(port: int, timeout_s: int = HEALTH_TIMEOUT_S) -> bool:
    client = AsyncOpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="dummy")
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            await client.models.list()
            log.info("Server healthy on port %d", port)
            return True
        except Exception:
            await asyncio.sleep(HEALTH_POLL_INTERVAL_S)
    return False


def kill_server(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    log.info("Sending SIGTERM to server (pid %d)", proc.pid)
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=SERVER_SHUTDOWN_WAIT_S)
        log.info("Server exited cleanly")
    except subprocess.TimeoutExpired:
        log.warning("Server did not exit, sending SIGKILL")
        proc.kill()
        proc.wait(timeout=5)


def get_server_stderr(proc: subprocess.Popen) -> str:
    if proc.stderr:
        try:
            return proc.stderr.read().decode("utf-8", errors="replace")[-2000:]
        except Exception:
            return "(could not read stderr)"
    return ""


async def wait_for_gpu_release() -> None:
    deadline = time.monotonic() + GPU_RELEASE_TIMEOUT_S
    while time.monotonic() < deadline:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            mem_used = int(result.stdout.strip().split("\n")[0])
            if mem_used < GPU_MEMORY_THRESHOLD_MB:
                log.info("GPU memory released (%d MB used)", mem_used)
                return
        except (FileNotFoundError, ValueError, subprocess.TimeoutExpired):
            log.info("nvidia-smi not available, waiting 10s")
            await asyncio.sleep(10)
            return
        await asyncio.sleep(GPU_RELEASE_POLL_S)
    log.warning("GPU memory did not drop below %d MB within %ds",
                GPU_MEMORY_THRESHOLD_MB, GPU_RELEASE_TIMEOUT_S)


# ---------------------------------------------------------------------------
# Request execution
# ---------------------------------------------------------------------------


async def send_request(
    client: AsyncOpenAI,
    req: PromptRequest,
    model: str,
) -> RequestResult:
    """Send a single streaming chat completion and record timing."""
    t_start = time.perf_counter()
    t_first_token = None
    token_timestamps: list[float] = []
    tokens_generated = 0

    try:
        stream = await asyncio.wait_for(
            client.chat.completions.create(
                model=model,
                messages=req.messages,
                max_tokens=req.max_tokens,
                stream=True,
                temperature=0.0,
            ),
            timeout=REQUEST_TIMEOUT_S,
        )

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                now = time.perf_counter()
                if t_first_token is None:
                    t_first_token = now
                token_timestamps.append(now)
                tokens_generated += 1

    except asyncio.TimeoutError:
        return RequestResult(
            request_id=req.request_id, ttft_ms=0, tpot_ms=0,
            e2e_latency_ms=0, tokens_generated=0,
            error=f"Request timed out after {REQUEST_TIMEOUT_S}s",
        )
    except Exception as e:
        return RequestResult(
            request_id=req.request_id, ttft_ms=0, tpot_ms=0,
            e2e_latency_ms=0, tokens_generated=0, error=str(e),
        )

    t_end = time.perf_counter()

    if t_first_token is None or tokens_generated == 0:
        return RequestResult(
            request_id=req.request_id, ttft_ms=0, tpot_ms=0,
            e2e_latency_ms=(t_end - t_start) * 1000,
            tokens_generated=0, error="No tokens generated",
        )

    ttft_ms = (t_first_token - t_start) * 1000
    e2e_ms = (t_end - t_start) * 1000

    if len(token_timestamps) > 1:
        inter_token = [
            (token_timestamps[i] - token_timestamps[i - 1]) * 1000
            for i in range(1, len(token_timestamps))
        ]
        tpot_ms = float(np.mean(inter_token))
    else:
        tpot_ms = 0.0

    return RequestResult(
        request_id=req.request_id,
        ttft_ms=ttft_ms, tpot_ms=tpot_ms,
        e2e_latency_ms=e2e_ms, tokens_generated=tokens_generated,
    )


async def run_warmup_phase(
    client: AsyncOpenAI,
    warmup_requests: list[PromptRequest],
    model: str,
) -> None:
    """Run warm-up requests (distinct prompts from test scenarios)."""
    n = min(WARMUP_REQUESTS, len(warmup_requests))
    log.info("Running %d warm-up requests...", n)
    tasks = [send_request(client, req, model) for req in warmup_requests[:n]]
    results = await asyncio.gather(*tasks)
    errors = sum(1 for r in results if r.error)
    if errors:
        log.warning("Warm-up: %d/%d requests had errors", errors, len(results))
    else:
        log.info("Warm-up complete")


async def run_scenario(
    client: AsyncOpenAI,
    scenario: LoadedScenario,
    model: str,
    framework: str,
) -> ScenarioResult:
    """Run a scenario and collect timing data."""
    log.info("Running scenario '%s' on %s (%d requests, concurrency=%d)",
             scenario.name, framework, len(scenario.requests), scenario.concurrency)

    result = ScenarioResult(framework=framework, scenario=scenario.name)
    sem = asyncio.Semaphore(scenario.concurrency)

    async def bounded(req: PromptRequest) -> RequestResult:
        async with sem:
            return await send_request(client, req, model)

    t_start = time.perf_counter()
    request_results = await asyncio.gather(*[bounded(r) for r in scenario.requests])
    result.wall_clock_s = time.perf_counter() - t_start
    result.requests = list(request_results)

    compute_metrics(result)
    return result


async def run_concurrency_ramp(
    framework: str,
    model_path: str,
    scenario: LoadedScenario,
    warmup_requests: list[PromptRequest],
    port: int,
    model_name: str,
    log_dir: Path | None = None,
) -> list[ScenarioResult]:
    """Run high-concurrency scenario with per-level server restarts."""
    results = []

    for level in CONCURRENCY_LEVELS:
        log.info("=== Concurrency level: %d ===", level)

        proc, server_log = launch_server(framework, model_path, port, log_dir=log_dir)
        healthy = await wait_for_health(port)

        if not healthy:
            if server_log:
                log.error("Server failed at concurrency=%d. Check log: %s", level, server_log)
            else:
                log.error("Server failed at concurrency=%d. stderr:\n%s", level, get_server_stderr(proc))
            kill_server(proc)
            await wait_for_gpu_release()
            log.info("Concurrency ceiling detected at level %d", level)
            break

        client = AsyncOpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="dummy")
        await run_warmup_phase(client, warmup_requests, model_name)

        # Create a copy of the scenario with this concurrency level
        level_scenario = LoadedScenario(
            name=f"high_concurrency_c{level}",
            requests=scenario.requests,
            concurrency=level,
            description=f"High concurrency at level {level}",
        )

        result = await run_scenario(client, level_scenario, model_name, framework)
        results.append(result)

        kill_server(proc)
        await wait_for_gpu_release()

    return results


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------


def compute_metrics(result: ScenarioResult) -> None:
    """Compute aggregate percentile metrics from request results."""
    successful = [r for r in result.requests if r.error is None]
    result.error_count = len(result.requests) - len(successful)

    if not successful:
        log.warning("No successful requests for %s/%s", result.framework, result.scenario)
        return

    ttfts = [r.ttft_ms for r in successful]
    tpots = [r.tpot_ms for r in successful if r.tpot_ms > 0]
    e2es = [r.e2e_latency_ms for r in successful]

    result.ttft_p50_ms = float(np.percentile(ttfts, 50))
    result.ttft_p90_ms = float(np.percentile(ttfts, 90))
    result.ttft_p99_ms = float(np.percentile(ttfts, 99))

    if tpots:
        result.tpot_p50_ms = float(np.percentile(tpots, 50))
        result.tpot_p90_ms = float(np.percentile(tpots, 90))
        result.tpot_p99_ms = float(np.percentile(tpots, 99))

    result.e2e_p50_ms = float(np.percentile(e2es, 50))
    result.e2e_p90_ms = float(np.percentile(e2es, 90))

    result.total_tokens = sum(r.tokens_generated for r in successful)

    if result.wall_clock_s > 0:
        result.throughput_tok_s = result.total_tokens / result.wall_clock_s


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_comparison_table(
    sglang_results: dict[str, ScenarioResult],
    vllm_results: dict[str, ScenarioResult],
) -> None:
    try:
        from rich.console import Console
        from rich.table import Table
    except ImportError:
        log.warning("rich not installed, using plain text output")
        print_comparison_plain(sglang_results, vllm_results)
        return

    console = Console()

    for scenario_name in SCENARIO_ORDER:
        sg = sglang_results.get(scenario_name)
        vl = vllm_results.get(scenario_name)
        if not sg and not vl:
            continue

        table = Table(
            title=f"DeepSeek-V2-Lite-Chat: {scenario_name}",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Metric", style="bold")
        table.add_column("SGLang", justify="right")
        table.add_column("vLLM", justify="right")
        table.add_column("Delta", justify="right", style="green")

        metrics = [
            ("TTFT p50 (ms)", "ttft_p50_ms"),
            ("TTFT p90 (ms)", "ttft_p90_ms"),
            ("TPOT p50 (ms)", "tpot_p50_ms"),
            ("TPOT p90 (ms)", "tpot_p90_ms"),
            ("E2E p50 (ms)", "e2e_p50_ms"),
            ("Throughput (tok/s)", "throughput_tok_s"),
            ("Errors", "error_count"),
        ]

        for label, attr in metrics:
            sg_val = getattr(sg, attr, 0) if sg else 0
            vl_val = getattr(vl, attr, 0) if vl else 0

            if vl_val and sg_val:
                if attr == "throughput_tok_s":
                    delta_pct = ((sg_val - vl_val) / vl_val) * 100
                    delta_str = f"+{delta_pct:.1f}%" if delta_pct > 0 else f"{delta_pct:.1f}%"
                elif attr == "error_count":
                    delta_str = ""
                else:
                    delta_pct = ((sg_val - vl_val) / vl_val) * 100
                    delta_str = f"{delta_pct:+.1f}%"
            else:
                delta_str = "N/A"

            table.add_row(
                label,
                f"{sg_val:.1f}" if sg else "---",
                f"{vl_val:.1f}" if vl else "---",
                delta_str,
            )

        console.print(table)
        console.print()


def print_comparison_plain(
    sglang_results: dict[str, ScenarioResult],
    vllm_results: dict[str, ScenarioResult],
) -> None:
    for scenario_name in SCENARIO_ORDER:
        sg = sglang_results.get(scenario_name)
        vl = vllm_results.get(scenario_name)
        if not sg and not vl:
            continue

        print(f"\n{'=' * 60}")
        print(f"  {scenario_name}")
        print(f"{'=' * 60}")
        print(f"  {'Metric':<25} {'SGLang':>10} {'vLLM':>10}")
        print(f"  {'-' * 45}")

        for label, attr in [
            ("TTFT p50 (ms)", "ttft_p50_ms"),
            ("TPOT p50 (ms)", "tpot_p50_ms"),
            ("Throughput (tok/s)", "throughput_tok_s"),
            ("Errors", "error_count"),
        ]:
            sg_val = getattr(sg, attr, 0) if sg else 0
            vl_val = getattr(vl, attr, 0) if vl else 0
            print(f"  {label:<25} {sg_val:>10.1f} {vl_val:>10.1f}")


def write_results_json(
    all_results: dict[str, dict[str, ScenarioResult]],
    output_path: Path,
    metadata: dict,
) -> None:
    data = {"metadata": metadata, "results": {}}

    for framework, scenarios in all_results.items():
        data["results"][framework] = {}
        for scenario_name, result in scenarios.items():
            data["results"][framework][scenario_name] = {
                "ttft_p50_ms": result.ttft_p50_ms,
                "ttft_p90_ms": result.ttft_p90_ms,
                "ttft_p99_ms": result.ttft_p99_ms,
                "tpot_p50_ms": result.tpot_p50_ms,
                "tpot_p90_ms": result.tpot_p90_ms,
                "tpot_p99_ms": result.tpot_p99_ms,
                "e2e_p50_ms": result.e2e_p50_ms,
                "e2e_p90_ms": result.e2e_p90_ms,
                "throughput_tok_s": result.throughput_tok_s,
                "total_tokens": result.total_tokens,
                "wall_clock_s": result.wall_clock_s,
                "error_count": result.error_count,
                "num_requests": len(result.requests),
                "per_request": [asdict(r) for r in result.requests],
            }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    log.info("Results written to %s", output_path)


def write_comparison_csv(
    all_results: dict[str, dict[str, ScenarioResult]],
    output_path: Path,
) -> None:
    fieldnames = [
        "scenario", "framework",
        "ttft_p50_ms", "ttft_p90_ms", "ttft_p99_ms",
        "tpot_p50_ms", "tpot_p90_ms", "tpot_p99_ms",
        "e2e_p50_ms", "throughput_tok_s",
        "total_tokens", "error_count",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for framework, scenarios in all_results.items():
            for scenario_name, result in scenarios.items():
                writer.writerow({
                    "scenario": scenario_name,
                    "framework": framework,
                    "ttft_p50_ms": f"{result.ttft_p50_ms:.2f}",
                    "ttft_p90_ms": f"{result.ttft_p90_ms:.2f}",
                    "ttft_p99_ms": f"{result.ttft_p99_ms:.2f}",
                    "tpot_p50_ms": f"{result.tpot_p50_ms:.2f}",
                    "tpot_p90_ms": f"{result.tpot_p90_ms:.2f}",
                    "tpot_p99_ms": f"{result.tpot_p99_ms:.2f}",
                    "e2e_p50_ms": f"{result.e2e_p50_ms:.2f}",
                    "throughput_tok_s": f"{result.throughput_tok_s:.1f}",
                    "total_tokens": result.total_tokens,
                    "error_count": result.error_count,
                })
    log.info("Comparison CSV written to %s", output_path)


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------


async def run_validation(
    model_path: str,
    framework: str,
    port: int,
) -> None:
    """Run SGLang's bench_serving.py and log output for manual comparison."""
    log.info("Cross-validating with bench_serving.py for %s...", framework)

    backend = "sglang" if framework == "sglang" else "vllm"
    cmd = [
        sys.executable, "-m", "sglang.bench_serving",
        "--backend", backend,
        "--model", model_path,
        "--num-prompts", "20",
        "--request-rate", "1",
        "--port", str(port),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            log.warning("bench_serving.py failed (exit %d): %s",
                        result.returncode, result.stderr[:500])
            return
        log.info("bench_serving.py output:\n%s", result.stdout[-1000:])
    except FileNotFoundError:
        log.warning("bench_serving.py not found -- skipping cross-validation")
    except subprocess.TimeoutExpired:
        log.warning("bench_serving.py timed out after 300s")


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def collect_metadata(model_path: str) -> dict:
    meta = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model_path,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(", ")
            meta["gpu_model"] = parts[0] if parts else "unknown"
            meta["gpu_memory"] = parts[1] if len(parts) > 1 else "unknown"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        meta["gpu_model"] = "unknown"

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            meta["nvidia_driver"] = result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    for pkg in ["sglang", "vllm"]:
        try:
            import importlib.metadata as importlib_metadata
            meta[f"{pkg}_version"] = importlib_metadata.version(pkg)
        except Exception:
            meta[f"{pkg}_version"] = "not installed"

    return meta


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


async def run_framework(
    framework: str,
    model_path: str,
    scenarios_to_run: list[str],
    prompts_data: dict,
    port: int,
    validate: bool,
    log_dir: Path | None = None,
) -> dict[str, ScenarioResult]:
    results: dict[str, ScenarioResult] = {}
    model_name = Path(model_path).name
    warmup_requests = load_warmup(prompts_data)

    for scenario_name in scenarios_to_run:
        scenario = load_scenario(prompts_data, scenario_name)

        # High concurrency uses per-level server restarts
        if scenario_name == "high_concurrency":
            level_results = await run_concurrency_ramp(
                framework, model_path, scenario,
                warmup_requests, port, model_name,
                log_dir=log_dir,
            )
            for lr in level_results:
                results[lr.scenario] = lr
            continue

        # Standard scenario: single server lifecycle
        log.info("=== %s / %s ===", framework.upper(), scenario_name)

        proc, server_log = launch_server(framework, model_path, port, log_dir=log_dir)
        healthy = await wait_for_health(port)

        if not healthy:
            if server_log:
                log.error("Server failed to start. Check log: %s", server_log)
            else:
                log.error("Server failed to start. stderr:\n%s", get_server_stderr(proc))
            kill_server(proc)
            await wait_for_gpu_release()
            continue

        client = AsyncOpenAI(
            base_url=f"http://127.0.0.1:{port}/v1", api_key="dummy"
        )

        await run_warmup_phase(client, warmup_requests, model_name)
        result = await run_scenario(client, scenario, model_name, framework)
        results[scenario_name] = result

        log.info(
            "Completed %s/%s: %d reqs, %.1f tok/s, TTFT p50=%.1fms, "
            "TPOT p50=%.1fms, %d errors",
            framework, scenario_name, len(result.requests),
            result.throughput_tok_s, result.ttft_p50_ms,
            result.tpot_p50_ms, result.error_count,
        )

        if validate:
            await run_validation(model_path, framework, port)

        kill_server(proc)
        await wait_for_gpu_release()

    return results


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="SGLang vs vLLM A/B Benchmark for DeepSeek-V2-Lite-Chat"
    )
    parser.add_argument(
        "--model-path", required=True,
        help="Path to the model weights directory",
    )
    parser.add_argument(
        "--framework", choices=["sglang", "vllm"],
        help="Run only one framework (default: both)",
    )
    parser.add_argument(
        "--scenario", choices=SCENARIO_ORDER,
        help="Run only one scenario (default: all)",
    )
    parser.add_argument(
        "--prompts", type=Path, default=Path(__file__).parent / "prompts.json",
        help="Path to pre-generated prompts.json (default: benchmark/prompts.json)",
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT,
        help=f"Port for inference server (default: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).parent,
        help="Directory for output files (default: benchmark/)",
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Cross-validate against SGLang's bench_serving.py",
    )
    parser.add_argument(
        "--log-dir", type=Path, default=Path(__file__).parent / "logs",
        help="Directory for server stdout/stderr logs (default: benchmark/logs/)",
    )
    args = parser.parse_args()

    # Set up file logging for the benchmark harness itself
    args.log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(args.log_dir / f"benchmark_{ts}.log")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    )
    log.addHandler(file_handler)
    log.info("Logging to %s", args.log_dir)

    frameworks = [args.framework] if args.framework else ["sglang", "vllm"]
    scenarios = [args.scenario] if args.scenario else SCENARIO_ORDER

    # Load pre-generated prompts
    if not args.prompts.exists():
        log.error("Prompts file not found: %s", args.prompts)
        log.error("Generate prompts first with generate_prompts.py on a machine with transformers installed.")
        sys.exit(1)

    log.info("Loading prompts from %s...", args.prompts)
    prompts_data = load_prompts(args.prompts)

    metadata = collect_metadata(args.model_path)
    all_results: dict[str, dict[str, ScenarioResult]] = {}

    for framework in frameworks:
        log.info("=" * 60)
        log.info("Starting benchmark for %s", framework.upper())
        log.info("=" * 60)

        results = await run_framework(
            framework, args.model_path, scenarios,
            prompts_data, args.port, args.validate,
            log_dir=args.log_dir,
        )
        all_results[framework] = results

    # Write outputs with unique filenames based on run parameters
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    fw_label = "_".join(frameworks)
    sc_label = "_".join(scenarios)
    run_tag = f"{fw_label}_{sc_label}_{ts}"

    write_results_json(all_results, output_dir / f"results_{run_tag}.json", metadata)
    write_comparison_csv(all_results, output_dir / f"comparison_{run_tag}.csv")

    if "sglang" in all_results and "vllm" in all_results:
        print_comparison_table(all_results["sglang"], all_results["vllm"])
    else:
        for framework, results in all_results.items():
            log.info("\nResults for %s:", framework)
            for name, result in results.items():
                log.info(
                    "  %s: TTFT p50=%.1fms, TPOT p50=%.1fms, throughput=%.1f tok/s",
                    name, result.ttft_p50_ms, result.tpot_p50_ms,
                    result.throughput_tok_s,
                )


if __name__ == "__main__":
    asyncio.run(main())
