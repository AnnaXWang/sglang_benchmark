# Plan: SGLang vs vLLM A/B Test with DeepSeek-V2-Lite-Chat on RunPod

## Context

You want to demonstrate SGLang's performance advantages over vLLM firsthand using DeepSeek-V2-Lite-Chat on a RunPod A100 GPU. This plan creates a single self-contained benchmark script and RunPod setup guide that lets you spin up both servers, run identical workloads, collect telemetry, and compare results side-by-side.

DeepSeek-V2-Lite uses MLA (Multi-head Latent Attention) — an architecture SGLang has specific optimizations for, making it ideal for showcasing the difference.

---

## What we'll build

A single directory `benchmark/` in `full_sglang/` with:

1. **`setup.sh`** — RunPod bootstrap script (venv, install both frameworks, download model)
2. **`run_benchmark.py`** — unified benchmark harness that tests both servers with identical workloads and outputs a comparison report
3. **`scenarios.py`** — realistic workload scenarios (single-turn, multi-turn, high-concurrency, long-context)

---

## Step 1: `setup.sh` — RunPod Environment Setup

Automates the full RunPod pod setup:

```
- Create two venvs: /workspace/sglang-env and /workspace/vllm-env
- Install sglang[all] and vllm respectively
- Pre-download DeepSeek-V2-Lite-Chat weights to /workspace/models/ via huggingface-cli
- Install shared benchmark deps (openai, pandas, matplotlib, rich) in both venvs
```

Key decisions:
- Separate venvs to avoid dependency conflicts between SGLang and vLLM
- Pre-download model once to a shared path so both frameworks load from disk (no download variance)
- Target GPU: 1x A100 40GB or 80GB

## Step 2: `scenarios.py` — Realistic Workload Definitions

Four test scenarios that exercise different inference patterns:

| Scenario | What it tests | Why SGLang should win |
|---|---|---|
| **Single-turn QA** | 200 independent short prompts (100-500 tok input, 128 tok output), low concurrency (4) | Baseline latency comparison |
| **Multi-turn chat** | 50 conversations with 3-5 turns each, simulating real chat sessions | SGLang's RadixAttention prefix caching reuses KV cache across turns |
| **High concurrency** | 500 requests fired at scale 0.4x-1.0x using Qwen trace replay pattern | SGLang's continuous batching + overlap scheduling under load |
| **Long context** | 50 prompts with 2K-4K token inputs, 256 tok output | MLA-optimized attention kernels in SGLang |

Each scenario is a list of `BenchmarkTrace`-compatible dicts with timestamp, prompt, and expected output length. We'll use the same prompts for both frameworks (seeded RNG).

Prompt source: synthetic prompts generated from a fixed seed (same approach as mini-sglang's `generate_prompt()`). No external dataset download required — keeps it self-contained.

## Step 3: `run_benchmark.py` — Unified Benchmark Harness

Single script that:

1. **Launches server** — starts SGLang or vLLM in a subprocess with the right args
2. **Waits for health** — polls `/v1/models` until server is ready
3. **Runs scenarios** — fires requests via async OpenAI client, streams responses, records per-token timestamps
4. **Collects telemetry** per request:
   - TTFT (Time to First Token)
   - TPOT (Time Per Output Token)
   - E2E latency
   - Total tokens generated
   - For vLLM: also scrape `/metrics` for GPU utilization, KV cache usage
5. **Kills server** and repeats for the other framework
6. **Outputs comparison report** as a rich terminal table + JSON file

Server launch commands:
```bash
# SGLang
python -m sglang.launch_server --model-path /workspace/models/DeepSeek-V2-Lite-Chat --host 127.0.0.1 --port 30000

# vLLM
python -m vllm.entrypoints.openai.api_server --model /workspace/models/DeepSeek-V2-Lite-Chat --host 127.0.0.1 --port 30000 --trust-remote-code
```

Usage:
```bash
# Run full comparison
python run_benchmark.py --model-path /workspace/models/DeepSeek-V2-Lite-Chat

# Run just one framework
python run_benchmark.py --model-path /workspace/models/DeepSeek-V2-Lite-Chat --framework sglang

# Run specific scenario
python run_benchmark.py --model-path /workspace/models/DeepSeek-V2-Lite-Chat --scenario multi_turn
```

### Telemetry approach

Follow mini-sglang's existing pattern from `benchmark/client.py`:
- Use `time.perf_counter()` around streaming responses
- Record timestamp at first token (TTFT) and each subsequent token (TPOT)
- Compute percentiles (p50, p90, p99, max) using numpy
- Both frameworks use the same OpenAI-compatible client — the only variable is the server

### Output format

Terminal output (via `rich`):
```
╔══════════════════════════════════════════════════════════════╗
║          DeepSeek-V2-Lite-Chat: SGLang vs vLLM              ║
╠═══════════════╦═══════════╦═══════════╦══════════════════════╣
║ Metric        ║  SGLang   ║   vLLM    ║  Δ (SGLang advantage)║
╠═══════════════╬═══════════╬═══════════╬══════════════════════╣
║ TTFT p50 (ms) ║   45.2    ║   62.1    ║  -27.2%              ║
║ TPOT p50 (ms) ║   12.3    ║   18.7    ║  -34.2%              ║
║ Throughput     ║  1842 t/s ║  1205 t/s ║  +52.9%              ║
╚═══════════════╩═══════════╩═══════════╩══════════════════════╝
```

Also writes `results.json` with all raw data for further analysis.

---

## Files to create

| File | Description |
|---|---|
| `/Users/annawang/repos/full_sglang/benchmark/setup.sh` | RunPod bootstrap: venvs, installs, model download |
| `/Users/annawang/repos/full_sglang/benchmark/scenarios.py` | Workload scenario definitions |
| `/Users/annawang/repos/full_sglang/benchmark/run_benchmark.py` | Main harness: launch servers, run tests, compare results |

---

## Key design decisions

- **Same port, sequential runs** — avoids needing two GPUs; one server at a time on port 30000
- **OpenAI client for both** — both expose `/v1/chat/completions`, so the benchmark code is framework-agnostic
- **No external datasets** — synthetic prompts from a seeded RNG keep it reproducible and avoid download issues on RunPod
- **Reuse mini-sglang patterns** — the timing approach (perf_counter per streamed token) matches `mini-sglang/python/minisgl/benchmark/client.py`
- **Separate venvs** — SGLang and vLLM have conflicting torch/CUDA dependencies; separate venvs avoid version hell

---

## Verification

After implementation, test locally (CPU, small scale) to verify:
1. `setup.sh` runs without errors (can dry-run the install commands)
2. `run_benchmark.py --framework sglang --scenario single_turn` completes and outputs a results table
3. `run_benchmark.py` (full run) produces `results.json` with data for all scenarios and both frameworks
4. Output comparison table renders correctly with `rich`

On RunPod:
1. SSH into pod, run `bash setup.sh`
2. Run `python run_benchmark.py --model-path /workspace/models/DeepSeek-V2-Lite-Chat`
3. Verify both servers start/stop cleanly and results are collected for all 4 scenarios

---

## CEO Plan Review Amendments (from /plan-ceo-review on 2026-03-30)

### Design doc
Key changes from original plan: hybrid approach (custom harness + upstream cross-validation), warm-up protocol, interleaved run order, output artifacts (JSON + CSV + rich table), risk mitigations.

### Amendments from outside voice review

**1. Warm-up must use DIFFERENT prompts than test scenarios.** If warm-up requests share the same 2048-token prefix as the prefix-heavy scenario, they pre-populate the RadixAttention cache and invalidate the cold-start measurement. Use random/distinct prompts for warm-up.

**2. Concurrency ceiling: empirical, not prescribed.** Replace fixed concurrency levels (8, 16, 32, 64) with: start at concurrency 8, double until OOM, record the ceiling. 64 concurrent requests with 2048+ token inputs will likely OOM even on 80GB.

**3. Pin framework versions.** `setup.sh` must pin exact versions (e.g., `sglang==0.4.x`, `vllm==0.6.x`) for reproducibility. Unpinned installs make the benchmark non-reproducible even for the author.

### Additional findings to address during implementation

- **Verify `bench_serving.py --backend vllm` actually works** with the installed SGLang version before building `validate.py` around it
- **Capture server stderr** during launch for diagnostics when health check fails
- **Health check timeout:** 60 seconds. If server isn't healthy by then, log stderr and abort scenario.
- **Add metadata to results.json:** GPU model, GPU memory, SGLang version, vLLM version, Python version, CUDA version, model name, timestamp
- **Server process management:** `subprocess.Popen`, `SIGTERM`, wait 10s, `SIGKILL`. Poll nvidia-smi or sleep 10s between server swaps.

---

## "NOT in scope" section

| Item | Rationale |
|---|---|
| Production-grade error recovery | Learning project; fail-fast is acceptable |
| Multiple GPU support | Single A100 is the constraint |
| Automated CI/CD for benchmarks | No deployment pipeline needed |
| KV-cache per-request instrumentation | Requires framework source modification; nvidia-smi dmon is the stretch goal |
| Support for models beyond DeepSeek-V2-Lite | Out of scope; one model comparison is sufficient for learning |
| Graphical UI for results | Terminal + JSON/CSV is sufficient |

## "What already exists" section

| Existing code/tool | Relevance | Reused? |
|---|---|---|
| SGLang `bench_serving.py` | Battle-tested measurement, supports both backends | Yes — cross-validation target |
| vLLM `benchmark_serving.py` | vLLM-specific benchmark tool | No — using SGLang's tool for both |
| mini-sglang `benchmark/client.py` | Timing pattern (perf_counter per streamed token) | Pattern only — not directly imported |
| OpenAI Python client | Framework-agnostic request client | Yes — both frameworks expose /v1/chat/completions |

## "Dream state delta" section

```
12-MONTH IDEAL: Deep understanding of inference optimization, ability to
                contribute to SGLang/vLLM or build production inference services.

THIS PLAN DELIVERS: Hands-on understanding of server lifecycle, streaming
                    measurement, MLA performance characteristics, and framework
                    comparison methodology.

REMAINING GAP: Production deployment experience, multi-GPU serving, model
               optimization (quantization, speculative decoding), contributing
               patches upstream.
```

## Error & Rescue Registry

| Method | Exception | Rescued? | Rescue Action | User Sees |
|---|---|---|---|---|
| launch_server() | FileNotFoundError | Plan: Y | Log error, abort scenario | "Framework not installed" |
| launch_server() | ServerStartError (OOM) | Plan: Y | Log GPU memory, suggest 80GB | "OOM during model load" |
| wait_health() | TimeoutError (60s) | Plan: Y | Log server stderr, abort | "Server failed to start" |
| run_scenario() | TimeoutError (30s/req) | Plan: Y | Save partial results, continue | "Request N timed out" |
| run_scenario() | ConnectionError | Plan: Y | Save partial, skip request | "Server crashed mid-request" |
| collect_metrics() | ZeroDivisionError | Plan: Y | Skip percentile, report "no data" | "Insufficient data" |
| kill_server() | Process zombie | Plan: Y | SIGKILL after 10s | (silent, logged) |
| validate.py | FileNotFoundError | Plan: Y | Skip validation, warn | "bench_serving.py not found" |
| validate.py | ParseError | Plan: Y | Skip validation, warn | "Upstream output format changed" |

## Failure Modes Registry

| Codepath | Failure Mode | Rescued? | Test? | User Sees? | Logged? |
|---|---|---|---|---|---|
| Server launch | OOM during model load | Y | Manual | Error message | Y |
| Server launch | Port in use | Y | Manual | Error message | Y |
| Health check | Server never healthy | Y | Manual | Timeout + stderr | Y |
| Streaming | 0 tokens returned | Y | Cross-val | "No data" | Y |
| Streaming | Partial response | Y | Cross-val | Partial metrics | Y |
| Warm-up | Warm-up requests fail | Y | Manual | Extended warm-up or abort | Y |
| Concurrency | OOM at high concurrency | Y | Empirical | Ceiling detection | Y |
| Cross-validation | bench_serving.py missing | Y | Manual | Warning, skip validation | Y |
| Cross-validation | >15% divergence | Y | Automated | Warning flag | Y |

**0 CRITICAL GAPS** (all failure modes have planned rescue actions).

## TODOS.md updates

Since there's no TODOS.md and this is a greenfield learning project, no TODOs to propose. All work is captured in the plan and design doc.

## Diagrams

### 1. System Architecture
```
  scenarios.py          run_benchmark.py              validate.py
  ┌──────────┐         ┌─────────────────────┐       ┌───────────────┐
  │ Define    │────────>│ Server Lifecycle     │       │ Run upstream  │
  │ scenarios │         │   launch_server()    │       │ bench_serving │
  │ (4 types) │         │   wait_health()      │       │ Compare vs    │
  └──────────┘         │   kill_server()      │       │ custom results│
                        │                     │       └───────────────┘
                        │ Measurement Loop     │              │
                        │   warm_up()          │              ▼
                        │   run_scenario()     │       divergence report
                        │   collect_metrics()  │
                        │                     │
                        │ Report Generation    │
                        │   compare_results()  │
                        │   write_json/csv()   │
                        │   rich_table()       │
                        └─────────────────────┘
                                 │
                                 ▼
                          results.json / comparison.csv / terminal table
```

### 2. Data Flow (including shadow paths)
```
  INPUT (scenario) ──▶ WARM-UP ──▶ SEND REQUEST ──▶ STREAM TOKENS ──▶ AGGREGATE ──▶ REPORT
       │                  │              │                │               │            │
       ▼                  ▼              ▼                ▼               ▼            ▼
    [empty          [warm-up        [connection       [stream          [0 data      [file write
     scenario?       fails?          refused?          interrupted?     points?       fails?
     → skip]         → extend/       → abort           → record         → skip        → stderr]
                      abort]          scenario]          partial]        percentile]
```

### 3. Server Lifecycle
```
  ┌─────────┐    ┌──────────┐    ┌─────────┐    ┌──────────┐    ┌─────────┐
  │ LAUNCH  │───>│ WAIT FOR │───>│ WARM UP │───>│   RUN    │───>│  KILL   │
  │ (Popen) │    │ HEALTH   │    │ (diff   │    │ SCENARIO │    │ (TERM/  │
  │         │    │ (60s max)│    │ prompts)│    │          │    │  KILL)  │
  └─────────┘    └──────────┘    └─────────┘    └──────────┘    └─────────┘
       │              │               │               │               │
       ▼              ▼               ▼               ▼               ▼
    [not found]   [timeout →      [failures →     [OOM/crash →   [zombie →
                   log stderr,     extend or       save partial,   SIGKILL
                   abort]          abort]          continue]       after 10s]
                                                                      │
                                                                      ▼
                                                               [wait for GPU
                                                                memory release]
```

### 4. Deployment Sequence
```
  N/A — learning project, no deployment.
  "Deployment" = ssh runpod && python run_benchmark.py
```

## Stale Diagram Audit
No existing diagrams in the codebase (empty directory).

## Eng Review Amendments (from /plan-eng-review on 2026-03-30)

### Structural changes
1. **validate.py merged into run_benchmark.py** as a `--validate` flag. 3 files total: setup.sh, scenarios.py, run_benchmark.py + test_benchmark.py.
2. **pytest tests added** for scenarios data validation, metrics computation (including edge cases: empty array, single point), and report output schema.

### Amendments from outside voice review
1. **Tokenizer dependency required.** `pip install transformers` in setup.sh. Use `AutoTokenizer.from_pretrained('deepseek-ai/DeepSeek-V2-Lite-Chat')` for prompt generation to get exact token counts.
2. **Each concurrency level = separate server launch.** OOM kills the CUDA context, so the concurrency ramp (8, double until OOM) requires restarting the server at each level. This is N server lifecycles, not one continuous run.

### Files (final)

| File | Description |
|---|---|
| `benchmark/setup.sh` | RunPod bootstrap: venvs (with pinned versions), installs (including transformers), model download |
| `benchmark/scenarios.py` | Workload scenario definitions (uses tokenizer for exact token counts) |
| `benchmark/run_benchmark.py` | Server lifecycle + measurement + cross-validation (--validate) + reporting |
| `benchmark/test_benchmark.py` | pytest tests for scenarios, metrics, report output |
