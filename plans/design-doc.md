# Design: SGLang vs vLLM A/B Benchmark with DeepSeek-V2-Lite-Chat

## Problem Statement

Learn how LLM inference frameworks differ in practice by building a hands-on benchmark comparing SGLang and vLLM on DeepSeek-V2-Lite-Chat (MLA architecture) running on a RunPod A100. The goal is understanding internals, not producing publishable numbers.

## What Makes This Cool

DeepSeek-V2-Lite uses Multi-head Latent Attention (MLA), which compresses KV-cache from multi-head down to a low-rank latent space. SGLang has specific kernel optimizations for MLA + RadixAttention for prefix caching. This means the performance delta between SGLang and vLLM on this model should be dramatic and instructive, not just "5% faster." You can see the architectural difference in the numbers.

## Constraints

- Single A100 GPU (RunPod, 40GB or 80GB)
- No local GPU access, all work happens on the pod
- Sequential runs (one server at a time) for apples-to-apples comparison
- Self-contained: no external dataset downloads, synthetic prompts only
- Learning context: measurement quality matters, but the primary goal is understanding, not publication-grade rigor

## Premises

1. Building from scratch (rather than only using existing benchmark tools) is the right approach because the learning IS the building
2. DeepSeek-V2-Lite-Chat is the right model: MLA architecture + fits single A100 + SGLang has specific optimizations
3. RunPod A100 is the right environment (no local GPU)
4. Sequential runs on same GPU/port gives a clean apples-to-apples comparison

## Cross-Model Perspective

An independent Claude subagent reviewed the plan cold and contributed these insights:

- **Coolest version not considered:** A live KV-cache memory visualization that shows SGLang's cache footprint staying flat while vLLM's grows linearly during MLA workloads. "The numbers tell you WHAT. The visualization tells you WHY."
- **Key insight:** "Build for maximum divergence-generation, not maximum measurement accuracy." The benchmark is a pretext for spelunking framework internals. Choose scenarios that maximally stress the MLA/RadixAttention difference.
- **Existing tool leverage:** SGLang's `bench_serving.py` already supports both SGLang AND vLLM backends via `--backend` flag. This simplifies cross-validation significantly.
- **Build order:** Start with measurement code + one scenario (prefix-heavy). Do NOT start with `setup.sh`. Install by hand the first time. Automate after you know what you're setting up.

## Approaches Considered

### Approach A: Build From Scratch
Custom harness (setup.sh, scenarios.py, run_benchmark.py) that launches servers, runs scenarios, collects timing. Maximum learning but risk of measurement bugs producing misleading results. Completeness: 6/10.

### Approach B: Wrapper Around Existing Tools
Thin orchestrator around SGLang's `bench_serving.py` and vLLM's `benchmark_serving.py`. Fastest, results comparable to published benchmarks, but least learning about internals. Completeness: 7/10.

### Approach C: Hybrid (chosen)
Build custom harness AND run upstream benchmark tools on the same workload. Cross-validate measurements. If numbers diverge, debugging THAT is the deepest learning. Completeness: 9/10.

## Recommended Approach

**Approach C: Hybrid.** Build the custom benchmark harness (4 files total), plus a validation step that runs SGLang's `bench_serving.py` (which supports both backends via `--backend sglang` / `--backend vllm`) on the same scenarios. The cross-validation catches measurement bugs and produces both custom and comparable-to-published results.

Upstream tool invocation: `python -m sglang.bench_serving --backend sglang --model /workspace/models/DeepSeek-V2-Lite-Chat --num-prompts 20 --request-rate 1`

Key modifications from the original plan:

1. **Add warm-up phase.** Run 10 throwaway requests before each scenario to warm caches and JIT-compile kernels. The original plan didn't mention this, and it's critical for valid measurements.

2. **Start with the prefix-heavy scenario.** This produces the most dramatic SGLang-vs-vLLM divergence (RadixAttention + MLA KV-cache compression). Build measurement code + this one scenario first. Other scenarios layer on top.

3. **Add upstream validation.** After custom harness runs, also run `python -m sglang.bench_serving` with equivalent parameters. Compare results. Flag divergences > 15% on p50 metrics as potential measurement bugs (> 10% warrants investigation).

4. **Don't start with setup.sh.** Install by hand on the first pod. Write setup.sh after you know exactly what dependencies and configurations you need.

5. **Stretch goal: KV-cache memory visualization.** Track GPU memory during each scenario using `nvidia-smi dmon` (sampling at ~100ms) or framework-specific metrics endpoints (version-dependent, check each framework's stats/metrics API). Note: per-request KV-cache tracking requires framework instrumentation, not just GPU-level polling. `nvidia-smi dmon` is the reliable cross-framework fallback. Produce a matplotlib timeline showing memory divergence between frameworks.

### Files to create

| File | Description | Build order |
|---|---|---|
| `benchmark/run_benchmark.py` | Server lifecycle + measurement + comparison report | 1st |
| `benchmark/scenarios.py` | Workload scenario definitions (start with prefix-heavy) | 1st (alongside) |
| `benchmark/validate.py` | Cross-validation against upstream bench_serving.py | 2nd |
| `benchmark/setup.sh` | RunPod bootstrap (venvs, installs, model download) | 3rd (after manual run) |

### Run order

For each scenario: launch SGLang, run warm-up (10 throwaway requests), run scenario, kill server, poll `nvidia-smi` until GPU memory < 1 GB (ensures CUDA context cleanup), launch vLLM, run warm-up, run scenario, kill server. This interleaved ordering controls for thermal throttling and GPU memory fragmentation.

All requests use the OpenAI-compatible `/v1/chat/completions` endpoint (both frameworks expose it). Concurrency=1 means each request completes end-to-end (all output tokens received) before the next begins.

### Four test scenarios (in priority order for learning)

| # | Scenario | Requests | Concurrency | Input tokens | Output tokens | Why it's interesting for MLA |
|---|---|---|---|---|---|---|
| 1 | **Prefix-heavy chat** | 20 sequential requests, all sharing the same 2048-token system prompt prefix, each with a unique 50-100 token user query. Request 1 is cold start; requests 2-20 should hit prefix cache. | 1 (sequential) | 2048 + 50-100 | 128 | RadixAttention prefix caching + MLA KV-cache compression. Most dramatic divergence. |
| 2 | **Multi-turn chat** | 50 conversations, 3-5 turns each (150-250 total requests). Each turn appends to conversation history. | 4 | Growing (100-2000) | 128 | SGLang reuses KV cache across turns via RadixAttention. |
| 3 | **High concurrency** | 200 requests total: 4 sub-runs of 50 requests each at concurrency levels 8, 16, 32, 64. | 8, 16, 32, 64 | 100-500 | 128 | Continuous batching + overlap scheduling under load. |
| 4 | **Long context** | 50 requests. | 4 | 2048-4096 | 256 | MLA-optimized attention kernels in SGLang. |

### Metrics collected

- TTFT (Time to First Token) — p50, p90, p99
- TPOT (Time Per Output Token) — p50, p90, p99
- End-to-end latency
- Throughput (tokens/second)
- Peak GPU memory (for KV-cache visualization stretch goal)

### Output artifacts

- Terminal: rich table with side-by-side comparison (TTFT, TPOT, throughput per scenario per framework)
- `benchmark/results.json`: all raw per-request timing data for further analysis
- `benchmark/comparison.csv`: summary metrics for easy import into spreadsheets

## Known Risks / Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| **OOM on A100-40GB** | High concurrency (200 requests at up to 64 concurrency) may OOM on 40GB | Reduced from 500 to 200 requests. Use `--mem-fraction-static 0.8` (SGLang) / `--gpu-memory-utilization 0.8` (vLLM). If still OOM, use 80GB SKU. |
| **vLLM incompatibility with DeepSeek-V2-Lite** | Entire benchmark collapses | Step 0: verify both frameworks can serve the model before building anything. Fallback chain: DeepSeek-V2-Lite (base) then Qwen2-7B (less MLA divergence). |
| **Pod disconnect mid-run** | Lose partial results | Write per-scenario results to disk immediately after each scenario completes. Use `tmux` or `screen`. |
| **Misleading measurements (no warm-up)** | Learn wrong things | 10-request warm-up phase before every scenario. Cross-validate against upstream tools. |

## Open Questions

1. Does vLLM support DeepSeek-V2-Lite-Chat well, or will there be compatibility issues? Need to verify `--trust-remote-code` is sufficient.
2. Is 10 warm-up requests sufficient, or do some scenarios need more? Determine empirically by checking whether TTFT stabilizes after warm-up.
3. For the prefix-heavy scenario, what's the optimal prefix length to maximize RadixAttention benefit?

## Success Criteria

1. Custom harness completes all 4 scenarios for both frameworks without errors
2. Cross-validation against upstream tools shows < 15% divergence on p50 metrics (> 10% warrants investigation)
3. At least one scenario shows a clear, explainable performance delta between SGLang and vLLM
4. You can explain WHY the performance differs, not just that it does

