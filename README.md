# SGLang vs vLLM Benchmark

A/B benchmark comparing SGLang and vLLM inference performance on DeepSeek-V2-Lite-Chat (MLA architecture) running on a RunPod A100.

## Why DeepSeek-V2-Lite?

DeepSeek-V2-Lite uses Multi-head Latent Attention (MLA), which compresses KV-cache into a low-rank latent space. SGLang has specific kernel optimizations for MLA + RadixAttention prefix caching, so the performance delta between frameworks should be dramatic and instructive.

## Quick Start

### RunPod Setup

1. Create a RunPod pod:
   - **GPU**: A100 80GB
   - **Template**: RunPod PyTorch (CUDA 12.1+)
   - **Container disk**: 20 GB
   - **Volume disk**: 100 GB

2. Copy files and run setup:
```bash
# From local machine
scp -P <PORT> -i ~/.ssh/id_ed25519 -r benchmark/ root@<IP>:/workspace/benchmark/

# On the pod (inside tmux)
tmux new -s bench
cd /workspace/benchmark
bash setup.sh
```

Setup creates two venvs (sglang + vllm have conflicting dependencies), downloads the model (~30GB), and copies benchmark scripts.

### Run the Benchmark

```bash
source /workspace/sglang-env/bin/activate
cd /workspace/benchmark

# Smoke test (one scenario, one framework, ~5 min)
python run_benchmark.py --model-path /workspace/models/DeepSeek-V2-Lite-Chat \
    --scenario prefix_heavy --framework sglang

# Full comparison (all scenarios, both frameworks)
python run_benchmark.py --model-path /workspace/models/DeepSeek-V2-Lite-Chat
```

Note: Each server startup has a ~3 minute silent period for CUDA graph capture. This is normal.

### CLI Options

```
--model-path    Path to model weights (required)
--framework     sglang or vllm (default: both)
--scenario      prefix_heavy, multi_turn, high_concurrency, long_context (default: all)
--prompts       Path to prompts.json (default: benchmark/prompts.json)
--port          Server port (default: 30000)
--output-dir    Output directory (default: benchmark/)
--validate      Cross-validate against SGLang's bench_serving.py
```

## Scenarios

| Scenario | Requests | Concurrency | What it tests |
|---|---|---|---|
| **prefix_heavy** | 20 sequential, shared 2048-token system prompt | 1 | RadixAttention prefix caching + MLA KV-cache compression |
| **multi_turn** | ~200 across 50 conversations, 3-5 turns each | 4 | KV cache reuse across conversation turns |
| **high_concurrency** | 50 per level, ramps 8/16/32/64 | 8-64 | Continuous batching under load (restarts server per level) |
| **long_context** | 50 with 2048-4096 token inputs | 4 | MLA-optimized attention kernels on long sequences |

## Metrics

- **TTFT** (Time to First Token) — p50, p90, p99
- **TPOT** (Time Per Output Token) — p50, p90, p99
- **E2E latency** — p50, p90
- **Throughput** — tokens/second

## Output

- **Terminal**: rich comparison table (SGLang vs vLLM side-by-side)
- **results.json**: all raw per-request timing data + system metadata
- **comparison.csv**: summary metrics for spreadsheets

## Files

| File | Runs on | Description |
|---|---|---|
| `run_benchmark.py` | RunPod | Main harness: server lifecycle, measurement, reporting |
| `scenarios.py` | Local | Scenario definitions + tokenizer-based prompt generation |
| `generate_prompts.py` | Local | Generates prompts.json from scenarios using the model tokenizer |
| `prompts.json` | RunPod | Pre-generated prompts (checked in, no tokenizer needed at runtime) |
| `setup.sh` | RunPod | RunPod bootstrap: venvs, installs, model download |
| `test_benchmark.py` | Both | pytest tests (scenario tests need transformers, harness tests don't) |

## Architecture

```
Local machine                          RunPod A100
─────────────                          ──────────
scenarios.py ──► generate_prompts.py   run_benchmark.py
                      │                     │
                      ▼                     ├── launch server (sglang or vllm venv)
                 prompts.json ──scp──►      ├── wait for health
                                            ├── warm-up (distinct prompts)
                                            ├── run scenario (async streaming)
                                            ├── kill server
                                            ├── wait for GPU memory release
                                            └── repeat for next framework/scenario
                                                    │
                                                    ▼
                                            results.json + comparison.csv + rich table
```

## Pinned Versions

| Package | Version |
|---|---|
| SGLang | 0.5.9 |
| vLLM | 0.6.6 |
| PyTorch | 2.5.1 |
| CUDA | 12.4 |

## Running Tests

```bash
# Harness tests (no GPU or transformers needed)
pip install pytest numpy openai
pytest test_benchmark.py

# Scenario tests (needs transformers + model tokenizer access)
pip install transformers
pytest test_benchmark.py
```
