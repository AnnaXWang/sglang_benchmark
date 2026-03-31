#!/bin/bash
# RunPod bootstrap script for SGLang vs vLLM benchmark
# Target: RunPod A100 (40GB or 80GB) with CUDA pre-installed
#
# Usage: bash setup.sh
#
# Creates two separate venvs to avoid dependency conflicts,
# pre-downloads the model once to a shared path.

set -euo pipefail

MODEL_ID="deepseek-ai/DeepSeek-V2-Lite-Chat"
MODEL_DIR="/workspace/models/DeepSeek-V2-Lite-Chat"
TORCH_VERSION="2.5.1"
VLLM_VERSION="0.18.0"

echo "============================================"
echo "  SGLang vs vLLM Benchmark Setup"
echo "============================================"
echo ""
echo "Model:   $MODEL_ID"
echo "PyTorch: $TORCH_VERSION"
echo "SGLang:  latest"
echo "vLLM:    $VLLM_VERSION"
echo ""

# -------------------------------------------------------------------
# 1. Create SGLang venv
# -------------------------------------------------------------------
echo "[1/4] Creating SGLang venv..."
if [ ! -d /workspace/sglang-env ]; then
    python3 -m venv /workspace/sglang-env
fi
source /workspace/sglang-env/bin/activate
pip install --upgrade pip
pip install "torch==${TORCH_VERSION}"
# flashinfer-python wheels aren't on PyPI — must use their custom index
pip install "sglang[all]==0.5.9" \
    --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer/
pip install openai numpy pandas rich aiohttp
deactivate
echo "  SGLang venv ready."

# -------------------------------------------------------------------
# 2. Create vLLM venv
# -------------------------------------------------------------------
echo "[2/4] Creating vLLM venv..."
if [ ! -d /workspace/vllm-env ]; then
    python3 -m venv /workspace/vllm-env
fi
source /workspace/vllm-env/bin/activate
pip install --upgrade pip
pip install "torch==${TORCH_VERSION}"
pip install "vllm==${VLLM_VERSION}"
pip install openai numpy pandas rich aiohttp
deactivate
echo "  vLLM venv ready."

# -------------------------------------------------------------------
# 3. Download model weights (shared between both venvs)
# -------------------------------------------------------------------
echo "[3/4] Downloading model weights..."
if [ -d "$MODEL_DIR" ] && [ "$(ls -A "$MODEL_DIR")" ]; then
    echo "  Model already exists at $MODEL_DIR, skipping download."
else
    mkdir -p "$(dirname "$MODEL_DIR")"
    # Use sglang venv for huggingface-cli
    source /workspace/sglang-env/bin/activate
    pip install huggingface_hub[cli]
    huggingface-cli download "$MODEL_ID" --local-dir "$MODEL_DIR"
    deactivate
    echo "  Model downloaded to $MODEL_DIR"
fi

# -------------------------------------------------------------------
# 4. Copy benchmark scripts
# -------------------------------------------------------------------
echo "[4/4] Setting up benchmark scripts..."
BENCH_DIR="/workspace/benchmark"
mkdir -p "$BENCH_DIR"

# Copy scripts from wherever this setup.sh lives
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for f in run_benchmark.py scenarios.py test_benchmark.py prompts.json; do
    if [ -f "$SCRIPT_DIR/$f" ]; then
        cp "$SCRIPT_DIR/$f" "$BENCH_DIR/$f"
        echo "  Copied $f"
    fi
done

echo ""
echo "============================================"
echo "  Setup complete!"
echo "============================================"
echo ""
echo "Usage:"
echo "  # Full benchmark (both frameworks)"
echo "  source /workspace/sglang-env/bin/activate"
echo "  cd /workspace/benchmark"
echo "  python run_benchmark.py --model-path $MODEL_DIR"
echo ""
echo "  # Single framework"
echo "  python run_benchmark.py --model-path $MODEL_DIR --framework sglang"
echo ""
echo "  # With cross-validation"
echo "  python run_benchmark.py --model-path $MODEL_DIR --validate"
echo ""
echo "Note: Run inside a tmux/screen session to survive disconnects."
echo "  tmux new -s bench"
