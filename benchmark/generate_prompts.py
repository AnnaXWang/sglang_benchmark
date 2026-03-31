#!/usr/bin/env python3
"""Generate all benchmark prompts and save to prompts.json for inspection.

Usage:
    python generate_prompts.py
    python generate_prompts.py --output prompts.json
    python generate_prompts.py --scenario prefix_heavy  # just one scenario
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from transformers import AutoTokenizer

from scenarios import (
    SCENARIO_ORDER,
    generate_warmup,
    get_scenario,
    token_count,
)

MODEL_NAME = "deepseek-ai/DeepSeek-V2-Lite-Chat"


def format_request(req, tokenizer) -> dict:
    """Format a Request into a human-readable dict."""
    # Count total input tokens across all messages
    total_input_tokens = sum(
        token_count(tokenizer, msg["content"]) for msg in req.messages
    )
    return {
        "request_id": req.request_id,
        "messages": req.messages,
        "max_tokens": req.max_tokens,
        "total_input_tokens": total_input_tokens,
        "metadata": req.metadata,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark prompts")
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("prompts.json"),
        help="Output file path (default: prompts.json)",
    )
    parser.add_argument(
        "--scenario", "-s",
        choices=SCENARIO_ORDER,
        help="Generate only one scenario (default: all)",
    )
    args = parser.parse_args()

    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    print(f"Vocab size: {tokenizer.vocab_size:,}")

    scenarios_to_gen = [args.scenario] if args.scenario else SCENARIO_ORDER
    output = {}

    for name in scenarios_to_gen:
        print(f"\nGenerating: {name}")
        scenario = get_scenario(name, tokenizer)
        print(f"  {len(scenario.requests)} requests, concurrency={scenario.concurrency}")
        print(f"  {scenario.description}")

        output[name] = {
            "description": scenario.description,
            "concurrency": scenario.concurrency,
            "num_requests": len(scenario.requests),
            "requests": [format_request(r, tokenizer) for r in scenario.requests],
        }

    # Warmup
    print("\nGenerating: warmup")
    warmup = generate_warmup(tokenizer)
    print(f"  {len(warmup)} requests")
    output["warmup"] = {
        "description": "Warmup requests with distinct prompts (no prefix overlap with test scenarios)",
        "num_requests": len(warmup),
        "requests": [format_request(r, tokenizer) for r in warmup],
    }

    # Write
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    size_mb = args.output.stat().st_size / (1024 * 1024)
    print(f"\nWrote {args.output} ({size_mb:.1f} MB)")

    # Print summary
    print("\n--- Summary ---")
    for name, data in output.items():
        n = data["num_requests"]
        if n > 0:
            tokens = [r["total_input_tokens"] for r in data["requests"]]
            print(f"  {name}: {n} requests, input tokens min={min(tokens)} max={max(tokens)} avg={sum(tokens)//n}")


if __name__ == "__main__":
    main()
