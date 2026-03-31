"""
Benchmark scenario definitions for SGLang vs vLLM comparison.

Each scenario produces a list of request dicts compatible with the
OpenAI chat completions API. Prompts use exact token counts via the
model's tokenizer.

Scenario architecture:

    get_scenario(name, tokenizer)
        │
        ├── prefix_heavy()     20 sequential reqs, shared 2048-tok prefix
        ├── multi_turn()       50 convos × 3-5 turns, growing history
        ├── high_concurrency() 50 reqs per concurrency level (8, 16, ...)
        └── long_context()     50 reqs with 2048-4096 tok inputs

    generate_warmup(tokenizer)
        └── 10 requests with DISTINCT prompts (never shares prefixes
            with test scenarios to avoid RadixAttention cache poisoning)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional


WARMUP_SEED_OFFSET = 999_999  # Ensures warmup RNG never overlaps with scenarios


@dataclass
class Request:
    """A single benchmark request."""
    messages: list[dict[str, str]]
    max_tokens: int
    scenario: str
    request_id: int
    metadata: dict = field(default_factory=dict)


@dataclass
class Scenario:
    """A benchmark scenario with its requests and config."""
    name: str
    requests: list[Request]
    concurrency: int
    description: str


def _generate_text(tokenizer, target_tokens: int, rng: random.Random) -> str:
    """Generate text that encodes to approximately `target_tokens` tokens.

    Picks random token IDs from the vocab and decodes them. Produces
    incoherent text, which is fine for throughput benchmarking.
    """
    vocab_size = tokenizer.vocab_size
    # Generate more tokens than needed, then trim
    token_ids = [rng.randint(100, vocab_size - 1) for _ in range(target_tokens + 20)]
    text = tokenizer.decode(token_ids, skip_special_tokens=True)

    # Trim to hit exact token count
    actual = tokenizer.encode(text, add_special_tokens=False)
    if len(actual) > target_tokens:
        actual = actual[:target_tokens]
        text = tokenizer.decode(actual, skip_special_tokens=True)
    elif len(actual) < target_tokens:
        # Pad with more random tokens
        extra = [rng.randint(100, vocab_size - 1) for _ in range(target_tokens - len(actual) + 10)]
        extra_text = tokenizer.decode(extra, skip_special_tokens=True)
        combined_ids = tokenizer.encode(text + " " + extra_text, add_special_tokens=False)
        combined_ids = combined_ids[:target_tokens]
        text = tokenizer.decode(combined_ids, skip_special_tokens=True)

    return text


def token_count(tokenizer, text: str) -> int:
    """Count tokens in text (no special tokens)."""
    return len(tokenizer.encode(text, add_special_tokens=False))


def prefix_heavy(tokenizer, seed: int = 42) -> Scenario:
    """20 sequential requests sharing a 2048-token system prompt.

    Request 1 is a cold start. Requests 2-20 should hit the
    RadixAttention prefix cache. Concurrency=1 so cache effects
    are visible in TTFT differences.
    """
    rng = random.Random(seed)

    system_text = _generate_text(tokenizer, 2048, rng)

    requests = []
    for i in range(20):
        query_len = rng.randint(50, 100)
        user_text = _generate_text(tokenizer, query_len, rng)

        requests.append(Request(
            messages=[
                {"role": "system", "content": system_text},
                {"role": "user", "content": user_text},
            ],
            max_tokens=128,
            scenario="prefix_heavy",
            request_id=i,
            metadata={
                "system_tokens": token_count(tokenizer, system_text),
                "user_tokens": token_count(tokenizer, user_text),
            },
        ))

    return Scenario(
        name="prefix_heavy",
        requests=requests,
        concurrency=1,
        description=(
            "20 sequential requests with shared 2048-token system prefix. "
            "Tests RadixAttention prefix caching + MLA KV-cache compression."
        ),
    )


def multi_turn(tokenizer, seed: int = 43) -> Scenario:
    """50 conversations with 3-5 turns each.

    Each turn sends the FULL conversation history. This exercises
    prefix caching because each turn's prompt starts with all
    previous turns (increasingly long prefix to match).
    """
    rng = random.Random(seed)

    requests = []
    request_id = 0

    for conv_idx in range(50):
        num_turns = rng.randint(3, 5)
        messages: list[dict[str, str]] = []

        system_text = _generate_text(tokenizer, 200, rng)
        messages.append({"role": "system", "content": system_text})

        for turn in range(num_turns):
            user_len = rng.randint(50, 150)
            user_text = _generate_text(tokenizer, user_len, rng)
            messages.append({"role": "user", "content": user_text})

            requests.append(Request(
                messages=list(messages),
                max_tokens=128,
                scenario="multi_turn",
                request_id=request_id,
                metadata={
                    "conversation": conv_idx,
                    "turn": turn,
                    "history_messages": len(messages),
                },
            ))
            request_id += 1

            # Fake assistant response for next turn's history
            assistant_len = rng.randint(50, 100)
            assistant_text = _generate_text(tokenizer, assistant_len, rng)
            messages.append({"role": "assistant", "content": assistant_text})

    return Scenario(
        name="multi_turn",
        requests=requests,
        concurrency=4,
        description=(
            "50 conversations x 3-5 turns. Each turn sends full history. "
            "Tests KV-cache reuse via prefix matching on growing prompts."
        ),
    )


def high_concurrency(tokenizer, seed: int = 44) -> Scenario:
    """50 requests for use at a single concurrency level.

    The harness runs this multiple times at increasing concurrency
    (8, 16, 32, ...) until OOM. Each level is a separate server
    launch since OOM kills the CUDA context.
    """
    rng = random.Random(seed)

    requests = []
    for i in range(50):
        input_len = rng.randint(100, 500)
        user_text = _generate_text(tokenizer, input_len, rng)

        requests.append(Request(
            messages=[{"role": "user", "content": user_text}],
            max_tokens=128,
            scenario="high_concurrency",
            request_id=i,
            metadata={"input_tokens": token_count(tokenizer, user_text)},
        ))

    return Scenario(
        name="high_concurrency",
        requests=requests,
        concurrency=8,  # starting level; harness doubles until OOM
        description=(
            "50 requests per concurrency level. Harness runs at 8, 16, 32, ... "
            "until OOM, restarting server each time. Tests continuous batching."
        ),
    )


def long_context(tokenizer, seed: int = 45) -> Scenario:
    """50 requests with 2048-4096 token inputs.

    Long inputs exercise MLA-optimized attention kernels. MLA compresses
    KV-cache from multi-head to low-rank latent space, so SGLang should
    use less memory per token.
    """
    rng = random.Random(seed)

    requests = []
    for i in range(50):
        input_len = rng.randint(2048, 4096)
        user_text = _generate_text(tokenizer, input_len, rng)

        requests.append(Request(
            messages=[{"role": "user", "content": user_text}],
            max_tokens=256,
            scenario="long_context",
            request_id=i,
            metadata={"input_tokens": token_count(tokenizer, user_text)},
        ))

    return Scenario(
        name="long_context",
        requests=requests,
        concurrency=4,
        description=(
            "50 requests with 2048-4096 token inputs, 256 token outputs. "
            "Tests MLA-optimized attention kernels."
        ),
    )


def generate_warmup(tokenizer, n: int = 10, seed: Optional[int] = None) -> list[Request]:
    """Generate warmup requests with DISTINCT prompts.

    Uses a completely separate RNG seed and different prompt length
    range (300-600 tokens) than any test scenario to avoid
    pre-populating the RadixAttention cache.
    """
    if seed is None:
        seed = WARMUP_SEED_OFFSET
    rng = random.Random(seed)

    requests = []
    for i in range(n):
        input_len = rng.randint(300, 600)
        user_text = _generate_text(tokenizer, input_len, rng)

        requests.append(Request(
            messages=[{"role": "user", "content": user_text}],
            max_tokens=32,
            scenario="warmup",
            request_id=i,
        ))

    return requests


SCENARIOS = {
    "prefix_heavy": prefix_heavy,
    "multi_turn": multi_turn,
    "high_concurrency": high_concurrency,
    "long_context": long_context,
}

# Priority order (most instructive first)
SCENARIO_ORDER = ["prefix_heavy", "multi_turn", "high_concurrency", "long_context"]


def get_scenario(name: str, tokenizer, seed: Optional[int] = None) -> Scenario:
    """Get a scenario by name."""
    if name not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {name}. Available: {list(SCENARIOS.keys())}")
    fn = SCENARIOS[name]
    kwargs = {"tokenizer": tokenizer}
    if seed is not None:
        kwargs["seed"] = seed
    return fn(**kwargs)


def get_all_scenarios(tokenizer) -> list[Scenario]:
    """Get all scenarios in priority order."""
    return [SCENARIOS[name](tokenizer) for name in SCENARIO_ORDER]
