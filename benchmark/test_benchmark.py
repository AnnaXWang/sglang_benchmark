"""Tests for benchmark scenarios and metrics computation."""

import random

import pytest

# Scenario tests require transformers + model tokenizer (run locally, not on RunPod).
# Guard the import so harness-only tests can run without transformers.
try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

if HAS_TRANSFORMERS:
    from scenarios import (
        Request,
        Scenario,
        SCENARIOS,
        SCENARIO_ORDER,
        WARMUP_SEED_OFFSET,
        _generate_text,
        generate_warmup,
        get_all_scenarios,
        get_scenario,
        prefix_heavy,
        multi_turn,
        high_concurrency,
        long_context,
        token_count,
    )

requires_transformers = pytest.mark.skipif(
    not HAS_TRANSFORMERS, reason="transformers not installed"
)


@pytest.fixture(scope="session")
def tokenizer():
    """Load the DeepSeek-V2-Lite-Chat tokenizer once for all tests."""
    if not HAS_TRANSFORMERS:
        pytest.skip("transformers not installed")
    return AutoTokenizer.from_pretrained(
        "deepseek-ai/DeepSeek-V2-Lite-Chat", trust_remote_code=True
    )


# --- _generate_text tests ---


@requires_transformers
class TestGenerateText:
    def test_produces_approximate_token_count(self, tokenizer):
        rng = random.Random(42)
        text = _generate_text(tokenizer, 100, rng)
        actual = token_count(tokenizer, text)
        assert abs(actual - 100) <= 5, f"Expected ~100 tokens, got {actual}"

    def test_different_seeds_produce_different_text(self, tokenizer):
        text1 = _generate_text(tokenizer, 50, random.Random(1))
        text2 = _generate_text(tokenizer, 50, random.Random(2))
        assert text1 != text2

    def test_same_seed_produces_same_text(self, tokenizer):
        text1 = _generate_text(tokenizer, 50, random.Random(42))
        text2 = _generate_text(tokenizer, 50, random.Random(42))
        assert text1 == text2

    def test_small_token_count(self, tokenizer):
        text = _generate_text(tokenizer, 5, random.Random(42))
        actual = token_count(tokenizer, text)
        assert actual <= 10, f"Expected <=10 tokens for target=5, got {actual}"

    def test_large_token_count(self, tokenizer):
        text = _generate_text(tokenizer, 2048, random.Random(42))
        actual = token_count(tokenizer, text)
        assert abs(actual - 2048) <= 5, f"Expected ~2048 tokens, got {actual}"


# --- Scenario tests ---


@requires_transformers
class TestPrefixHeavy:
    def test_produces_20_requests(self, tokenizer):
        scenario = prefix_heavy(tokenizer)
        assert len(scenario.requests) == 20

    def test_concurrency_is_1(self, tokenizer):
        scenario = prefix_heavy(tokenizer)
        assert scenario.concurrency == 1

    def test_all_requests_share_system_prompt(self, tokenizer):
        scenario = prefix_heavy(tokenizer)
        system_prompts = [r.messages[0]["content"] for r in scenario.requests]
        assert len(set(system_prompts)) == 1, "All requests should share the same system prompt"

    def test_system_prompt_is_approximately_2048_tokens(self, tokenizer):
        scenario = prefix_heavy(tokenizer)
        system_text = scenario.requests[0].messages[0]["content"]
        actual = token_count(tokenizer, system_text)
        assert abs(actual - 2048) <= 5, f"System prompt should be ~2048 tokens, got {actual}"

    def test_user_queries_are_unique(self, tokenizer):
        scenario = prefix_heavy(tokenizer)
        user_queries = [r.messages[1]["content"] for r in scenario.requests]
        assert len(set(user_queries)) == 20, "Each user query should be unique"

    def test_max_tokens_is_128(self, tokenizer):
        scenario = prefix_heavy(tokenizer)
        for r in scenario.requests:
            assert r.max_tokens == 128

    def test_deterministic_with_same_seed(self, tokenizer):
        s1 = prefix_heavy(tokenizer, seed=42)
        s2 = prefix_heavy(tokenizer, seed=42)
        for r1, r2 in zip(s1.requests, s2.requests):
            assert r1.messages == r2.messages


@requires_transformers
class TestMultiTurn:
    def test_produces_between_150_and_250_requests(self, tokenizer):
        scenario = multi_turn(tokenizer)
        # 50 convos * 3-5 turns = 150-250 requests
        assert 150 <= len(scenario.requests) <= 250

    def test_concurrency_is_4(self, tokenizer):
        scenario = multi_turn(tokenizer)
        assert scenario.concurrency == 4

    def test_conversation_history_grows(self, tokenizer):
        scenario = multi_turn(tokenizer)
        # Find requests from the same conversation
        conv0_requests = [r for r in scenario.requests if r.metadata["conversation"] == 0]
        msg_counts = [r.metadata["history_messages"] for r in conv0_requests]
        # History should grow with each turn
        assert msg_counts == sorted(msg_counts), "Message history should grow per turn"
        assert msg_counts[-1] > msg_counts[0], "Last turn should have more messages"

    def test_each_turn_includes_full_history(self, tokenizer):
        scenario = multi_turn(tokenizer)
        conv0_requests = [r for r in scenario.requests if r.metadata["conversation"] == 0]
        for i in range(1, len(conv0_requests)):
            prev_msgs = conv0_requests[i - 1].messages
            curr_msgs = conv0_requests[i].messages
            # Current messages should start with all previous messages
            for j, msg in enumerate(prev_msgs):
                assert curr_msgs[j] == msg, (
                    f"Turn {i} should include all messages from turn {i-1}"
                )


@requires_transformers
class TestHighConcurrency:
    def test_produces_50_requests(self, tokenizer):
        scenario = high_concurrency(tokenizer)
        assert len(scenario.requests) == 50

    def test_starting_concurrency_is_8(self, tokenizer):
        scenario = high_concurrency(tokenizer)
        assert scenario.concurrency == 8

    def test_input_tokens_in_range(self, tokenizer):
        scenario = high_concurrency(tokenizer)
        for r in scenario.requests:
            actual = r.metadata["input_tokens"]
            assert 80 <= actual <= 520, f"Input tokens {actual} outside expected range"


@requires_transformers
class TestLongContext:
    def test_produces_50_requests(self, tokenizer):
        scenario = long_context(tokenizer)
        assert len(scenario.requests) == 50

    def test_max_tokens_is_256(self, tokenizer):
        scenario = long_context(tokenizer)
        for r in scenario.requests:
            assert r.max_tokens == 256

    def test_input_tokens_in_range(self, tokenizer):
        scenario = long_context(tokenizer)
        for r in scenario.requests:
            actual = r.metadata["input_tokens"]
            assert 2000 <= actual <= 4150, f"Input tokens {actual} outside expected range"


# --- Warmup tests ---


@requires_transformers
class TestWarmup:
    def test_produces_10_requests_by_default(self, tokenizer):
        warmup = generate_warmup(tokenizer)
        assert len(warmup) == 10

    def test_custom_count(self, tokenizer):
        warmup = generate_warmup(tokenizer, n=5)
        assert len(warmup) == 5

    def test_warmup_prompts_differ_from_prefix_heavy(self, tokenizer):
        warmup = generate_warmup(tokenizer)
        scenario = prefix_heavy(tokenizer)
        warmup_texts = {r.messages[0]["content"] for r in warmup}
        scenario_texts = set()
        for r in scenario.requests:
            for msg in r.messages:
                scenario_texts.add(msg["content"])
        overlap = warmup_texts & scenario_texts
        assert len(overlap) == 0, "Warmup prompts must not overlap with test prompts"

    def test_warmup_max_tokens_is_short(self, tokenizer):
        warmup = generate_warmup(tokenizer)
        for r in warmup:
            assert r.max_tokens == 32

    def test_warmup_scenario_label(self, tokenizer):
        warmup = generate_warmup(tokenizer)
        for r in warmup:
            assert r.scenario == "warmup"


# --- Utility tests ---


@requires_transformers
class TestGetScenario:
    def test_known_scenario(self, tokenizer):
        s = get_scenario("prefix_heavy", tokenizer)
        assert s.name == "prefix_heavy"

    def test_unknown_scenario_raises(self, tokenizer):
        with pytest.raises(ValueError, match="Unknown scenario"):
            get_scenario("nonexistent", tokenizer)

    def test_custom_seed(self, tokenizer):
        s1 = get_scenario("prefix_heavy", tokenizer, seed=100)
        s2 = get_scenario("prefix_heavy", tokenizer, seed=200)
        assert s1.requests[0].messages != s2.requests[0].messages


@requires_transformers
class TestGetAllScenarios:
    def test_returns_all_four(self, tokenizer):
        scenarios = get_all_scenarios(tokenizer)
        assert len(scenarios) == 4
        names = [s.name for s in scenarios]
        assert names == SCENARIO_ORDER


# ===================================================================
# run_benchmark.py tests
# ===================================================================

import csv
import json
from pathlib import Path

from run_benchmark import (
    LoadedScenario,
    PromptRequest,
    RequestResult,
    ScenarioResult,
    build_server_cmd,
    compute_metrics,
    load_prompts,
    load_scenario,
    load_warmup,
    write_comparison_csv,
    write_results_json,
)


# --- Metrics computation tests ---


class TestComputeMetrics:
    def _make_result(self, request_results):
        result = ScenarioResult(framework="test", scenario="test")
        result.requests = request_results
        result.wall_clock_s = 10.0
        return result

    def test_basic_metrics(self):
        requests = [
            RequestResult(i, ttft_ms=10 + i, tpot_ms=5 + i,
                          e2e_latency_ms=100 + i * 10, tokens_generated=50)
            for i in range(10)
        ]
        result = self._make_result(requests)
        compute_metrics(result)

        assert result.ttft_p50_ms > 0
        assert result.tpot_p50_ms > 0
        assert result.throughput_tok_s > 0
        assert result.error_count == 0
        assert result.total_tokens == 500

    def test_empty_results(self):
        result = self._make_result([])
        compute_metrics(result)
        assert result.ttft_p50_ms == 0
        assert result.throughput_tok_s == 0

    def test_all_errors(self):
        requests = [
            RequestResult(i, ttft_ms=0, tpot_ms=0, e2e_latency_ms=0,
                          tokens_generated=0, error="connection refused")
            for i in range(5)
        ]
        result = self._make_result(requests)
        compute_metrics(result)
        assert result.error_count == 5
        assert result.ttft_p50_ms == 0

    def test_single_request(self):
        requests = [
            RequestResult(0, ttft_ms=42.0, tpot_ms=8.5,
                          e2e_latency_ms=500.0, tokens_generated=50)
        ]
        result = self._make_result(requests)
        compute_metrics(result)
        assert result.ttft_p50_ms == 42.0
        assert result.ttft_p90_ms == 42.0
        assert result.ttft_p99_ms == 42.0
        assert result.tpot_p50_ms == 8.5

    def test_mixed_success_and_errors(self):
        requests = [
            RequestResult(0, ttft_ms=10, tpot_ms=5, e2e_latency_ms=100, tokens_generated=20),
            RequestResult(1, ttft_ms=0, tpot_ms=0, e2e_latency_ms=0,
                          tokens_generated=0, error="timeout"),
            RequestResult(2, ttft_ms=20, tpot_ms=8, e2e_latency_ms=200, tokens_generated=30),
        ]
        result = self._make_result(requests)
        compute_metrics(result)
        assert result.error_count == 1
        assert result.total_tokens == 50
        assert result.ttft_p50_ms == pytest.approx(15.0)

    def test_zero_tpot_excluded(self):
        requests = [
            RequestResult(0, ttft_ms=10, tpot_ms=0, e2e_latency_ms=100, tokens_generated=1),
            RequestResult(1, ttft_ms=20, tpot_ms=5, e2e_latency_ms=200, tokens_generated=10),
        ]
        result = self._make_result(requests)
        compute_metrics(result)
        assert result.tpot_p50_ms == 5.0

    def test_zero_wall_clock(self):
        requests = [
            RequestResult(0, ttft_ms=10, tpot_ms=5, e2e_latency_ms=100, tokens_generated=50)
        ]
        result = self._make_result(requests)
        result.wall_clock_s = 0.0
        compute_metrics(result)
        assert result.throughput_tok_s == 0.0


# --- Report output tests ---


class TestResultsJson:
    def test_schema(self, tmp_path):
        results = {
            "sglang": {
                "prefix_heavy": ScenarioResult(
                    framework="sglang", scenario="prefix_heavy",
                    requests=[RequestResult(0, 10, 5, 100, 50)],
                    total_tokens=50, wall_clock_s=1.0,
                    ttft_p50_ms=10, throughput_tok_s=50,
                ),
            },
        }
        metadata = {"timestamp": "2026-01-01T00:00:00Z", "model": "test"}
        out = tmp_path / "results.json"

        write_results_json(results, out, metadata)

        data = json.loads(out.read_text())
        assert "metadata" in data
        assert "results" in data
        assert "sglang" in data["results"]
        scenario_data = data["results"]["sglang"]["prefix_heavy"]
        for key in ["ttft_p50_ms", "tpot_p50_ms", "throughput_tok_s",
                     "total_tokens", "error_count", "per_request"]:
            assert key in scenario_data

    def test_per_request_data(self, tmp_path):
        results = {
            "sglang": {
                "test": ScenarioResult(
                    framework="sglang", scenario="test",
                    requests=[
                        RequestResult(0, 10, 5, 100, 50),
                        RequestResult(1, 20, 8, 200, 60, error="timeout"),
                    ],
                ),
            },
        }
        out = tmp_path / "results.json"
        write_results_json(results, out, {})

        data = json.loads(out.read_text())
        per_req = data["results"]["sglang"]["test"]["per_request"]
        assert len(per_req) == 2
        assert per_req[1]["error"] == "timeout"


class TestComparisonCsv:
    def test_schema(self, tmp_path):
        results = {
            "sglang": {
                "prefix_heavy": ScenarioResult(
                    framework="sglang", scenario="prefix_heavy",
                    ttft_p50_ms=10, tpot_p50_ms=5, throughput_tok_s=100,
                ),
            },
            "vllm": {
                "prefix_heavy": ScenarioResult(
                    framework="vllm", scenario="prefix_heavy",
                    ttft_p50_ms=20, tpot_p50_ms=10, throughput_tok_s=60,
                ),
            },
        }
        out = tmp_path / "comparison.csv"
        write_comparison_csv(results, out)

        with open(out) as f:
            rows = list(csv.DictReader(f))

        assert len(rows) == 2
        assert rows[0]["framework"] == "sglang"
        assert float(rows[0]["ttft_p50_ms"]) == pytest.approx(10.0)


# --- CLI tests ---


class TestBuildServerCmd:
    def test_sglang_cmd(self):
        cmd = build_server_cmd("sglang", "/workspace/models/test", 30000)
        joined = " ".join(cmd)
        assert "sglang.launch_server" in joined
        assert "--model-path" in cmd
        assert "30000" in cmd

    def test_vllm_cmd(self):
        cmd = build_server_cmd("vllm", "/workspace/models/test", 30000)
        joined = " ".join(cmd)
        assert "vllm.entrypoints.openai.api_server" in joined
        assert "--trust-remote-code" in cmd

    def test_unknown_framework_raises(self):
        with pytest.raises(ValueError, match="Unknown framework"):
            build_server_cmd("unknown", "/test", 30000)


# --- Prompt loading tests ---


class TestLoadPrompts:
    @pytest.fixture
    def sample_prompts(self, tmp_path):
        data = {
            "prefix_heavy": {
                "description": "test scenario",
                "concurrency": 1,
                "num_requests": 2,
                "requests": [
                    {"request_id": 0, "messages": [{"role": "user", "content": "hello"}], "max_tokens": 128},
                    {"request_id": 1, "messages": [{"role": "user", "content": "world"}], "max_tokens": 128},
                ],
            },
            "warmup": {
                "description": "warmup",
                "num_requests": 1,
                "requests": [
                    {"request_id": 0, "messages": [{"role": "user", "content": "warmup"}], "max_tokens": 32},
                ],
            },
        }
        path = tmp_path / "prompts.json"
        path.write_text(json.dumps(data))
        return path

    def test_load_prompts(self, sample_prompts):
        data = load_prompts(sample_prompts)
        assert "prefix_heavy" in data
        assert "warmup" in data

    def test_load_scenario(self, sample_prompts):
        data = load_prompts(sample_prompts)
        scenario = load_scenario(data, "prefix_heavy")
        assert isinstance(scenario, LoadedScenario)
        assert scenario.name == "prefix_heavy"
        assert scenario.concurrency == 1
        assert len(scenario.requests) == 2
        assert isinstance(scenario.requests[0], PromptRequest)
        assert scenario.requests[0].messages == [{"role": "user", "content": "hello"}]

    def test_load_scenario_unknown(self, sample_prompts):
        data = load_prompts(sample_prompts)
        with pytest.raises(KeyError, match="nonexistent"):
            load_scenario(data, "nonexistent")

    def test_load_warmup(self, sample_prompts):
        data = load_prompts(sample_prompts)
        warmup = load_warmup(data)
        assert len(warmup) == 1
        assert warmup[0].messages == [{"role": "user", "content": "warmup"}]
        assert warmup[0].max_tokens == 32

    def test_load_warmup_missing(self, tmp_path):
        path = tmp_path / "prompts.json"
        path.write_text('{"prefix_heavy": {}}')
        data = load_prompts(path)
        with pytest.raises(KeyError, match="warmup"):
            load_warmup(data)
