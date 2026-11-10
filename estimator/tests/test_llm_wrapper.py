from types import SimpleNamespace
from unittest.mock import patch

import fakeredis
import pytest

from app.services.cache import EstimationCache
from app.services.llm_wrapper import FALLBACK_GROUP, PRIMARY_GROUP, LLMWrapper, _estimate_cost


def _fake_completion(
    model: str, content: str = "the answer", input_tokens: int = 100, output_tokens: int = 50
):
    """Build a SimpleNamespace shaped like a litellm.ModelResponse."""
    return SimpleNamespace(
        model=model,
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        ),
    )


@pytest.fixture
def wrapper() -> LLMWrapper:
    cache = EstimationCache(fakeredis.FakeRedis(decode_responses=True), ttl=60)
    return LLMWrapper(
        openai_api_key="fake-openai",
        anthropic_api_key="fake-anthropic",
        primary_model="gpt-4o-mini",
        fallback_model="claude-haiku-4-5-20251001",
        timeout=30,
        num_retries=2,
        cache=cache,
    )


def test_estimate_cost_uses_pricing_table() -> None:
    cost = _estimate_cost("gpt-4o-mini", 1_000_000, 1_000_000)
    # 1M input * 0.15 + 1M output * 0.60 = 0.75 USD
    assert cost == pytest.approx(0.75)


def test_complete_returns_normalised_dict_and_caches(wrapper: LLMWrapper) -> None:
    fake = _fake_completion(model="gpt-4o-mini", content="hello world")
    with patch.object(wrapper.router, "completion", return_value=fake) as mocked:
        result = wrapper.complete(
            system_prompt="sys",
            user_message="usr",
            model_override=None,
            max_tokens=4000,
            thinking_budget=None,
        )
    assert mocked.call_count == 1
    assert result["estimation"] == "hello world"
    assert result["model"] == "gpt-4o-mini"
    assert result["provider"] == "openai"
    assert result["finish_reason"] == "stop"
    assert result["usage"]["input_tokens"] == 100
    assert result["usage"]["output_tokens"] == 50
    assert result["cache_hit"] is False
    assert result["cost_usd"] > 0

    # Second call with the same inputs should hit the cache without invoking the router.
    with patch.object(wrapper.router, "completion") as mocked_again:
        cached = wrapper.complete(
            system_prompt="sys",
            user_message="usr",
            model_override=None,
            max_tokens=4000,
            thinking_budget=None,
        )
    assert mocked_again.call_count == 0
    assert cached["cache_hit"] is True
    assert cached["estimation"] == "hello world"


def test_complete_with_model_override_bypasses_router(wrapper: LLMWrapper) -> None:
    fake = _fake_completion(model="gpt-4o", content="overridden")
    with (
        patch("app.services.llm_wrapper.litellm.completion", return_value=fake) as direct,
        patch.object(wrapper.router, "completion") as router_call,
    ):
        result = wrapper.complete(
            system_prompt="sys",
            user_message="usr",
            model_override="gpt-4o",
            max_tokens=4000,
            thinking_budget=None,
        )
    assert direct.call_count == 1
    assert router_call.call_count == 0
    assert direct.call_args.kwargs["model"] == "gpt-4o"
    assert result["model"] == "gpt-4o"


def test_thinking_budget_passed_for_anthropic_fallback(wrapper: LLMWrapper) -> None:
    fake = _fake_completion(model="claude-haiku-4-5-20251001", content="ok")
    with patch.object(wrapper.router, "completion", return_value=fake) as mocked:
        wrapper.complete(
            system_prompt="sys",
            user_message="usr",
            model_override=None,
            max_tokens=4000,
            thinking_budget=2048,
        )
    # primary is OpenAI (gpt-4o-mini), so thinking budget is *ignored* in kwargs.
    assert "thinking" not in mocked.call_args.kwargs


def test_thinking_budget_pads_max_tokens_when_anthropic_override(wrapper: LLMWrapper) -> None:
    fake = _fake_completion(model="claude-haiku-4-5-20251001", content="ok")
    with patch("app.services.llm_wrapper.litellm.completion", return_value=fake) as direct:
        wrapper.complete(
            system_prompt="sys",
            user_message="usr",
            model_override="claude-haiku-4-5-20251001",
            max_tokens=1000,
            thinking_budget=4096,
        )
    kwargs = direct.call_args.kwargs
    assert kwargs["thinking"] == {"type": "enabled", "budget_tokens": 4096}
    assert kwargs["max_tokens"] == 4096 + 1024


def test_complete_stream_yields_chunks_and_caches(wrapper: LLMWrapper) -> None:
    chunks = [
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="Hello "))]),
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="world"))]),
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=None))]),
    ]
    with patch.object(wrapper.router, "completion", return_value=iter(chunks)):
        emitted = list(
            wrapper.complete_stream(
                system_prompt="sys",
                user_message="usr",
                model_override=None,
                max_tokens=4000,
            )
        )
    assert "".join(emitted) == "Hello world"

    # Now the same request hits the cache and replays the full text as one chunk.
    with patch.object(wrapper.router, "completion") as router_call:
        replayed = list(
            wrapper.complete_stream(
                system_prompt="sys",
                user_message="usr",
                model_override=None,
                max_tokens=4000,
            )
        )
    assert router_call.call_count == 0
    assert "".join(replayed) == "Hello world"


def test_complete_stream_reports_usage_metrics_and_caches_them(wrapper: LLMWrapper) -> None:
    chunks = [
        SimpleNamespace(
            model="gpt-4o-mini-2024-07-18",
            choices=[SimpleNamespace(delta=SimpleNamespace(content="Hi"))],
            usage=None,
        ),
        # OpenAI sends usage in a final chunk with no choices when include_usage is set.
        SimpleNamespace(
            model="gpt-4o-mini-2024-07-18",
            choices=[],
            usage=SimpleNamespace(prompt_tokens=1000, completion_tokens=200),
        ),
    ]
    metrics: dict = {}
    with patch.object(wrapper.router, "completion", return_value=iter(chunks)) as call:
        list(
            wrapper.complete_stream(
                system_prompt="sys",
                user_message="usr",
                max_tokens=4000,
                metrics=metrics,
            )
        )
    assert call.call_args.kwargs["stream_options"] == {"include_usage": True}
    assert metrics["model"] == "gpt-4o-mini-2024-07-18"
    assert metrics["usage"] == {"input_tokens": 1000, "output_tokens": 200, "total_tokens": 1200}
    assert metrics["cost_usd"] == _estimate_cost("gpt-4o-mini", 1000, 200)
    assert metrics["cache_hit"] is False

    cached_metrics: dict = {}
    list(
        wrapper.complete_stream(
            system_prompt="sys",
            user_message="usr",
            max_tokens=4000,
            metrics=cached_metrics,
        )
    )
    assert cached_metrics["cache_hit"] is True
    assert cached_metrics["usage"] == metrics["usage"]


def test_estimate_cost_resolves_dated_snapshots() -> None:
    assert _estimate_cost("gpt-4o-mini-2024-07-18", 1_000_000, 0) == 0.15


def test_router_always_picks_primary_and_falls_back_on_failure(wrapper: LLMWrapper) -> None:
    # The primary group must hold only the primary model, otherwise the Router
    # load-balances and sends healthy traffic to the (pricier) fallback.
    picked = {
        wrapper.router.get_available_deployment(model=PRIMARY_GROUP)["litellm_params"]["model"]
        for _ in range(50)
    }
    assert picked == {"gpt-4o-mini"}
    assert wrapper.router.fallbacks == [{PRIMARY_GROUP: [FALLBACK_GROUP]}]

    fallback = wrapper.router.get_available_deployment(model=FALLBACK_GROUP)
    assert fallback["litellm_params"]["model"] == "claude-haiku-4-5-20251001"


def test_complete_messages_cache_key_covers_the_whole_history(wrapper):
    """The same user message with a different history must not hit the cache."""
    turn_1 = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "Add payments"},
    ]
    turn_2 = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "Booking app"},
        {"role": "assistant", "content": "120 h"},
        {"role": "user", "content": "Add payments"},
    ]
    with patch.object(
        wrapper.router, "completion", return_value=_fake_completion("gpt-4o-mini", "ok")
    ) as completion:
        wrapper.complete_messages(turn_1)
        second = wrapper.complete_messages(turn_2)
        replay = wrapper.complete_messages(turn_2)

    assert completion.call_count == 2
    assert second["cache_hit"] is False
    assert replay["cache_hit"] is True
    assert completion.call_args.kwargs["messages"] == turn_2


def test_complete_keeps_the_session_4_cache_key(wrapper):
    """complete() delegates to complete_messages() without changing its cache key."""
    with patch.object(
        wrapper.router, "completion", return_value=_fake_completion("gpt-4o-mini", "ok")
    ):
        wrapper.complete(system_prompt="sys", user_message="hello")

    key = EstimationCache.make_key(
        system_prompt="sys",
        user_message="hello",
        model=wrapper.primary_model,
        max_tokens=4000,
        thinking_budget=None,
    )
    assert wrapper.cache.get(key) is not None
