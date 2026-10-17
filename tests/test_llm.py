"""Request limit and retries of the LLM client, without an API key and without really waiting."""
from types import SimpleNamespace

import httpx
import openai
import pytest
from fastapi.testclient import TestClient

from workshop_a2a import llm, registry
from workshop_a2a.llm import WINDOW, OpenAIChat, RequestWindow

# shortened 429 answer of the Gemini free tier
GEMINI_PER_MINUTE = (
    "Error code: 429 - [{'error': {'code': 429, 'message': 'You exceeded your current quota. "
    "Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, "
    "limit: 5, model: gemini-3.6-flash\\nPlease retry in 9.343594093s.', 'status': 'RESOURCE_EXHAUSTED', "
    "'details': [{'quotaId': 'GenerateRequestsPerMinutePerProjectPerModel-FreeTier'}, {'retryDelay': '9s'}]}}]")
GEMINI_PER_DAY = GEMINI_PER_MINUTE.replace("PerMinute", "PerDay")


def rate_limit(text, headers=None):
    response = httpx.Response(429, headers=headers, request=httpx.Request("POST", "http://llm.test/chat/completions"))
    return openai.RateLimitError(text, response=response, body=None)


class FakeCompletions:
    """Raises the scripted errors in order and answers "ok" otherwise."""

    def __init__(self, errors=()):
        self.errors, self.calls = list(errors), 0

    def create(self, **kwargs):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        message = SimpleNamespace(content="ok", tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


@pytest.fixture
def sleeps(monkeypatch):
    waited = []
    monkeypatch.setattr(llm.time, "sleep", waited.append)
    return waited


@pytest.fixture
def make_chat(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://llm.test/v1/")
    monkeypatch.setenv("LLM_API_KEY", "test")
    monkeypatch.setenv("LLM_MODEL", "gemini-test")

    def make(errors=(), rpm=0):
        monkeypatch.setenv("LLM_RPM", str(rpm))
        chat = OpenAIChat()
        chat.client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions(errors)))
        return chat

    return make


def ask(chat):
    return chat.complete([{"role": "user", "content": "Pump vibrates."}])["content"]


def test_request_window_allows_rpm_requests_per_window():
    window = RequestWindow()
    assert [window.reserve("gemini", 5, now=1000) for _ in range(6)] == [0, 0, 0, 0, 0, WINDOW]
    assert window.reserve("other-model", 5, now=1000) == 0
    # once the first five have left the window, a new request starts right away
    assert window.reserve("gemini", 5, now=1000 + WINDOW + 1) == 0


def test_registry_hands_out_llm_slots():
    with TestClient(registry.create_app()) as client:
        waits = [client.post("/llm-slots", json={"model": "gemini", "rpm": 5}).json()["wait"] for _ in range(6)]
        assert waits[:5] == [0] * 5 and waits[5] > WINDOW - 2
        assert client.post("/llm-slots", json={"model": "gemini", "rpm": 0}).status_code == 422


def test_agents_share_the_limit_through_the_registry(make_chat, sleeps, monkeypatch):
    with TestClient(registry.create_app()) as client:
        monkeypatch.setattr(llm.httpx, "post", lambda url, json, timeout: client.post(url, json=json))
        diagnosis_llm, parts_llm = make_chat(rpm=5), make_chat(rpm=5)
        for chat in (diagnosis_llm, parts_llm, diagnosis_llm, parts_llm, diagnosis_llm):
            assert ask(chat) == "ok"
        assert sleeps == []
        # the sixth request in a minute waits, although this agent has only sent two so far
        assert ask(parts_llm) == "ok"
    assert len(sleeps) == 1 and sleeps[0] > WINDOW - 2


def test_without_registry_the_limit_holds_per_process(make_chat, sleeps, monkeypatch):
    def unreachable(*args, **kwargs):
        raise httpx.ConnectError("registry is down")

    monkeypatch.setattr(llm.httpx, "post", unreachable)
    monkeypatch.setattr(llm, "_local_window", RequestWindow())
    chat = make_chat(rpm=5)
    for _ in range(6):
        ask(chat)
    assert len(sleeps) == 1 and sleeps[0] > WINDOW - 2


def test_rate_limit_waits_as_long_as_gemini_asks(make_chat, sleeps):
    chat = make_chat([rate_limit(GEMINI_PER_MINUTE)])
    assert ask(chat) == "ok"
    assert sleeps == [pytest.approx(10.34, abs=0.01)]


def test_rate_limit_uses_retry_after_header(make_chat, sleeps):
    chat = make_chat([rate_limit("Error code: 429 - slow down", {"retry-after": "20"})])
    assert ask(chat) == "ok"
    assert sleeps == [21]


def test_used_up_daily_quota_fails_at_once(make_chat, sleeps):
    chat = make_chat([rate_limit(GEMINI_PER_DAY)])
    with pytest.raises(openai.RateLimitError):
        ask(chat)
    assert chat.client.chat.completions.calls == 1 and sleeps == []


def test_gives_up_after_max_attempts(make_chat, sleeps):
    chat = make_chat([rate_limit(GEMINI_PER_MINUTE)] * llm.MAX_ATTEMPTS)
    with pytest.raises(openai.RateLimitError):
        ask(chat)
    assert chat.client.chat.completions.calls == llm.MAX_ATTEMPTS and len(sleeps) == llm.MAX_ATTEMPTS - 1
