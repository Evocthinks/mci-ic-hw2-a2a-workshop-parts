"""OpenAI-compatible chat client (LiteLLM, Gemini, Azure, ...) with a shared request limit and a small tool-calling loop."""
import json
import os
import re
import threading
import time

import httpx

try:
    from dotenv import load_dotenv
except ImportError:  # dotenv is optional, plain environment variables work too
    def load_dotenv():
        return False

load_dotenv()

# 127.0.0.1, not localhost: on Windows localhost tries IPv6 first and costs 2 s per request
REGISTRY_URL = os.environ.get("REGISTRY_URL", "http://127.0.0.1:8100")
WINDOW = 61  # seconds: the provider's one-minute window plus a margin for network delays
MAX_ATTEMPTS = 4
MAX_RETRY_WAIT = 120  # a longer wait means a daily quota or billing problem, not the per-minute limit


class RequestWindow:
    """Hands out start times so that at most `limit` requests per key fall into any WINDOW seconds."""

    def __init__(self):
        self.starts = {}
        self.lock = threading.Lock()

    def reserve(self, key, limit, now=None):
        """Reserves the next free slot and returns the seconds to wait for it."""
        with self.lock:
            now = time.time() if now is None else now
            starts = [t for t in self.starts.get(key, []) if t > now - WINDOW]
            # first come, first served, so the start times stay sorted
            start = max(now, starts[-1]) if starts else now
            if len(starts) >= limit:
                start = max(start, starts[-limit] + WINDOW)
            self.starts[key] = starts + [start]
            return start - now


_local_window = RequestWindow()  # used when the registry cannot be reached


def retry_delay(exc, attempt):
    """Seconds to wait before the next attempt, or None if waiting will not help."""
    if getattr(exc, "status_code", None) != 429:
        return 2 ** attempt  # overloaded model or network problem
    text = str(exc)
    if "PerDay" in text:  # Gemini: a used-up daily quota does not come back within minutes
        return None
    # Gemini writes "Please retry in 9.34s", other servers send a Retry-After header
    match = re.search(r"retry in ([\d.]+)s", text)
    try:
        delay = float(match[1]) if match else float(exc.response.headers["retry-after"])
    except (AttributeError, KeyError, ValueError):
        delay = 30
    return delay + 1 if delay <= MAX_RETRY_WAIT else None


class OpenAIChat:
    def __init__(self, model_env=None, registry_url=REGISTRY_URL):
        from openai import OpenAI
        # complete() retries itself, so that every attempt counts against the request limit
        self.client = OpenAI(base_url=os.environ["LLM_BASE_URL"], api_key=os.environ["LLM_API_KEY"], max_retries=0)
        # every agent can run on its own model, e.g. PARTS_MODEL=gpt-4o-mini
        self.model = os.environ.get(model_env or "", "") or os.environ["LLM_MODEL"]
        # free tiers allow only a few requests per minute and model, e.g. LLM_RPM=5 for Gemini
        self.rpm = int(os.environ.get("LLM_RPM") or 0)
        self.registry_url = registry_url

    def wait_for_slot(self):
        if not self.rpm:
            return
        try:
            # the agents run as separate processes on one quota, so the registry hands out the slots
            response = httpx.post(f"{self.registry_url}/llm-slots",
                                  json={"model": self.model, "rpm": self.rpm}, timeout=5)
            response.raise_for_status()
            wait = response.json()["wait"]
        except httpx.HTTPError:
            wait = _local_window.reserve(self.model, self.rpm)
        if wait > 0:
            print(f"[llm] {self.model}: waiting {wait:.0f} s for a free request slot (LLM_RPM={self.rpm})", flush=True)
            time.sleep(wait)

    def complete(self, messages, tools=None):
        from openai import APIConnectionError, InternalServerError, RateLimitError
        kwargs = {"model": self.model, "messages": messages}
        if tools:
            kwargs["tools"] = tools
        for attempt in range(1, MAX_ATTEMPTS + 1):
            self.wait_for_slot()
            try:
                message = self.client.chat.completions.create(**kwargs).choices[0].message
                break
            except (RateLimitError, InternalServerError, APIConnectionError) as exc:
                delay = retry_delay(exc, attempt)
                if delay is None or attempt == MAX_ATTEMPTS:
                    raise
                print(f"[llm] {self.model}: {type(exc).__name__}, retry in {delay:.0f} s", flush=True)
                time.sleep(delay)
        # provider fields of a call go back unchanged, Gemini 3 rejects calls without its thought_signature
        calls = [{"id": c.id, "name": c.function.name,
                  "arguments": json.loads(c.function.arguments or "{}"), "extra": c.model_extra or {}}
                 for c in (message.tool_calls or [])]
        return {"content": message.content, "tool_calls": calls}


def tool(function, description, parameters):
    return {"function": function, "description": description, "parameters": parameters}


def run_tool_agent(llm, system_prompt, tools, user_text, max_steps=6):
    specs = [{"type": "function", "function": {"name": name, "description": t["description"],
                                               "parameters": t["parameters"]}}
             for name, t in tools.items()]
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_text}]
    for _ in range(max_steps):
        reply = llm.complete(messages, specs)
        if not reply["tool_calls"]:
            return reply["content"] or ""
        messages.append({"role": "assistant", "content": reply["content"], "tool_calls": [
            {"id": c["id"], "type": "function",
             "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}, **c.get("extra", {})}
            for c in reply["tool_calls"]]})
        for call in reply["tool_calls"]:
            try:
                result = tools[call["name"]]["function"](**call["arguments"])
            except Exception as exc:  # wrong tool name or arguments go back to the model
                result = {"error": f"{type(exc).__name__}: {exc}"}
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result)})
    return "Stopped after too many tool calls."
