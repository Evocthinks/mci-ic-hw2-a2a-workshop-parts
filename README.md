# Workshop A2A: distributed maintenance agents with registry lookup

A fault report from the shop floor ("pump P-200 vibrates, bearing noise") is answered by three remote agents that run as separate processes on their own ports. A coordinator looks them up in an agent registry for every request, lets its LLM pick the next agent and calls it over A2A (JSON-RPC `message/send`) until cause, spare parts and repair date are known.

| Service | Port | Job |
| --- | --- | --- |
| Registry | 8100 | Agents post their A2A agent card on startup; `GET /agents` lists them. Hands out LLM request slots (`POST /llm-slots`), so that all agents together stay below the request limit. |
| Diagnosis Agent | 8101 | Symptom to cause, spare parts, skill and work hours. |
| Parts Agent | 8102 | Stock, storage location, price and supplier lead time. |
| Scheduling Agent | 8103 | First free technician with the right skill after the parts arrive. |
| Maintenance Coordinator | 8110 | Entry point, registry lookup, delegation, checks if the request is fully answered. |

Every agent publishes its card at `/.well-known/agent-card.json` and can use its own model (`DIAGNOSIS_MODEL`, `PARTS_MODEL`, `SCHEDULING_MODEL`, `COORDINATOR_MODEL`).

## Setup

Clone the repository, then in its folder:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create a file `.env` in this folder for any OpenAI-compatible endpoint (LiteLLM proxy, Gemini free tier, Azure). For the Gemini free tier, get a key in [Google AI Studio](https://aistudio.google.com/):

```
LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
LLM_API_KEY=your-key
LLM_MODEL=gemini-3.6-flash
LLM_RPM=5
```

`LLM_RPM` is the request limit per minute of your model, see [Request limit](#request-limit-gemini-free-tier). Leave it out for endpoints without a tight limit.

## Run locally (five processes)

```bash
python run_all.py
```

In a second terminal:

```bash
python ask.py
python ask.py "The belt conveyor in hall 1 keeps slipping and squeals. When can it be fixed?"
```

## Request limit (Gemini free tier)

Free tiers allow only a few LLM requests per minute and model. The Gemini free tier allows 5 per minute for `gemini-3.6-flash` ([rate limits](https://ai.google.dev/gemini-api/docs/rate-limits), your current usage at [ai.dev/rate-limit](https://ai.dev/rate-limit)). One fault report needs about ten LLM requests: four for the coordinator and two for each agent. Without a limit, the sixth request fails with `429 RESOURCE_EXHAUSTED`.

- **Shared limit.** With `LLM_RPM=5`, all agents together send at most 5 requests in any 61 seconds. The agents run as separate processes on one quota, so before every LLM call each agent reserves a slot at the registry (`POST /llm-slots`) and waits if needed. Agents with their own model (`PARTS_MODEL`, ...) get their own slots, because Google counts per model. If the registry cannot be reached, the limit only holds per process.
- **Retry after 429.** If the provider still answers 429, the agent waits as long as the provider asks (`Please retry in 9s` from Gemini or a `Retry-After` header) and tries up to four times. Overloaded models (5xx) and network errors also get short retries. A used-up daily quota (`PerDay` in the Gemini error) fails at once, because waiting would not help.
- **Duration.** A request takes one to two minutes. `ask.py` prints the registry list and the final answer; the terminal of `run_all.py` shows the agent calls and lines like `[llm] gemini-3.6-flash: waiting 55 s for a free request slot (LLM_RPM=5)`. The coordinator waits up to 5 minutes for each agent, `ask.py` up to 10 minutes for the answer.
- **Daily quota.** The free tier also limits the requests per day and model. Every run of `ask.py` uses about ten of them.

## Gemini 3 tool calls

Gemini 3 attaches a `thought_signature` to every tool call and rejects the next request with `400 INVALID_ARGUMENT` if the signature is missing. The tool loop in `workshop_a2a/llm.py` therefore sends every provider field of a tool call back unchanged. Other OpenAI-compatible endpoints are not affected.

## Run with Docker

```bash
docker compose up --build
docker compose exec coordinator python ask.py
```

## Tests

```bash
python -m pytest -q
```

The tests need no API key. They start all five services as real HTTP servers with scripted LLMs and check the full chain registry, coordinator, diagnosis, parts, scheduling. `tests/test_llm.py` checks the shared request limit, the retries after a 429 (without waiting) and that Gemini's thought signatures go back with the tool calls.
