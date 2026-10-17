"""Coordinator agent: looks up the available agents in the registry for every request, lets the LLM
pick the next agent, calls it over A2A and repeats until the request is fully answered."""
import json
import os

from .a2a import agent_card, create_agent_app, list_agents, send_message, serve, skill
from .llm import REGISTRY_URL, OpenAIChat

NAME = "Maintenance Coordinator"
SYSTEM_PROMPT = """You coordinate maintenance requests in a workshop. You do not solve tasks yourself, you delegate them to the agents listed below. The list was loaded from the agent registry just now.
Reply with exactly one JSON object and nothing else, either
{"action": "call", "agent": "<agent name>", "message": "<task for this agent with all facts it needs>"}
or, when every part of the user's request is answered,
{"action": "finish", "answer": "<final answer for the user>"}
The agents do not see each other's results, so pass on part numbers, quantities, lead times, skill and work hours yourself.

Available agents:
"""


def parse_decision(text):
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        decision = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    return decision if decision.get("action") in ("call", "finish") else None


def coordinate(llm, registry_url, request, max_rounds=8, log=print):
    agents = {a["name"]: a for a in list_agents(registry_url) if a["name"] != NAME}
    log(f"[coordinator] registry lookup: {', '.join(agents) or 'no agents'}")
    catalog = "\n".join(
        f"- {a['name']}: {a['description']} Skills: "
        + "; ".join(f"{s['name']} ({s['description']})" for s in a.get("skills", []))
        for a in agents.values())
    messages = [{"role": "system", "content": SYSTEM_PROMPT + catalog}, {"role": "user", "content": request}]
    used = []
    for _ in range(max_rounds):
        reply = llm.complete(messages)["content"] or ""
        messages.append({"role": "assistant", "content": reply})
        decision = parse_decision(reply)
        if decision is None:
            messages.append({"role": "user", "content": "Reply with one JSON object as described."})
            continue
        if decision["action"] == "finish":
            return f"{decision.get('answer', '')}\n\nAgents used: {' -> '.join(used) or 'none'}"
        agent = agents.get(decision.get("agent"))
        if agent is None:
            result = f"Unknown agent {decision.get('agent')!r}. Choose one from the list."
        else:
            log(f"[coordinator] -> {agent['name']} ({agent['url']}): {decision.get('message', '')}")
            try:
                # generous timeout: with LLM_RPM the agent may wait for free request slots
                result = send_message(agent["url"], decision.get("message", ""), timeout=300)
            except Exception as exc:
                result = f"Call failed: {exc}"
            used.append(agent["name"])
            log(f"[coordinator] <- {agent['name']}: {result}")
        messages.append({"role": "user", "content": f"Result from {decision.get('agent')}:\n{result}"})
    return f"No complete answer after {max_rounds} rounds.\n\nAgents used: {' -> '.join(used) or 'none'}"


def create_app(llm=None, registry_url=REGISTRY_URL, url=None):
    llm = llm or OpenAIChat("COORDINATOR_MODEL")
    card = agent_card(
        NAME, "Entry point for maintenance requests. Finds the right agents in the registry and combines their answers.",
        url or os.environ.get("PUBLIC_URL", "http://localhost:8110/"),
        [skill("coordinate-maintenance-request", "Coordinate maintenance request",
               "Takes a fault report and returns cause, spare parts status and repair date.",
               ["coordinator", "maintenance"])])
    return create_agent_app(card, lambda text: coordinate(llm, registry_url, text), registry_url)


def main():
    serve(create_app(), int(os.environ.get("PORT", 8110)))


if __name__ == "__main__":
    main()
