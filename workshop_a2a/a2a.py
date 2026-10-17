"""Small A2A helpers (JSON-RPC binding, protocol 0.3): agent card, server app, client, registry calls."""
import asyncio
import uuid
from contextlib import asynccontextmanager

import httpx
import uvicorn
from fastapi import FastAPI, Request
from starlette.concurrency import run_in_threadpool

PROTOCOL_VERSION = "0.3.0"


class A2AError(Exception):
    pass


def skill(skill_id, name, description, tags, examples=()):
    return {"id": skill_id, "name": name, "description": description,
            "tags": list(tags), "examples": list(examples)}


def agent_card(name, description, url, skills):
    return {
        "protocolVersion": PROTOCOL_VERSION,
        "name": name,
        "description": description,
        "url": url,
        "preferredTransport": "JSONRPC",
        "version": "1.0.0",
        "capabilities": {"streaming": False, "pushNotifications": False},
        "defaultInputModes": ["text/plain"],
        "defaultOutputModes": ["text/plain"],
        "skills": skills,
    }


def text_message(role, text, context_id=None):
    message = {"kind": "message", "messageId": str(uuid.uuid4()), "role": role,
               "parts": [{"kind": "text", "text": text}]}
    if context_id:
        message["contextId"] = context_id
    return message


def _error(req_id, code, text):
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": text}}


def create_agent_app(card, handler, registry_url=None):
    """FastAPI app that serves the agent card and answers message/send with handler(text) -> text."""

    @asynccontextmanager
    async def lifespan(app):
        if registry_url:
            await register(registry_url, card)
        yield
        if registry_url:
            await unregister(registry_url, card["name"])

    app = FastAPI(title=card["name"], lifespan=lifespan)

    @app.get("/.well-known/agent-card.json")
    def get_card():
        return card

    @app.post("/")
    async def rpc(request: Request):
        try:
            body = await request.json()
        except ValueError:
            return _error(None, -32700, "Parse error")
        req_id = body.get("id")
        if body.get("method") != "message/send":
            return _error(req_id, -32601, "Method not found")
        message = (body.get("params") or {}).get("message") or {}
        text = "\n".join(p.get("text", "") for p in message.get("parts", [])
                         if p.get("kind") == "text").strip()
        if not text:
            return _error(req_id, -32602, "Invalid params: message has no text part")
        try:
            # handlers call the LLM synchronously, so they run in the thread pool
            reply = await run_in_threadpool(handler, text)
        except Exception as exc:
            return _error(req_id, -32603, f"Internal error: {exc}")
        return {"jsonrpc": "2.0", "id": req_id,
                "result": text_message("agent", reply, message.get("contextId"))}

    return app


def _result_text(result):
    if result.get("kind") == "task":
        parts = [p for a in result.get("artifacts", []) for p in a.get("parts", [])]
        if not parts:
            parts = ((result.get("status") or {}).get("message") or {}).get("parts", [])
    else:
        parts = result.get("parts", [])
    return "\n".join(p["text"] for p in parts if p.get("kind") == "text")


def send_message(url, text, context_id=None, timeout=180):
    payload = {"jsonrpc": "2.0", "id": str(uuid.uuid4()), "method": "message/send",
               "params": {"message": text_message("user", text, context_id)}}
    response = httpx.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    if "error" in data:
        raise A2AError(data["error"]["message"])
    return _result_text(data["result"])


async def register(registry_url, card, attempts=15):
    # the registry may still be starting, so retry for a few seconds
    async with httpx.AsyncClient(timeout=5) as client:
        for _ in range(attempts):
            try:
                response = await client.post(f"{registry_url}/agents", json=card)
                response.raise_for_status()
                print(f"[{card['name']}] registered at {registry_url} as {card['url']}", flush=True)
                return True
            except httpx.HTTPError:
                await asyncio.sleep(1)
    print(f"[{card['name']}] WARNING: registry {registry_url} not reachable", flush=True)
    return False


async def unregister(registry_url, name):
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            await client.delete(f"{registry_url}/agents/{name}")
    except httpx.HTTPError:
        pass


def list_agents(registry_url):
    response = httpx.get(f"{registry_url}/agents", timeout=10)
    response.raise_for_status()
    return response.json()


def find_by_skill(agents, skill_id):
    for agent in agents:
        if any(s["id"] == skill_id for s in agent.get("skills", [])):
            return agent
    return None


def serve(app, port):
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
