"""Tests without an API key: tools directly, then all five services as real HTTP servers with scripted LLMs."""
import json
import socket
import threading
import time
from datetime import date

import pytest
import uvicorn
from fastapi.testclient import TestClient

from workshop_a2a import coordinator, diagnosis, parts, registry, scheduling
from workshop_a2a.a2a import list_agents, send_message


def test_lookup_fault_finds_bearing():
    match = diagnosis.lookup_fault("Centrifugal pump", "strong vibration, bearing noise")["matches"][0]
    assert match["parts"] == [{"part_no": "BRG-6205", "qty": 2}]
    assert diagnosis.lookup_fault("lathe", "noise")["matches"] == []


def test_check_parts_reports_shortage_and_lead_time():
    result = parts.check_parts([{"part_no": "BRG-6205", "qty": 2}, {"part_no": "ORING-40", "qty": 2}])
    assert [i["available"] for i in result["items"]] == [False, True]
    assert result["parts_lead_time_days"] == 3


def test_find_slot_waits_for_parts_and_skips_busy_days():
    # Tuesday 13.10.2026 + 3 days lead + 1 -> Monday 19.10.; Lena is busy Mon-Wed, Tobias is free
    slot = scheduling.find_slot("mechanical", 3, 3, today=date(2026, 10, 13))
    assert slot["start_date"] == "2026-10-19" and slot["technician"] == "Tobias Wolf"


def test_a2a_endpoint_and_errors():
    app = parts.create_app(llm=EchoToolLLM("search_parts", {"text": "bearing"}), registry_url=None)
    with TestClient(app) as client:
        assert client.get("/.well-known/agent-card.json").json()["name"] == "Parts Agent"
        assert client.post("/", json={"jsonrpc": "2.0", "id": 1, "method": "tasks/get"}).json()["error"]["code"] == -32601
        ok = client.post("/", json={"jsonrpc": "2.0", "id": 2, "method": "message/send", "params": {
            "message": {"kind": "message", "messageId": "m1", "role": "user", "parts": [{"kind": "text", "text": "bearing?"}]}}})
        assert "BRG-6205" in ok.json()["result"]["parts"][0]["text"]


def test_registry_rejects_incomplete_card():
    with TestClient(registry.create_app()) as client:
        assert client.post("/agents", json={"name": "x"}).status_code == 422


class EchoToolLLM:
    """Calls one tool, then answers with the tool result."""

    def __init__(self, name, arguments):
        self.name, self.arguments = name, arguments

    def complete(self, messages, tools=None):
        if messages[-1]["role"] == "tool":
            return {"content": messages[-1]["content"], "tool_calls": []}
        return {"content": None, "tool_calls": [{"id": "c1", "name": self.name, "arguments": self.arguments}]}


class ScriptedCoordinatorLLM:
    def __init__(self):
        self.prompts = []
        self.replies = [
            {"action": "call", "agent": "Diagnosis Agent", "message": "Pump vibrates, bearing noise."},
            {"action": "call", "agent": "Parts Agent", "message": "Check 2 x BRG-6205."},
            {"action": "call", "agent": "Scheduling Agent", "message": "mechanical, 3 h, parts in 3 days"},
            {"action": "finish", "answer": "Bearing worn, parts in 3 days, repair planned."},
        ]

    def complete(self, messages, tools=None):
        self.prompts.append(messages[-1]["content"])
        return {"content": "```json\n" + json.dumps(self.replies.pop(0)) + "\n```", "tool_calls": []}


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start(app, port):
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(200):
        if server.started:
            return server
        time.sleep(0.05)
    raise RuntimeError("server did not start")


@pytest.fixture
def system():
    reg_port = free_port()
    reg_url = f"http://127.0.0.1:{reg_port}"
    servers = [start(registry.create_app(), reg_port)]
    coord_llm = ScriptedCoordinatorLLM()
    apps = [
        (diagnosis, EchoToolLLM("lookup_fault", {"machine": "centrifugal pump", "symptom": "bearing noise"})),
        (parts, EchoToolLLM("check_parts", {"items": [{"part_no": "BRG-6205", "qty": 2}]})),
        (scheduling, EchoToolLLM("find_slot", {"skill": "mechanical", "work_hours": 3, "parts_lead_days": 3})),
        (coordinator, coord_llm),
    ]
    for module, llm in apps:
        port = free_port()
        servers.append(start(module.create_app(llm, reg_url, f"http://127.0.0.1:{port}/"), port))
    yield reg_url, coord_llm
    for server in servers:
        server.should_exit = True


def test_end_to_end_over_http(system):
    reg_url, coord_llm = system
    agents = {a["name"]: a for a in list_agents(reg_url)}
    assert len(agents) == 4
    answer = send_message(agents["Maintenance Coordinator"]["url"], "Pump P-200 vibrates.")
    assert answer.endswith("Agents used: Diagnosis Agent -> Parts Agent -> Scheduling Agent")
    # the coordinator got the real tool results of the remote agents back
    assert "Worn ball bearings" in coord_llm.prompts[1]
    assert '"parts_lead_time_days": 3' in coord_llm.prompts[2]
    assert '"technician"' in coord_llm.prompts[3]
