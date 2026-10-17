"""Diagnosis agent: maps a machine symptom to the likely cause, spare parts, skill and work hours."""
import os
import re

from .a2a import agent_card, create_agent_app, serve, skill
from .llm import REGISTRY_URL, OpenAIChat, run_tool_agent, tool

NAME = "Diagnosis Agent"
FAULTS = [
    {"machine": "centrifugal pump", "keywords": ["vibration", "vibrates", "noise", "grinding", "bearing"],
     "cause": "Worn ball bearings on the drive side", "parts": [{"part_no": "BRG-6205", "qty": 2}],
     "skill": "mechanical", "work_hours": 3},
    {"machine": "centrifugal pump", "keywords": ["leak", "leaking", "leaks", "drip", "seal"],
     "cause": "Worn mechanical shaft seal", "skill": "mechanical", "work_hours": 4,
     "parts": [{"part_no": "SEAL-MS35", "qty": 1}, {"part_no": "ORING-40", "qty": 2}]},
    {"machine": "belt conveyor", "keywords": ["slip", "slips", "slipping", "belt", "squeal", "squeaks"],
     "cause": "Stretched drive belt", "parts": [{"part_no": "BELT-V1250", "qty": 1}],
     "skill": "mechanical", "work_hours": 2},
    {"machine": "belt conveyor", "keywords": ["stops", "trips", "overload", "contactor", "motor"],
     "cause": "Defective motor contactor", "parts": [{"part_no": "CONT-24V-18A", "qty": 1}],
     "skill": "electrical", "work_hours": 2},
    {"machine": "air compressor", "keywords": ["pressure", "slow", "filter", "clogged"],
     "cause": "Clogged intake air filter", "parts": [{"part_no": "FLT-AIR-160", "qty": 1}],
     "skill": "mechanical", "work_hours": 1},
    {"machine": "air compressor", "keywords": ["hot", "temperature", "overheat", "overheating", "shuts"],
     "cause": "Faulty PT100 temperature sensor", "parts": [{"part_no": "SNS-PT100", "qty": 1}],
     "skill": "electrical", "work_hours": 2},
]


def lookup_fault(machine, symptom):
    machine_words = set(re.findall(r"[a-z]+", machine.lower()))
    symptom_words = set(re.findall(r"[a-z]+", symptom.lower()))
    scored = []
    for fault in FAULTS:
        if not machine_words & set(fault["machine"].split()):
            continue
        score = len(symptom_words & set(fault["keywords"]))
        if score:
            scored.append((score, fault))
    scored.sort(key=lambda s: -s[0])
    matches = [{k: v for k, v in f.items() if k != "keywords"} for _, f in scored[:2]]
    if not matches:
        return {"matches": [], "note": "No known fault pattern for this machine and symptom."}
    return {"matches": matches}


TOOLS = {"lookup_fault": tool(
    lookup_fault, "Look up known fault patterns for a machine type and a symptom description.",
    {"type": "object", "properties": {
        "machine": {"type": "string", "description": "Machine type, e.g. centrifugal pump"},
        "symptom": {"type": "string", "description": "Symptom in the words of the operator"}},
     "required": ["machine", "symptom"]})}

SYSTEM_PROMPT = (
    "You are the diagnosis agent of a maintenance workshop. Use lookup_fault for every request. "
    "Report the most likely cause, the spare parts with part numbers and quantities, the required "
    "skill and the work hours. Only use data returned by the tool; if nothing matches, say so.")


def create_app(llm=None, registry_url=REGISTRY_URL, url=None):
    llm = llm or OpenAIChat("DIAGNOSIS_MODEL")
    card = agent_card(
        NAME, "Finds the likely cause of a machine fault and the spare parts and work needed to fix it.",
        url or os.environ.get("PUBLIC_URL", "http://localhost:8101/"),
        [skill("diagnose-machine-fault", "Diagnose machine fault",
               "Maps a symptom of a pump, conveyor or compressor to cause, spare parts, skill and work hours.",
               ["diagnosis", "maintenance"], ["The pump vibrates and the bearing is noisy."])])
    return create_agent_app(card, lambda text: run_tool_agent(llm, SYSTEM_PROMPT, TOOLS, text), registry_url)


def main():
    serve(create_app(), int(os.environ.get("PORT", 8101)))


if __name__ == "__main__":
    main()
