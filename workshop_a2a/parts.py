"""Parts agent: checks stock, storage location, price and supplier lead time of spare parts."""
import os

from .a2a import agent_card, create_agent_app, serve, skill
from .llm import REGISTRY_URL, OpenAIChat, run_tool_agent, tool

NAME = "Parts Agent"
# part_no: (name, stock, location, unit price EUR, supplier lead time in business days)
INVENTORY = {
    "BRG-6205": ("Ball bearing 6205 2RS", 1, "A-03-2", 12.40, 3),
    "SEAL-MS35": ("Mechanical shaft seal 35 mm", 0, "A-05-1", 89.00, 10),
    "ORING-40": ("O-ring 40x3 NBR", 50, "B-01-4", 0.80, 2),
    "BELT-V1250": ("V-belt SPA 1250", 4, "C-02-1", 18.90, 4),
    "CONT-24V-18A": ("Contactor 24 V DC 18 A", 2, "E-01-3", 46.50, 5),
    "FLT-AIR-160": ("Intake air filter 160 mm", 6, "D-04-2", 23.00, 3),
    "SNS-PT100": ("PT100 temperature sensor", 0, "E-02-1", 31.70, 5),
}


def check_parts(items):
    result, max_lead = [], 0
    for item in items:
        part_no, qty = item["part_no"].upper(), int(item.get("qty", 1))
        if part_no not in INVENTORY:
            result.append({"part_no": part_no, "error": "unknown part number"})
            continue
        name, stock, location, price, lead = INVENTORY[part_no]
        available = stock >= qty
        result.append({"part_no": part_no, "name": name, "needed": qty, "in_stock": stock,
                       "location": location, "available": available,
                       "lead_time_days": 0 if available else lead, "total_price_eur": round(qty * price, 2)})
        max_lead = max(max_lead, 0 if available else lead)
    return {"items": result, "parts_lead_time_days": max_lead}


def search_parts(text):
    words = text.lower().split()
    return [{"part_no": no, "name": data[0], "in_stock": data[1]} for no, data in INVENTORY.items()
            if all(w in data[0].lower() for w in words)]


TOOLS = {
    "check_parts": tool(
        check_parts, "Check stock, location, price and lead time for a list of part numbers.",
        {"type": "object", "properties": {"items": {"type": "array", "items": {
            "type": "object", "properties": {"part_no": {"type": "string"}, "qty": {"type": "integer"}},
            "required": ["part_no", "qty"]}}}, "required": ["items"]}),
    "search_parts": tool(
        search_parts, "Find part numbers by words in the part name.",
        {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}),
}

SYSTEM_PROMPT = (
    "You are the spare parts agent of a maintenance workshop. Use check_parts for every part number "
    "you get and search_parts if only a part name is given. Report for each part whether it is in stock, "
    "the storage location, missing quantities, the total price and the overall parts lead time in "
    "business days. Only use data returned by the tools.")


def create_app(llm=None, registry_url=REGISTRY_URL, url=None):
    llm = llm or OpenAIChat("PARTS_MODEL")
    card = agent_card(
        NAME, "Knows the spare parts store: stock, storage location, price and supplier lead time.",
        url or os.environ.get("PUBLIC_URL", "http://localhost:8102/"),
        [skill("check-spare-parts", "Check spare parts",
               "Checks availability of part numbers and returns how many business days until all parts are there.",
               ["inventory", "spare parts"], ["Check 2 x BRG-6205."])])
    return create_agent_app(card, lambda text: run_tool_agent(llm, SYSTEM_PROMPT, TOOLS, text), registry_url)


def main():
    serve(create_app(), int(os.environ.get("PORT", 8102)))


if __name__ == "__main__":
    main()
