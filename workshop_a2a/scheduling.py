"""Scheduling agent: finds the first day a technician with the right skill can do the repair."""
import math
import os
from datetime import date, timedelta

from .a2a import agent_card, create_agent_app, serve, skill
from .llm import REGISTRY_URL, OpenAIChat, run_tool_agent, tool

NAME = "Scheduling Agent"
TECHNICIANS = [
    {"name": "Lena Gruber", "skills": ["mechanical", "electrical"], "busy_weekdays": [0, 1, 2]},
    {"name": "Tobias Wolf", "skills": ["mechanical"], "busy_weekdays": [3]},
    {"name": "Sara Novak", "skills": ["electrical"], "busy_weekdays": [4]},
]
HOURS_PER_DAY = 8


def add_business_days(start, days):
    current = start
    while days > 0:
        current += timedelta(days=1)
        if current.weekday() < 5:
            days -= 1
    return current


def find_slot(skill, work_hours, parts_lead_days, today=None):
    today = today or date.fromisoformat(os.environ.get("PLANNING_DATE", date.today().isoformat()))
    # the repair can start the business day after the last part has arrived
    earliest = add_business_days(today, int(parts_lead_days) + 1)
    days_needed = math.ceil(float(work_hours) / HOURS_PER_DAY)
    for offset in range(60):
        start = earliest + timedelta(days=offset)
        days = [add_business_days(start, i) for i in range(days_needed)]
        if start.weekday() >= 5:
            continue
        for tech in TECHNICIANS:
            if skill in tech["skills"] and all(d.weekday() not in tech["busy_weekdays"] for d in days):
                return {"technician": tech["name"], "start_date": start.isoformat(),
                        "weekday": start.strftime("%A"), "days_needed": days_needed,
                        "earliest_possible": earliest.isoformat()}
    return {"error": f"no technician with skill {skill!r} free in the next 60 days"}


TOOLS = {"find_slot": tool(
    find_slot, "Find the first free technician and date for a repair.",
    {"type": "object", "properties": {
        "skill": {"type": "string", "enum": ["mechanical", "electrical"]},
        "work_hours": {"type": "number"},
        "parts_lead_days": {"type": "integer", "description": "Business days until all parts are available, 0 if in stock"}},
     "required": ["skill", "work_hours", "parts_lead_days"]})}

SYSTEM_PROMPT = (
    "You are the scheduling agent of a maintenance workshop. Use find_slot with the skill, the work "
    "hours and the parts lead time from the request. Report technician, start date with weekday and "
    "the number of days. Only use data returned by the tool.")


def create_app(llm=None, registry_url=REGISTRY_URL, url=None):
    llm = llm or OpenAIChat("SCHEDULING_MODEL")
    card = agent_card(
        NAME, "Plans repairs: finds a technician with the right skill and the first possible date.",
        url or os.environ.get("PUBLIC_URL", "http://localhost:8103/"),
        [skill("schedule-repair", "Schedule repair",
               "Needs skill (mechanical or electrical), work hours and parts lead time; returns technician and date.",
               ["scheduling", "maintenance"], ["Mechanical job, 3 hours, parts in 3 days."])])
    return create_agent_app(card, lambda text: run_tool_agent(llm, SYSTEM_PROMPT, TOOLS, text), registry_url)


def main():
    serve(create_app(), int(os.environ.get("PORT", 8103)))


if __name__ == "__main__":
    main()
