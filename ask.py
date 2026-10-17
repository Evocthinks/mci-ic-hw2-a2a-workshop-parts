"""Sends a fault report to the coordinator, which is itself looked up in the registry by its skill."""
import sys

from workshop_a2a.a2a import find_by_skill, list_agents, send_message
from workshop_a2a.llm import REGISTRY_URL

DEFAULT_REQUEST = ("Our centrifugal pump P-200 in hall 2 vibrates strongly and the bearing makes a grinding noise. "
                   "What is the likely cause, are the spare parts in stock and when can a technician repair it?")


def main():
    request = " ".join(sys.argv[1:]) or DEFAULT_REQUEST
    agents = list_agents(REGISTRY_URL)
    print(f"Registry {REGISTRY_URL}:")
    for agent in agents:
        print(f"  {agent['name']:<24} {agent['url']}")
    coordinator = find_by_skill(agents, "coordinate-maintenance-request")
    if coordinator is None:
        sys.exit("No coordinator registered. Start the agents first.")
    print(f"\nRequest: {request}\n")
    print(send_message(coordinator["url"], request, timeout=600))


if __name__ == "__main__":
    main()
