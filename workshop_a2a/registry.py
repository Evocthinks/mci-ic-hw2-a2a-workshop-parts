"""Agent registry: agents post their A2A agent card on startup, clients look them up at runtime.
It also hands out LLM request slots, so that all agents together stay below the provider's request limit."""
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .a2a import serve
from .llm import RequestWindow


class SlotRequest(BaseModel):
    model: str
    rpm: int = Field(ge=1)


def create_app():
    app = FastAPI(title="Agent Registry")
    agents = {}
    llm_window = RequestWindow()

    @app.post("/agents", status_code=201)
    def register(card: dict):
        missing = [k for k in ("name", "description", "url", "skills") if not card.get(k)]
        if missing:
            raise HTTPException(422, f"agent card is missing {', '.join(missing)}")
        agents[card["name"]] = card
        print(f"[registry] {card['name']} -> {card['url']}", flush=True)
        return {"registered": card["name"]}

    @app.get("/agents")
    def list_agents():
        return list(agents.values())

    @app.get("/agents/{name}")
    def get_agent(name: str):
        if name not in agents:
            raise HTTPException(404, "unknown agent")
        return agents[name]

    @app.delete("/agents/{name}")
    def remove_agent(name: str):
        agents.pop(name, None)
        return {"removed": name}

    @app.post("/llm-slots")
    def reserve_llm_slot(request: SlotRequest):
        # every agent asks here before an LLM call and waits the returned seconds
        return {"wait": llm_window.reserve(request.model, request.rpm)}

    return app


def main():
    serve(create_app(), int(os.environ.get("PORT", 8100)))


if __name__ == "__main__":
    main()
