"""OSIRIS web server — chat API + live state for the frontend.

Run:  python server.py   →   http://localhost:8765
"""
import sys

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

from osiris import config
from osiris.agent import Agent
from osiris.agents import pick_agent
from osiris.memory import Memory
from osiris.planner import Planner
from osiris.router import ModelRouter
from osiris.tools import build_registry

try:
    llm = ModelRouter()
except RuntimeError as e:
    sys.exit(str(e))

memory = Memory(llm)
planner = Planner(llm, memory)
agent = Agent(llm, memory, build_registry(memory, planner))

app = FastAPI(title="OSIRIS")


class ChatRequest(BaseModel):
    message: str


@app.post("/api/chat")
def chat(req: ChatRequest):
    profile = pick_agent(req.message)
    trace: list[dict] = []
    try:
        reply = agent.run(req.message, profile=profile,
                          on_tool=lambda name, args: trace.append({"name": name, "args": args}))
        return {"reply": reply, "agent": profile.key, "tools": trace, "error": False}
    except Exception as e:
        return {"reply": f"{type(e).__name__}: {e}", "agent": profile.key,
                "tools": trace, "error": True}


@app.get("/api/state")
def state():
    return {
        "providers": llm.describe(),
        "goals": memory.list_goals(),
        "tasks": memory.list_tasks(),
        "memories": memory.db.execute("SELECT COUNT(*) FROM memories").fetchone()[0],
    }


@app.get("/")
def index():
    return FileResponse(config.ROOT / "frontend" / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8765)
