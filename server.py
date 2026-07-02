"""OSIRIS web server — chat API + live state for the frontend.

Run:  python server.py   →   http://localhost:8765
"""
import sys
import threading

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

from osiris import config
from osiris.agent import Agent
from osiris.agents import pick_agent
from osiris.memory import Memory
from osiris.planner import Planner
from osiris.reflection import reflect
from osiris.router import ModelRouter
from osiris.tools import build_registry

REFLECT_EVERY = 8  # user turns between automatic background reflections

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


_turns = 0


def _background_reflect():
    try:
        stored = reflect(llm, memory, agent.history)
        if stored:
            print(f"[reflection] remembered {len(stored)} new fact(s)")
    except Exception as e:
        print(f"[reflection failed: {e}]")


@app.post("/api/chat")
def chat(req: ChatRequest):
    global _turns
    profile = pick_agent(req.message)
    trace: list[dict] = []
    try:
        reply = agent.run(req.message, profile=profile,
                          on_tool=lambda name, args: trace.append({"name": name, "args": args}))
        _turns += 1
        if _turns % REFLECT_EVERY == 0:  # learn in the background, never block a reply
            threading.Thread(target=_background_reflect, daemon=True).start()
        return {"reply": reply, "agent": profile.key, "tools": trace, "error": False}
    except Exception as e:
        return {"reply": f"{type(e).__name__}: {e}", "agent": profile.key,
                "tools": trace, "error": True}


@app.post("/api/reflect")
def manual_reflect():
    stored = reflect(llm, memory, agent.history)
    return {"stored": stored}


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
