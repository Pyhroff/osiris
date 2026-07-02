"""Central config — every knob lives here, loaded from .env at project root."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")

# Model router: providers tried in this order, failing over on any error
PROVIDER_CHAIN = os.getenv("OSIRIS_PROVIDERS", "gemini,groq,ollama,openai")

CHAT_MODEL = os.getenv("OSIRIS_CHAT_MODEL", "gemini-2.5-flash")
GROQ_MODEL = os.getenv("OSIRIS_GROQ_MODEL", "llama-3.3-70b-versatile")
OLLAMA_MODEL = os.getenv("OSIRIS_OLLAMA_MODEL", "llama3.2")  # prefix-matched against pulled models
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OPENAI_MODEL = os.getenv("OSIRIS_OPENAI_MODEL", "gpt-4o-mini")

EMBED_MODEL = os.getenv("OSIRIS_EMBED_MODEL", "gemini-embedding-001")
EMBED_DIM = 768  # keep vectors small; free tier is generous but not infinite

DB_PATH = ROOT / "osiris.db"
WORKSPACE = ROOT / "workspace"  # write_file/read_file sandbox
WORKSPACE.mkdir(exist_ok=True)

# Read/list/git access is limited to these roots (fs_* and git_* tools).
# Override: OSIRIS_ALLOWED_DIRS=C:\path\one;C:\path\two
_default_roots = [Path.home() / "OneDrive" / "Desktop", Path.home() / "Desktop"]
ALLOWED_DIRS = ([Path(p).resolve() for p in os.getenv("OSIRIS_ALLOWED_DIRS", "").split(";") if p.strip()]
                or [p.resolve() for p in _default_roots if p.exists()])

# Commit identity for git_commit (falls back to your global git config if unset)
GIT_NAME = os.getenv("OSIRIS_GIT_NAME", "")
GIT_EMAIL = os.getenv("OSIRIS_GIT_EMAIL", "")

MAX_AGENT_STEPS = 8   # hard cap on tool-call loops per user turn
MAX_HISTORY = 60      # keep sessions from growing unbounded (trimmed at turn boundaries)
