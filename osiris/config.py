"""Central config — every knob lives here, loaded from .env at project root."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", "")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
GITHUB_MODELS_TOKEN = os.getenv("GITHUB_MODELS_TOKEN", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")  # free, no card: tavily.com

# Model router: providers tried in this order, failing over on any error.
# Unconfigured ones are skipped, so listing them all costs nothing.
PROVIDER_CHAIN = os.getenv(
    "OSIRIS_PROVIDERS", "gemini,groq,cerebras,mistral,openrouter,github,ollama,openai")

CHAT_MODEL = os.getenv("OSIRIS_CHAT_MODEL", "gemini-2.5-flash")
GROQ_MODEL = os.getenv("OSIRIS_GROQ_MODEL", "openai/gpt-oss-120b")  # far better at tools than llama-3.3
CEREBRAS_MODEL = os.getenv("OSIRIS_CEREBRAS_MODEL", "llama-3.3-70b")
MISTRAL_MODEL = os.getenv("OSIRIS_MISTRAL_MODEL", "mistral-small-latest")
OPENROUTER_MODEL = os.getenv("OSIRIS_OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free")
GITHUB_MODELS_MODEL = os.getenv("OSIRIS_GITHUB_MODELS_MODEL", "gpt-4o-mini")
OLLAMA_MODEL = os.getenv("OSIRIS_OLLAMA_MODEL", "")  # empty = auto-rank best pulled model
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OPENAI_MODEL = os.getenv("OSIRIS_OPENAI_MODEL", "gpt-4o-mini")

# Show provider failover chatter in the terminal. Off = clean output; the
# [provider · model] tag under each reply still shows who answered.
DEBUG = os.getenv("OSIRIS_DEBUG", "").lower() in ("1", "on", "true")

EMBED_MODEL = os.getenv("OSIRIS_EMBED_MODEL", "gemini-embedding-001")
EMBED_DIM = 768  # keep vectors small; free tier is generous but not infinite

# Set OSIRIS_EMBEDDINGS=off to stop spending Gemini quota on memory embeddings;
# recall falls back to keyword search (less smart, zero API cost).
EMBEDDINGS_ENABLED = os.getenv("OSIRIS_EMBEDDINGS", "on").lower() not in ("off", "0", "false")

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

# Free-text identity/context injected into every agent's system prompt (not just
# recalled memories, which only surface probabilistically). Empty = skipped.
USER_PROFILE = os.getenv("OSIRIS_USER_PROFILE", "")

# Hosts/IPs the redteam/blueteam agent's run_recon tool may target, beyond
# localhost/private IPs (always allowed). Only add targets you have explicit
# authorization to test. Comma-separated, e.g. OSIRIS_AUTHORIZED_TARGETS=lab.example.com
AUTHORIZED_TARGETS = os.getenv("OSIRIS_AUTHORIZED_TARGETS", "")

MAX_AGENT_STEPS = 12  # hard cap on tool-call loops per user turn
MAX_HISTORY = 60      # keep sessions from growing unbounded (trimmed at turn boundaries)
