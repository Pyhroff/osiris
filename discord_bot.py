"""OSIRIS Discord bot — the mobile dispatcher. Message OSIRIS from your phone,
from anywhere with internet, no VPN/tunnel needed.

Run in a second terminal:  python discord_bot.py  (or `osiris bot`)

Security model: only DMs from OSIRIS_DISCORD_ALLOWED_USERS are processed.
Everything else (server messages, DMs from anyone else) is silently ignored —
no reply, no hint the bot even exists. This is a new front door, not new
permissions: every tool's own guardrails (run_recon's authorization gate,
run_project_command's allowlist, confirm-before-act persona rules) apply
identically here, same as the REPL/web UI.

Your PC must be on and this process running for the bot to respond — it's a
personal assistant on your machine, not a hosted cloud service.
"""
import sys

from osiris import config

try:
    import discord
except ImportError:
    sys.exit("discord.py not installed — run: pip install -r requirements.txt")

from osiris.agent import Agent
from osiris.agents import pick_agent
from osiris.memory import Memory
from osiris.planner import Planner
from osiris.router import ModelRouter
from osiris.tools import build_registry

DISCORD_MSG_LIMIT = 2000


def _allowed_ids() -> set[int]:
    return {int(u.strip()) for u in config.DISCORD_ALLOWED_USERS.split(",") if u.strip()}


def _chunks(text: str, size: int = DISCORD_MSG_LIMIT) -> list[str]:
    return [text[i:i + size] for i in range(0, len(text), size)] or ["(no response)"]


def main():
    if not config.DISCORD_BOT_TOKEN:
        sys.exit("DISCORD_BOT_TOKEN not set in .env — see HANDBOOK.md for the Discord bot setup steps.")
    allowed = _allowed_ids()
    if not allowed:
        sys.exit("OSIRIS_DISCORD_ALLOWED_USERS not set in .env — add your Discord user id, "
                 "comma-separated for more than one. Without this the bot would answer anyone who DMs it.")

    try:
        llm = ModelRouter(verbose=config.DEBUG)
    except RuntimeError as e:
        sys.exit(str(e))
    memory = Memory(llm)
    agent = Agent(llm, memory, build_registry(memory, Planner(llm, memory)))

    intents = discord.Intents.default()
    intents.message_content = True
    client = discord.Client(intents=intents)

    @client.event
    async def on_ready():
        print(f"[discord bot] online as {client.user} — DMs from {sorted(allowed)} will be answered")

    @client.event
    async def on_message(message: discord.Message):
        if message.author.bot:
            return
        if not isinstance(message.channel, discord.DMChannel):
            return  # server messages ignored — DM-only surface, smallest exposure
        if message.author.id not in allowed:
            return  # silently ignored: no reply, no hint the bot exists

        async with message.channel.typing():
            profile = pick_agent(message.content)
            try:
                reply = agent.run(message.content, profile=profile)
            except Exception as e:
                reply = f"[error] {type(e).__name__}: {e}"
        for chunk in _chunks(reply):
            await message.channel.send(chunk)

    client.run(config.DISCORD_BOT_TOKEN)


if __name__ == "__main__":
    main()
