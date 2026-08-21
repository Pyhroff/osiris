"""OSIRIS Discord bot — the mobile dispatcher. Message OSIRIS from your phone,
from anywhere with internet, no VPN/tunnel needed.

Run in a second terminal:  python discord_bot.py  (or `osiris bot`)

Security model: only DMs from OSIRIS_DISCORD_ALLOWED_USERS are processed.
Everything else (server messages, DMs from anyone else) is silently ignored —
no reply, no hint the bot even exists. This is a new front door, not new
permissions: every tool's own guardrails (run_recon's authorization gate,
run_project_command's allowlist, confirm-before-act persona rules) apply
identically here, same as the REPL/web UI.

Image attachments: a photo/screenshot sent in the DM is saved to
workspace/inbox/ and handed to the personal agent (which always carries
analyze_image, unlike some of the narrower specialist profiles) so "what's
this bug" + a screenshot from your phone actually works, not just text.

This process ALSO polls tasks/routines internally (same cycle as daemon.py) —
one process now covers both the chat surface and the proactive side, instead
of needing `osiris daemon` running in a second terminal too. Due
reminders/routines get DMed to you directly, on top of the usual toast/webhook.

Your PC must be on and this process running for the bot to respond — it's a
personal assistant on your machine, not a hosted cloud service.
"""
import asyncio
import sys
from datetime import datetime
from pathlib import Path

from osiris import config

try:
    import discord
    from discord.ext import tasks
except ImportError:
    sys.exit("discord.py not installed — run: pip install -r requirements.txt")

import daemon
from osiris.agent import Agent
from osiris.agents import PROFILES, pick_agent
from osiris.memory import Memory
from osiris.planner import Planner
from osiris.router import ModelRouter
from osiris.tools import build_registry

DISCORD_MSG_LIMIT = 2000
MAX_IMAGE_BYTES = 20 * 1024 * 1024  # matches analyze_image's own cap — no point saving a file it'll reject
_IMAGE_INBOX = config.WORKSPACE / "inbox"
# Keyed by Discord's content_type, not the user-supplied filename — the
# filename is never trusted for anything, including the extension.
_IMAGE_EXTENSIONS = {"image/png": ".png", "image/jpeg": ".jpg",
                     "image/webp": ".webp", "image/gif": ".gif"}


def _allowed_ids() -> set[int]:
    return {int(u.strip()) for u in config.DISCORD_ALLOWED_USERS.split(",") if u.strip()}


def _save_image_attachment(data: bytes, content_type: str, index: int) -> Path | None:
    """Save downloaded attachment bytes under workspace/inbox/ with a name we
    fully control (timestamp + index + extension from content_type) — never
    the attacker/user-controlled original filename, so there's no path to
    escape the inbox folder with a crafted name. Returns None for a
    content_type analyze_image wouldn't accept anyway (skip saving it)."""
    ext = _IMAGE_EXTENSIONS.get((content_type or "").split(";")[0].strip().lower())
    if not ext:
        return None
    _IMAGE_INBOX.mkdir(parents=True, exist_ok=True)
    path = _IMAGE_INBOX / f"{datetime.now():%Y%m%d-%H%M%S}-{index}{ext}"
    path.write_bytes(data)
    return path


def _build_content_with_images(text: str, image_paths: list[Path]) -> str:
    """Fold saved image paths into the message text as an explicit instruction
    — the agent has no other way to know a DM attachment landed on disk."""
    if not image_paths:
        return text
    note = "; ".join(f'analyze_image path="{p}"' for p in image_paths)
    prefix = f"[{len(image_paths)} image attachment(s) saved — {note}]"
    return f"{text}\n\n{prefix}" if text else prefix


def _chunks(text: str, size: int = DISCORD_MSG_LIMIT) -> list[str]:
    """Split on the last whitespace before the limit, not a blind character
    cut — otherwise a long reply can slice a word in half mid-sentence."""
    if not text:
        return ["(no response)"]
    out = []
    while len(text) > size:
        cut = text.rfind(" ", 0, size)
        if cut <= 0:  # no whitespace to break on (one giant token) — fall back to a hard cut
            cut = size
        out.append(text[:cut])
        text = text[cut:].lstrip()
    out.append(text)
    return out


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

    # Reminder/routine poller — separate from the chat agent's memory, same
    # split daemon.py uses (reminders never need an LLM, so this stays free).
    poll_memory = Memory(daemon._NoLLM(), db_path=config.DB_PATH)
    routines = daemon.load_routines()

    intents = discord.Intents.default()
    intents.message_content = True
    client = discord.Client(intents=intents)

    @tasks.loop(seconds=daemon.POLL_SECONDS)
    async def poll_loop():
        # poll_once is blocking (SQLite, an LLM call if a routine fires) — run
        # it off-thread so it can't stall the Gateway heartbeat.
        fired = await asyncio.to_thread(daemon.poll_once, poll_memory, routines)
        if not fired:
            return
        for uid in allowed:
            try:
                user = await client.fetch_user(uid)
                for line in fired:
                    await user.send(line[:DISCORD_MSG_LIMIT])
            except Exception as e:
                print(f"[discord bot] couldn't DM {uid}: {e}")

    @client.event
    async def on_ready():
        print(f"[discord bot] online as {client.user} — DMs from {sorted(allowed)} will be answered")
        if not poll_loop.is_running():
            poll_loop.start()

    @client.event
    async def on_message(message: discord.Message):
        if message.author.bot:
            return
        if not isinstance(message.channel, discord.DMChannel):
            return  # server messages ignored — DM-only surface, smallest exposure
        if message.author.id not in allowed:
            return  # silently ignored: no reply, no hint the bot exists

        image_paths = []
        for i, att in enumerate(message.attachments):
            if not (att.content_type or "").startswith("image/"):
                continue  # non-image attachments (pdfs, etc.) aren't handled here yet
            if att.size > MAX_IMAGE_BYTES:
                print(f"[discord bot] skipping oversized attachment from {message.author.id}: {att.size} bytes")
                continue
            try:
                data = await att.read()
                saved = _save_image_attachment(data, att.content_type, i)
                if saved:
                    image_paths.append(saved)
            except Exception as e:
                print(f"[discord bot] couldn't download attachment: {e}")

        content = _build_content_with_images(message.content, image_paths)
        if not content:
            return  # e.g. a sticker or unsupported file with no caption text — nothing to act on

        print(f"[discord bot] DM from {message.author.id}: {message.content!r}" +
             (f" (+{len(image_paths)} image attachment(s))" if image_paths else ""))
        # Images bypass keyword routing straight to the personal profile — it's
        # the one profile guaranteed to carry analyze_image, so a photo works
        # regardless of what text (if any) came with it, instead of possibly
        # landing on a narrower specialist agent that can't see the tool.
        profile = PROFILES["personal"] if image_paths else pick_agent(content)
        async with message.channel.typing():
            try:
                # agent.run is blocking (LLM HTTP calls, tool I/O, sometimes an
                # 8s cross-provider retry sleep) — off-thread so it can't stall
                # the Gateway heartbeat and get the connection dropped mid-reply.
                reply = await asyncio.to_thread(agent.run, content, profile=profile)
            except Exception as e:
                reply = f"[error] {type(e).__name__}: {e}"
        print(f"[discord bot] replying: {reply[:100]!r}")
        for chunk in _chunks(reply):
            await message.channel.send(chunk)

    client.run(config.DISCORD_BOT_TOKEN)


if __name__ == "__main__":
    main()
