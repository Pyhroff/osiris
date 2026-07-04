@echo off
rem OSIRIS launcher — `osiris` for the chat REPL, `osiris serve` for the web UI,
rem `osiris daemon` for background reminders, `osiris bot` for the Discord
rem mobile dispatcher.
rem Install (makes `osiris` work from any terminal):
rem   copy osiris.cmd "%LOCALAPPDATA%\Microsoft\WindowsApps\"
rem then IN THE COPIED FILE replace the line below with your clone's absolute
rem path, e.g.  set OSIRIS_HOME=C:\path\to\OSIRIS\
set OSIRIS_HOME=%~dp0
if "%1"=="serve" (
    python "%OSIRIS_HOME%server.py"
) else if "%1"=="daemon" (
    python "%OSIRIS_HOME%daemon.py"
) else if "%1"=="bot" (
    python "%OSIRIS_HOME%discord_bot.py"
) else (
    python "%OSIRIS_HOME%main.py"
)
