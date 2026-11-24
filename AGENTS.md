# AGENTS.md — Pachinko

## Replies to the owner
- Always reply to me in **English**, short and plain. End every task with: what changed, how it was verified, what you skipped and any risk.

## Project
- Pachinko: a Python 3.12 Discord **text + voice** companion bot ("Pachinko"), one process (`main.py`, asyncio), run by the owner on **one Windows 11 PC** (AMD RX 9070 16 GB) for **one friends' server** (~5–50 members, one voice channel at a time).
- LLM: local **Ollama**, model from `OLLAMA_MODEL` in `.env` (currently `qwen3-8k` = qwen3:8b with `num_ctx 8192`). Small 8B model: keep prompts short and explicit. Optional fallbacks: OpenAI-compatible, Gemini.
- Voice: py-cord (pinned commit of unmerged PR #3159 for DAVE receive) + faster-whisper on CPU + Edge/Piper TTS. Panel: FastAPI on `127.0.0.1:7878`. Data: SQLite WAL `data/pachinko.db`.
- Map: `brain/` (prompt, runner, llm, tools, gate, triage, profile) · `heart/` (emotions, bond) · `memory/` · `voice/` + `speech/` (turn taking, STT, TTS) · `sing/` · `play/` (economy) · `discord_bot/` · `web/` · `core/` (store, bus, safety) · `profiles/default/` (persona) · `tools/lang_eval.py` · `tests/`.

## Hard rules
- **Never open, print or edit `.env`**, and never touch `data/` except via the code paths below or a temporary copy. Never put tokens or keys in code, logs, tests or commits. If a new setting is needed, add it to `config.py` + `.env.example` and tell me what to add to my `.env`.
- **Language split:** code, comments, logs, docs, panel = English. Everything members read or hear stays **Indonesian**: Pachinko's replies, slash-command texts, reminders, errors, `profiles/default/*.md`.
- **Never translate or "clean up" Indonesian matching data**: wake aliases, nicknames, backchannel words, tool-need keywords, reminder parser, fact-capture regexes, STT hallucination filters, pronoun register rules. Changing them silently breaks features.
- **Database:** never drop tables or columns from existing DBs; add migrations that keep old `pachinko.db` working. All writes go through `Store.write()` / `Store.write_tx()`; never call `Store.write()`, `write_tx()` or `bus.emit()` inside a `write_tx()` block (deadlock).
- **Prompts:** exactly one system message at index 0 (strict templates reject more). `LLM_COMPACT_PROMPT=true` is the default path for 8B models; `false` must keep producing the old prompt. Keep tag formats (`<emo:...>`, `<feel>`, `<silence/>`, `<data>`, `<now>`, `<style_note>`, `<turn>`) parseable.
- **Tools:** only names in `TOOL_SCHEMAS` may run; the gate rejects unknown capabilities. Never claim an action happened unless a tool result confirms it.
- **Security:** keep the panel's escaping (`esc()` / `textContent`), Origin/Host checks and optional `PANEL_TOKEN`; Discord messages keep `allowed_mentions` without everyone/roles; admin commands keep `is_admin`.
- **Dependencies:** no new packages unless I approve. Keep `websockets` (uvicorn needs it). Pins live in `requirements.txt`; `requirements-lock.txt` is the known-good snapshot. Install only with `.venv\Scripts\python.exe -m pip ...`, never globally.
- Keep changes small and on task; prefer deleting code to adding layers. Long-lived shortcuts get a `shortcut:` comment naming the limit.

## Code navigation
- Use the **codegraph** MCP tools (list them first) to find callers/callees and impact before editing; read only the parts you need. If the index looks stale (files missing), tell me to run `codegraph init`.

## Run & check (Windows, PowerShell, project root)
- Python: always `.venv\Scripts\python.exe`. The sandbox warning "Failed to find real location of ...python.exe" is harmless; output is still valid.
- One command per call: no `;`, `&&` or `Write-Host` chaining (chained commands break the permission sandbox).
- Checks: `check.bat` (compileall + all tests), or `.venv\Scripts\python.exe -m unittest discover tests`. Tests are stdlib `unittest`, offline, no Discord, no Ollama, no real keys; use fakes and temp DBs.
- Bug fixes: add a test that FAILS on the old code first, then fix.
- Prompt/persona/model changes: run `.venv\Scripts\python.exe tools\lang_eval.py --model <name> --runs 3` before and after and report leaks, wrong pronouns, English sentences, tool calls and time.
- Smoke run when startup code changes: `.venv\Scripts\python.exe main.py` for ~20 s, then Ctrl+C; show a clean start and shutdown.

## Git & commit
- After a finished, verified task: run `ponytail-review` on the diff, fix Bug/Risk findings (only report Speed/Lean), then commit.
- Inspect `git status` and `git diff` first; `git add <files>` for this task only; never commit `.env`, `data/`, `models/`, `.venv/`, `.codegraph/`, `AUDIT*.md` or logs.
- Conventional Commits in English, ≤72 chars, `type(scope): summary` (feat, fix, refactor, test, docs, chore); one concern per commit; never `--no-verify`.
- **No push** unless I ask.
