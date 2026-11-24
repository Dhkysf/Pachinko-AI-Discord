# Pachinko — Autonomous Discord Voice & Chat Companion

A pure-Python Discord voice & chat companion bot (inspired by the architecture of owenliu0924/Miyoria, no code copied)—tailored for easy editing in VS Code on **Windows 11 + AMD Radeon RX 9070 (16 GB VRAM)** and 100% safe for Discord accounts (using **Official Bot Token + DAVE E2EE**).

---

## ✨ Included Subsystems

1. **Brain + 7-Block Prompt + Triage (`brain/`)**
   - One unified mind for **Voice** & **Text Chat** (replies in casual Indonesian).
   - Per-channel autonomy modes (`/mode`): `off`, `mention-only`, `quiet` (2x/hr), `normal` (6x/hr), `lively` (14x/hr).
   - Quoted `<data>` prompt-injection defense (`core/safety.py`) + **Action Gate** (`brain/gate.py`).
   - Built-in tools: Web Search (`ddgs`), Living Notebook, Timed Reminders, & Song Queue.

2. **Heart Engine & 9-Axis Relationship (`heart/`)**
   - Simulates **12 emotions** with half-life decay, valence, arousal, & energy.
   - Hidden `<emo:...>` & `<feel>...</feel>` tags update character mood and a **10-Tier Bond Ladder** (`-100..1000`) per member with daily geometric damping.

3. **Layered SQLite WAL Memory (`core/store.py`, `memory/`)**
   - **L0 Event Log**, **L1 Persistent Transcripts** (records partial spoken text when interrupted by barge-in), **L2 Hybrid Recall** (SQLite FTS5 + local n-gram vector `<1ms`), and automatic **L3 Living Notebook**.

4. **Natural Voice Pipeline (`speech/`, `voice/`)**
   - **Smart Prosodic Turn-Taking + Parked Stream**: Doesn't cut you off when you pause to think mid-sentence, yet responds snappily when your intonation finishes.
   - **Sentence Streaming + `LinePrefetch`**: Sentence 1 is spoken as soon as it leaves the LLM *while* sentence 2 is synthesized in the background.
   - **Cached Filler** (*"Hmm?"*, *"Eh, bentar..."*) + **SelfEchoGuard** (prevents speaker-to-mic feedback).
   - **Two-Stage Smart Barge-In**: Distinguishes short backchannels (*"hmm"*, *"iya"*, *"oke"*) from real interruptions (supports 28% volume ducking, pause, and seamless resume!).

5. **Web Admin Panel (`web/`)**
   - **Admin Studio (`http://127.0.0.1:7878`)**: Live emotion monitor, WebSocket event stream, hot-reload Persona & Lorebook editor, Sandbox Playground chat, Member Bond explorer, Notebook, & AI engine controls.

6. **AI Singing / Karaoke + RX 9070 VRAM Coordinator (`sing/`, `core/vram.py`)**
   - Request songs from YouTube (`/sing`), temporarily unloads Ollama VRAM (`keep_alive: 0`) when RVC AI Cover mode is active, and plays music in the Discord voice channel.

7. **Play & Economy (`play/`)**
   - Coin wallet (`/daily` streak + fortune, `/gift` to boost Bond, `/card` SVG collectible gacha, & `/trivia` mini-game).

---

## 🚀 Running on Windows (AMD Radeon RX 9070 16GB)

### 1. Set Up Ollama for the Local LLM Brain
In Windows PowerShell:
```powershell
winget install Ollama.Ollama
ollama pull qwen3:8b
```
*(With 16 GB VRAM on the RX 9070, `qwen3:8b` uses only ~5.5 GB VRAM, leaving >10 GB free. You can also use `gemma3:12b`.)*

### 2. Enable Intent in Discord Developer Portal
Open https://discord.com/developers/applications → select your bot application → **Bot** menu → enable **Message Content Intent** → Save Changes. *(If disabled, the bot still runs safely in Voice + Slash Command mode.)*

### 3. Run Setup & Start
1. Extract the `pachinko` folder (e.g. to `C:\pachinko`).
2. Double-click **`setup.bat`** → fill in `DISCORD_TOKEN` and `GUILD_ID` when Notepad opens → save.
   - **Reproducible fallback:** If a fresh `setup.bat` install breaks due to a newer transitive dependency, install the known-good snapshot instead: `.venv\Scripts\python.exe -m pip install -r requirements-lock.txt`.
3. Double-click **`Pachinko.lnk`** (or **`Pachinko.pyw`**) to open **Pachinko Control**, then click **Start bot** (`run.bat` remains available as a console fallback).
4. Open **`http://127.0.0.1:7878`** in your browser to open **Pachinko Studio**!
   - **Panel Auth (`.env`, optional):** If `PANEL_TOKEN` is set, open `http://127.0.0.1:7878/?token=<PANEL_TOKEN>` (the token is saved in `sessionStorage` and stripped from the URL).
   - **Configuration (`.env`):** Set `LLM_TEMPERATURE` (`0.0`–`2.0`; leave empty to use the model/Ollama Modelfile default).

### Control App (`Pachinko.lnk` / `Pachinko.pyw`)
- <!-- Screenshot placeholder: Pachinko Control window & system tray -->
- **System tray & status:** Shows live bot state (`Stopped`, `Starting`, `Running`, `Stopping`, `Crashed`, `Already running`), uptime, Discord readiness, Ollama health, and voice session count. Closing with **X** minimizes to the tray when enabled.
- **Quick actions:** Start / Stop / Restart bot, Open web panel, Run checks, Update `yt-dlp`, Repair install (`setup.bat`), Edit settings (`.env` in Notepad), and open logs/project folders.
- **Emergency fallbacks:** `run.bat` and `check.bat` remain in the root folder if you ever need a plain console run.
- **Out of scope:** Single-file `.exe` packaging (PyInstaller) and automatic bot code self-updates.

---

## 🎮 Discord Slash Commands (17 Commands)

| Command | Description |
|---|---|
| `/join` / `/leave` | Join / leave voice channel (DAVE E2EE + Smart Turn) |
| `/say <pesan>` | Chat via text (also spoken in voice if connected) |
| `/mode <level>` | Set autonomous reply frequency in this channel (`off`, `mention-only`, `quiet`, `normal`, `lively`) |
| `/mood` | View character's emotion spectrum, valence, & energy |
| `/bond` | View closeness tier (9-Axis Bond) & character notes about you |
| `/sing <lagu>` / `/skip` / `/queue` | Request a YouTube song / karaoke, skip track, view queue |
| `/daily` / `/card` / `/gift` | Claim daily coins + fortune, draw SVG collectible card, gift coins |
| `/trivia` | Start a quick coin-reward trivia mini-game in chat |
| `/remind <menit> <catatan>` | Set a reminder (delivered in chat & voice!) |
| `/privacy` / `/reset` / `/reload` | Toggle auto-greeting privacy, clear short-term chat history, reload persona |

---

## 🛠️ Development

Double-click **`check.bat`** (or run `.venv\Scripts\python.exe -m compileall -q ...` and `.venv\Scripts\python.exe -m unittest discover tests -v`) to syntax-check all modules and run the offline unit test suite.

### Runtime Files & Exit Codes
- **Logs:** `data/logs/pachinko.log` (UTF-8, rotated at 5 MB × 3 backups).
- **Single-instance lock:** `data/run/bot.lock` (`pid`, `started`). Removed on clean exit; stale locks are replaced.
- **Status file:** `data/run/status.json` (updated on start and every 10 s with `state`, `pid`, `started`, `discord_ready`, `voice_sessions`, `llm_ok`). Removed on clean exit.
- **Stop request:** Create `data/run/stop.request` to trigger a clean shutdown (deleted automatically).
- **Exit codes:** `0` clean stop, `3` already running, `1` crash.
