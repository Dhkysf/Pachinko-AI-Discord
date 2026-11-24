# Miyoria-Py Codebase Audit

**What this repo does:** Miyoria-Py is a single-process Python Discord Voice & VTuber AI character bot built for Windows 11 (AMD Radeon RX 9070 16 GB VRAM). It combines a local-first LLM brain (Ollama / OpenAI-compatible / Gemini fallback), a 12-emotion + 9-axis relationship engine, layered SQLite memory, a real-time Discord DAVE E2EE voice pipeline (Faster-Whisper STT + Edge/Piper TTS + optional RVC), YouTube karaoke/singing, a mini-game coin economy, VTube Studio/OBS WebSocket bridges, and a FastAPI Web Admin Studio (`:7878`) + OBS overlay (`/overlay`).

**Assumed load:** Single local Windows 11 host running one bot process for one personal/community Discord server (~5–50 active members, 1 active voice channel at a time, several text channels) plus the owner using the local Web Admin Studio (`127.0.0.1:7878`) and OBS Browser Overlay.

## Checks Run (No Network, No Secrets)

- **CodeGraph MCP tools listed & queried:** `codegraph_explore` (only tool exposed by `codegraph` MCP server). Index is present (`.codegraph/`) and current across Python and JS modules.
- **Syntax / bytecode compile:** `.venv\Scripts\python.exe -m compileall -q main.py config.py setup_models.py brain core discord_bot heart memory play sing speech stage voice web` → **PASS** (exit code 0, no errors).
- **Entrypoint import check:** `.venv\Scripts\python.exe -c "import main, config, setup_models; print('All entrypoints imported OK')"` → **PASS** (`All entrypoints imported OK`).
- **Tests & linters:** **NONE PRESENT.** 0 project test files (`*test*.py` exists only inside `.venv/Lib/site-packages`); `pytest`, `ruff`, `flake8`, and `mypy` are not installed in `.venv` or `PATH`.
- **Targeted read-only runtime verifications (`.venv\Scripts\python.exe`):**
  - Inspected `websockets.asyncio.connection.Connection.recv` (v17.4): confirmed concurrent `recv()` raises `ConcurrencyError`.
  - Inspected `py-cord` `VoiceClient.start_recording` & `PacketRouter._do_run`: confirmed `Sink.write(data, data.source)` receives `(AudioData, Member/User)`.
  - Ran `ffmpeg -f m4a -i pipe:0` and `ffmpeg -f opus -i pipe:0`: confirmed `m4a` fails with `moov atom not found` and `opus` fails with `Unknown input format: 'opus'`.
  - Reproduced `brain/llm.py:L177-184` with `['<think>ok</think>Halo', ' dunia!']`: confirmed `in_think=True`, output `[]`.
  - Ran `speech.turn.analyze_end_of_turn(bytes(1000 * BYTES_PER_MS))`: confirmed return value `0.35` (< `0.58` threshold).
  - Verified unused symbols/dependencies via both `codegraph_explore` and full-repo grep.
- **Main flows traced end to end:**
  1. **Discord text chat:** `discord_bot/client.py:on_message` → `EconomyAndPlay.check_trivia_answer` → `TriageEngine.decide_text` → `BrainRunner.run_text_turn` → `LLMClient` + `ToolBox` + `ActionGate` → `HeartEngine.apply_feel_block` + `MemoryManager` → `Store` (SQLite) & `EventBus` → Discord reply.
  2. **Discord voice:** `SmartTurnSink.write` / `_clock_loop` → `NaturalVoiceSession._worker_loop` → `STTEngine.transcribe` → `BrainRunner.stream_voice_turn` → `TTSEngine.synthesize_pcm` → `ExpressivePCMSource.read` → Discord VC + `stage.lipsync` → `StageBridge` (VTube Studio) & `/overlay`.
  3. **Singing / karaoke:** `/sing` or `queue_song` or `/api/sing/enqueue` → `SingingEngine._queue_loop` → `yt-dlp` → LRCLIB → optional `vram_guard.lease_heavy_gpu` + RVC → `audio_bytes_to_pcm48k` → `NaturalVoiceSession.play_raw_pcm` + `overlay.karaoke`.
  4. **Web Admin Panel:** `web/server.py` (REST `/api/*` + WebSocket `/ws/live`) ↔ `web/static/index.html`, `panel.js`, `overlay.html`.
  5. **Economy & reminders:** `/daily`, `/card`, `/gift`, `/trivia`, `/remind` → `EconomyAndPlay` / `MemoryManager` → `Store` (SQLite) + `_schedule_loop`.

## Audit Areas Checked

| # | Area | Status | Findings |
|---|---|---|---|
| 1 | Bugs | Issues found | #3, #4, #5, #6, #7, #12, #15, #20 |
| 2 | Security & secrets | Issues found (no hardcoded secrets in tracked source; `.env` ignored by `.gitignore`) | #1, #2, #8, #10 |
| 3 | Reliability | Issues found | #6, #7, #13, #14, #15 |
| 4 | Scale (at assumed load) | Issues found | #9, #13 |
| 5 | Missing tests | Issues found | #11 |
| 6 | Speed | Issues found | #5, #12, #13, #14, #20 |
| 7 | Lean | Issues found | #10, #13, #16, #17 |
| 8 | Docs drift | Issues found | #18 |
| 9 | UX / accessibility | Issues found | #2, #19 |
| 10 | Dependencies & setup | Issues found | #16 |

*(Note: `.gitignore` does NOT ignore `AUDIT.md`; `.gitignore` was left untouched per instructions.)*

---

## Must Fix

1. **Stored XSS across the Web Admin Panel** (`web/static/panel.js:L70, L104, L135-145, L154-158, L206-218`)
   - **What this is:** The Admin Studio renders live Discord events, member names, impressions, notebook facts, episodic memories, song queue items, and playground replies in the browser.
   - **Problem:** All of these untrusted strings are inserted directly via `innerHTML` without HTML escaping. Any Discord member who sends `<img src=x onerror=...>` in chat or puts HTML in their nickname executes arbitrary JavaScript in the owner's browser, which can call `POST /api/persona` or `POST /api/runtime` to hijack the bot.
   - **Fix:** Add a small `escapeHtml(s)` helper (or use DOM elements + `textContent`, matching `web/static/overlay.html`) and escape every dynamic value rendered into HTML in `panel.js`.
   - **If we skip it:** Any member in the Discord server can execute scripts in the owner's Admin Studio session whenever the dashboard is open.
   - **Effort:** S | **Verified:** yes (code read of `web/static/panel.js` + dataflow trace from `memory.transcript` and `heart_person`).

2. **WebSocket `/ws/live` ignores `PANEL_TOKEN`, and enabling `PANEL_TOKEN` breaks the Admin UI** (`web/server.py:L44, L47-52, L213-226`, `web/static/panel.js:L13-20, L268-270`)
   - **What this is:** `PANEL_TOKEN` is the optional auth secret protecting the Admin Studio API and live event feed.
   - **Problem:** `@app.websocket("/ws/live")` never checks `config.panel_token` (nor does `/api/docs`), so any unauthenticated client that can reach port 7878 can stream all Discord transcripts, voice utterances, and tool calls even when `PANEL_TOKEN` is set. Conversely, `panel.js` never sends `Authorization` or `?token=`, so setting `PANEL_TOKEN` breaks every Admin Studio call with HTTP 401; `_check_auth` also uses `!=` instead of constant-time comparison.
   - **Fix:** Validate `config.panel_token` with `hmac.compare_digest` in both `_check_auth` and `ws_live` (closing unauthorized sockets with code `1008`), disable `docs_url` when `PANEL_TOKEN` is set, and read/send the token from `sessionStorage` or URL query in `panel.js` and `overlay.html`.
   - **If we skip it:** Setting `PANEL_TOKEN` leaves live chat/voice traffic exposed over WebSocket while breaking the owner's own Admin Panel.
   - **Effort:** S | **Verified:** yes (code read of `web/server.py` and `web/static/panel.js`).

3. **Two-Stage Smart Barge-In is deadlocked during normal voice replies** (`voice/session.py:L242-250, L278-295, L304-310`)
   - **What this is:** `_worker_loop` processes user voice utterances and runs Stage 2 barge-in (distinguishing short backchannels like "hmm"/"oke" from real interruptions) while the bot is speaking.
   - **Problem:** `_process_utterance` spawns `self._active_turn_task` and immediately `await`s it (`L308`), blocking the single `_worker_loop` until the entire bot reply finishes playing. Any user speech during playback sits unread in `_utterance_q`, Stage 2 (`L278-295`) never executes, playback stalls paused for 2.2 seconds until `_auto_resume_if_still_paused` resumes the old reply, and the interruption is handled only after the bot finishes talking.
   - **Fix:** Do not `await self._active_turn_task` inside `_process_utterance`; track it as a background task with an exception-logging callback and cancel any running `_active_turn_task` right before starting a new turn.
   - **If we skip it:** Real-time interruption and backchannel resume do not work; interrupting the bot causes a 2.2-second freeze followed by the bot talking over the user.
   - **Effort:** S | **Verified:** yes (code read and async control-flow trace of `voice/session.py:L173-310`).

4. **Interrupted voice replies are lost and corrupt the previous transcript** (`voice/session.py:L376-378`, `brain/runner.py:L261-272`)
   - **What this is:** When a user interrupts the bot mid-sentence, Miyoria should store only what the user actually heard (`heard_content`, `interrupted = 1`).
   - **Problem:** `BrainRunner.stream_voice_turn` only inserts the assistant transcript and sets `self._last_voice_transcript_id` *after* the LLM stream finishes (`brain/runner.py:L263-272`). When barge-in cancels `prod_task` mid-stream, the current reply is never written to SQLite, and `voice/session.py:L378` calls `mark_interrupted` on the stale `_last_voice_transcript_id` from the *previous* completed turn (which can even belong to another guild since `BrainRunner` is shared).
   - **Fix:** Write the assistant voice transcript inside `NaturalVoiceSession._speak_streaming_turn` after playback ends (using `self._spoken_so_far`, `full_sentences`, and `self._barge_triggered.is_set()`), removing `_last_voice_transcript_id` from `BrainRunner`.
   - **If we skip it:** Every mid-stream barge-in drops the interrupted reply from memory and silently marks an older completed reply as interrupted with the wrong text.
   - **Effort:** S | **Verified:** yes (code read of `voice/session.py:L332-386` and `brain/runner.py:L222-273`).

5. **Parked-stream silence padding injects ~17 seconds of zeros and breaks fast turn-taking** (`speech/turn.py:L38-50, L116-126`)
   - **What this is:** `SmartTurnSink.write` pads short packet gaps while a speaker is talking, and `analyze_end_of_turn` inspects the audio tail to end completed sentences quickly (`550ms`) instead of waiting `1350ms`.
   - **Problem:** `gap_s = now - tr.last_voice_ts` compares packet arrival time to the last *loud* frame rather than the last *received* packet. While a user is silent with an open mic (sending quiet 20ms frames), `0.08 < gap_s < max_silence_s` fires on every quiet frame, injecting up to 300ms of zero bytes per 20ms frame (~17+ seconds of silence in 1.35s). `analyze_end_of_turn` then reads only the last 600ms of pure zeros (`drop_ratio = 1.0`), returning `0.35` (< `0.58`), which disables fast prosodic endpointing and sends bloated silent buffers to STT.
   - **Fix:** Track `tr.last_packet_ts` separately so padding runs only on real packet-delivery gaps (`now - tr.last_packet_ts`), and run `analyze_end_of_turn` on the last 600ms of *voiced* audio before trailing silence.
   - **If we skip it:** Every voice turn suffers ~800ms extra latency and burns Faster-Whisper CPU / Groq upload bandwidth on multi-second blocks of fake silence.
   - **Effort:** S | **Verified:** yes (code read + ran `analyze_end_of_turn(bytes(1000 * BYTES_PER_MS))` → `0.35`).

6. **Singing decoder fails on `.opus` / standard `.m4a` and loads whole songs into RAM** (`sing/engine.py:L151, L177-185`, `speech/tts.py:L36-48`)
   - **What this is:** `SingingEngine._perform_song` downloads requested tracks with `yt-dlp` and decodes them to 48kHz stereo PCM via `audio_bytes_to_pcm48k`.
   - **Problem:** It reads the entire downloaded file into RAM (`audio_file.read_bytes()`) and pipes it to `ffmpeg -f <ext> -i pipe:0`. FFmpeg fails on `.opus` (`Unknown input format: 'opus'`) and fails on standard `.m4a` files (`moov atom not found`) because `pipe:0` is non-seekable. Additionally, `bestaudio[filesize<25M]/bestaudio/best` falls back to arbitrarily long videos (a 2-hour video decodes to ~1.4 GB of raw PCM in RAM), and arbitrary `http(s)://` URLs (including localhost) are passed straight to `yt-dlp`.
   - **Fix:** Decode directly from the on-disk file path (`ffmpeg -i <audio_file>` without `-f`), enforce a duration cap in `yt-dlp` (e.g., `match_filter` ≤ 600s), and restrict URL inputs to allowed YouTube hosts (or search queries only).
   - **If we skip it:** `/sing` crashes on common YouTube `.m4a`/`.opus` audio downloads, and one long video link can spike memory usage by >1 GB.
   - **Effort:** M | **Verified:** yes (code read + tested `ffmpeg -f opus` → `Unknown input format: 'opus'` and `ffmpeg -f m4a -i pipe:0` → `moov atom not found`).

7. **VTube Studio WebSocket crashes with `ConcurrencyError` when an emotion hotkey fires** (`stage/bridge.py:L80-85, L100, L152-164`)
   - **What this is:** `StageBridge` drives Live2D mouth/head movement at 25 FPS in `_vts_loop` and triggers expression hotkeys in `_on_emotion` over the same `self._vts_ws` socket.
   - **Problem:** `_vts_loop` (`L163`) and `_on_emotion` → `_vts_send` (`L100`) call `await self._vts_ws.recv()` concurrently with no lock. In the installed `websockets 17.4`, concurrent `recv()` raises `ConcurrencyError`, killing `_vts_loop` and forcing a 10-second reconnect cycle every time a matching emotion hotkey triggers.
   - **Fix:** Serialize all `_vts_ws` send/recv operations behind a single `self._vts_lock = asyncio.Lock()`.
   - **If we skip it:** Whenever `VTS_ENABLED=true` and the Live2D model has emotion hotkeys, avatar tracking disconnects for 10 seconds on every mood shift.
   - **Effort:** S | **Verified:** yes (code read of `stage/bridge.py` + inspected `websockets.asyncio.connection.Connection.recv` in `.venv`).

8. **Admin/destructive slash commands lack permission checks, and bot messages allow `@everyone` pings** (`discord_bot/client.py:L57, L84-90, L235, L248-254, L289, L291-296, L310, L380-390`)
   - **What this is:** Slash commands like `/mode`, `/reset`, `/reload`, and `/skip` alter shared server state, while `/remind`, `/say`, `/sing`, and `/queue` echo user text back into channels.
   - **Problem:** Any server member can run `/reset` (wiping room and voice transcripts for everyone), switch `/mode` to `off`/`lively`, reload personas, or `/skip` songs. Meanwhile `_schedule_loop` (`L90`), `/say` (`L235`), `/sing` (`L289`), and `/queue` (`L310`) send unescaped user text without `allowed_mentions`, letting any member trigger `@everyone`/`@here` pings (e.g., `/remind 1 @everyone ...`).
   - **Fix:** Require `ctx.author.id in config.owner_ids` or `ctx.author.guild_permissions.manage_guild` for `/mode`, `/reset`, and `/reload`, and pass `allowed_mentions=discord.AllowedMentions(everyone=False, roles=False)` when constructing `discord.Bot`.
   - **If we skip it:** Any member in the server can wipe short-term memory, silence or spam the bot, or ping `@everyone` through reminders.
   - **Effort:** S | **Verified:** yes (code read of `discord_bot/client.py`).

9. **Wallet double-spend / double-daily race and unserialized shared SQLite writes** (`play/economy.py:L93-164`, `core/store.py:L174-228`)
   - **What this is:** Economy commands update coin balances, ledgers, gacha cards, and bond scores using one shared `aiosqlite.Connection`.
   - **Problem:** `daily_checkin`, `draw_card`, and `gift_to_character` read balance, yield across `await`, and write updates without a lock or atomic SQL update. Two rapid `/daily` or `/gift` calls both pass the check (double rewards or free bond farming because `add_coins` clamps balance at `0`). In `draw_card`, coins are committed (`L141`) before inserting the card (`L144`), and because every module shares `store.conn` with no write lock, `EventBus` logging an event can `commit()` another coroutine's half-finished transaction.
   - **Fix:** Add an `asyncio.Lock`-backed `write_tx()` context manager on `Store`, and enforce coin deductions atomically in SQL (`UPDATE economy_wallets SET balance = balance - ? WHERE user_id = ? AND balance >= ?`, checking `cursor.rowcount == 1`).
   - **If we skip it:** Rapid clicks or concurrent commands can double-claim `/daily`, farm free Bond with `/gift`, lose coins on partial failures, or commit half-written DB updates.
   - **Effort:** M | **Verified:** yes (code read of `play/economy.py` and `core/store.py`).

10. **Unadvertised `web_read` tool allows localhost SSRF without owner approval** (`brain/tools.py:L116-117, L159-170`, `brain/gate.py:L34`, `brain/runner.py:L145`)
    - **What this is:** `ToolBox.execute` handles `name == "web_read"` by fetching any `http://` or `https://` URL with `follow_redirects=True` at `risk = "low"`.
    - **Problem:** Although `web_read` is not listed in `TOOL_SCHEMAS`, `BrainRunner` passes any tool name emitted by the LLM straight into `tools.execute()`. A prompt-injected message can coerce the LLM into calling `web_read("http://127.0.0.1:7878/api/memory")` or `http://127.0.0.1:11434/...`, reading local services with no private-IP block and no owner approval.
    - **Fix:** Delete `_web_read` and its `brain/gate.py` entry (nothing advertises it), and reject any `tc.name` not present in `TOOL_SCHEMAS` inside `ToolBox.execute`.
    - **If we skip it:** Prompt injection can read local Admin Studio endpoints or other localhost services and leak them into Discord chat.
    - **Effort:** S | **Verified:** yes (code read + `codegraph_explore` + full-repo grep confirmed `web_read` is absent from `TOOL_SCHEMAS`).

---

## Should Fix

11. **Zero automated tests for risky core logic** (repo-wide: `core/safety.py`, `heart/engine.py`, `brain/triage.py`, `play/economy.py`, `speech/turn.py`, `sing/engine.py`)
    - **What this is:** Input sanitization, secret redaction, bond decay/damping, coin ledger math, turn endpointing, and LRC parsing run continuously on untrusted input.
    - **Problem:** No project tests exist anywhere in the repo, so logic bugs like the stuck `<think>` parser (#12) or broken `analyze_end_of_turn` (#5) go completely unnoticed.
    - **Fix:** Add one `tests/test_core.py` using Python's built-in `unittest` (runnable via `.venv\Scripts\python.exe -m unittest discover tests`, zero new dependencies) with focused tests for `safety`, `heart`, `triage`, `economy`, `turn`, and `parse_lrc`.
    - **If we skip it:** Every future change risks silently breaking security filters, bond math, or voice processing.
    - **Effort:** M | **Verified:** yes (repo scan confirmed 0 project test files).

12. **`_ollama_stream` silences replies on single-chunk `<think>...</think>` and wastes a full LLM call on voice tool checks** (`brain/llm.py:L158-184`, `brain/runner.py:L205-215`)
    - **What this is:** `LLMClient._ollama_stream` strips `<think>` blocks for voice streaming, and `BrainRunner.stream_voice_turn` pre-checks keywords to decide whether tools are needed.
    - **Problem:** When a single chunk contains `<think>...</think>Halo`, `L179` strips `tok` to `""` before `L180` checks for `</think>`, locking `in_think = True` for the entire rest of the stream (0 words spoken). On HTTP 400 retry (`L158-168`), `<think>` tags are not stripped at all. Additionally, in `brain/runner.py:L205-215`, if `need_tools` matches a keyword (e.g., "berita", "lagu") and the LLM replies directly with no tool calls, `first.text` is discarded and `stream_text` runs a second full LLM call.
    - **Fix:** Use a small shared stream parser buffer that handles `<think>`/`</think>` across arbitrary chunk boundaries (including on retry), and emit `first.text` directly in `stream_voice_turn` when `not first.tool_calls`.
    - **If we skip it:** Thinking models like `qwen3:8b` can produce silent voice turns or speak `<think>` reasoning aloud, and keyword-matched turns pay double LLM latency.
    - **Effort:** S | **Verified:** yes (reproduced single-chunk `<think>ok</think>Halo` → `in_think=True, output=[]` in Python; code read of `brain/runner.py:L205-215`).

13. **Unbounded `events_log` writes (never read), unbounded song cache, and leaked RVC temp files** (`core/store.py:L17-24, L197-202`, `core/bus.py:L62-66`, `sing/engine.py:L33, L166-195, L251`, `speech/tts.py:L176-188`)
    - **What this is:** `EventBus` persists almost every event to `events_log` in SQLite; `sing/engine.py` caches songs in `data/songs/`; `speech/tts.py` creates temp files for RVC conversion.
    - **Problem:** Nothing in the repo ever queries `events_log` (`SELECT` count = 0), yet every event triggers a SQLite `INSERT` + `commit()` and grows `miyoria.db` forever. `data/songs/` keeps all downloaded tracks and RVC intermediate `.wav` files forever, and `speech/tts.py:L176-188` skips `in_path.unlink()` whenever the RVC HTTP call raises an exception, leaking files in `%TEMP%`.
    - **Fix:** Remove the unused `events_log` persistence hook (or add periodic retention pruning), wrap `in_path.unlink(missing_ok=True)` in `try/finally` in `speech/tts.py`, and cap `data/songs/` disk usage (e.g. evict oldest files above 500 MB) while cleaning up RVC intermediate stems.
    - **If we skip it:** Disk usage in `data/` and `%TEMP%` grows endlessly, and every event pays unnecessary SQLite commit overhead.
    - **Effort:** S | **Verified:** yes (`codegraph_explore` + full-repo grep confirmed 0 reads of `events_log`; code read of `tts.py` and `sing/engine.py`).

14. **Blocking FFmpeg on the event loop, swallowed bus exceptions, and abrupt shutdown** (`sing/engine.py:L253-261`, `core/bus.py:L62-86`, `main.py:L154-160`)
    - **What this is:** RVC audio mixing, event dispatch, and process shutdown all share one asyncio event loop.
    - **Problem:** `sing/engine.py:L253` calls `subprocess.run(["ffmpeg", ...])` directly on the event loop, freezing Discord voice heartbeats and the Web Panel during mixing. `core/bus.py:L65, L85` catches `Exception: pass` and spawns unmonitored tasks (`asyncio.create_task(res)`), hiding handler bugs silently. `main.py:L154-160` closes `store` immediately on exit without stopping active voice sessions or awaiting `web_task` / `bot.close()`.
    - **Fix:** Offload `subprocess.run` to `asyncio.to_thread`, log handler errors in `EventBus.emit` (with a done-callback on spawned tasks), and cleanly stop voice sessions + await `web_task` before `store.close()` in `main.py`.
    - **If we skip it:** RVC mixing freezes the bot, background crashes vanish without logs, and Ctrl+C shutdown can raise DB-closed errors or leave voice sessions dangling.
    - **Effort:** S | **Verified:** yes (code read of `sing/engine.py`, `core/bus.py`, `main.py`).

15. **DM crashes on slash commands, 2000-char overflows, and unvalidated JSON/float inputs** (`discord_bot/client.py:L185-244, L284, L310, L383`, `web/server.py:L112-120, L144-157`, `brain/tools.py:L126`)
    - **What this is:** User input handling across Discord slash commands, LLM tool calls, and Admin Panel POST routes.
    - **Problem:** If `GUILD_ID` is unset (global commands), running `/join`, `/leave`, `/say`, `/sing`, or `/reset` in a DM crashes with `AttributeError` because `ctx.guild` is `None`. `/say` and `/queue` can exceed Discord's 2000-char limit and fail with HTTP 400. Submitting non-list `lore_entries` (e.g. `{}`) to `POST /api/persona` causes `ProfileManager.matches_lore` to crash on every subsequent chat/voice turn; non-numeric `minutes` in `set_reminder` or `bond` in `POST /api/people/{id}` throws unhandled `ValueError`.
    - **Fix:** Guard `ctx.guild is None` in guild-only commands (and skip guild voice lookup in `/say` & `/reset` when in DM), slice Discord messages to `[:1990]`, validate `lore_entries` is a `list[dict]`, and catch `ValueError`/`TypeError` on numeric casts.
    - **If we skip it:** A single DM command, long reply, malformed LLM argument, or typo in Lorebook JSON crashes handlers or breaks every turn until fixed in SQLite.
    - **Effort:** S | **Verified:** yes (code read of `discord_bot/client.py`, `web/server.py`, `brain/profile.py`, `brain/tools.py`).

16. **Unpinned dependencies, mutable GitHub PR ref, and 3 unused packages in `requirements.txt`** (`requirements.txt:L2-27`, `setup.bat:L30`)
    - **What this is:** Python dependency specification installed by `setup.bat` on fresh Windows setups.
    - **Problem:** `requirements.txt:L2` pulls `py-cord[voice]` from `@refs/pull/3159/head`—a mutable PR branch that breaks when merged, closed, or force-pushed. Almost every other dependency has no version pin at all, and 3 installed packages (`tomli-w`, `soundfile`, `scipy` — ~100 MB+) are never imported anywhere in the repo.
    - **Fix:** Pin `py-cord` to a fixed commit SHA (or released version once DAVE support ships), add compatible version bounds for core packages, and remove `tomli-w`, `soundfile`, and `scipy` from `requirements.txt`.
    - **If we skip it:** Fresh installs can break unpredictably when PR #3159 or upstream libraries change, while downloading >100 MB of unused packages.
    - **Effort:** S | **Verified:** yes (full-repo grep confirmed 0 imports of `tomli_w`, `soundfile`, `scipy`; verified installed versions via `pip list`).

17. **Dead code, unused tables, and no-op configuration options** (`core/store.py:L42-47, L111`, `config.py:L90, L105, L115-116`, `brain/tools.py:L74, L88-89`, `brain/gate.py:L102-108`, `stage/bridge.py:L174-194`, `web/server.py:L40, L42`, `main.py:L6`)
    - **What this is:** Leftover tables, config fields, methods, and parameters from unimplemented or refactored features.
    - **Problem:** Confirmed unused by both `codegraph_explore` and full-repo grep: `transcript_summaries` table (only deleted, never populated/read), `heart_person.opt_out_voice`, config fields `kokoro_voice`, `parked_stream_ttl_ms`, `mind_loop_interval_sec`, `max_history_turns` (runner hardcodes 20/18), `VOICE_TOOL_NAMES`, `ToolBox.set_song_callback`, `ActionGate.resolve_approval` (no route/command calls it), `StageBridge.switch_obs_scene`, unused `economy` & `tts` params in `create_web_app`, and `import signal` in `main.py`.
    - **Fix:** Delete the unused symbols, imports, and dead config options (or wire `max_history_turns` into `brain/runner.py:L91` if you want it configurable).
    - **If we skip it:** Changing `.env` settings like `MAX_HISTORY_TURNS` or `PARKED_STREAM_TTL_MS` silently has no effect, and dead code clutters maintenance.
    - **Effort:** S | **Verified:** yes (`codegraph_explore` + full-repo grep).

18. **Docs and character profile drift** (`README.md:L64`, `.env.example:L7, L22`, `config.py:L64-66`, `profiles/default/identity.md:L1-3`, `profiles/default/profile.toml:L2-27`)
    - **What this is:** Setup documentation, `.env.example`, and default persona files.
    - **Problem:** (1) `README.md:L64` and `.env.example:L7` say **15** slash commands, while `discord_bot/client.py` defines **17**. (2) `profiles/default/identity.md` names the character **Pachinko**, whereas `profile.toml`, `examples.md`, and `lorebook.toml` still say **Aira**, so voice/text wake-word triage responds to "Aira" instead of "Pachinko". (3) `profile.toml` fields `baseline_valence`, `baseline_arousal`, `baseline_energy`, and `[voice]` are never applied to `HeartEngine` or `TTSEngine`. (4) `config.py:L64-66` & `.env.example:L22` set default Gemini fallback models to `gemini-3.8-flash` and `gemini-3.5-flash-lite` (unverified against live Gemini API without network call—double-check model IDs before relying on cloud fallback).
    - **Fix:** Sync the character name across `profiles/default/*`, update README/env command counts to 17, and remove or wire unused `profile.toml` keys.
    - **If we skip it:** Calling the character "Pachinko" fails wake-word detection, and editing `profile.toml` voice/mood baselines silently does nothing.
    - **Effort:** S | **Verified:** yes (code read of all profile, config, and README files; Gemini model ID validity marked unverified because no network/API calls were run).

---

## Nice to Have

19. **Admin Studio UX & accessibility gaps (silent API errors, wrong model label, missing labels/focus)** (`web/static/panel.js:L75-86, L163-265`, `web/static/index.html:L116, L121, L203`, `web/static/panel.css:L279-284`)
    - **What this is:** Browser Admin Studio forms, buttons, and status cards.
    - **Problem:** Button handlers (`btn-save-persona`, `btn-save-runtime`, `btn-add-memory`, `btn-song-add`, `btn-toggle-autonomy`) don't catch `api()` rejections, so failed actions give no toast or error feedback. The Overview card always shows `ollama_model` even when `llm_provider` is `gemini` or `openai_compat`. Three inputs (`#inp-playground`, `#inp-lore-json`, `#inp-song-query`) lack `<label>`/`aria-label`, and buttons/tabs lack `:focus-visible` outlines.
    - **Fix:** Add error handling with `toast(...)` on all action clicks, show the active provider's model in Overview, add `aria-label` to unlabeled inputs, and add `:focus-visible` rules in `panel.css`.
    - **If we skip it:** Failed actions fail silently in the browser console, keyboard navigation has weak focus cues, and non-Ollama setups show a misleading model name.
    - **Effort:** S | **Verified:** yes (code read of `index.html`, `panel.js`, `panel.css`).

20. **Per-request cloud client allocation & unquoted FTS5 search terms** (`brain/llm.py:L244`, `speech/stt.py:L162`, `memory/manager.py:L132-142`)
    - **What this is:** Optional Gemini/Groq fallback clients and SQLite FTS5 keyword memory search.
    - **Problem:** `_gemini_once` and `_transcribe_groq` create a new SDK client on every call (leaving `AsyncGroq` HTTP pools unclosed). In `memory/manager.py:L134`, joining raw words with `OR` causes an FTS5 syntax error whenever a message contains boolean words like `"not"`, `"and"`, or `"near"`, which is swallowed at `L141` and drops all keyword matches for that query.
    - **Fix:** Cache the Gemini/Groq clients lazily on `LLMClient`/`STTEngine` (closing them on shutdown) and quote FTS5 terms (`" OR ".join(f'"{w}"' for w in words)`).
    - **If we skip it:** Cloud fallback leaks sockets per call, and queries containing words like "not" or "and" silently miss FTS5 recall hits.
    - **Effort:** S | **Verified:** yes (code read of `brain/llm.py`, `speech/stt.py`, `memory/manager.py`).

---

## Improvement Ideas (Separate from Findings)

1. **Wire Owner Approval UI in Admin Studio** — Connect the existing `ActionGate.approvals` and `resolve_approval()` to a small approval banner on the Overview tab (`POST /api/approvals/{id}`).
   - **Value for you:** Turns the currently half-wired high-risk Action Gate (`proactive_dm`) into a working one-click approve/deny workflow.
   - **Effort:** S | **New deps:** none.
2. **Add a Zero-Dep `unittest` Suite + `check.bat`** — Create `tests/` using Python's built-in `unittest` and a quick `check.bat` script.
   - **Value for you:** Verifies safety filters, bond math, economy transactions, and voice endpointing in <2 seconds offline before launching the bot.
   - **Effort:** M | **New deps:** none.
3. **Inline Bond & Notebook Editor in Admin Studio** — Expose the already-implemented `POST /api/people/{user_id}` endpoint in the Bonds tab and add soft-delete (`active=0`) for Notebook rows.
   - **Value for you:** Lets you correct bad auto-captured facts or adjust nicknames/impressions without touching SQLite by hand.
   - **Effort:** S | **New deps:** none.
4. **Initialize Git + Export Tested Lockfile** — Run `git init` (add `AUDIT.md` to `.gitignore`) and save a tested `requirements-lock.txt`.
   - **Value for you:** Gives safe rollbacks when tuning prompts/voice code and reproducible installs on a fresh Windows PC.
   - **Effort:** S | **New deps:** none.
5. **Streaming PCM Source for `/sing`** — Stream 20ms PCM chunks straight from an FFmpeg subprocess in `discord.AudioSource` instead of decoding entire songs into memory first.
   - **Value for you:** Near-instant playback start and flat ~1 MB RAM usage regardless of track length.
   - **Effort:** M | **New deps:** none.

---

## Do These First (Top 3)

1. **Fix #1 & #2 (Web Admin Panel XSS + `/ws/live` auth):** Highest security impact—any Discord user can currently run JS in your Admin Panel browser session, and setting `PANEL_TOKEN` leaves `/ws/live` open while breaking the UI.
2. **Fix #3, #4 & #5 (Voice Barge-In deadlock, corrupted interrupted transcripts, and broken silence padding):** Biggest everyday quality-of-life impact—right now Stage 2 Smart Barge-In cannot run during replies, interruptions overwrite older transcripts, and every utterance gets padded with ~17s of fake zeros that disable fast turn-taking.
3. **Fix #6 & #8 (Singing `.m4a`/`.opus` decoder crash + Discord command permissions & `@everyone` protection):** Prevents `/sing` from crashing on common YouTube audio formats and stops non-admin members from wiping memory or pinging `@everyone`.

### Ready-to-Paste Fix Prompts (All Must-Fix Findings)

#### Prompt 1 — Web Admin Panel XSS & `PANEL_TOKEN` Auth (Findings #1, #2)
```text
/ponytail
Task: Fix stored XSS in the Web Admin Panel and enforce PANEL_TOKEN consistently across REST, WebSocket, and the frontend UI.
Scope:
- `web/static/panel.js`: Add a small `escapeHtml` helper (or use `textContent` DOM nodes) and escape all dynamic fields rendered in `appendEvent`, `loadStatus` (singing queue & stage), `loadPeople`, `loadMemory`, and `sendPlayground`.
- `web/server.py`: Use `hmac.compare_digest` in `_check_auth`, enforce the same token check on `@app.websocket("/ws/live")` (via `?token=` query param, closing with code 1008 if invalid), and disable `docs_url` when `config.panel_token` is set.
- `web/static/panel.js` & `web/static/overlay.html`: Read an optional `?token=` query param (persisting in `sessionStorage` in `panel.js`) and send it via `Authorization: Bearer <token>` on `fetch` and `?token=` on `/ws/live`.
Verify: Run `.venv\Scripts\python.exe -m compileall -q web`, and run a small local FastAPI `TestClient` / Python check confirming (a) `/ws/live` rejects missing/wrong token when `PANEL_TOKEN` is set and accepts the right token, and (b) `<img src=x onerror=alert(1)>` in event/person data is escaped in `panel.js`.
```

#### Prompt 2 — Voice Barge-In, Interrupted Transcripts & Silence Padding (Findings #3, #4, #5)
```text
/ponytail
Task: Fix the voice pipeline so Stage 2 Smart Barge-In actually runs during playback, interrupted turns save accurately without corrupting older transcripts, and SmartTurnSink stops injecting seconds of fake zeros.
Scope:
- `voice/session.py`: Do not `await self._active_turn_task` inside `_process_utterance`; track it as a background task with an exception callback and cancel any active turn before starting a new one. Move assistant voice transcript saving into `_speak_streaming_turn` (saving `heard_content` + `interrupted=1` when `_barge_triggered` is set) and remove usage of `brain._last_voice_transcript_id`.
- `brain/runner.py`: Remove the duplicate assistant transcript insert / `_last_voice_transcript_id` state from `stream_voice_turn` (return/emit emotion as before, letting `NaturalVoiceSession` own the final spoken/interrupted transcript write).
- `speech/turn.py`: Track `last_packet_ts` on `_SpeakerTrack` so synthetic silence padding only runs on genuine packet gaps (`now - tr.last_packet_ts`), and run `analyze_end_of_turn` over the last 600ms of voiced audio (excluding trailing quiet frames).
Verify: Run `.venv\Scripts\python.exe -m compileall -q voice brain speech` and a quick Python self-check confirming (a) feeding 20ms quiet frames to `SmartTurnSink.write` does NOT inflate `tr.buf` with synthetic zeros and `analyze_end_of_turn` returns >= 0.58 on a falling-tail signal, and (b) `_worker_loop` dequeues a second utterance while `_speak_streaming_turn` is still running.
```

#### Prompt 3 — Singing FFmpeg File Decoding, Duration Cap & Safe URLs (Finding #6)
```text
/ponytail
Task: Fix `/sing` failing on `.m4a` and `.opus` files, prevent huge RAM spikes from long videos, and block non-YouTube/localhost URLs in `yt-dlp`.
Scope:
- `sing/engine.py`: Decode `audio_file` directly from its disk path via FFmpeg (`ffmpeg -hide_banner -loglevel error -i <path> -f s16le -acodec pcm_s16le -ar 48000 -ac 2 pipe:1`) instead of piping bytes with `-f <ext>`. Add a `yt-dlp` duration limit (e.g., reject tracks > 600s) and allow only YouTube URLs (`youtube.com`, `youtu.be`) or plain text search queries.
Verify: Run `.venv\Scripts\python.exe -m compileall -q sing` and a quick Python check decoding a small generated `.m4a` and `.ogg`/opus file via the new helper + confirming `http://127.0.0.1:7878/` is rejected.
```

#### Prompt 4 — VTube Studio WebSocket Concurrency Lock (Finding #7)
```text
/ponytail
Task: Prevent VTube Studio WebSocket `ConcurrencyError` crashes when emotion hotkeys fire concurrently with the 25 FPS parameter loop.
Scope:
- `stage/bridge.py`: Add `self._vts_lock = asyncio.Lock()` and acquire it around every `send` + `recv` exchange on `self._vts_ws` (both `_vts_send` and the 25 FPS loop in `_vts_loop`).
Verify: Run `.venv\Scripts\python.exe -m compileall -q stage` and a small asyncio test with a mock WebSocket asserting concurrent `_vts_send` and parameter injection never overlap `recv()`.
```

#### Prompt 5 — Discord Command Permissions & Mention Safety (Finding #8)
```text
/ponytail
Task: Restrict admin/destructive slash commands to owners or server admins and disable `@everyone`/`@here`/role pings across bot messages.
Scope:
- `discord_bot/client.py`: Pass `allowed_mentions=discord.AllowedMentions(everyone=False, roles=False)` when creating `discord.Bot`. Add a small permission helper checking `ctx.author.id in config.owner_ids` or `manage_guild` permission, and enforce it on `/mode`, `/reset`, and `/reload` (and allow `/skip` only for the requester or owner/admin).
Verify: Run `.venv\Scripts\python.exe -m compileall -q discord_bot` and verify `build_discord_bot(...)` sets `bot.allowed_mentions.everyone is False` and `roles is False`.
```

#### Prompt 6 — Serialized SQLite Writes & Atomic Economy Transactions (Finding #9)
```text
/ponytail
Task: Prevent coin double-spend / double-daily races and cross-coroutine SQLite commit collisions.
Scope:
- `core/store.py`: Add an `asyncio.Lock` and an `@asynccontextmanager async def write_tx(self)` helper on `Store` to serialize multi-statement write transactions cleanly.
- `play/economy.py`: Run `daily_checkin`, `draw_card`, and `gift_to_character` inside `store.write_tx()`, use conditional SQL (`UPDATE economy_wallets SET balance = balance - ? WHERE user_id = ? AND balance >= ?`, checking `rowcount == 1`), and return the actual post-commit balance.
Verify: Run `.venv\Scripts\python.exe -m compileall -q core play` and an async Python test firing two concurrent `daily_checkin` and two concurrent `gift_to_character` calls against an in-memory/temp SQLite DB, asserting only one succeeds.
```

#### Prompt 7 — Remove Unadvertised `web_read` SSRF & Enforce Tool Allow-list (Finding #10)
```text
/ponytail
Task: Remove the unadvertised `web_read` SSRF path and restrict `ToolBox.execute` to tools declared in `TOOL_SCHEMAS`.
Scope:
- `brain/tools.py`: Delete `_web_read` and the `name == "web_read"` branch; validate `name` against `{t["function"]["name"] for t in TOOL_SCHEMAS}` before executing.
- `brain/gate.py`: Remove `"web_read"` from `RISK_LEVELS`.
Verify: Run `.venv\Scripts\python.exe -m compileall -q brain` and a quick Python check confirming `tools.execute("web_read", {"url": "http://127.0.0.1:7878/api/status"}, ...)` is rejected.
```

---

**Verdict:** Do not expose beyond localhost or rely on voice barge-in/singing yet—fix #1, #2, #3, #4, #5, #6, #7, #8, #9, and #10 first.
**Lean:** -~90 lines, -3 dependencies (`tomli-w`, `soundfile`, `scipy`) possible.
**Not checked:** `.env` (intentionally unread to avoid exposing real tokens/keys); live runtime integrations requiring network or local desktop apps (Discord gateway/DAVE voice handshake, Ollama inference, Faster-Whisper model download, Edge/Piper/ElevenLabs/RVC synthesis, YouTube/LRCLIB fetch, VTube Studio, OBS Studio WebSocket, Gemini/Groq cloud APIs).
