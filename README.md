# Pachinko — Autonomous Discord Voice & Chat Companion

Bot companion suara & chat Discord berbahasa Python penuh (inspired by the architecture of owenliu0924/Miyoria, no code copied)—dirancang khusus supaya mudah diedit di VS Code pada **Windows 11 + GPU AMD Radeon RX 9070 (16 GB VRAM)** dan 100% aman untuk akun Discord (memakai **Official Bot Token + DAVE E2EE**).

---

## ✨ Seluruh Subsistem yang Termasuk

1. **Brain + 7-Block Prompt + Triage (`brain/`)**
   - Satu pikiran menyatu untuk **Voice** & **Text Chat**.
   - Mode otonomi per channel (`/mode`): `off`, `mention-only`, `quiet` (2x/jam), `normal` (6x/jam), `lively` (14x/jam).
   - Pertahanan anti-*prompt-injection* berbasis kutipan `<data>` (`core/safety.py`) + **Action Gate** (`brain/gate.py`).
   - Tools terpasang: Pencarian internet (`ddgs`), Buku Catatan, Pengingat waktu, & Antrean Lagu.

2. **Heart Engine & 9-Axis Relationship (`heart/`)**
   - Simulasi **12 emosi** dengan peluruhan waktu (*half-life decay*), *valence*, *arousal*, & *energi*.
   - Tag tersembunyi `<emo:...>` & `<feel>...</feel>` meng-update mood karakter dan **10 Tingkat Bond Ladder** (`-100..1000`) per member dengan redaman geometris harian.

3. **Memori Berlapis SQLite WAL (`core/store.py`, `memory/`)**
   - **L0 Event Log**, **L1 Persistent Transcripts** (mencatat kalimat parsial bila terpotong *barge-in*), **L2 Hybrid Recall** (SQLite FTS5 + vektor lokal n-gram `<1ms`), dan **L3 Living Notebook** otomatis.

4. **Natural Voice Pipeline (`speech/`, `voice/`)**
   - **Smart Prosodic Turn-Taking + Parked Stream**: Tidak memotong saat kamu jeda mikir di tengah kalimat, tapi responsif saat intonasi kalimat selesai.
   - **Streaming Kalimat + `LinePrefetch`**: Kalimat ke-1 langsung diucapkan begitu keluar dari LLM *sambil* kalimat ke-2 disintesis di latar belakang.
   - **Cached Filler** (*"Hmm?"*, *"Eh, bentar..."*) + **SelfEchoGuard** (anti-feedback bocoran speaker).
   - **Two-Stage Smart Barge-In**: Membedakan gumaman pendek (*"hmm"*, *"iya"*, *"oke"*) vs interupsi sungguhan (bisa *ducking* 28%, *pause*, lalu *resume* tanpa putus!).

5. **Web Admin Panel (`web/`)**
   - **Admin Studio (`http://127.0.0.1:7878`)**: Monitor emosi live, aliran event WebSocket, editor Persona & Lorebook *hot-reload*, Sandbox Playground chat, penjelajah Bond member, Notebook, & kontrol mesin AI.

6. **AI Singing / Karaoke + Koordinator VRAM RX 9070 (`sing/`, `core/vram.py`)**
   - Request lagu dari YouTube (`/sing`), mengosongkan VRAM Ollama sementara (`keep_alive: 0`) bila mode AI Cover RVC aktif, dan memutar lagu di voice channel Discord.

7. **Play & Economy (`play/`)**
   - Dompet koin (`/daily` streak + ramalan, `/gift` penambah Bond, `/card` gacha kartu koleksi SVG, & `/trivia` mini-game).

---

## 🚀 Cara Menjalankan di Windows (AMD Radeon RX 9070 16GB)

### 1. Siapkan Ollama untuk Otak LLM Lokal
Di PowerShell Windows:
```powershell
winget install Ollama.Ollama
ollama pull qwen3:8b
```
*(Dengan VRAM 16 GB di RX 9070, `qwen3:8b` hanya memakai ~5.5 GB VRAM sehingga sangat kencang dan menyisakan >10 GB VRAM longgar. Kamu juga bisa memakai `gemma3:12b`).*

### 2. Aktifkan Intent di Discord Developer Portal
Buka https://discord.com/developers/applications → pilih aplikasi bot kamu → menu **Bot** → nyalakan **Message Content Intent** → Save Changes. *(Jika belum dinyalakan, bot tetap berjalan aman dalam mode Voice + Slash Command).*

### 3. Jalankan Setup & Start
1. Ekstrak folder `pachinko` (mis. ke `C:\pachinko`).
2. Klik dua kali **`setup.bat`** → isi `DISCORD_TOKEN` dan `GUILD_ID` saat Notepad terbuka → simpan.
3. Klik dua kali **`run.bat`**.
4. Buka **`http://127.0.0.1:7878`** di browser untuk membuka **Pachinko Studio**!

---

## 🎮 Daftar Slash Command Discord (15 Command)

| Command | Fungsi |
|---|---|
| `/join` / `/leave` | Masuk / keluar voice channel (DAVE E2EE + Smart Turn) |
| `/say <pesan>` | Ajak ngobrol lewat teks (sekaligus diucapkan di voice bila sedang join) |
| `/mode <level>` | Atur keaktifan nimbrung di channel (`off`, `mention-only`, `quiet`, `normal`, `lively`) |
| `/mood` | Lihat spektrum emosi, *valence*, & energi karakter |
| `/bond` | Lihat tingkat kedekatan (9-Axis Bond) & catatan karakter tentangmu |
| `/sing <lagu>` / `/skip` / `/queue` | Request lagu / karaoke YouTube, lewati lagu, lihat antrean |
| `/daily` / `/card` / `/gift` | Klaim koin harian + ramalan, gacha kartu koleksi SVG, traktir karakter |
| `/trivia` | Mulai mini-game kuis cepat berhadiah koin di chat |
| `/remind <menit> <catatan>` | Pasang pengingat (diingatkan lewat chat & suara!) |
| `/privacy` / `/reset` / `/reload` | Atur privasi sapaan otomatis, segarkan obrolan pendek, muat ulang persona |
