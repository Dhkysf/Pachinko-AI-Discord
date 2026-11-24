const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const ESC_MAP = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };
function esc(v) {
  return String(v ?? '').replace(/[&<>"']/g, (c) => ESC_MAP[c]);
}

const TOKEN_KEY = 'pachinko_panel_token';
function loadPanelToken() {
  const url = new URL(location.href);
  const fromUrl = url.searchParams.get('token');
  if (fromUrl !== null) {
    if (fromUrl) sessionStorage.setItem(TOKEN_KEY, fromUrl);
    else sessionStorage.removeItem(TOKEN_KEY);
    url.searchParams.delete('token');
    history.replaceState(null, '', url.pathname + url.search + url.hash);
  }
  return sessionStorage.getItem(TOKEN_KEY) || '';
}
const PANEL_TOKEN = loadPanelToken();
let authToastShown = false;

let autonomyPaused = false;

function toast(msg) {
  const el = $('#toast');
  el.textContent = msg;
  el.classList.add('show');
  setTimeout(() => el.classList.remove('show'), 2400);
}

async function api(path, opts = {}) {
  const headers = { 'Content-Type': 'application/json', ...(opts.headers || {}) };
  if (PANEL_TOKEN) headers.Authorization = `Bearer ${PANEL_TOKEN}`;
  const res = await fetch(path, { ...opts, headers });
  if (res.status === 401) {
    if (!authToastShown) {
      authToastShown = true;
      toast('Token panel salah atau belum diisi. Buka panel dengan ?token=...');
    }
    throw new Error('Unauthorized (401)');
  }
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

/* Tab Navigation */
const HEADINGS = {
  overview: 'Character Monitor & Mind',
  persona: 'Persona Studio, Lorebook & Playground',
  bonds: 'Social Relationships & Bond Ladder',
  memory: 'Hybrid Memory & Living Notebook',
  voice: 'Voice & Karaoke Configuration',
};

$$('.nav-btn').forEach((btn) => {
  btn.addEventListener('click', () => {
    $$('.nav-btn').forEach((b) => b.classList.remove('active'));
    $$('.tab-panel').forEach((p) => p.classList.remove('active'));
    btn.classList.add('active');
    const tab = btn.dataset.tab;
    $(`#tab-${tab}`).classList.add('active');
    $('#page-heading').textContent = HEADINGS[tab] || 'Pachinko Studio';
    if (tab === 'persona') loadPersona();
    if (tab === 'bonds') loadPeople();
    if (tab === 'memory') loadMemory();
  });
});

function renderEmotions(heart) {
  const box = $('#emotion-bars');
  const entries = Object.entries(heart.active_emotions || {});
  if (!entries.length) {
    box.innerHTML = '<p style="color:var(--text-muted); font-size:0.86rem;">Mood is calm (no strong emotion spikes yet).</p>';
    return;
  }
  box.innerHTML = entries
    .sort((a, b) => b[1] - a[1])
    .map(
      ([k, v]) => {
        const pct = Math.max(0, Math.min(100, Number(v) * 100 || 0));
        return `
      <div class="meter-row">
        <div class="meter-head"><span>${esc(k)}</span><span>${Math.round(pct)}%</span></div>
        <div class="meter-track"><div class="meter-fill" style="width:${pct}%"></div></div>
      </div>`;
      }
    )
    .join('');
}

function appendEvent(ev) {
  const log = $('#live-event-log');
  if (!log) return;
  const d = new Date(ev.ts * 1000).toLocaleTimeString('en-GB', { hour12: false });
  const row = document.createElement('div');
  row.className = 'event-item';
  row.innerHTML = `<span class="event-ts">${esc(d)}</span><span class="event-topic">${esc(ev.topic)}</span><span class="event-body">${esc(JSON.stringify(ev.payload))}</span>`;
  log.prepend(row);
  while (log.children.length > 90) log.lastChild.remove();
}

async function loadStatus() {
  const s = await api('/api/status');
  $('#brand-char-name').textContent = `${s.character.display_name} Studio`;
  $('#stat-emotion').textContent = s.heart.dominant;
  $('#stat-valence').textContent = `Valence ${s.heart.valence >= 0 ? '+' : ''}${s.heart.valence} · Energy ${Math.round(s.heart.energy * 100)}%`;
  $('#stat-llm').textContent = s.runtime.llm_provider;
  $('#stat-llm-model').textContent = s.runtime.ollama_model;
  $('#stat-tts').textContent = s.runtime.tts_provider;
  $('#stat-vram').textContent = s.runtime.vram_busy_with
    ? `Busy: ${s.runtime.vram_busy_with}`
    : `Idle (${s.runtime.gpu_vram_gb} GB VRAM)`;

  autonomyPaused = s.autonomy.paused;
  const abtn = $('#btn-toggle-autonomy');
  abtn.textContent = autonomyPaused ? '▶️ Resume Autonomy' : '⏸️ Pause Autonomy';

  renderEmotions(s.heart);

  $('#sel-llm-provider').value = s.runtime.llm_provider;
  $('#inp-ollama-model').value = s.runtime.ollama_model;
  $('#sel-tts-provider').value = s.runtime.tts_provider;
  $('#inp-edge-voice').value = s.runtime.edge_tts_voice;

  const qLines = [];
  if (s.singing.current) {
    qLines.push(`▶️ Playing: ${s.singing.current.title} (${s.singing.current.status})`);
  }
  (s.singing.queue || []).forEach((it, i) => qLines.push(`${i + 1}. ${it.query} — req: ${it.requester_name}`));
  $('#sing-queue-box').innerHTML = qLines.length
    ? qLines.map((l) => `<div class="event-item"><span class="event-body">${esc(l)}</span></div>`).join('')
    : '<div style="color:var(--text-muted)">Queue is empty.</div>';
}

async function loadPersona() {
  const p = await api('/api/persona');
  $('#inp-char-name').value = p.display_name || '';
  $('#inp-identity').value = p.identity_md || '';
  $('#inp-style').value = p.style_md || '';
  $('#inp-examples').value = p.examples_md || '';
  $('#inp-directives').value = p.directives || '';
  $('#inp-lore-json').value = JSON.stringify(p.lore_entries || [], null, 2);
}

async function loadPeople() {
  const { people } = await api('/api/people');
  const tbody = $('#people-tbody');
  if (!people.length) {
    tbody.innerHTML = '<tr><td colspan="6">No member interactions recorded yet.</td></tr>';
    return;
  }
  tbody.innerHTML = people
    .map(
      (p) => {
        const bond = Number(p.bond) || 0;
        const trust = Number(p.trust) || 0;
        const warmth = Number(p.warmth) || 0;
        return `
      <tr>
        <td><strong style="color:var(--text-primary)">${esc(p.display_name)}</strong></td>
        <td><span class="tag">${esc(p.tier)}</span></td>
        <td>
          <div style="min-width:110px;">
            <div>${bond.toFixed(1)} / 1000</div>
            <div class="meter-track"><div class="meter-fill jade" style="width:${Math.max(2, Math.min(100, bond / 10))}%"></div></div>
          </div>
        </td>
        <td>T: ${trust.toFixed(0)} · W: ${warmth.toFixed(2)}</td>
        <td>${esc(p.interactions)}x</td>
        <td>${esc(p.impression)}</td>
      </tr>`;
      }
    )
    .join('');
}

async function loadMemory() {
  const { notebook, memories } = await api('/api/memory');
  $('#notebook-list').innerHTML = notebook.length
    ? notebook.map((n) => `<div class="event-item"><span class="event-topic">${esc(n.doc_key)}</span><span class="event-body">${esc(n.fact)}</span></div>`).join('')
    : '<div style="color:var(--text-muted)">No notes in Living Notebook yet.</div>';

  $('#memories-list').innerHTML = memories.length
    ? memories.map((m) => `<div class="event-item"><span class="event-topic">${esc(m.scope)}</span><span class="event-body">${esc(m.content)}</span></div>`).join('')
    : '<div style="color:var(--text-muted)">No episodic memories yet.</div>';
}

/* Actions */
$('#btn-refresh-all').addEventListener('click', () => {
  loadStatus();
  toast('Status refreshed');
});

$('#btn-toggle-autonomy').addEventListener('click', async () => {
  await api('/api/runtime', {
    method: 'POST',
    body: JSON.stringify({ autonomy_paused: !autonomyPaused }),
  });
  await loadStatus();
  toast(autonomyPaused ? 'Autonomy paused' : 'Autonomy active');
});

$('#btn-save-persona').addEventListener('click', async () => {
  let lore = null;
  try {
    lore = JSON.parse($('#inp-lore-json').value || '[]');
  } catch (e) {
    alert('Invalid Lorebook JSON format: ' + e.message);
    return;
  }
  await api('/api/persona', {
    method: 'POST',
    body: JSON.stringify({
      display_name: $('#inp-char-name').value,
      identity_md: $('#inp-identity').value,
      style_md: $('#inp-style').value,
      examples_md: $('#inp-examples').value,
      directives: $('#inp-directives').value,
      lore_entries: lore,
    }),
  });
  await loadStatus();
  toast('Persona & Lorebook applied live!');
});

async function sendPlayground() {
  const inp = $('#inp-playground');
  const msg = inp.value.trim();
  if (!msg) return;
  inp.value = '';
  const box = $('#playground-log');
  box.innerHTML += `<div class="chat-msg"><strong>You:</strong> ${esc(msg)}</div>`;
  box.scrollTop = box.scrollHeight;
  try {
    const res = await api('/api/playground', {
      method: 'POST',
      body: JSON.stringify({ message: msg }),
    });
    const reply = res.silent ? '*(chose silence / <silence/>)*' : res.reply;
    box.innerHTML += `<div class="chat-msg self"><strong>[${esc(res.emotion)}]:</strong> ${esc(reply)}</div>`;
    box.scrollTop = box.scrollHeight;
    loadStatus();
  } catch (e) {
    box.innerHTML += `<div class="chat-msg" style="color:var(--accent-coral)">Error: ${esc(e.message)}</div>`;
  }
}

$('#btn-send-playground').addEventListener('click', sendPlayground);
$('#inp-playground').addEventListener('keydown', (e) => {
  if (e.key === 'Enter') sendPlayground();
});

$('#btn-add-memory').addEventListener('click', async () => {
  const fact = $('#inp-mem-fact').value.trim();
  if (!fact) return;
  await api('/api/memory', {
    method: 'POST',
    body: JSON.stringify({ doc_key: $('#inp-mem-doc').value.trim(), fact }),
  });
  $('#inp-mem-fact').value = '';
  await loadMemory();
  toast('Note saved to Notebook!');
});

$('#btn-save-runtime').addEventListener('click', async () => {
  await api('/api/runtime', {
    method: 'POST',
    body: JSON.stringify({
      llm_provider: $('#sel-llm-provider').value,
      ollama_model: $('#inp-ollama-model').value,
      tts_provider: $('#sel-tts-provider').value,
      edge_tts_voice: $('#inp-edge-voice').value,
    }),
  });
  await loadStatus();
  toast('AI engine settings applied!');
});

$('#btn-song-add').addEventListener('click', async () => {
  const q = $('#inp-song-query').value.trim();
  if (!q) return;
  const res = await api('/api/sing/enqueue', { method: 'POST', body: JSON.stringify({ query: q }) });
  $('#inp-song-query').value = '';
  await loadStatus();
  toast(res.message);
});

$('#btn-song-skip').addEventListener('click', async () => {
  await api('/api/sing/skip', { method: 'POST', body: '{}' });
  toast('Song skipped');
});

/* WebSocket Live Stream */
function connectWS() {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const qs = PANEL_TOKEN ? `?token=${encodeURIComponent(PANEL_TOKEN)}` : '';
  const ws = new WebSocket(`${proto}//${location.host}/ws/live${qs}`);
  ws.onopen = () => {
    $('#ws-dot').classList.add('live');
    $('#ws-state').textContent = 'Connected';
  };
  ws.onmessage = (e) => {
    const ev = JSON.parse(e.data);
    appendEvent(ev);
    if (ev.topic.startsWith('heart.') || ev.topic.startsWith('sing.') || ev.topic === 'runtime.updated') {
      loadStatus();
    }
  };
  ws.onclose = (e) => {
    $('#ws-dot').classList.remove('live');
    if (e && e.code === 1008) {
      $('#ws-state').textContent = 'Auth failed';
      return;
    }
    $('#ws-state').textContent = 'Reconnecting...';
    setTimeout(connectWS, 3000);
  };
}

(async function init() {
  try {
    await loadStatus();
    const { events } = await api('/api/events?limit=40');
    events.forEach(appendEvent);
    connectWS();
  } catch (e) {
    console.error(e);
  }
})();
