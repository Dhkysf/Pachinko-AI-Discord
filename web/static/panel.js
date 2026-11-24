const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

let autonomyPaused = false;

function toast(msg) {
  const el = $('#toast');
  el.textContent = msg;
  el.classList.add('show');
  setTimeout(() => el.classList.remove('show'), 2400);
}

async function api(path, opts = {}) {
  const res = await fetch(path, {
    headers: { 'Content-Type': 'application/json' },
    ...opts,
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

/* Navigasi Tab */
const HEADINGS = {
  overview: 'Monitor & Pikiran Karakter',
  persona: 'Studio Persona, Lorebook & Playground',
  bonds: 'Hubungan Sosial & Bond Ladder',
  memory: 'Memori Hibrida & Living Notebook',
  voice: 'Konfigurasi Suara, Karaoke & VTuber Stage',
};

$$('.nav-btn').forEach((btn) => {
  btn.addEventListener('click', () => {
    $$('.nav-btn').forEach((b) => b.classList.remove('active'));
    $$('.tab-panel').forEach((p) => p.classList.remove('active'));
    btn.classList.add('active');
    const tab = btn.dataset.tab;
    $(`#tab-${tab}`).classList.add('active');
    $('#page-heading').textContent = HEADINGS[tab] || 'Miyoria Studio';
    if (tab === 'persona') loadPersona();
    if (tab === 'bonds') loadPeople();
    if (tab === 'memory') loadMemory();
  });
});

function renderEmotions(heart) {
  const box = $('#emotion-bars');
  const entries = Object.entries(heart.active_emotions || {});
  if (!entries.length) {
    box.innerHTML = '<p style="color:var(--text-muted); font-size:0.86rem;">Suasana hati tenang (belum ada lonjakan emosi kuat).</p>';
    return;
  }
  box.innerHTML = entries
    .sort((a, b) => b[1] - a[1])
    .map(
      ([k, v]) => `
      <div class="meter-row">
        <div class="meter-head"><span>${k}</span><span>${Math.round(v * 100)}%</span></div>
        <div class="meter-track"><div class="meter-fill" style="width:${Math.min(100, v * 100)}%"></div></div>
      </div>`
    )
    .join('');
}

function appendEvent(ev) {
  const log = $('#live-event-log');
  if (!log) return;
  const d = new Date(ev.ts * 1000).toLocaleTimeString('id-ID', { hour12: false });
  const row = document.createElement('div');
  row.className = 'event-item';
  row.innerHTML = `<span class="event-ts">${d}</span><span class="event-topic">${ev.topic}</span><span class="event-body">${JSON.stringify(ev.payload)}</span>`;
  log.prepend(row);
  while (log.children.length > 90) log.lastChild.remove();
}

async function loadStatus() {
  const s = await api('/api/status');
  $('#brand-char-name').textContent = `${s.character.display_name} Studio`;
  $('#stat-emotion').textContent = s.heart.dominant;
  $('#stat-valence').textContent = `Valence ${s.heart.valence >= 0 ? '+' : ''}${s.heart.valence} · Energi ${Math.round(s.heart.energy * 100)}%`;
  $('#stat-llm').textContent = s.runtime.llm_provider;
  $('#stat-llm-model').textContent = s.runtime.ollama_model;
  $('#stat-tts').textContent = s.runtime.tts_provider;
  $('#stat-vram').textContent = s.runtime.vram_busy_with
    ? `Dipakai: ${s.runtime.vram_busy_with}`
    : `Siaga (${s.runtime.gpu_vram_gb} GB VRAM)`;

  autonomyPaused = s.autonomy.paused;
  const abtn = $('#btn-toggle-autonomy');
  abtn.textContent = autonomyPaused ? '▶️ Lanjutkan Otonomi' : '⏸️ Jeda Otonomi';

  renderEmotions(s.heart);

  $('#sel-llm-provider').value = s.runtime.llm_provider;
  $('#inp-ollama-model').value = s.runtime.ollama_model;
  $('#sel-tts-provider').value = s.runtime.tts_provider;
  $('#inp-edge-voice').value = s.runtime.edge_tts_voice;

  const qLines = [];
  if (s.singing.current) {
    qLines.push(`▶️ Sedang: ${s.singing.current.title} (${s.singing.current.status})`);
  }
  (s.singing.queue || []).forEach((it, i) => qLines.push(`${i + 1}. ${it.query} — req: ${it.requester_name}`));
  $('#sing-queue-box').innerHTML = qLines.length
    ? qLines.map((l) => `<div class="event-item"><span class="event-body">${l}</span></div>`).join('')
    : '<div style="color:var(--text-muted)">Antrean kosong.</div>';

  $('#stage-status-box').innerHTML = `
    VTube Studio Live2D: <strong>${s.stage.vts_connected ? 'Terhubung' : (s.stage.vts_enabled ? 'Menunggu aplikasi...' : 'Nonaktif di .env')}</strong><br>
    OBS Studio WebSocket: <strong>${s.stage.obs_connected ? 'Terhubung' : (s.stage.obs_enabled ? 'Menunggu OBS...' : 'Nonaktif di .env')}</strong><br>
    Scene Aktif: <strong>${s.stage.scene}</strong> · Ekspresi Wajah: <strong>${s.stage.emotion}</strong>
  `;
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
    tbody.innerHTML = '<tr><td colspan="6">Belum ada interaksi member tercatat.</td></tr>';
    return;
  }
  tbody.innerHTML = people
    .map(
      (p) => `
      <tr>
        <td><strong style="color:var(--text-primary)">${p.display_name}</strong></td>
        <td><span class="tag">${p.tier}</span></td>
        <td>
          <div style="min-width:110px;">
            <div>${p.bond.toFixed(1)} / 1000</div>
            <div class="meter-track"><div class="meter-fill jade" style="width:${Math.max(2, Math.min(100, p.bond / 10))}%"></div></div>
          </div>
        </td>
        <td>T: ${p.trust.toFixed(0)} · W: ${p.warmth.toFixed(2)}</td>
        <td>${p.interactions}x</td>
        <td>${p.impression}</td>
      </tr>`
    )
    .join('');
}

async function loadMemory() {
  const { notebook, memories } = await api('/api/memory');
  $('#notebook-list').innerHTML = notebook.length
    ? notebook.map((n) => `<div class="event-item"><span class="event-topic">${n.doc_key}</span><span class="event-body">${n.fact}</span></div>`).join('')
    : '<div style="color:var(--text-muted)">Belum ada catatan di Living Notebook.</div>';

  $('#memories-list').innerHTML = memories.length
    ? memories.map((m) => `<div class="event-item"><span class="event-topic">${m.scope}</span><span class="event-body">${m.content}</span></div>`).join('')
    : '<div style="color:var(--text-muted)">Belum ada memori episodik.</div>';
}

/* Actions */
$('#btn-refresh-all').addEventListener('click', () => {
  loadStatus();
  toast('Status diperbarui');
});

$('#btn-toggle-autonomy').addEventListener('click', async () => {
  await api('/api/runtime', {
    method: 'POST',
    body: JSON.stringify({ autonomy_paused: !autonomyPaused }),
  });
  await loadStatus();
  toast(autonomyPaused ? 'Otonomi dijeda' : 'Otonomi aktif');
});

$('#btn-save-persona').addEventListener('click', async () => {
  let lore = null;
  try {
    lore = JSON.parse($('#inp-lore-json').value || '[]');
  } catch (e) {
    alert('Format JSON Lorebook kurang tepat: ' + e.message);
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
  toast('Persona & Lorebook diterapkan live!');
});

async function sendPlayground() {
  const inp = $('#inp-playground');
  const msg = inp.value.trim();
  if (!msg) return;
  inp.value = '';
  const box = $('#playground-log');
  box.innerHTML += `<div class="chat-msg"><strong>Kamu:</strong> ${msg}</div>`;
  box.scrollTop = box.scrollHeight;
  try {
    const res = await api('/api/playground', {
      method: 'POST',
      body: JSON.stringify({ message: msg }),
    });
    const reply = res.silent ? '*(memilih diam / <silence/>)*' : res.reply;
    box.innerHTML += `<div class="chat-msg self"><strong>[${res.emotion}]:</strong> ${reply}</div>`;
    box.scrollTop = box.scrollHeight;
    loadStatus();
  } catch (e) {
    box.innerHTML += `<div class="chat-msg" style="color:var(--accent-coral)">Error: ${e.message}</div>`;
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
  toast('Catatan disimpan ke Notebook!');
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
  toast('Pengaturan mesin AI diterapkan!');
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
  toast('Lagu di-skip');
});

/* WebSocket Live Stream */
function connectWS() {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const ws = new WebSocket(`${proto}//${location.host}/ws/live`);
  ws.onopen = () => {
    $('#ws-dot').classList.add('live');
    $('#ws-state').textContent = 'Terhubung';
  };
  ws.onmessage = (e) => {
    const ev = JSON.parse(e.data);
    if (ev.topic !== 'stage.lipsync') {
      appendEvent(ev);
    }
    if (ev.topic.startsWith('heart.') || ev.topic.startsWith('sing.') || ev.topic === 'runtime.updated') {
      loadStatus();
    }
  };
  ws.onclose = () => {
    $('#ws-dot').classList.remove('live');
    $('#ws-state').textContent = 'Rekoneksi...';
    setTimeout(connectWS, 3000);
  };
}

(async function init() {
  await loadStatus();
  const { events } = await api('/api/events?limit=40');
  events.forEach(appendEvent);
  connectWS();
})();
