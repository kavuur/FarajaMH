// app.js — FULL, with:
// - No dependency on removed page-level "Search Query / Max Results" UI
// - Question Bank only shows results after Search; Print/Export disabled until results exist
// - Live Mic streaming + Voice Note intact
// - SSE agent chat + screening intact

// ---------------------------------------------
// Boot: small helpers
// ---------------------------------------------
function $(id) { return document.getElementById(id); }
function on(el, ev, fn) { el && el.addEventListener(ev, fn); }
function show(el) { if (el) el.style.display = ''; }
function hide(el) { if (el) el.style.display = 'none'; }

// ---------------------------------------------
// Screening wiring (kept)
// ---------------------------------------------
window.MH_TRANSCRIPT = window.MH_TRANSCRIPT || "";   // accumulate patient-only text
window.MH_ANSWERS   = window.MH_ANSWERS   || {};     // PHQ-9/GAD-7 answers (if any)
window.MH_SAFETY    = !!window.MH_SAFETY;            // suicidality / safety flag

async function runScreening() {
  try {
    const r = await fetch('/mh/screen', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' }, // csrf-exempt on server
      body: JSON.stringify({
        transcript: MH_TRANSCRIPT,
        responses: MH_ANSWERS,
        safety_concerns: MH_SAFETY
      })
    });
    if (!r.ok) return;
    const data = await r.json();
    renderScreeningPanel(data);
  } catch (e) { console.error('mh/screen failed', e); }
}
window.runScreening = runScreening;

let _mhTimer = null;
function debounceScreening() {
  clearTimeout(_mhTimer);
  _mhTimer = setTimeout(runScreening, 1200);
}
window.debounceScreening = debounceScreening;

function renderScreeningPanel(data) {
  const mount = $('mh-screening') || (() => {
    const d = document.createElement('div');
    d.id = 'mh-screening';
    d.className = 'alert alert-secondary mt-2';
    ($('agentsConversation') || document.body).prepend(d);
    return d;
  })();

  const chips = (data.results || []).map(r => {
    const sev  = String(r.severity || '').replace('_',' ');
    const conf = Math.round((r.confidence || 0) * 100);
    let extra = '';
    if (r.name === 'depression') extra = ` • PHQ-9≈${Math.round((r.score || 0) * 27)}/27`;
    if (r.name === 'anxiety')    extra = ` • GAD-7≈${Math.round((r.score || 0) * 21)}/21`;
    return `<span class="badge bg-info text-dark me-1">${r.name}: ${sev}${extra} (${conf}%)</span>`;
  }).join(' ');

  const why  = (data.results || []).flatMap(r =>
    (r.rationale || []).slice(0,5).map(e =>
      `<li><em>${e.feature}</em>: ${e.text} <small class="text-muted">(${e.source})</small></li>`)).join('');

  const steps = (data.results || []).flatMap(r =>
    (r.next_steps || []).map(s => `<li>${s}</li>`)).join('');

  mount.innerHTML = `
    <div><strong>Screening:</strong> ${data.overall_flag || ''}</div>
    <div class="mt-1">${chips || '<span class="text-muted">No signals yet</span>'}</div>
    <details class="mt-2"><summary>Why & Next steps</summary>
      <ul>${why || '<li class="text-muted">—</li>'}</ul>
      <strong>Next steps</strong>
      <ul>${steps || '<li class="text-muted">—</li>'}</ul>
    </details>`;
}
window.renderScreeningPanel = renderScreeningPanel;

// ---------------------------------------------
// Auth & CSRF helpers (kept)
// ---------------------------------------------
let CSRF_TOKEN = null;
async function loadCsrf() {
  try {
    const r = await fetch('/csrf-token', { credentials: 'same-origin' });
    const j = await r.json();
    CSRF_TOKEN = j.csrf_token || j.csrfToken || null;
    window.CSRF_TOKEN = CSRF_TOKEN;
  } catch (_) {}
}
function authHeaders() {
  return { 'Content-Type': 'application/json', 'X-CSRFToken': CSRF_TOKEN || '' };
}
async function getMe() {
  try {
    const r = await fetch('/auth/me', { credentials: 'same-origin' });
    return await r.json();
  } catch { return { authenticated: false }; }
}
async function login(email, password, remember=true) {
  const r = await fetch('/auth/login', {
    method: 'POST', headers: authHeaders(), credentials: 'same-origin',
    body: JSON.stringify({ email, password, remember })
  });
  return r.json();
}
async function signup(email, password) {
  const r = await fetch('/auth/signup', {
    method: 'POST', headers: authHeaders(), credentials: 'same-origin',
    body: JSON.stringify({ email, password })
  });
  return r.json();
}
async function logout() {
  const r = await fetch('/auth/logout', {
    method: 'POST', headers: authHeaders(), credentials: 'same-origin'
  });
  return r.json();
}
function showAuth() {
  $('auth-gate')?.setAttribute('style','');
  const app = $('app-wrapper'); if (app) app.style.display = 'none';
}
function showApp(user) {
  const gate = $('auth-gate'); if (gate) gate.style.display = 'none';
  const app  = $('app-wrapper'); if (app) app.style.display = '';
  const who  = $('whoami');
  if (who) who.textContent = `${user?.email || user?.name || 'User'} — roles: ${(user?.roles||[]).join(', ')}`;
}

// ---------------------------------------------
// App init (auth gate + toggles)
// ---------------------------------------------
window.addEventListener('DOMContentLoaded', async () => {
  await loadCsrf();
  const me = await getMe();
  if (me.authenticated) showApp(me.user); else showAuth();

  const loginForm = $('login-form');
  on(loginForm, 'submit', async (e) => {
    e.preventDefault();
    const email = $('login-email')?.value.trim().toLowerCase();
    const password = $('login-password')?.value;
    try {
      const res = await login(email, password);
      if (res.ok || res.authenticated) { showApp(res.user || me.user); location.reload(); }
      else {
        const el = $('auth-error');
        if (el) { el.textContent = res.error || 'Login failed'; el.classList.remove('d-none'); }
        await loadCsrf();
      }
    } catch {
      const el = $('auth-error');
      if (el) { el.textContent = 'Network error'; el.classList.remove('d-none'); }
    }
  });

  const signupForm = $('signup-form');
  on(signupForm, 'submit', async (e) => {
    e.preventDefault();
    const email = $('signup-email')?.value.trim().toLowerCase();
    const password = $('signup-password')?.value;
    try {
      const res = await signup(email, password);
      if (res.ok) {
        const res2 = await login(email, password);
        if (res2.ok || res2.authenticated) { showApp(res2.user || me.user); location.reload(); }
      } else {
        const el = $('auth-error');
        if (el) { el.textContent = res.error || 'Signup failed'; el.classList.remove('d-none'); }
        await loadCsrf();
      }
    } catch {
      const el = $('auth-error');
      if (el) { el.textContent = 'Network error'; el.classList.remove('d-none'); }
    }
  });

  on(document, 'click', (e) => {
    const el = e.target.closest('[data-action="show-signup"], [data-action="show-login"]');
    if (!el) return;
    e.preventDefault();
    if (el.dataset.action === 'show-signup') {
      $('signup-card')?.classList.remove('d-none');
      $('login-card')?.classList.add('d-none');
    } else {
      $('login-card')?.classList.remove('d-none');
      $('signup-card')?.classList.add('d-none');
    }
    const err = $('auth-error'); if (err) { err.classList.add('d-none'); err.textContent = ''; }
  });

  on($('logout-btn'), 'click', async () => {
    try { await logout(); } finally { location.reload(); }
  });
});

// Keep login card visible when showing auth gate
const _origShowAuth = showAuth;
window.showAuth = function() { _origShowAuth(); $('login-card')?.classList.remove('d-none'); $('signup-card')?.classList.add('d-none'); const err=$('auth-error'); if (err){err.classList.add('d-none'); err.textContent='';} };

// ---------------------------------------------
// Agents UI: mode & toggles (kept)
// ---------------------------------------------
document.addEventListener('DOMContentLoaded', () => {
  const convWrap   = $('agentsConversation');
  const chatModeEl = $('chatMode');
  const modeBadge  = $('modeBadge');
  const modeTipTxt = $('modeTipText');
  const msgInput   = $('agentMessage');
  if (!chatModeEl || !convWrap) return;

  // Ensure "Live (Mic)" option exists
  if (![...chatModeEl.options].some(o => o.value === 'live')) {
    chatModeEl.add(new Option('Live (Mic)', 'live'));
  }

  function ensureLiveUI() {
    let bar = $('liveBar');
    if (!bar) {
      bar = document.createElement('div');
      bar.id = 'liveBar';
      bar.className = 'd-flex gap-2 align-items-center mb-2';
      (convWrap || document.body).prepend(bar);
    }
    if (!$('startLiveBtn')) {
      const s = document.createElement('button');
      s.id = 'startLiveBtn'; s.type = 'button';
      s.className = 'btn btn-sm btn-primary'; s.textContent = '▶ Start Live';
      bar.appendChild(s);
    }
    if (!$('stopLiveBtn')) {
      const t = document.createElement('button');
      t.id = 'stopLiveBtn'; t.type = 'button';
      t.className = 'btn btn-sm btn-danger'; t.textContent = '■ Stop'; t.disabled = true;
      bar.appendChild(t);
    }
    if (!$('liveStatus')) {
      const sp = document.createElement('span');
      sp.id = 'liveStatus'; sp.className = 'ms-2 text-muted'; sp.textContent = '';
      bar.appendChild(sp);
    }
    if (!$('liveMessage')) {
      const sp = document.createElement('span');
      sp.id = 'liveMessage'; sp.className = 'ms-2'; sp.textContent = '';
      bar.appendChild(sp);
    }
    if (!$('liveMeter')) {
      const sp = document.createElement('span');
      sp.id = 'liveMeter'; sp.className = 'ms-2 text-muted small'; sp.textContent = '';
      bar.appendChild(sp);
    }

    if (!$('liveSuggestMode')) {
      const sel = document.createElement('select');
      sel.id = 'liveSuggestMode';
      sel.className = 'form-select form-select-sm w-auto';
      sel.innerHTML = `<option value="stream">Suggest during conversation</option>
                       <option value="final">Suggest at the end</option>`;
      bar.appendChild(sel);
    }

    return bar;
  }

  function toggleLiveControls(showIt) { const bar = ensureLiveUI(); bar.style.display = showIt ? '' : 'none'; }

  function applyModeUI(val) {
    const turn = $('turnPane') || convWrap;
    const live = $('livePane');
    if (turn) show(turn);
    if (live) hide(live);

    convWrap.classList.remove('mode-real', 'mode-simulated', 'mode-live');

    if (val === 'simulated') {
      convWrap.classList.add('mode-simulated');
      if (modeBadge)  modeBadge.textContent = 'Simulated';
      if (modeTipTxt) modeTipTxt.textContent = 'Simulated chat: the system can generate patient responses. Please reset conversation before continuing';
      if (msgInput)   msgInput.placeholder = 'Say something to the doctor...';
      toggleLiveControls(false);
    } else if (val === 'live') {
      convWrap.classList.add('mode-live');
      if (modeBadge)  modeBadge.textContent = 'Live (Mic)';
      if (modeTipTxt) modeTipTxt.textContent = 'Speak and get continuous recommendations. Press Stop to end; use Finalize for summary/plan.';
      toggleLiveControls(true);
      if (live) { const turnP = $('turnPane'); if (turnP) hide(turnP); show(live); }
    } else {
      convWrap.classList.add('mode-real');
      if (modeBadge)  modeBadge.textContent = 'Real Actors';
      if (modeTipTxt) modeTipTxt.textContent = 'Turn-based chat: alternate between Clinician and Patient. Please reset conversation before continuing';
      if (msgInput)   msgInput.placeholder = 'Say something to the doctor...';
      toggleLiveControls(false);
    }
  }

  applyModeUI(chatModeEl.value);
  on(chatModeEl, 'change', () => applyModeUI(chatModeEl.value));
});

// ---------------------------------------------
// Role switches + finalize + reset (kept)
// ---------------------------------------------
document.addEventListener("DOMContentLoaded", () => {
  let currentRole = "clinician";
  const roleDisplay = $("currentRoleDisplay");
  on($("roleClinicianBtn"), "click", () => {
    currentRole = "clinician"; if (roleDisplay) roleDisplay.textContent = `(Current: Clinician)`;
  });
  on($("rolePatientBtn"), "click", () => {
    currentRole = "patient"; if (roleDisplay) roleDisplay.textContent = `(Current: Patient)`;
  });

  // Finalize (SSE)
  on($("finalizeBtn"), "click", () => {
    const language = $("languageMode")?.value || "bilingual";
    const transcriptDiv = $("agentChatTranscript");
    const mode = $("chatMode")?.value || "real";
    const es = new EventSource(`/agent_chat_stream?message=${encodeURIComponent('[Finalize]')}&lang=${language}&role=finalize&mode=${mode}`);
    es.onmessage = (event) => {
      const item = JSON.parse(event.data);
      if (item.type === 'question_recommender') return;
      const p = document.createElement('p');
      p.innerHTML = `<strong>${item.role}:</strong><br>${(item.message || '').replaceAll('\n','<br>')}<br>
                     <small class="text-muted">${item.timestamp || ''}</small>`;
      transcriptDiv?.appendChild(p);
      transcriptDiv?.scrollTo({ top: transcriptDiv.scrollHeight, behavior: 'smooth' });
    };
    es.onerror = () => es.close();
  });

  // Reset conversation (+ live plan)
  on($("resetBtn"), "click", async () => {
    try {
      const res1 = await fetch('/reset_conv', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-CSRFToken': (window.CSRF_TOKEN || '') },
        credentials: 'same-origin'
      });
      await fetch('/live/reset_plan', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-CSRFToken': (window.CSRF_TOKEN || '') },
        credentials: 'same-origin'
      });
      const data1 = await res1.json();
      if (data1.ok) {
        $("agentChatTranscript")?.replaceChildren();
        $("chatSuggestedQuestions")?.replaceChildren();
        $("liveTranscript")?.replaceChildren();
        $("liveSuggestedQuestions")?.replaceChildren();
        const badge = $("unasked-badge"); if (badge) badge.textContent = '0';
        MH_TRANSCRIPT = ""; MH_SAFETY = false;
        $("mh-screening")?.replaceChildren();
      }
    } catch (err) { console.error('Reset error:', err); }
  });

  // Turn-based SSE chat send
  on($("agentChatForm"), "submit", (e) => {
    e.preventDefault();
    const messageInput = $("agentMessage");
    const transcriptDiv = $("agentChatTranscript");
    const typingIndicator = $("typingIndicator");
    const message = (messageInput?.value || "").trim();
    const language = $("languageMode")?.value || "bilingual";
    if (!message) { alert("Please enter a message!"); return; }

    typingIndicator && (typingIndicator.style.display = "block");
    const mode = $("chatMode")?.value || "real";
    const es = new EventSource(`/agent_chat_stream?message=${encodeURIComponent(message)}&lang=${language}&role=${encodeURIComponent(currentRole)}&mode=${mode}`);

    es.onmessage = (event) => {
      const item = JSON.parse(event.data);
      if (item.type === "question_recommender") {
        const qContainer = $("chatSuggestedQuestions");
        const li = document.createElement("li");
        li.innerHTML = `<strong>English:</strong> ${item.question?.english || ""}<br>
                        <strong>Swahili:</strong> ${item.question?.swahili || ""}`;
        qContainer?.appendChild(li);
        return;
      }
      const p = document.createElement("p");
      p.innerHTML = `<strong>${item.role}:</strong><br>${(item.message || "").replaceAll("\n", "<br>")}<br>
                    <small class="text-muted">${item.timestamp || ""}</small>`;
      transcriptDiv?.appendChild(p);
      transcriptDiv?.scrollTo({ top: transcriptDiv.scrollHeight, behavior: "smooth" });
      runScreening();

      const role = (item.role || "").toLowerCase().trim();
      if (role === "patient") {
        const msg = (item.message || "").trim();
        if (msg) {
          MH_TRANSCRIPT += (MH_TRANSCRIPT ? "\n" : "") + msg;
          const t = msg.toLowerCase();
          if (t.includes("suicid") || t.includes("kujiua") || t.includes("kill myself")) MH_SAFETY = true;
          debounceScreening();
        }
      }
    };

    es.onerror = () => {
      typingIndicator && (typingIndicator.style.display = "none");
      es.close();
    };
    es.onopen = () => typingIndicator && (typingIndicator.style.display = "block");
    es.addEventListener("message", () => setTimeout(() => typingIndicator && (typingIndicator.style.display = "none"), 500));
    if (messageInput) messageInput.value = "";
  });
});

// ---------------------------------------------
// Voice note (batch transcription) — kept
// ---------------------------------------------
document.addEventListener("DOMContentLoaded", () => {
  let mediaRecorder;
  let audioChunks = [];
  const recordBtn = $("recordAudioBtn");
  const audioElement = $("recordedAudio");

  on(recordBtn, "click", async () => {
    if (!mediaRecorder || mediaRecorder.state === "inactive") {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: { channelCount: 1, sampleRate: 48000, noiseSuppression: true, echoCancellation: true, autoGainControl: true }
        });
        const options = { mimeType: "audio/webm;codecs=opus", audioBitsPerSecond: 128000 };
        mediaRecorder = new MediaRecorder(stream, options);
        mediaRecorder.start(1000);
        audioChunks = [];
        mediaRecorder.ondataavailable = e => { if (e.data?.size) audioChunks.push(e.data); };
        mediaRecorder.onstop = async () => {
          try { stream.getTracks().forEach(t => t.stop()); } catch(_) {}
          const audioBlob = new Blob(audioChunks, { type: "audio/webm" });
          if (audioElement) {
            audioElement.src = URL.createObjectURL(audioBlob);
            audioElement.style.display = "block";
          }
          const formData = new FormData();
          const lang = $("languageMode")?.value || "bilingual";
          formData.append("audio", audioBlob);
          formData.append("lang", lang);
          formData.append("role", "patient");

          recordBtn.textContent = "⌛ Transcribing...";
          try {
            const response = await fetch("/transcribe_audio", { method: "POST", body: formData });
            const data = await response.json();
            if (data.text) {
              const input = $("agentMessage"); if (input) input.value = data.text;
              const form = $("agentChatForm");
              form?.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true }));
            } else {
              alert("Failed to transcribe audio.");
            }
          } catch (err) {
            alert("Transcription error.");
            console.error(err);
          }
          recordBtn.textContent = "🎤 Voice Note";
        };
        recordBtn.textContent = "⏹ Stop Recording";
      } catch (error) {
        console.error(error);
        alert(error?.name === "NotAllowedError" ? "Microphone access denied in the browser." : "Audio capture failed.");
      }
    } else if (mediaRecorder.state === "recording") {
      mediaRecorder.stop();
    }
  });
});

// ---------------------------------------------
// Live (Mic) via WebSocket — kept + tuned
// ---------------------------------------------
(() => {
  const qs = (id) => $(id);

  function wsURL(path) {
    const base = new URL(window.location.origin);
    base.protocol = base.protocol === 'https:' ? 'wss:' : 'ws:';
    return `${base.origin}${path.startsWith('/') ? path : '/' + path}`;
  }

  function setLiveRole(_role) { /* reserved */ }
  window.setLiveRole = window.setLiveRole || setLiveRole;

  let liveMediaStream = null;
  let liveRecorder = null;
  let liveWS = null;
  let liveActive = false;

  async function startLive() {
    if (liveActive) return;
    try {
      liveMediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate: 48000,
          noiseSuppression: true,
          echoCancellation: true,
          autoGainControl: true
        }
      });

      const mime = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
        ? 'audio/webm;codecs=opus'
        : (MediaRecorder.isTypeSupported('audio/webm') ? 'audio/webm' : '');

      liveRecorder = new MediaRecorder(liveMediaStream, { mimeType: mime, audioBitsPerSecond: 128000 });

      const lang = $('languageMode')?.value || 'bilingual';
      liveWS = new WebSocket(wsURL(`/ws/stt?lang=${encodeURIComponent(lang)}`));
      liveWS.binaryType = 'arraybuffer';

      liveWS.onopen = () => {
        qs('liveStatus') && (qs('liveStatus').textContent = 'Connected');
        qs('startLiveBtn')?.setAttribute('disabled','');
        qs('stopLiveBtn')?.removeAttribute('disabled');
        liveActive = true;
        liveRecorder.start(300); // ~300ms chunks
      };

      liveRecorder.ondataavailable = async (e) => {
        if (!e.data || !e.data.size) return;
        if (!liveWS || liveWS.readyState !== WebSocket.OPEN) return;
        try {
          const buf = await e.data.arrayBuffer();
          liveWS.send(buf);
        } catch (err) { console.warn('WS send failed', err); }
      };

      liveRecorder.onerror = (e) => { console.error('Recorder error:', e); stopLive(); };

      liveWS.onmessage = (ev) => {
        let j; try { j = JSON.parse(ev.data); } catch { return; }
        if (j.type === 'meter') {
          const m = qs('liveMeter'); if (m) m.textContent = `in=${j.bytes_in}B pcm=${j.bytes_pcm}B`;
        } else if (j.type === 'partial') {
          const lm = qs('liveMessage'); if (lm) lm.textContent = j.text || '';
        } else if (j.type === 'final') {
          const host = qs('liveTranscript');
          if (host) {
            const p = document.createElement('p');
            p.textContent = j.text || '';
            host.appendChild(p);
            host.scrollTo({ top: host.scrollHeight, behavior: 'smooth' });
          }
          const lm = qs('liveMessage'); if (lm) lm.textContent = '';

          // Forward to agents in live mode for suggestions
          try {
            const language = $('languageMode')?.value || 'bilingual';
            const modeSel = $('liveSuggestMode');
            const suggest = (modeSel ? modeSel.value : 'stream');
            const msg = encodeURIComponent(j.text || '');
            const es = new EventSource(`/agent_chat_stream?message=${msg}&lang=${language}&role=live&mode=live&suggest=${suggest}`);
            window.__LIVE_QSET = window.__LIVE_QSET || new Set();
            es.onmessage = (event) => {
              try {
                const item = JSON.parse(event.data);
                if (item.type === 'question_recommender') {
                  const qContainer = $('liveSuggestedQuestions') || $('chatSuggestedQuestions');
                  if (!qContainer) return;
                  const en = (item.question?.english || '').trim();
                  const sw = (item.question?.swahili || '').trim();
                  const key = (en + '|' + sw).toLowerCase();
                  if (!en || window.__LIVE_QSET.has(key)) return;
                  window.__LIVE_QSET.add(key);
                  const li = document.createElement('li');
                  li.innerHTML = `<strong>English:</strong> ${en}${sw ? `<br><strong>Swahili:</strong> ${sw}` : ''}`;
                  qContainer.appendChild(li);
                }
              } catch(_) {}
            };
            es.onerror = () => es.close();
          } catch (e) { console.warn('live SSE trigger failed', e); }

          // Accumulate for screening
          if (j.text) {
            window.MH_TRANSCRIPT = (window.MH_TRANSCRIPT ? window.MH_TRANSCRIPT + ' ' : '') + j.text;
            window.debounceScreening && window.debounceScreening();
          }
          window.dispatchEvent(new CustomEvent('live:final', { detail: { text: j.text || '' } }));

          // De-duplicate adjacent identical finals
          (function dedupeLive() {
            const host2 = $('liveTranscript');
            if (!host2) return;
            const nodes = host2.querySelectorAll('p');
            if (nodes.length < 2) return;
            const a = nodes[nodes.length-1];
            const b = nodes[nodes.length-2];
            if (a.textContent && b.textContent && a.textContent.trim() === b.textContent.trim()) {
              b.remove();
            }
          })();

        } else if (j.type === 'error') {
          const s = qs('liveStatus'); if (s) s.textContent = `Error: ${j.message || 'unknown'}`;
        }
      };

      liveWS.onerror = () => { const s = qs('liveStatus'); if (s) s.textContent = 'WS error'; stopLive(); };
      liveWS.onclose  = () => { const s = qs('liveStatus'); if (s) s.textContent = 'WS closed'; stopLive(); };

      // Clean up on tab hide
      document.addEventListener('visibilitychange', () => { if (document.hidden && liveActive) stopLive(); }, { once: true });
    } catch (err) {
      console.error('Live start error', err);
      const s = qs('liveStatus'); if (s) s.textContent = (err?.name === 'NotAllowedError') ? 'Mic denied' : 'Mic/WS failed';
      stopLive();
    }
  }

  function stopLive() {
    try {
      if (liveRecorder && liveRecorder.state === 'recording') liveRecorder.stop();
      if (liveMediaStream) liveMediaStream.getTracks().forEach(t => t.stop());
      if (liveWS && liveWS.readyState === WebSocket.OPEN) liveWS.close(1000, 'stop');
    } catch (e) {
      console.warn('stopLive error', e);
    } finally {
      liveRecorder = null; liveMediaStream = null; liveWS = null; liveActive = false;
      const s = qs('liveStatus'); if (s) s.textContent = 'Stopped';
      qs('startLiveBtn')?.removeAttribute('disabled');
      qs('stopLiveBtn')?.setAttribute('disabled','');
    }
  }

  // Buttons & keyboard
  window.addEventListener('DOMContentLoaded', () => {
    document.addEventListener('keydown', (e) => {
      const bar = $('liveBar');
      if (!bar || bar.style.display === 'none') return;
      if (e.code === 'Space' && !e.shiftKey) { setLiveRole('patient'); e.preventDefault(); }
      if (e.code === 'Space' && e.shiftKey)  { setLiveRole('clinician'); e.preventDefault(); }
    });
    on($('startLiveBtn'), 'click', (e) => { e.preventDefault(); startLive(); });
    on($('stopLiveBtn'),  'click', (e) => { e.preventDefault(); stopLive();  });
  });

  // Auto-stop if mode switches away from Live
  on($('chatMode'), 'change', (e) => {
    const turn = $('turnPane');
    const live = $('livePane');
    if (turn && live) {
      if (e.target.value === 'live') { hide(turn); show(live); }
      else { show(turn); hide(live); if (liveActive) stopLive(); }
    }
  });

  // Hook to mark asked, update badges, etc.
  window.addEventListener('live:final', (ev) => {
    const { text } = ev.detail || {};
    fetch('/live/mark_asked', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-CSRFToken': (window.CSRF_TOKEN || '') },
      credentials: 'same-origin',
      body: JSON.stringify({ text })
    }).then(() => {
      try { typeof refreshUnaskedBadge === 'function' && refreshUnaskedBadge(); } catch {}
    }).catch(()=>{});
  });
})();

// ---------------------------------------------
// Question Bank wiring — NO auto-load on page open
// ---------------------------------------------
document.addEventListener('DOMContentLoaded', () => {
  const elCat = $('qbCategory');
  const elQ   = $('qbQuery');
  const btnS  = $('qbSearchBtn');
  const btnP  = $('qbPrintBtn');
  const btnE  = $('qbExportBtn');
  const host  = $('qbResults');

  if (!host || (!btnS && !btnP && !btnE)) return;

  function setActionsEnabled(enabled) {
    if (btnP) btnP.disabled = !enabled;
    if (btnE) btnE.disabled = !enabled;
  }
  function hasResults() {
    return !!host.querySelector('.border-bottom');
  }
  function renderItems(items) {
    if (!items || !items.length) {
      host.innerHTML = '<div class="text-muted">No questions found.</div>';
      setActionsEnabled(false);
      return;
    }
    const html = items.map(it => `
      <div class="border-bottom py-2">
        <div class="small text-muted">${it.id} · ${it.category || ''}</div>
        <div><strong>English:</strong> ${it.english || ''}</div>
        <div><strong>Swahili:</strong> ${it.swahili || ''}</div>
      </div>`).join('');
    host.innerHTML = html;
    setActionsEnabled(true);
  }

  async function doSearch() {
    host.innerHTML = '<div class="text-muted">Searching…</div>';
    setActionsEnabled(false);
    const cat = elCat?.value || '';
    const q   = (elQ?.value || '').trim();
    const body = { query: q || (cat ? cat : ''), category: cat || null, k: 50 };
    try {
      const r = await fetch('/questions/search', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      });
      const j = await r.json();
      if (j.error) {
        host.innerHTML = `<div class="text-danger">Search failed: ${j.error}</div>`;
        return;
      }
      renderItems(j.items || []);
    } catch (e) {
      host.innerHTML = `<div class="text-danger">Search failed.</div>`;
    }
  }

  on(btnS, 'click', (e) => { e.preventDefault(); doSearch(); });

  on(btnP, 'click', (e) => {
    e.preventDefault();
    if (!hasResults()) return; // only after search results
    const cat = elCat?.value || '';
    const url = `/questions/print${cat ? ('?category=' + encodeURIComponent(cat)) : ''}`;
    window.open(url, '_blank');
  });

  on(btnE, 'click', (e) => {
    e.preventDefault();
    if (!hasResults()) return; // only after search results
    const cat = elCat?.value || '';
    const q   = (elQ?.value || '').trim();
    const params = new URLSearchParams();
    if (cat) params.set('category', cat);
    if (q) params.set('q', q);
    window.location.href = `/questions/export?${params.toString()}`;
  });

  // Initial state: instruct user to search (no auto-load)
  host.innerHTML = '<div class="text-muted">Search to see questions.</div>';
  setActionsEnabled(false);
});