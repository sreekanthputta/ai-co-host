const $ = (id) => document.getElementById(id);
const statePill = $('state-pill');
const latencyEl = $('latency');
const logEl = $('log');
const transcript = $('transcript');
const decisions = $('decisions');

let port = '8765';
let ws = null;

function setState(label) {
  statePill.className = `pill ${label}`;
  statePill.textContent = label;
}

function appendLog(channel, text) {
  logEl.textContent += `[${channel}] ${text}`;
  logEl.scrollTop = logEl.scrollHeight;
}

function appendItem(list, text, cls = '') {
  const li = document.createElement('li');
  if (cls) li.className = cls;
  li.textContent = text;
  list.appendChild(li);
  list.parentElement.scrollTop = list.parentElement.scrollHeight;
  while (list.children.length > 100) list.removeChild(list.firstChild);
}

async function api(path, body) {
  const opts = { method: 'POST', headers: { 'Content-Type': 'application/json' } };
  if (body) opts.body = JSON.stringify(body);
  try {
    const res = await fetch(`http://127.0.0.1:${port}${path}`, opts);
    return await res.json();
  } catch (err) {
    appendLog('api-error', `${path}: ${err.message}\n`);
    return null;
  }
}

function openSocket() {
  if (ws) try { ws.close(); } catch {}
  ws = new WebSocket(`ws://127.0.0.1:${port}/events`);
  ws.onopen = () => appendLog('ws', 'connected\n');
  ws.onclose = () => { appendLog('ws', 'disconnected\n'); setTimeout(openSocket, 2000); };
  ws.onerror = () => {};
  ws.onmessage = (msg) => {
    let evt;
    try { evt = JSON.parse(msg.data); } catch { return; }
    handleEvent(evt);
  };
}

function handleEvent(evt) {
  switch (evt.name) {
    case 'turn.indexed':
      appendItem(transcript, evt.data.text);
      setState('listening');
      break;
    case 'decision.skip':
      appendItem(decisions, `skip · ${evt.data.reason || ''}`, 'skip');
      break;
    case 'decision.llm':
      latencyEl.textContent = `${Math.round(evt.data.latency_ms)}ms`;
      setState('thinking');
      break;
    case 'chime.emitted':
      appendItem(decisions, `chime · ${evt.data.line}`, 'chime');
      setState('speaking');
      setTimeout(() => setState('listening'), 1500);
      break;
    case 'trigger.fired':
      appendItem(decisions, `forced · ${evt.data.hint || '(no hint)'}`, 'chime');
      break;
    case 'session.started':
      setState('listening');
      break;
  }
}

$('start-btn').onclick = async () => {
  await window.riff.start();
  setState('listening');
};
$('stop-btn').onclick = async () => {
  await window.riff.stop();
  setState('idle');
};
$('trigger-btn').onclick = () => api('/trigger', { hint: '' });
$('mute-btn').onclick = async () => {
  await api('/mute', { seconds: 30 });
  setState('muted');
};

window.riff.onLog(({ channel, text }) => appendLog(channel, text));
window.riff.port().then(({ port: p }) => { port = p; openSocket(); });
