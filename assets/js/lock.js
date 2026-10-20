/* GRAVITY site lock (browser side). Every data file is AES-256-GCM encrypted; the data key (DEK) is
   wrapped in lock.json under a key derived from the owner's passphrase with Argon2id. Nothing readable
   is ever fetched before the passphrase unwraps the DEK. See gravity/sitelock.py for the formats.
   The key lives only in this page's memory: nothing is stored on the device, so every opening of the
   site asks for the passphrase, and so does coming back after more than a minute in the background. */

const MAGIC = [0x47, 0x52, 0x56, 0x31];                 // "GRV1"
const VERSION = 1;
const HEADER_LEN = 25;
const DB = 'gravity-lock';             // the old "remember this device" store — deleted on every load
const AWAY_MS = 60e3;                  // hidden longer than this → locked again
const enc = new TextEncoder();

let DEK = null;          // non-extractable CryptoKey (decrypt only)
let EPOCH = null;
let INDEX = null;        // {seq, files: {name: {seq, sha256}}}
let INDEX_AT = 0;

const b64 = (s) => Uint8Array.from(atob(s), (c) => c.charCodeAt(0));
const hex = (buf) => Array.from(new Uint8Array(buf), (b) => b.toString(16).padStart(2, '0')).join('');

/* ── nothing persists: drop any key an older version of this page saved on the device ── */
export async function forgetDevice() {
  try {
    if (indexedDB.databases) {
      const dbs = await indexedDB.databases();
      if (!dbs.some((d) => d.name === DB)) return;
    }
    await new Promise((res) => { const r = indexedDB.deleteDatabase(DB); r.onsuccess = res; r.onerror = res; r.onblocked = res; });
  } catch { /* storage blocked: nothing stored */ }
}

/* ── key derivation and unwrap ── */
function deriveKEK(pw, lock) {
  return new Promise((resolve, reject) => {
    const w = new Worker(new URL('./kdf-worker.js', import.meta.url));
    w.onmessage = (ev) => { w.terminate(); ev.data && ev.data.ok ? resolve(ev.data.key) : reject(new Error(ev.data ? ev.data.error : 'kdf')); };
    w.onerror = () => { w.terminate(); reject(new Error('worker')); };
    w.postMessage({ pw, salt: b64(lock.kdf.salt).buffer, kdf: { alg: lock.kdf.alg, version: lock.kdf.version, m: lock.kdf.m, t: lock.kdf.t, p: lock.kdf.p, len: lock.kdf.len } });
  });
}

async function unwrapDEK(pw, lock) {
  const kekBytes = await deriveKEK(pw, lock);
  const kek = await crypto.subtle.importKey('raw', kekBytes, 'AES-GCM', false, ['decrypt']);
  kekBytes.fill(0);
  let raw;
  try {
    raw = new Uint8Array(await crypto.subtle.decrypt(
      { name: 'AES-GCM', iv: b64(lock.wrap.iv), additionalData: enc.encode(`gravity|dek-wrap|v1|${lock.epoch}`) },
      kek, b64(lock.wrap.ct)));
  } catch { throw new Error('wrong'); }
  const ep = new Uint8Array(4); new DataView(ep.buffer).setUint32(0, lock.epoch);
  const commit = hex(await crypto.subtle.digest('SHA-256', new Uint8Array([...enc.encode('gravity-dek-commit-v1'), ...ep, ...raw])));
  if (commit !== lock.commit) { raw.fill(0); throw new Error('commit'); }
  const key = await crypto.subtle.importKey('raw', raw, 'AES-GCM', false, ['decrypt']);
  raw.fill(0);
  return key;
}

/* ── envelopes ── */
async function openEnvelope(name, buf) {
  const u = new Uint8Array(buf);
  if (u.length < HEADER_LEN + 16 || MAGIC.some((b, i) => u[i] !== b) || u[4] !== VERSION) throw new Error('format');
  const dv = new DataView(buf);
  if (dv.getUint32(5) !== EPOCH) throw new Error('epoch');
  const header = u.slice(0, HEADER_LEN);
  const aad = new Uint8Array([...header, ...enc.encode(`data/${name}`)]);
  const body = new Uint8Array(await crypto.subtle.decrypt({ name: 'AES-GCM', iv: u.slice(13, 25), additionalData: aad }, DEK, u.slice(HEADER_LEN)));
  const n = new DataView(body.buffer).getUint32(0);
  const gz = body.slice(4, 4 + n);
  const text = await new Response(new Blob([gz]).stream().pipeThrough(new DecompressionStream('gzip'))).text();
  return { text, seq: dv.getUint32(9) };
}

async function fetchBin(path, cache) {
  const r = await fetch(path, { cache, credentials: 'omit' });
  if (!r.ok) { const e = new Error(`HTTP ${r.status}`); e.status = r.status; throw e; }
  return r;
}

async function loadIndex(force) {
  if (INDEX && !force && Date.now() - INDEX_AT < 60e3) return INDEX;
  const r = await fetchBin('data/index.bin', 'no-cache');
  const { text } = await openEnvelope('index.json', await r.arrayBuffer());
  const idx = JSON.parse(text);
  if (INDEX && idx.seq < INDEX.seq) throw new Error('stale index');           // never go backwards
  INDEX = idx; INDEX_AT = Date.now();
  return INDEX;
}

/** Decrypted JSON for a published data file ("today.json", "history/2026-10-20.json"), or throws. */
export async function siteJSON(name, meta = {}) {
  if (!DEK) throw new Error('locked');
  for (let attempt = 0; attempt < 3; attempt += 1) {
    const idx = await loadIndex(attempt > 0);
    const want = idx.files[name];
    if (!want) { const e = new Error('HTTP 404'); e.status = 404; throw e; }
    const r = await fetchBin(`data/${name}.bin`, attempt === 0 ? 'no-cache' : 'reload');
    const buf = await r.arrayBuffer();
    if (hex(await crypto.subtle.digest('SHA-256', buf)) !== want.sha256) continue;   // a stale copy — refetch
    if (meta) meta.lastModified = r.headers.get('last-modified');
    return JSON.parse((await openEnvelope(name, buf)).text);
  }
  throw new Error('stale copy (try again in a minute)');
}

/** Has the site published something newer than what we hold? (cheap: one small encrypted index) */
export async function newerThan(seq) {
  try { const idx = await loadIndex(true); return isFinite(seq) && idx.seq > seq; } catch { return false; }
}
export const currentSeq = () => (INDEX ? INDEX.seq : null);

/* ── the lock screen ── */
function screen() {
  const el = document.createElement('div');
  el.className = 'lockscreen';
  el.setAttribute('role', 'dialog');
  el.setAttribute('aria-modal', 'true');
  el.setAttribute('aria-labelledby', 'lock-title');
  el.innerHTML = `
    <div class="lockscreen__card">
      <p class="lockscreen__mark" aria-hidden="true"></p>
      <h1 class="lockscreen__title" id="lock-title">Gravity</h1>
      <p class="lockscreen__sub">Private. Enter the passphrase to open today's radar.</p>
      <div class="lockscreen__form">
        <label class="sr-only" for="lock-pw">Passphrase</label>
        <input id="lock-pw" type="password" autocomplete="off" autocapitalize="off" autocorrect="off" spellcheck="false" placeholder="Passphrase">
        <button id="lock-go" type="button" class="btn">Unlock</button>
        <p class="lockscreen__msg" id="lock-msg" role="status" aria-live="polite"></p>
      </div>
    </div>`;
  return el;
}

const MSG = {
  wrong: "That's not it.",
  memory: 'This device ran out of memory unlocking — close other tabs and try again.',
  worker: 'This browser blocked the unlock step (WebAssembly). In Lockdown Mode, exclude this site and try again.',
  lock: 'The site lock could not be loaded — check the connection and reload.',
  commit: 'The site key did not verify — reload the page.',
};

/** Resolve once the owner has typed the passphrase (asked on every load — no key is ever kept). */
export async function unlockSite() {
  let lock;
  try {
    const r = await fetch('lock.json', { cache: 'no-cache', credentials: 'omit' });
    if (!r.ok) throw new Error(String(r.status));
    lock = await r.json();
    if (lock.v !== 1 || !lock.kdf || lock.kdf.m < 65536 || lock.kdf.t < 3) throw new Error('lock format');
  } catch {
    lock = null;
  }
  await forgetDevice();
  const el = screen();
  document.body.appendChild(el);
  const pw = el.querySelector('#lock-pw');
  const go = el.querySelector('#lock-go');
  const msg = el.querySelector('#lock-msg');
  if (!lock) { msg.textContent = MSG.lock; go.disabled = true; pw.disabled = true; }
  pw.focus();
  await new Promise((resolve) => {
    const attempt = async (ev) => {
      if (ev) ev.preventDefault();
      if (!lock || !pw.value || go.disabled) return;
      go.disabled = true; pw.disabled = true; msg.textContent = 'Unlocking…';
      try {
        const key = await unwrapDEK(pw.value, lock);
        DEK = key; EPOCH = lock.epoch;
        await loadIndex(true);
        pw.value = '';
        el.remove();
        resolve();
      } catch (e) {
        DEK = null;
        pw.value = '';
        msg.textContent = MSG[e && e.message] || (e && /kdf|memory/i.test(e.message) ? MSG.memory : MSG.wrong);
        go.disabled = false; pw.disabled = false; pw.focus();
      }
    };
    go.addEventListener('click', attempt);
    pw.addEventListener('keydown', (ev) => { if (ev.key === 'Enter') attempt(ev); });
  });
}

/** Lock now: drop the key and everything decrypted. */
export function lockNow() {
  DEK = null; INDEX = null;
  document.documentElement.classList.add('is-locked');
  location.reload();
}

/* Closed, navigated away, or in the background for more than a minute → the passphrase is needed again.
   - pagehide: the key is dropped and the page blanked, so a back/forward-cache restore shows nothing;
     pageshow from that cache reloads straight into the lock screen.
   - hidden > AWAY_MS (another tab or app, phone locked): blanked while away, reloaded on return. */
let hiddenAt = 0;
let awayTimer = 0;
const blank = () => {                    // every path after this reloads into the lock screen
  DEK = null; INDEX = null;
  document.documentElement.classList.add('is-locked');
  try { document.body.replaceChildren(); } catch { /* already gone */ }
};
window.addEventListener('pagehide', blank);
window.addEventListener('pageshow', (ev) => { if (ev.persisted) location.reload(); });
document.addEventListener('visibilitychange', () => {
  if (document.hidden) {
    hiddenAt = Date.now();
    clearTimeout(awayTimer);
    awayTimer = setTimeout(() => { if (document.hidden && DEK) blank(); }, AWAY_MS);
    return;
  }
  clearTimeout(awayTimer);
  if (hiddenAt && (Date.now() - hiddenAt > AWAY_MS || !DEK) && !document.querySelector('.lockscreen')) {
    blank();
    location.reload();
  }
});
