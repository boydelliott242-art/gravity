/* GRAVITY site lock (browser side). Every data file is AES-256-GCM encrypted; the data key (DEK) is
   wrapped in lock.json under a key derived from the owner's passphrase with Argon2id. Nothing readable
   is ever fetched before the passphrase unwraps the DEK. See gravity/sitelock.py for the formats. */

const MAGIC = [0x47, 0x52, 0x56, 0x31];                 // "GRV1"
const VERSION = 1;
const HEADER_LEN = 25;
const DB = 'gravity-lock';
const STORE = 'keys';
const enc = new TextEncoder();

let DEK = null;          // non-extractable CryptoKey (decrypt only)
let EPOCH = null;
let INDEX = null;        // {seq, files: {name: {seq, sha256}}}
let INDEX_AT = 0;
let REMEMBERED = false;

const b64 = (s) => Uint8Array.from(atob(s), (c) => c.charCodeAt(0));
const hex = (buf) => Array.from(new Uint8Array(buf), (b) => b.toString(16).padStart(2, '0')).join('');

/* ── IndexedDB (only when the owner ticks "Remember this device") ── */
function idb() {
  return new Promise((resolve, reject) => {
    const r = indexedDB.open(DB, 1);
    r.onupgradeneeded = () => r.result.createObjectStore(STORE);
    r.onsuccess = () => resolve(r.result);
    r.onerror = () => reject(r.error);
  });
}
async function idbGet(k) {
  try {
    const db = await idb();
    const v = await new Promise((res) => { const q = db.transaction(STORE).objectStore(STORE).get(k); q.onsuccess = () => res(q.result); q.onerror = () => res(null); });
    db.close();
    return v;
  } catch { return null; }
}
async function idbPut(k, v) {
  try {
    const db = await idb();
    await new Promise((res) => { const t = db.transaction(STORE, 'readwrite'); t.objectStore(STORE).put(v, k); t.oncomplete = res; t.onerror = res; });
    db.close();
  } catch { /* storage blocked: stay session-only */ }
}
export async function forgetDevice() {
  try { await new Promise((res) => { const r = indexedDB.deleteDatabase(DB); r.onsuccess = res; r.onerror = res; r.onblocked = res; }); } catch { /* nothing stored */ }
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
export const isRemembered = () => REMEMBERED;

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
        <input id="lock-pw" type="password" autocomplete="current-password" autocapitalize="off" autocorrect="off" spellcheck="false" placeholder="Passphrase">
        <label class="lockscreen__remember"><input id="lock-remember" type="checkbox"> Remember this device</label>
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

/** Resolve once the site is unlocked (prompting if no remembered key fits the current epoch). */
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
  const known = indexedDB.databases ? await indexedDB.databases().catch(() => [{ name: DB }]) : [{ name: DB }];
  const saved = lock && known.some((d) => d.name === DB) ? await idbGet('dek') : null;
  if (saved && saved.epoch === lock.epoch && saved.commit === lock.commit && saved.key) {
    DEK = saved.key; EPOCH = lock.epoch; REMEMBERED = true;
    try { await loadIndex(true); return; } catch { DEK = null; REMEMBERED = false; await forgetDevice(); }
  }
  const el = screen();
  document.body.appendChild(el);
  const pw = el.querySelector('#lock-pw');
  const go = el.querySelector('#lock-go');
  const msg = el.querySelector('#lock-msg');
  const rem = el.querySelector('#lock-remember');
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
        if (rem.checked) { await idbPut('dek', { epoch: lock.epoch, commit: lock.commit, key }); REMEMBERED = true; }
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

/** Lock now: forget this device and drop everything decrypted. */
export async function lockNow() {
  DEK = null; INDEX = null;
  await forgetDevice();
  location.reload();
}

/* Not remembered: lock again after 30 minutes in the background (the decrypted page is dropped). */
let hiddenAt = 0;
document.addEventListener('visibilitychange', () => {
  if (document.hidden) { hiddenAt = Date.now(); return; }
  if (DEK && !REMEMBERED && hiddenAt && Date.now() - hiddenAt > 30 * 60e3) location.reload();
});
