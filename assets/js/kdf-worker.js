/* Argon2id in a worker (keeps the page responsive; the worker is terminated after one use so its
   128 MiB of WebAssembly memory is released). hash-wasm is served from this site (assets/vendor). */
/* global hashwasm */
importScripts('../vendor/argon2.umd.min.js');

self.onmessage = async (ev) => {
  const { pw, salt, kdf } = ev.data || {};
  try {
    if (!kdf || kdf.alg !== 'argon2id' || kdf.version !== 19 || kdf.m < 65536 || kdf.t < 3 || kdf.len !== 32) {
      throw new Error('kdf parameters refused');
    }
    const key = await hashwasm.argon2id({
      password: String(pw).normalize('NFC'),
      salt: new Uint8Array(salt),
      iterations: kdf.t,
      memorySize: kdf.m,
      parallelism: kdf.p,
      hashLength: kdf.len,
      outputType: 'binary',
    });
    const out = new Uint8Array(key);            // own buffer, then hand it over (no copy left behind here)
    key.fill(0);
    self.postMessage({ ok: true, key: out }, [out.buffer]);
  } catch (e) {
    self.postMessage({ ok: false, error: e && e.name === 'RangeError' ? 'memory' : String(e && e.message || e) });
  }
};
