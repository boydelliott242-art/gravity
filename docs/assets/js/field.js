/**
 * The hero signature — "What goes up."
 *
 * The #1 pick's last ~120 daily closes are drawn as a dense particle field
 * (a tight band on the price line plus a fading dust "area" beneath it).
 * After a beat — or as soon as the reader scrolls — the particles lose their
 * hold, highest first, and fall under gravity into a heap on the floor.
 *
 * Performance contract: WebGL1 points, CPU physics over typed arrays (one
 * pass per frame, no allocation), DPR capped at 2 and total backing store
 * capped at ~6 MP, the loop stops when nothing moves, and it pauses when the
 * canvas is off-screen or the tab is hidden. Reduced-motion and no-WebGL get
 * a calm static render of the same field (Canvas2D, drawn once).
 */

import { prefersReducedMotion } from './util.js';

const PLATINUM = [0.933, 0.945, 0.957];
const RED = [1.0, 0.18, 0.302];
const POINT = 1.35; // CSS px

const VS = `
attribute vec3 a_p;
attribute vec4 a_c;
uniform vec2 u_res;
uniform float u_size;
varying vec4 v_c;
void main() {
  vec2 c = (a_p.xy / u_res) * 2.0 - 1.0;
  gl_Position = vec4(c.x, -c.y, 0.0, 1.0);
  gl_PointSize = u_size;
  v_c = vec4(a_c.rgb, a_c.a * a_p.z);
}`;
const FS = `
precision mediump float;
varying vec4 v_c;
void main() { gl_FragColor = vec4(v_c.rgb * v_c.a, v_c.a); }`;

const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);
const easeOutCubic = (t) => 1 - (1 - t) ** 3;
const easeInOutCubic = (t) => (t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2);
const easeInQuad = (t) => t * t;

function gauss() {
  let u = 0;
  let v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function initGL(canvas) {
  let gl = null;
  try {
    gl = canvas.getContext('webgl', {
      alpha: true, premultipliedAlpha: true, antialias: false, depth: false, stencil: false,
      preserveDrawingBuffer: false, powerPreference: 'default',
    });
  } catch {
    gl = null;
  }
  if (!gl) return null;
  const sh = (type, src) => {
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    return gl.getShaderParameter(s, gl.COMPILE_STATUS) ? s : null;
  };
  const vs = sh(gl.VERTEX_SHADER, VS);
  const fs = sh(gl.FRAGMENT_SHADER, FS);
  if (!vs || !fs) return null;
  const prog = gl.createProgram();
  gl.attachShader(prog, vs);
  gl.attachShader(prog, fs);
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) return null;
  gl.useProgram(prog);
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
  gl.clearColor(0, 0, 0, 0);
  return {
    gl,
    prog,
    aP: gl.getAttribLocation(prog, 'a_p'),
    aC: gl.getAttribLocation(prog, 'a_c'),
    uRes: gl.getUniformLocation(prog, 'u_res'),
    uSize: gl.getUniformLocation(prog, 'u_size'),
    bufP: gl.createBuffer(),
    bufC: gl.createBuffer(),
  };
}

/**
 * Mount the field on ``canvas`` for a series of closes (nulls allowed).
 * Returns a controller {mode, replay(), release(), destroy()} or null when
 * the series is too short to draw.
 */
export function mountField(canvas, series, opts = {}) {
  const n = series.length;
  const pts = [];
  for (let i = 0; i < n; i++) if (Number.isFinite(series[i]) && series[i] > 0) pts.push([i, series[i]]);
  if (pts.length < 4) return null;

  const redTail = opts.redTail ?? 5;
  const reduced = opts.reduced ?? prefersReducedMotion();
  let G = reduced ? null : initGL(canvas);
  let mode = G ? 'webgl' : 'static';

  // geometry + particles
  let W = 0;
  let H = 0;
  let dpr = 1;
  let N = 0;
  let hx; let hy; let px; let py; let vx; let vy; let al; let sx; let sy; let dl; let gm; let rel; let st; let isRed;
  let posBuf; let colBuf;
  let heights; let CW = 2; let inc = 1; let cols = 0;
  let geom = null;

  // time + phase
  let phase = 'intro';
  let t = 0;
  let t0 = 0;
  let settled = 0;
  let scrollArmed = false;
  let running = false;
  let visible = true;
  let raf = 0;
  let last = 0;
  let destroyed = false;
  let dirty = true;

  const INTRO = 1.0;
  const INTRO_SPREAD = 1.1;
  const HOLD = 1.7;
  const SWEEP = 1.6;
  const RETURN = 1.25;

  function setPhase(p) {
    phase = p;
    t0 = t;
    if (opts.onPhase) opts.onPhase(p);
  }

  function layout() {
    const r = canvas.getBoundingClientRect();
    W = Math.max(1, Math.round(r.width));
    H = Math.max(1, Math.round(r.height));
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    if (W * H * dpr * dpr > 6e6) dpr = Math.max(1, Math.sqrt(6e6 / (W * H)));
    canvas.width = Math.round(W * dpr);
    canvas.height = Math.round(H * dpr);

    const mobile = W < 700;
    const top = H * (mobile ? 0.17 : 0.15);
    const bottom = H * (mobile ? 0.45 : 0.56);
    const left = Math.max(16, W * 0.04);
    const right = W - left;
    let lo = Infinity;
    let hi = -Infinity;
    let loI = 0;
    let hiI = 0;
    for (const [i, v] of pts) {
      if (v < lo) { lo = v; loI = i; }
      if (v > hi) { hi = v; hiI = i; }
    }
    const span = hi - lo || hi * 0.1 || 1;
    const xOf = (i) => left + ((right - left) * i) / Math.max(1, n - 1);
    const yOf = (v) => bottom - ((bottom - top) * (v - lo)) / span;
    geom = { W, H, top, bottom, left, right, xOf, yOf, lo, hi, loI, hiI };
    if (opts.onLayout) opts.onLayout(geom);
    build();
  }

  function build() {
    const { top, bottom, left, right, xOf, yOf } = geom;
    const target = Math.round(clamp((W * H) / 55, 4500, 18000));
    if (target !== N) {
      N = target;
      hx = new Float32Array(N); hy = new Float32Array(N); px = new Float32Array(N); py = new Float32Array(N);
      vx = new Float32Array(N); vy = new Float32Array(N); al = new Float32Array(N); sx = new Float32Array(N);
      sy = new Float32Array(N); dl = new Float32Array(N); gm = new Float32Array(N); rel = new Float32Array(N);
      st = new Uint8Array(N); isRed = new Uint8Array(N);
      posBuf = new Float32Array(N * 3);
      colBuf = new Float32Array(N * 4);
    }
    // polyline + arc length
    const P = pts.map(([i, v]) => [xOf(i), yOf(v)]);
    const L = new Float32Array(P.length);
    for (let k = 1; k < P.length; k++) L[k] = L[k - 1] + Math.hypot(P[k][0] - P[k - 1][0], P[k][1] - P[k - 1][1]);
    const total = L[L.length - 1] || 1;
    // curve y per pixel column (for the dust area)
    const ys = new Float32Array(Math.ceil(W) + 2).fill(NaN);
    for (let k = 1; k < P.length; k++) {
      const [x0, y0] = P[k - 1];
      const [x1, y1] = P[k];
      for (let xx = Math.floor(x0); xx <= Math.ceil(x1); xx++) {
        if (xx < 0 || xx >= ys.length) continue;
        const f = x1 === x0 ? 0 : clamp((xx - x0) / (x1 - x0), 0, 1);
        ys[xx] = y0 + (y1 - y0) * f;
      }
    }
    const redX = redTail > 0 ? xOf(Math.max(0, n - 1 - redTail)) : Infinity;
    const nLine = Math.round(N * 0.68);
    const lambda = (bottom - top) * 0.2 + 8;
    const floorLimit = H * 0.97;
    for (let i = 0; i < N; i++) {
      let x;
      let y;
      let a;
      if (i < nLine) {
        const s = Math.random() * total;
        let lo = 1;
        let hi = L.length - 1;
        while (lo < hi) { const mid = (lo + hi) >> 1; if (L[mid] < s) lo = mid + 1; else hi = mid; }
        const k = lo;
        const seg = L[k] - L[k - 1] || 1;
        const f = (s - L[k - 1]) / seg;
        const [x0, y0] = P[k - 1];
        const [x1, y1] = P[k];
        const dx = x1 - x0;
        const dy = y1 - y0;
        const len = Math.hypot(dx, dy) || 1;
        const off = gauss() * 1.05;
        x = x0 + dx * f + (-dy / len) * off;
        y = y0 + dy * f + (dx / len) * off;
        a = 0.5 + Math.random() * 0.45;
      } else {
        x = left + Math.random() * (right - left);
        const yc = ys[Math.round(x)];
        const base = Number.isFinite(yc) ? yc : bottom;
        const depth = -Math.log(1 - Math.random()) * lambda;
        y = Math.min(base + 2 + depth, floorLimit);
        a = 0.05 + 0.3 * Math.exp(-depth / lambda);
      }
      hx[i] = x;
      hy[i] = y;
      al[i] = a;
      isRed[i] = x >= redX ? 1 : 0;
      gm[i] = 0.8 + Math.random() * 0.45;
      rel[i] = y + (Math.random() - 0.5) * 36;
      dl[i] = ((x - left) / Math.max(1, right - left)) * INTRO_SPREAD * 0.8 + Math.random() * INTRO_SPREAD * 0.2;
      const c = isRed[i] ? RED : PLATINUM;
      colBuf[i * 4] = c[0];
      colBuf[i * 4 + 1] = c[1];
      colBuf[i * 4 + 2] = c[2];
      colBuf[i * 4 + 3] = a;
    }
    CW = 2;
    cols = Math.ceil(W / CW) + 1;
    heights = new Float32Array(cols);
    inc = clamp(((H * 0.055) * cols) / N, 0.25, 2.2);
    if (G) {
      const { gl, bufC, aC } = G;
      gl.bindBuffer(gl.ARRAY_BUFFER, bufC);
      gl.bufferData(gl.ARRAY_BUFFER, colBuf, gl.STATIC_DRAW);
      gl.enableVertexAttribArray(aC);
      gl.vertexAttribPointer(aC, 4, gl.FLOAT, false, 0, 0);
      gl.bindBuffer(gl.ARRAY_BUFFER, G.bufP);
      gl.bufferData(gl.ARRAY_BUFFER, posBuf.byteLength, gl.DYNAMIC_DRAW);
      gl.enableVertexAttribArray(G.aP);
      gl.vertexAttribPointer(G.aP, 3, gl.FLOAT, false, 0, 0);
      gl.viewport(0, 0, canvas.width, canvas.height);
    }
  }

  function startIntro() {
    for (let i = 0; i < N; i++) {
      sx[i] = hx[i] + gauss() * 22;
      sy[i] = hy[i] + gauss() * 22 - 10;
      px[i] = sx[i];
      py[i] = sy[i];
      st[i] = 0;
      posBuf[i * 3 + 2] = 0;
    }
    heights.fill(0);
    settled = 0;
    setPhase('intro');
  }

  function placeHome() {
    for (let i = 0; i < N; i++) {
      px[i] = hx[i];
      py[i] = hy[i];
      st[i] = 0;
      posBuf[i * 3] = px[i];
      posBuf[i * 3 + 1] = py[i];
      posBuf[i * 3 + 2] = 1;
    }
    heights.fill(0);
    settled = 0;
  }

  // Drop particle i onto the heap: slide down-hill, then stack.
  function land(i) {
    let c = clamp((px[i] / CW) | 0, 0, cols - 1);
    const T = inc * 2.2;
    for (let k = 0; k < 40; k++) {
      const h = heights[c];
      const l = c > 0 ? heights[c - 1] : Infinity;
      const r = c < cols - 1 ? heights[c + 1] : Infinity;
      let nc = c;
      if (l < h - T && r < h - T) nc = Math.random() < 0.5 ? c - 1 : c + 1;
      else if (l < h - T) nc = c - 1;
      else if (r < h - T) nc = c + 1;
      if (nc === c) break;
      c = nc;
    }
    heights[c] += inc;
    px[i] = c * CW + Math.random() * CW;
    py[i] = H - heights[c] + inc * 0.5;
    st[i] = 2;
    settled++;
  }

  function release() {
    if (phase === 'fall' || phase === 'rest') return;
    if (phase !== 'hold') { scrollArmed = true; return; }
    setPhase('fall');
    kick();
  }

  function settleInstantly() {
    placeHome();
    const order = Array.from({ length: N }, (_, i) => i).sort((a, b) => hy[a] - hy[b]);
    for (const i of order) {
      px[i] = hx[i];
      land(i);
    }
    for (let i = 0; i < N; i++) {
      posBuf[i * 3] = px[i];
      posBuf[i * 3 + 1] = py[i];
      posBuf[i * 3 + 2] = 1;
    }
  }

  function step(dt) {
    t += dt;
    const lt = t - t0;
    if (phase === 'intro' || phase === 'return') {
      const dur = phase === 'intro' ? INTRO : RETURN;
      const ease = phase === 'intro' ? easeOutCubic : easeInOutCubic;
      let done = true;
      for (let i = 0; i < N; i++) {
        const k = clamp((lt - dl[i]) / dur, 0, 1);
        if (k < 1) done = false;
        const e = ease(k);
        px[i] = sx[i] + (hx[i] - sx[i]) * e;
        py[i] = sy[i] + (hy[i] - sy[i]) * e;
        const j = i * 3;
        posBuf[j] = px[i];
        posBuf[j + 1] = py[i];
        posBuf[j + 2] = phase === 'intro' ? k : 1;
      }
      dirty = true;
      if (done) {
        placeHome();
        setPhase('hold');
      }
    } else if (phase === 'hold') {
      if (scrollArmed || lt >= HOLD) {
        scrollArmed = false;
        setPhase('fall');
      }
    } else if (phase === 'fall') {
      const sweep = easeInQuad(clamp(lt / SWEEP, 0, 1));
      const thr = geom.top - 24 + (H + 48 - geom.top) * sweep;
      const g = 1700 * clamp(H / 720, 0.7, 1.25);
      const damp = Math.max(0, 1 - 1.4 * dt);
      for (let i = 0; i < N; i++) {
        const s = st[i];
        if (s === 2) continue;
        if (s === 0) {
          if (rel[i] > thr) continue;
          st[i] = 1;
          vy[i] = -(15 + Math.random() * 55);
          vx[i] = (Math.random() - 0.5) * 60;
        }
        vy[i] += g * gm[i] * dt;
        vx[i] *= damp;
        let x = px[i] + vx[i] * dt;
        if (x < 0) { x = 0; vx[i] = -vx[i] * 0.3; } else if (x > W - 0.5) { x = W - 0.5; vx[i] = -vx[i] * 0.3; }
        px[i] = x;
        py[i] += vy[i] * dt;
        const c = clamp((x / CW) | 0, 0, cols - 1);
        if (py[i] >= H - heights[c]) land(i);
        const j = i * 3;
        posBuf[j] = px[i];
        posBuf[j + 1] = py[i];
      }
      dirty = true;
      if (settled >= N) setPhase('rest');
    }
  }

  function draw() {
    if (!G || !dirty) return;
    const { gl } = G;
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.uniform2f(G.uRes, W, H);
    gl.uniform1f(G.uSize, Math.max(1, POINT * dpr));
    gl.bindBuffer(gl.ARRAY_BUFFER, G.bufP);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, posBuf);
    gl.drawArrays(gl.POINTS, 0, N);
    dirty = false;
  }

  function drawStatic() {
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    const levels = 8;
    const buckets = Array.from({ length: levels * 2 }, () => []);
    for (let i = 0; i < N; i++) {
      const lv = Math.min(levels - 1, Math.floor(al[i] * levels));
      buckets[lv + (isRed[i] ? levels : 0)].push(i);
    }
    const s = POINT;
    buckets.forEach((idx, b) => {
      if (!idx.length) return;
      const red = b >= levels;
      const a = ((b % levels) + 0.5) / levels;
      const c = red ? RED : PLATINUM;
      ctx.fillStyle = `rgba(${Math.round(c[0] * 255)},${Math.round(c[1] * 255)},${Math.round(c[2] * 255)},${a.toFixed(3)})`;
      for (const i of idx) ctx.fillRect(hx[i] - s / 2, hy[i] - s / 2, s, s);
    });
  }

  const animating = () => phase === 'intro' || phase === 'hold' || phase === 'fall' || phase === 'return';

  function frame(now) {
    raf = 0;
    if (destroyed || !running) return;
    const dt = Math.min(0.033, Math.max(0, (now - last) / 1000));
    last = now;
    step(dt);
    draw();
    if (animating()) raf = requestAnimationFrame(frame);
    else running = false;
  }

  function kick() {
    if (mode !== 'webgl' || destroyed || !visible || document.hidden || running) return;
    running = true;
    last = performance.now();
    raf = requestAnimationFrame(frame);
  }

  function stop() {
    running = false;
    if (raf) cancelAnimationFrame(raf);
    raf = 0;
  }

  // ── lifecycle ──
  layout();
  if (mode === 'webgl') {
    startIntro();
    kick();
  } else {
    drawStatic();
    if (opts.onPhase) opts.onPhase('static');
  }

  const io = 'IntersectionObserver' in window
    ? new IntersectionObserver((es) => {
      visible = es.some((e) => e.isIntersecting);
      if (visible) kick(); else stop();
    }, { threshold: 0.02 })
    : null;
  if (io) io.observe(canvas);

  const onVis = () => { if (document.hidden) stop(); else kick(); };
  document.addEventListener('visibilitychange', onVis);

  const onScroll = () => {
    if (window.scrollY > 24 && (phase === 'intro' || phase === 'hold')) {
      release();
      if (phase === 'fall' || scrollArmed) window.removeEventListener('scroll', onScroll);
    }
  };
  if (mode === 'webgl') window.addEventListener('scroll', onScroll, { passive: true });

  let rsz = 0;
  const onResize = () => {
    clearTimeout(rsz);
    rsz = setTimeout(() => {
      const r = canvas.getBoundingClientRect();
      if (Math.abs(r.width - W) < 2 && Math.abs(r.height - H) < 80) return;
      layout();
      if (mode === 'static') { drawStatic(); return; }
      if (phase === 'fall' || phase === 'rest') {
        settleInstantly();
        stop();
        setPhase('rest');
      } else {
        placeHome();
        setPhase('hold');
      }
      dirty = true;
      draw();
      kick();
    }, 160);
  };
  window.addEventListener('resize', onResize);

  const onLost = (e) => {
    e.preventDefault();
    stop();
    // A canvas that had a WebGL context can't give a 2D one: swap in a fresh canvas.
    const fresh = canvas.cloneNode(false);
    canvas.replaceWith(fresh);
    G = null;
    mode = 'static';
    canvas = fresh; // eslint-disable-line no-param-reassign
    layout();
    drawStatic();
    if (opts.onPhase) opts.onPhase('static');
  };
  if (G) canvas.addEventListener('webglcontextlost', onLost, { once: true });

  return {
    get mode() { return mode; },
    get phase() { return phase; },
    release,
    replay() {
      if (mode !== 'webgl') return;
      if (phase !== 'rest' && phase !== 'fall') return;
      for (let i = 0; i < N; i++) {
        sx[i] = px[i];
        sy[i] = py[i];
        st[i] = 0;
        dl[i] = Math.random() * 0.45 + ((hy[i] - geom.top) / Math.max(1, H - geom.top)) * 0.2;
      }
      heights.fill(0);
      settled = 0;
      setPhase('return');
      kick();
    },
    destroy() {
      destroyed = true;
      stop();
      if (io) io.disconnect();
      document.removeEventListener('visibilitychange', onVis);
      window.removeEventListener('scroll', onScroll);
      window.removeEventListener('resize', onResize);
    },
  };
}
