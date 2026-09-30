/**
 * Small, dependency-free chart primitives (inline SVG / HTML).
 *
 * Rules: honest axes (zero line when the domain crosses zero, labelled
 * ticks), no chartjunk, and "—" instead of a mark when a value is missing.
 * Measured charts re-render on container resize so text never scales.
 */

import { DASH, esc, fmtDay, isNum, pct, price } from './util.js';

const NS = 'http://www.w3.org/2000/svg';

/* ── scales ────────────────────────────────────────────────────────────── */
export function linear(d0, d1, r0, r1) {
  const span = d1 - d0 || 1;
  const k = (r1 - r0) / span;
  return (v) => r0 + (v - d0) * k;
}

/** "Nice" tick values covering [min, max]. */
export function niceTicks(min, max, count = 4) {
  if (!isNum(min) || !isNum(max)) return [];
  if (min === max) {
    const pad = Math.abs(min) * 0.1 || 1;
    min -= pad;
    max += pad;
  }
  const raw = (max - min) / Math.max(1, count);
  const mag = 10 ** Math.floor(Math.log10(raw));
  const norm = raw / mag;
  const step = (norm >= 5 ? 10 : norm >= 2 ? 5 : norm >= 1 ? 2 : 1) * mag;
  const out = [];
  for (let v = Math.ceil(min / step) * step; v <= max + step * 1e-9; v += step) out.push(+v.toFixed(12));
  return out;
}

/* ── inline marks (HTML strings) ───────────────────────────────────────── */

/** 120-day close sparkline. rows = [[date,o,h,l,c,v], ...]. */
export function sparkline(rows, { w = 104, h = 28 } = {}) {
  const c = (rows || []).map((r) => (Array.isArray(r) ? r[4] : null)).filter(isNum);
  if (c.length < 2) return `<span class="muted" aria-label="No chart data">${DASH}</span>`;
  const lo = Math.min(...c);
  const hi = Math.max(...c);
  const x = linear(0, c.length - 1, 1, w - 3);
  const y = linear(lo, hi, h - 2, 2);
  let d = '';
  c.forEach((v, i) => { d += `${i ? 'L' : 'M'}${x(i).toFixed(1)} ${y(v).toFixed(1)}`; });
  const lx = x(c.length - 1).toFixed(1);
  const ly = y(c[c.length - 1]).toFixed(1);
  const chg = c[c.length - 1] / c[0] - 1;
  return `<svg class="spark" viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" role="img" aria-label="${c.length}-session closes, ${esc(pct(chg, 0, true))} over the window"><path d="${d}" fill="none" stroke="#858e99" stroke-width="1" vector-effect="non-scaling-stroke"/><circle cx="${lx}" cy="${ly}" r="2" fill="#ff2e4d"/></svg>`;
}

/** Probability bar on a 0–max scale with a base-rate tick. */
export function probBar(p, base, max = 1) {
  if (!isNum(p)) return `<span class="pbar" aria-hidden="true"></span>`;
  const w = Math.max(0, Math.min(1, p / max)) * 100;
  const tick = isNum(base) ? `<span class="pbar__base" style="left:${(Math.min(1, base / max) * 100).toFixed(2)}%"></span>` : '';
  return `<span class="pbar" aria-hidden="true"><span class="pbar__fill" style="width:${w.toFixed(2)}%"></span>${tick}</span>`;
}

export const FAMILIES = ['dilution', 'exhaustion', 'decay', 'flow', 'street', 'news'];
export const FAMILY_LABEL = {
  dilution: 'Dilution', exhaustion: 'Exhaustion', decay: 'Decay', flow: 'Flow', street: 'Street', news: 'News',
};
export const FAMILY_HELP = {
  dilution: 'Offerings, ATMs, shelf and resale registrations',
  exhaustion: 'Size of the recent run-up, gap and stretch',
  decay: 'Reverse splits, drawdown, deficiency notices',
  flow: 'Volume, short-sale share, liquidity',
  street: 'Analyst stance and downgrades',
  news: 'Tone of recent headlines',
};

/** Six tiny vertical bars, one per signal family (0–100, null = dashed). */
export function familyBars(fams) {
  const f = fams || {};
  const vals = FAMILIES.map((k) => (isNum(f[k]) ? f[k] : null));
  const top = Math.max(...vals.filter(isNum), -1);
  const label = FAMILIES.map((k, i) => `${FAMILY_LABEL[k]} ${vals[i] == null ? 'n/a' : Math.round(vals[i])}`).join(', ');
  const bars = vals.map((v) => {
    if (v == null) return '<i class="null"></i>';
    const h = Math.max(2, Math.round((v / 100) * 24));
    return `<i class="${v === top ? 'top' : ''}" style="--h:${h}px"></i>`;
  }).join('');
  return `<span class="fam" role="img" aria-label="${esc(label)}">${bars}</span>`;
}

/** Ten-cell squeeze-danger meter (ice-blue only when elevated). */
export function squeezeMeter(v, { large = false } = {}) {
  if (!isNum(v)) return `<span class="sqm"><span class="sqm__val muted">${DASH}</span></span>`;
  const on = Math.round(Math.max(0, Math.min(100, v)) / 10);
  const lvl = v >= 70 ? 'hi' : v >= 45 ? 'mid' : 'lo';
  const word = v >= 70 ? 'high' : v >= 45 ? 'elevated' : 'low';
  let cells = '';
  for (let i = 0; i < 10; i++) cells += `<i class="${i < on ? 'on' : ''}"></i>`;
  return `<span class="sqm sqm--${lvl}${large ? ' sqm--lg' : ''}" role="img" aria-label="Squeeze danger ${Math.round(v)} of 100, ${word}"><span class="sqm__cells" aria-hidden="true">${cells}</span><span class="sqm__val">${Math.round(v)}</span></span>`;
}

/** Severity 1–3 as three short red bars. */
export function severity(n) {
  const k = isNum(n) ? Math.max(0, Math.min(3, Math.round(n))) : 0;
  let s = '';
  for (let i = 0; i < 3; i++) s += `<i class="${i < k ? 'on' : ''}"></i>`;
  return `<span class="sev" role="img" aria-label="Severity ${k} of 3">${s}</span>`;
}

/* ── measured SVG charts ───────────────────────────────────────────────── */

/** Render with the container's real width now and whenever it changes. */
export function responsive(el, render) {
  let lastW = 0;
  const draw = () => {
    const w = Math.round(el.clientWidth);
    if (!w || w === lastW) return;
    lastW = w;
    try {
      render(w);
    } catch (e) {
      console.error('chart render failed', e);
    }
  };
  draw();
  if ('ResizeObserver' in window) {
    let raf = 0;
    const ro = new ResizeObserver(() => {
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(draw);
    });
    ro.observe(el);
    return () => ro.disconnect();
  }
  return () => {};
}

function svgOpen(w, h, label) {
  return `<svg xmlns="${NS}" viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" role="img" aria-label="${esc(label)}">`;
}

function dayTicks(dates, n) {
  if (dates.length < 2) return [];
  const out = [];
  const step = (dates.length - 1) / (n - 1);
  for (let i = 0; i < n; i++) out.push(Math.round(i * step));
  return [...new Set(out)];
}

/**
 * Close-price line chart with labelled axes and an optional vertical marker.
 * rows: [[date,o,h,l,c,v]], marker: {date, label}, ref: {price, label}.
 */
export function priceChart(el, rows, { marker = null, ref = null, height = 300, label = 'Price chart' } = {}) {
  const pts = (rows || []).filter((r) => Array.isArray(r) && isNum(r[4]));
  if (pts.length < 2) {
    el.innerHTML = `<p class="empty">No price history in this feed.</p>`;
    return () => {};
  }
  return responsive(el, (W) => {
    const H = W < 520 ? Math.min(height, 240) : height;
    const m = { t: 16, r: 56, b: 28, l: 8 };
    const closes = pts.map((r) => r[4]);
    let lo = Math.min(...closes);
    let hi = Math.max(...closes);
    if (ref && isNum(ref.price)) { lo = Math.min(lo, ref.price); hi = Math.max(hi, ref.price); }
    const pad = (hi - lo) * 0.08 || hi * 0.1 || 1;
    const x = linear(0, pts.length - 1, m.l, W - m.r);
    const y = linear(lo - pad, hi + pad, H - m.b, m.t);
    const ticks = niceTicks(lo, hi, 4);
    let s = svgOpen(W, H, label);
    for (const t of ticks) {
      const yy = y(t).toFixed(1);
      s += `<line class="grid" x1="${m.l}" x2="${W - m.r}" y1="${yy}" y2="${yy}"/><text x="${W - m.r + 8}" y="${+yy + 3}">${esc(price(t))}</text>`;
    }
    let d = '';
    pts.forEach((r, i) => { d += `${i ? 'L' : 'M'}${x(i).toFixed(1)} ${y(r[4]).toFixed(1)}`; });
    const area = `${d}L${x(pts.length - 1).toFixed(1)} ${H - m.b}L${x(0).toFixed(1)} ${H - m.b}Z`;
    s += `<path class="a-px" d="${area}"/><path class="l-px" d="${d}"/>`;
    s += `<line class="ax" x1="${m.l}" x2="${W - m.r}" y1="${H - m.b}" y2="${H - m.b}"/>`;
    for (const i of dayTicks(pts, W < 520 ? 3 : 5)) {
      const anchor = i === 0 ? 'start' : i === pts.length - 1 ? 'end' : 'middle';
      s += `<text x="${x(i).toFixed(1)}" y="${H - 8}" text-anchor="${anchor}">${esc(fmtDay(pts[i][0], { weekday: false }))}</text>`;
    }
    if (ref && isNum(ref.price)) {
      const yy = y(ref.price).toFixed(1);
      s += `<line class="mk" x1="${m.l}" x2="${W - m.r}" y1="${yy}" y2="${yy}" stroke-dasharray="2 4" opacity=".6"/>`;
    }
    if (marker && marker.date) {
      let idx = -1;
      for (let i = 0; i < pts.length; i++) if (pts[i][0] <= marker.date) idx = i;
      if (idx >= 0) {
        const xx = x(idx);
        const yy = y(pts[idx][4]);
        const anchor = xx > W * 0.7 ? 'end' : 'start';
        const tx = anchor === 'end' ? xx - 8 : xx + 8;
        s += `<line class="mk" x1="${xx.toFixed(1)}" x2="${xx.toFixed(1)}" y1="${m.t}" y2="${H - m.b}"/>`;
        s += `<circle class="dot-red" cx="${xx.toFixed(1)}" cy="${yy.toFixed(1)}" r="4"/>`;
        s += `<text class="lbl-red" x="${tx.toFixed(1)}" y="${m.t + 10}" text-anchor="${anchor}">${esc(marker.label || '')}</text>`;
      }
    }
    const last = pts[pts.length - 1];
    s += `<circle class="dot" cx="${x(pts.length - 1).toFixed(1)}" cy="${y(last[4]).toFixed(1)}" r="3"/>`;
    s += '</svg>';
    el.innerHTML = s;
  });
}

/**
 * Cumulative P&L of "short the #1 at the open, cover at the close".
 * series: [{name, cls, values: number[]}] (values in fraction units), dates: string[].
 */
export function equityChart(el, dates, series, { label = 'Backtest equity curve' } = {}) {
  if (!dates.length) {
    el.innerHTML = '<p class="empty">No backtest days in this feed.</p>';
    return () => {};
  }
  return responsive(el, (W) => {
    const H = W < 520 ? 220 : 280;
    const m = { t: 16, r: 64, b: 28, l: 8 };
    const all = series.flatMap((s) => s.values).filter(isNum);
    const lo = Math.min(0, ...all);
    const hi = Math.max(0, ...all);
    const pad = (hi - lo) * 0.06 || 0.01;
    const x = linear(0, dates.length - 1, m.l, W - m.r);
    const y = linear(lo - pad, hi + pad, H - m.b, m.t);
    let s = svgOpen(W, H, label);
    for (const t of niceTicks(lo, hi, 4)) {
      const yy = y(t).toFixed(1);
      s += `<line class="${t === 0 ? 'zero' : 'grid'}" x1="${m.l}" x2="${W - m.r}" y1="${yy}" y2="${yy}"/>`;
      s += `<text x="${W - m.r + 8}" y="${+yy + 3}">${esc(pct(t, 0, true))}</text>`;
    }
    if (!(lo <= 0 && hi >= 0)) {
      const yy = y(0).toFixed(1);
      s += `<line class="zero" x1="${m.l}" x2="${W - m.r}" y1="${yy}" y2="${yy}"/>`;
    }
    for (const se of series) {
      let d = '';
      se.values.forEach((v, i) => { if (isNum(v)) d += `${d ? 'L' : 'M'}${x(i).toFixed(1)} ${y(v).toFixed(1)}`; });
      s += `<path class="${se.cls}" d="${d}"/>`;
      const lv = se.values[se.values.length - 1];
      if (isNum(lv)) s += `<circle class="${se.cls === 'l-net' ? 'dot' : 'dot'}" cx="${x(dates.length - 1).toFixed(1)}" cy="${y(lv).toFixed(1)}" r="${se.cls === 'l-net' ? 3 : 2}" opacity="${se.cls === 'l-net' ? 1 : 0.6}"/>`;
    }
    for (const i of dayTicks(dates, W < 520 ? 3 : 5)) {
      const anchor = i === 0 ? 'start' : i === dates.length - 1 ? 'end' : 'middle';
      s += `<text x="${x(i).toFixed(1)}" y="${H - 8}" text-anchor="${anchor}">${esc(fmtDay(dates[i], { weekday: false, year: i === 0 || i === dates.length - 1 }))}</text>`;
    }
    s += '</svg>';
    el.innerHTML = s;
  });
}

/** Predicted vs actual dump rate per decile, with the y = x diagonal. */
export function calibrationChart(el, bins, { label = 'Calibration: predicted vs actual dump rate' } = {}) {
  const pts = (bins || []).filter((b) => isNum(b.pred) && isNum(b.actual));
  if (!pts.length) {
    el.innerHTML = '<p class="empty">No calibration table in this feed.</p>';
    return () => {};
  }
  return responsive(el, (W) => {
    const size = Math.min(W, 420);
    const H = size;
    const m = { t: 12, r: 12, b: 36, l: 44 };
    const mx = Math.max(...pts.map((b) => Math.max(b.pred, b.actual))) * 1.1;
    const top = niceTicks(0, mx, 4);
    const dmax = top[top.length - 1] >= mx ? top[top.length - 1] : mx;
    const x = linear(0, dmax, m.l, size - m.r);
    const y = linear(0, dmax, H - m.b, m.t);
    let s = svgOpen(size, H, label);
    for (const t of niceTicks(0, dmax, 4)) {
      s += `<line class="grid" x1="${m.l}" x2="${size - m.r}" y1="${y(t).toFixed(1)}" y2="${y(t).toFixed(1)}"/>`;
      s += `<text x="${m.l - 6}" y="${(y(t) + 3).toFixed(1)}" text-anchor="end">${esc(pct(t))}</text>`;
      s += `<text x="${x(t).toFixed(1)}" y="${H - m.b + 14}" text-anchor="middle">${esc(pct(t))}</text>`;
    }
    s += `<line class="ax" x1="${m.l}" x2="${m.l}" y1="${m.t}" y2="${H - m.b}"/><line class="ax" x1="${m.l}" x2="${size - m.r}" y1="${H - m.b}" y2="${H - m.b}"/>`;
    s += `<line class="diag" x1="${x(0)}" y1="${y(0)}" x2="${x(dmax).toFixed(1)}" y2="${y(dmax).toFixed(1)}"/>`;
    let d = '';
    pts.forEach((b, i) => { d += `${i ? 'L' : 'M'}${x(b.pred).toFixed(1)} ${y(b.actual).toFixed(1)}`; });
    s += `<path d="${d}" fill="none" stroke="#858e99" stroke-width="1"/>`;
    for (const b of pts) {
      s += `<circle class="dot" cx="${x(b.pred).toFixed(1)}" cy="${y(b.actual).toFixed(1)}" r="3.5"><title>Decile ${esc(b.bin)}: predicted ${esc(pct(b.pred, 1))}, actual ${esc(pct(b.actual, 1))}${isNum(b.n) ? `, n = ${b.n.toLocaleString('en-US')}` : ''}</title></circle>`;
    }
    s += `<text x="${((m.l + size - m.r) / 2).toFixed(1)}" y="${H - 4}" text-anchor="middle">Predicted</text>`;
    s += `<text x="12" y="${((m.t + H - m.b) / 2).toFixed(1)}" text-anchor="middle" transform="rotate(-90 12 ${((m.t + H - m.b) / 2).toFixed(1)})">Actual</text>`;
    s += '</svg>';
    el.innerHTML = s;
  });
}

/** CI whisker for one study on a shared 0–max scale, with the base rate. */
export function whisker(p, ci, base, max) {
  const W = 300;
  const H = 36;
  const x = linear(0, max, 4, W - 4);
  let s = `<svg class="whisk" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" aria-hidden="true">`;
  s += `<line class="rail" x1="4" x2="${W - 4}" y1="18" y2="18"/>`;
  if (isNum(base)) s += `<line class="base" x1="${x(base).toFixed(1)}" x2="${x(base).toFixed(1)}" y1="4" y2="32"/>`;
  if (Array.isArray(ci) && isNum(ci[0]) && isNum(ci[1])) {
    s += `<line class="ci" x1="${x(ci[0]).toFixed(1)}" x2="${x(ci[1]).toFixed(1)}" y1="18" y2="18"/>`;
    s += `<line class="cap" x1="${x(ci[0]).toFixed(1)}" x2="${x(ci[0]).toFixed(1)}" y1="12" y2="24"/>`;
    s += `<line class="cap" x1="${x(ci[1]).toFixed(1)}" x2="${x(ci[1]).toFixed(1)}" y1="12" y2="24"/>`;
  }
  if (isNum(p)) s += `<line class="pt" x1="${x(p).toFixed(1)}" x2="${x(p).toFixed(1)}" y1="8" y2="28"/>`;
  s += '</svg>';
  return s;
}
