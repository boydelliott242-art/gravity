/**
 * GRAVITY — site controller.
 *
 * Reads the static feeds written by the pipeline (CONTRACTS.md §11–14:
 * today, model, evidence, scorecard, plus the optional live tape and the
 * full-universe lookup), renders every section, and wires the board, the
 * lookup table and the dossier drawer.
 *
 * Honesty: every missing value renders as "—"; nothing is imputed, rounded
 * into a different claim, or invented here. Probabilities always sit next to
 * the base rate they should be read against. A section whose field is absent
 * from the feed is hidden or says so — it never fills in.
 */

import {
  DASH, isNum, esc, safeUrl, extLink, pct, pctUnits, price, compact, money, fixed, signedMoney,
  parseDay, fmtDay, parseTs, fmtTimeET, fmtStamp, etDate, ago, daysBetween, marketPhaseNow,
  CAT_LABEL, SUPPLY_CATS, humanize, ASIA, $, $$, prefersReducedMotion, fallbackLinks,
} from './util.js';
import {
  sparkline, probBar, familyBars, squeezeMeter, severity, priceChart, equityChart,
  calibrationChart, whisker, miniLine, intradayChart, sparkCloses, FAMILIES, FAMILY_LABEL, FAMILY_HELP,
  ATTR_FAMILIES, ATTR_LABEL, ATTR_HELP,
} from './charts.js';
import { mountField } from './field.js';

const LWC_URL = 'https://unpkg.com/lightweight-charts@4.2.0/dist/lightweight-charts.standalone.production.js';
const LWC_SRI = 'sha384-OK7vELvjHdhUFi31JYioPIcRHTROLdcDa6ZsNWgvgLaKj+9JqhU0Ad8g4wz3CXjA';
const WIRE_SHOW = 14;
const DAYS_SHOW = 20;
const LOOKUP_PAGE = 50;
const LIVE_POLL_MS = 5 * 60e3;

const S = {
  today: null,
  model: null,
  evidence: null,
  scorecard: null,
  live: null,
  liveAt: 0,
  errors: {},
  base: null,
  sample: false,
  newer: false,
  lastModified: null,
  recs: new Map(),
  sort: { key: 'rank', dir: 1 },
  filters: new Set(),
  tab: 'm0',
  field: null,
  disposers: { record: [], live: [] },
  uni: { status: 'idle', rows: [], bySym: new Map(), asof: null, session: null, fallback: false },
  look: { q: '', sort: 'prob', dir: -1, page: 0, filters: new Set() },
  size: 50000,
  sizeLab: null,
};

/* ── small helpers ─────────────────────────────────────────────────────── */
const arr = (x) => (Array.isArray(x) ? x : []);
const obj = (x) => (x && typeof x === 'object' && !Array.isArray(x) ? x : {});
const txt = (s) => (s == null || s === '' ? DASH : esc(s));
const liftTxt = (x) => (isNum(x) ? `${x.toFixed(x >= 10 ? 0 : 1)}×` : DASH);
const int = (x) => (isNum(x) ? Math.round(x).toLocaleString('en-US') : DASH);
const pad2 = (n) => String(n).padStart(2, '0');
const empty = (html) => `<p class="empty">${html}</p>`;


/** One honest sentence about how the #1 has fared historically (model.json walk-forward). */
function backtestLine(which) {
  const oos = (S.model && S.model.oos) || {};
  const m = oos.m0;  // the morning #1 is chosen on close-only features; M1 only re-scores
  if (!m) return '';
  const hit = isNum(m.pub1_hit) ? m.pub1_hit : m.top1_hit;
  const sqr = isNum(m.pub1_squeeze_rate) ? m.pub1_squeeze_rate : m.top1_squeeze_rate;
  const avg = isNum(m.pub1_mean_oc) ? m.pub1_mean_oc : m.top1_mean_oc;
  if (!isNum(hit)) return '';
  const r10 = obj(obj(obj(obj(S.model).sim).m0).realistic)['10000'];
  const net = r10 && isNum(r10.net_mean) ? ` — after each pick's own estimated cost at $10K that is ${Math.abs(r10.net_mean) < 0.0005 ? 'roughly breakeven' : `${pct(r10.net_mean, 1, true)} per trade`}` : '';
  return ` Backtest of this exact rule (point-in-time universe, no Rule-201 names, ≥$300K daily volume): the #1 fell 5%+ open→close on ${Math.round(hit * 100)}% of sessions${isNum(sqr) ? `, spiked 20%+ above the open on ${Math.round(sqr * 100)}%` : ''}${isNum(avg) ? `, averaging ${pct(avg, 1, true)} open→close before costs` : ''}${net}. A tilt in the odds, never a sure thing.`;
}

function host(u) {
  const s = safeUrl(u);
  if (!s) return null;
  try {
    return new URL(s).hostname.replace(/^www\./, '');
  } catch {
    return null;
  }
}

function setHTML(sel, html) {
  const el = typeof sel === 'string' ? $(sel) : sel;
  if (el) el.innerHTML = html;
  return el;
}

function safe(name, fn) {
  try {
    fn();
  } catch (e) {
    console.error(`[gravity] ${name} failed`, e);
  }
}

function dispose(list) {
  while (list.length) {
    const d = list.pop();
    try {
      if (typeof d === 'function') d();
    } catch { /* already gone */ }
  }
}

const PHASE_LABEL = {
  'pre-market': 'Pre-market', open: 'Market open', 'after-hours': 'After hours', closed: 'Market closed',
};
const STATUS_LABEL = { ETB: 'Easy to borrow', HTB: 'Hard to borrow', NONE: 'No borrow', UNKNOWN: 'Borrow unknown' };
const RED_FLAG = /^(OFFERING|ATM|DILUTION|PUMP|GAP|RUN)\b/;
const SOLID_FLAG = /^(NO BORROW|HALTED)$/;
const STRENGTH = { 1: 'mild', 2: 'moderate', 3: 'strong' };

const isAsia = (p) => arr(p.flags).includes('ASIA') || ASIA.has(p.country);
const sqWord = (v) => (!isNum(v) ? 'unknown' : v >= 70 ? 'high' : v >= 45 ? 'elevated' : 'low');

function lastBar(rows) {
  const r = arr(rows).filter((x) => Array.isArray(x) && isNum(x[4]));
  return r.length ? r[r.length - 1] : null;
}

/** A supply filing (offering / ATM / toxic financing / unregistered sale) in the 24 h before publish. */
function freshSupply(p) {
  const gen = parseTs(S.today && S.today.generated_at) || new Date();
  const cutoff = gen.getTime() - 24 * 3600e3;
  return arr(p.filings).some((e) => {
    if (!e || !SUPPLY_CATS.has(e.category)) return false;
    const t = parseTs(e.accepted);
    if (t) return t.getTime() >= cutoff && t.getTime() <= gen.getTime() + 3600e3;
    return typeof e.date === 'string' && e.date >= etDate(new Date(cutoff));
  });
}

/* ── shared fragments ──────────────────────────────────────────────────── */
function flagChips(flags, max = 99) {
  const f = arr(flags).filter(Boolean);
  if (!f.length) return '<span class="muted">No flags</span>';
  const shown = f.slice(0, max).map((x) => {
    const s = String(x);
    const cls = RED_FLAG.test(s) ? ' chip--red' : SOLID_FLAG.test(s) ? ' chip--solid' : '';
    return `<span class="chip${cls}">${esc(s)}</span>`;
  }).join('');
  const more = f.length > max ? `<span class="chip">+${f.length - max}</span>` : '';
  return `<span class="chips">${shown}${more}</span>`;
}

function borrowText(sh) {
  const s = obj(sh);
  const bits = [];
  if (isNum(s.fee_rate)) bits.push(`${pctUnits(s.fee_rate, s.fee_rate >= 10 ? 0 : 1)}/yr`);
  if (isNum(s.available)) bits.push(`${compact(s.available)} sh`);
  return bits.join(' · ') || (s.status === 'UNKNOWN' ? 'file unavailable' : DASH);
}

function borrowChip(sh) {
  const st = obj(sh).status || 'UNKNOWN';
  const cls = st === 'NONE' ? ' chip--solid' : '';
  return `<span class="chip${cls}" title="${esc(STATUS_LABEL[st] || st)}">${esc(st === 'NONE' ? 'No borrow' : st)}</span>`;
}

function borrowInline(sh) {
  return `<span class="borrow">${borrowChip(sh)}<span class="borrow__txt">${esc(borrowText(sh))}</span></span>`;
}

function reasonsList(reasons) {
  const rs = arr(reasons).filter((r) => r && r.text);
  if (!rs.length) return empty('No itemized reasons in this feed.');
  return `<ol class="reasons">${rs.map((r, i) => {
    const h = host(r.url);
    const fam = FAMILY_LABEL[r.family] || humanize(r.family) || 'Signal';
    const str = STRENGTH[r.strength] ? `<span class="str">${STRENGTH[r.strength]}</span>` : '';
    const src = h ? extLink(r.url, `${h} ↗`, 'src') : '';
    return `<li><span class="n">${pad2(i + 1)}</span><div><span class="fam">${esc(fam)}${str}</span><p>${esc(r.text)}</p>${src}</div></li>`;
  }).join('')}</ol>`;
}

function topReasons(reasons, n = 3) {
  return arr(reasons)
    .map((r, i) => ({ r, i }))
    .sort((a, b) => (b.r.strength || 0) - (a.r.strength || 0) || a.i - b.i)
    .slice(0, n)
    .map((x) => x.r);
}

function linksRow(p, label) {
  const own = obj(p && p.links);
  const entries = Object.keys(own).length ? Object.entries(own) : Object.entries(fallbackLinks(p.symbol));
  const a = entries.map(([k, u]) => (safeUrl(u) ? extLink(u, k) : '')).join('');
  return `<nav class="links" aria-label="${esc(label || `Research links for ${p.symbol}`)}">${a}</nav>`;
}

function filingTable(filings) {
  const fs = arr(filings).filter(Boolean);
  if (!fs.length) return empty('No filings in this feed.');
  const rows = fs.map((e) => {
    const when = parseTs(e.accepted) ? fmtStamp(e.accepted) : fmtDay(e.date, { weekday: false, year: true });
    const cat = CAT_LABEL[e.category] || humanize(e.category) || 'Filing';
    const extra = [];
    if (arr(e.items).length) extra.push(`Items ${arr(e.items).join(', ')}`);
    if (arr(e.text_tags).length) extra.push(arr(e.text_tags).map(humanize).join(', '));
    return `<tr><td>${esc(when)}</td><td class="f">${txt(e.form)}</td><td>${extLink(e.url, cat)}${extra.length ? ` <span class="muted">· ${esc(extra.join(' · '))}</span>` : ''}</td></tr>`;
  }).join('');
  return `<div class="table-scroll"><table class="ftable"><caption class="sr-only">SEC filings, newest first</caption><tbody>${rows}</tbody></table></div>`;
}

function newsList(news) {
  const ns = arr(news).filter((n) => n && n.title);
  if (!ns.length) return empty('No recent headlines in this feed.');
  return `<ul class="lst">${ns.map((n) => {
    const tone = n.polarity < 0 ? 'Bearish' : n.polarity > 0 ? 'Bullish' : '';
    const meta = [fmtStamp(n.published), n.source, tone, arr(n.tags).map(humanize).join(', ')].filter((x) => x && x !== DASH);
    return `<li><span class="meta">${esc(meta.join(' · ') || DASH)}</span>${extLink(n.url, n.title)}</li>`;
  }).join('')}</ul>`;
}

/* ── data loading ──────────────────────────────────────────────────────── */
async function getJSON(name) {
  try {
    const r = await fetch(`data/${name}`, { cache: 'no-cache', credentials: 'omit' });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    if (name === 'today.json') S.lastModified = r.headers.get('last-modified');
    return await r.json();
  } catch (e) {
    S.errors[name] = e && e.message ? e.message : String(e);
    return null;
  }
}

function indexRecords() {
  S.recs.clear();
  const t = S.today || {};
  const put = (p, ctx) => {
    if (p && p.symbol && !S.recs.has(p.symbol)) S.recs.set(p.symbol, { pick: p, ...ctx });
  };
  arr(t.board).forEach((p) => put(p, { where: 'board' }));
  if (t.top) put(t.top, { where: 'board' });
  arr(t.squeeze_zone).forEach((p) => put(p, { where: 'zone' }));
  arr(t.swing_board).forEach((p) => put(p, { where: 'swing' }));
  if (twinsAnchor()) {
    arr(t.twins).forEach((tw) => { if (tw && tw.pick) put(tw.pick, { where: 'twin' }); });
    arr(t.twins).forEach((tw) => {
      const r = tw && S.recs.get(tw.symbol);
      if (r) r.twin = tw;
    });
  }
  arr(t.swing_board).forEach((p) => {
    const r = p && S.recs.get(p.symbol);
    if (r && r.where !== 'swing') r.swing = p;
  });
}

/** The #1 the lookalikes were measured against — only feeds that say so (§14) get a twins section. */
function twinsAnchor() {
  const a = S.today && S.today.twins_anchor;
  return typeof a === 'string' && a ? a : null;
}

/* ── top bar & banners ─────────────────────────────────────────────────── */
function renderBar() {
  const t = S.today;
  setHTML('#st-session', t && t.session_date ? `Session <b>${esc(fmtDay(t.session_date))}</b>` : DASH);
  setHTML('#st-run', t && t.run ? `${esc(humanize(t.run))} run` : DASH);
  const btn = $('#feeds-btn');
  const src = arr(t && t.sources);
  if (btn) {
    if (!src.length) {
      btn.innerHTML = 'Feeds —';
      btn.setAttribute('aria-label', 'Data feed health: not reported');
    } else {
      const ok = src.filter((s) => s && s.ok).length;
      btn.innerHTML = `<span class="sq${ok < src.length ? ' bad' : ''}" aria-hidden="true"></span>Feeds ${ok}/${src.length}`;
      btn.setAttribute('aria-label', `Data feeds: ${ok} of ${src.length} worked in this run. Show details`);
    }
  }
  const pop = $('#feeds-pop');
  if (pop) {
    const list = src.map((s) => {
      const detail = [s.detail, s.asof ? fmtStamp(s.asof) : null].filter(Boolean).join(' · ');
      return `<li><span class="st ${s.ok ? 'ok' : 'bad'}">${s.ok ? 'OK' : 'FAILED'}</span><span>${txt(s.name)}</span>${detail ? `<span class="dt">${esc(detail)}</span>` : ''}</li>`;
    }).join('');
    const gen = t && t.generated_at ? `during the ${esc(t.run || '')} run published ${esc(fmtStamp(t.generated_at))}` : 'during the last run';
    pop.innerHTML = `<h2>Data sources</h2><p>Whether each public feed answered ${gen}. A failed feed means its fields show “—”, never a guess.</p>${list ? `<ul class="feeds-list">${list}</ul>` : empty('This feed did not report source health.')}`;
  }
  tickBar();
}

function tickBar() {
  const t = S.today;
  const phase = marketPhaseNow();
  setHTML('#st-phase', `<b>${esc(PHASE_LABEL[phase] || phase)}</b>`);
  const upd = $('#st-updated');
  if (upd) {
    upd.textContent = t && t.generated_at ? `Updated ${ago(t.generated_at)}` : 'Not published';
    if (t && t.generated_at) upd.title = fmtStamp(t.generated_at, { year: true });
  }
}

function renderBanners() {
  const sb = $('#sample-banner');
  if (sb) sb.hidden = !S.sample;
  const msgs = [];
  const t = S.today;
  if (!t) {
    msgs.push(`<b>Today's feed could not be loaded</b> (${esc(S.errors['today.json'] || 'unknown error')}). Sections show what is available.`);
  } else {
    if (t.late) msgs.push('<b>Published after the open</b> — not counted in the track record.');
    if (t.withheld && t.withheld.reason && (!t.generated_at || String(t.withheld.at) > String(t.generated_at))) msgs.push(`<b>The latest run was withheld</b> (${esc(t.withheld.reason)}) — this is the last page that passed the data checks.`);
    const nowEt = etDate();
    if (!S.sample && t.session_date && t.session_date < nowEt) {
      msgs.push(`<b>This radar is for ${esc(fmtDay(t.session_date))}.</b> Today's run has not published yet — treat every number here as stale.`);
    }
  }
  if (S.newer) msgs.push('A newer run has been published. <button class="btn btn--quiet" type="button" data-reload>Reload</button>');
  const el = $('#stale-banner');
  if (!el) return;
  el.hidden = !msgs.length;
  setHTML($('.wrap', el), msgs.map((m) => `<p>${m}</p>`).join(''));
}

function toggleFeeds(force) {
  const btn = $('#feeds-btn');
  const pop = $('#feeds-pop');
  if (!btn || !pop) return;
  const open = typeof force === 'boolean' ? force : pop.hidden;
  pop.hidden = !open;
  btn.setAttribute('aria-expanded', String(open));
}

/* ── hero ──────────────────────────────────────────────────────────────── */
function stat(label, value, note) {
  return `<div><dt class="eyebrow">${label}</dt><dd>${value}${note ? `<small>${note}</small>` : ''}</dd></div>`;
}

function renderHero() {
  const t = S.today;
  const top = t && t.top;
  const titleWrap = $('#hero-title-wrap');
  const body = $('#hero-body');
  const cap = $('#hero-caption');
  const eyebrow = $('#hero-eyebrow');
  const art = $('#hero-art');
  if (S.field) { S.field.destroy(); S.field = null; }
  $$('.hero-guide, .replay', art).forEach((n) => n.remove());

  if (eyebrow) eyebrow.textContent = (t && t.session_date ? `Today's #1 · ${fmtDay(t.session_date)}` : "Today's #1") + (t && t.top && isNum(t.top.rank) && t.top.rank > 1 ? ` · board rank ${t.top.rank}` : '');

  if (!top) {
    let why = "Today's feed could not be loaded. The raw file is at data/today.json.";
    if (t) {
      why = arr(t.board).length
        ? 'The feed has a board but no #1 was set.'
        : 'Nothing passed the borrow and squeeze-danger filters this session. The squeeze zone below lists what was held back.';
    }
    setHTML(titleWrap, `<div><h1 id="hero-title"><span class="hero-ticker hero-ticker--none">${t ? 'No clean #1 today' : 'Radar unavailable'}</span></h1><p class="hero-name">${esc(why)}</p></div>`);
    if (cap) cap.textContent = '';
    setHTML(body, '');
    return;
  }

  const sym = esc(top.symbol);
  const base = S.base;
  const probN = isNum(top.prob_dump) ? Math.round(top.prob_dump * 100) : null;
  const capV = obj(obj(t.model).prob_cap)[top.model_used === 'm1' ? 'm1' : 'm0'];
  const atCap = isNum(capV) && isNum(top.prob_dump) && top.prob_dump >= capV - 1e-4;
  const where = [top.exchange, top.country].filter(Boolean).map(esc).join(' · ');
  const lede = isNum(base)
    ? `The model's odds of a <b>5%+ drop from open to close</b> this session — <b>${esc(liftTxt(top.lift))}</b> the ${esc(pct(base, 1))} base rate for an average eligible name.`
    : 'The model\'s odds of a <b>5%+ drop from open to close</b> this session. Base rate not reported in this feed.';
  setHTML(titleWrap, `
    <div>
      <h1 id="hero-title"><button type="button" class="hero-ticker" data-open="${sym}">${sym}<span class="sr-only">, open the dossier</span></button></h1>
      <p class="hero-name">${txt(top.name)}${where ? ` <span class="mono">· ${where}</span>` : ''}</p>
    </div>
    <div class="hero-prob">
      <p class="hero-prob__num">${probN == null ? DASH : `${atCap ? '≥' : ''}${probN}<sup>%</sup>`}</p>
      <p class="hero-prob__lede">${lede}</p>
    </div>`);

  // particle field + caption + guide
  const rows = arr(top.chart).filter((r) => Array.isArray(r));
  const closes = rows.map((r) => (isNum(r[4]) ? r[4] : null));
  const valid = rows.filter((r) => isNum(r[4]));
  if (cap) {
    cap.innerHTML = valid.length >= 2
      ? `<b>${sym}</b> · ${valid.length} daily closes, ${esc(fmtDay(valid[0][0], { weekday: false }))} – ${esc(fmtDay(valid[valid.length - 1][0], { weekday: false }))}<br>Red marks the last 5 sessions.`
      : '';
  }
  const canvas = $('#field');
  if (canvas && art && valid.length >= 4) {
    const replay = document.createElement('button');
    replay.type = 'button';
    replay.className = 'replay';
    replay.textContent = 'Replay';
    replay.hidden = true;
    replay.setAttribute('aria-label', 'Replay the falling-particles animation');
    art.appendChild(replay);
    const guide = document.createElement('div');
    guide.className = 'hero-guide';
    guide.setAttribute('aria-hidden', 'true');
    guide.hidden = true;
    art.appendChild(guide);
    try {
      S.field = mountField(canvas, closes, {
        redTail: 5,
        onPhase: (ph) => { replay.hidden = ph !== 'rest'; },
        onLayout: (g) => {
          if (!g || !isNum(g.hi)) return;
          const hiRow = rows[g.hiI];
          guide.style.top = `${Math.round(g.yOf(g.hi))}px`;
          guide.innerHTML = `<span>Window high ${esc(price(g.hi))}${hiRow ? ` · ${esc(fmtDay(hiRow[0], { weekday: false }))}` : ''}</span>`;
          guide.hidden = false;
        },
      });
    } catch (e) {
      console.error('[gravity] hero field failed', e);
      S.field = null;
    }
    replay.addEventListener('click', () => { if (S.field) S.field.replay(); });
  }

  // body
  const pre = top.premarket;
  const gap = pre && isNum(pre.gap_pct) ? pct(pre.gap_pct, 0, true) : DASH;
  const preNote = pre
    ? `${esc(price(pre.price))} at ${esc(fmtTimeET(pre.asof))} ET vs ${esc(price(isNum(pre.prev_close) ? pre.prev_close : top.prev_close))} close`
    : (t.run === 'evening' ? 'Evening watchlist — no pre-market print yet' : 'No pre-market trade reported');
  const sh = obj(top.shortability);
  const shNote = sh.status === 'UNKNOWN'
    ? 'IBKR borrow file unavailable this run'
    : `${esc(borrowText(sh))}${sh.asof ? ` · IBKR ${esc(fmtStamp(sh.asof))}` : ' · IBKR'}`;
  const sqNote = isNum(top.squeeze_danger) ? `${sqWord(top.squeeze_danger)} — a crowding score, not a forecast` : 'Not enough data to score';
  const scored = t.universe && isNum(t.universe.scored) ? t.universe.scored : null;
  const modelLine = top.model_used === 'm1'
    ? 'M1 reading — uses the pre-market price as the expected open'
    : top.model_used === 'm0' ? 'M0 reading — close-only features (no pre-market print)' : 'Model version not reported';

  const aboutNote = `${esc(modelLine)}. Features as of the close on ${esc(fmtDay(t.features_asof, { year: true }))}.${isNum(top.score) ? (top.score >= 100 ? (t.top_tie_count > 1 ? ` Tied with ${esc(int(t.top_tie_count - 1))} other name${t.top_tie_count > 2 ? 's' : ''} for the highest odds of the ${esc(int(scored))} scored — ties are ordered by the model's raw score.` : ` The highest odds of the ${esc(int(scored))} names scored today.`) : ` Higher odds than ${esc(top.score)}% of the ${esc(int(scored))} names scored today.`) : ''}${backtestLine(top.model_used)}${isNum(top.rank) && top.rank > 1 ? ` The ${top.rank - 1} higher-ranked board name${top.rank > 2 ? 's are' : ' is'} skipped because ${top.rank > 2 ? 'they are' : 'it is'} under the Rule 201 short-sale restriction today or trade under $300K a day — the rule the backtest measured.` : ''}`;
  const weighed = attributionBars(top.attribution);

  setHTML(body, `
    <dl class="hstats reveal">
      ${stat('Squeeze odds', esc(pct(top.prob_squeeze, 0)), esc(isNum(obj(obj(S.model).base_rate).squeeze) ? `P(+20% above the open) · ${fixed(top.prob_squeeze / S.model.base_rate.squeeze, 1)}× normal — dump odds and squeeze odds rise together` : 'P(+20% above the open), same session'))}
      ${stat('Pre-market gap', esc(gap), preNote)}
      ${stat('Borrow', esc(sh.status === 'NONE' ? 'None' : sh.status === 'ETB' || sh.status === 'HTB' ? sh.status : DASH), shNote)}
      ${stat('Squeeze danger', squeezeMeter(top.squeeze_danger, { large: true }), esc(sqNote))}
    </dl>
    ${heroSub(top)}
    <div class="hero-cols">
      <div class="reveal">
        <p class="eyebrow" style="margin-bottom:16px">What stands out — rule-based context, not the model's own reasoning</p>
        ${reasonsList(topReasons(top.reasons))}
        <div class="hero-actions">
          <button class="btn" type="button" data-open="${sym}">Open the full dossier</button>
          <a class="btn btn--quiet" href="#board">See all ${arr(t.board).length} on the board</a>
        </div>
      </div>
      <aside class="reveal" aria-label="${sym} at a glance">
        <div class="side-block"><p class="eyebrow">Flags</p>${flagChips(top.flags)}</div>
        ${weighed ? `<div class="side-block"><p class="eyebrow">What the model weighed</p>${weighed}<p class="note" style="margin-top:8px">Share of today's reading that disappears when each family is swapped for a typical name's values. Describes the model, not the company.</p></div>` : ''}
        <div class="side-block"><p class="eyebrow">Research ${sym} elsewhere</p>${linksRow(top)}</div>
        <div class="side-block">
          <p class="eyebrow">About this reading</p>
          <p class="note">${aboutNote}</p>
        </div>
      </aside>
    </div>`);
}

/** Base rate for a target from model.json (dump/squeeze/swing/pump), or null. */
function baseOf(k) {
  const b = obj(obj(S.model).base_rate)[k];
  if (isNum(b)) return b;
  const sw = swingReport();
  if (k === 'swing' && isNum(sw.base_rate)) return sw.base_rate;
  return null;
}

/** "62% of its big-move odds point down" — the downside share of the two big-move odds. */
function skewChip(skew, { lead = '' } = {}) {
  if (!isNum(skew)) return '';
  const down = skew >= 0.5;
  const title = 'skew = dump odds ÷ (dump odds + odds of a +5% open→close rise)';
  return `<span class="chip${down ? ' chip--red' : ''}" title="${esc(title)}">${esc(`${lead}${pct(skew)} of its big-move odds point down`)}</span>`;
}

function heroSub(p) {
  const bits = [];
  const sw = p.prob_swing;
  const bs = baseOf('swing');
  if (isNum(sw)) {
    bits.push(`<p class="hero-sub__line"><span class="eyebrow">5-session swing</span><b>${esc(pct(sw))}</b> odds it closes 15%+ below the next open five sessions from now${isNum(bs) ? ` · ${esc(liftTxt(sw / bs))} the ${esc(pct(bs, 1))} base rate` : ''}</p>`);
  }
  if (isNum(p.prob_pump)) {
    bits.push(`<p class="hero-sub__line"><span class="eyebrow">Pump odds</span><b>${esc(pct(p.prob_pump))}</b> odds of a 5%+ rise open→close instead${isNum(baseOf('pump')) ? ` · base ${esc(pct(baseOf('pump'), 1))}` : ''}</p>`);
  }
  const chip = skewChip(p.skew);
  const size = sizeLine(p);
  if (!chip && !bits.length && !size) return '';
  return `<div class="hero-sub reveal">${chip ? `<div class="hero-sub__chip">${chip}</div>` : ''}${bits.join('')}<div id="hero-size">${size}</div></div>`;
}

/** Normalised attribution {family: 0–1} → hbars with plain labels, biggest first. */
function attributionBars(attr, { max = 6 } = {}) {
  const a = obj(attr);
  const items = Object.entries(a)
    .filter(([, v]) => isNum(v) && v > 0)
    .sort((x, y) => y[1] - x[1])
    .slice(0, max)
    .map(([k, v], i) => ({ label: ATTR_LABEL[k] || humanize(k), v, hot: i === 0, fmt: (x) => pct(x), title: ATTR_HELP[k] || '' }));
  if (!items.length) return '';
  return hbars(items, 1, { cls: 'hbars--mini' });
}

/* ── board ─────────────────────────────────────────────────────────────── */
const FILTERS = [
  {
    id: 'shortable', label: 'Shortable only', title: 'IBKR lists shares to borrow (ETB or HTB)',
    test: (p) => { const s = obj(p.shortability); return (s.status === 'ETB' || s.status === 'HTB') && !(isNum(s.available) && s.available <= 0); },
  },
  { id: 'nosq', label: 'No squeeze danger', title: 'Squeeze danger known and below 45 of 100', test: (p) => isNum(p.squeeze_danger) && p.squeeze_danger < 45 },
  { id: 'asia', label: 'Asia-linked', title: 'Issuer country or business address in Asia', test: isAsia },
  { id: 'offer24', label: 'Offering in 24h', title: 'Offering, ATM, toxic financing or unregistered sale filed in the 24 hours before publish', test: freshSupply },
  { id: 'sub1', label: 'Under $1', title: 'Last close below one dollar', test: (p) => isNum(p.price) && p.price < 1 },
  { id: 'fits', label: 'Fits my size', title: 'Capacity (volume, impact and IBKR borrow) at least the selected size', test: (p) => fitsSize(p) },
];

const SORTS = {
  rank: { label: 'Rank', get: (p) => p.rank, dir: 1 },
  prob: { label: 'Dump odds', get: (p) => p.prob_dump, dir: -1 },
  lift: { label: 'Lift', get: (p) => p.lift, dir: -1 },
  gap: { label: 'Pre-market gap', get: (p) => (p.premarket ? p.premarket.gap_pct : null), dir: -1 },
  fee: { label: 'Borrow fee', get: (p) => obj(p.shortability).fee_rate, dir: 1 },
  sq: { label: 'Squeeze danger', get: (p) => p.squeeze_danger, dir: 1 },
  size: { label: 'Cost at my size', get: (p) => costAt(p), dir: 1 },
  symbol: { label: 'Ticker', get: (p) => p.symbol, dir: 1 },
};

function probScale(board) {
  const mx = Math.max(0, ...board.map((p) => p.prob_dump).filter(isNum));
  return Math.max(0.2, Math.ceil((mx * 1.1) / 0.1) * 0.1);
}

function sortedFiltered(board) {
  const tests = FILTERS.filter((f) => S.filters.has(f.id));
  const rows = board.filter((p) => tests.every((f) => f.test(p)));
  const { key, dir } = S.sort;
  const get = (SORTS[key] || SORTS.rank).get;
  return rows.slice().sort((a, b) => {
    const va = get(a);
    const vb = get(b);
    const ma = va == null || (typeof va === 'number' && !Number.isFinite(va));
    const mb = vb == null || (typeof vb === 'number' && !Number.isFinite(vb));
    if (ma && mb) return (a.rank || 0) - (b.rank || 0);
    if (ma) return 1;
    if (mb) return -1;
    const c = typeof va === 'string' ? va.localeCompare(vb) : va - vb;
    return c * dir || (a.rank || 0) - (b.rank || 0);
  });
}

function boardRow(p, max) {
  const sym = esc(p.symbol);
  const pre = p.premarket;
  const gap = pre && isNum(pre.gap_pct) ? pct(pre.gap_pct, 0, true) : DASH;
  const nm = [p.name, p.country].filter(Boolean).join(' · ');
  const meta = `<div class="mcard-meta">${probBar(p.prob_dump, S.base, max)}<span>Lift ${esc(liftTxt(p.lift))}</span><span>Gap ${esc(gap)}</span>${borrowInline(p.shortability)}<span>Squeeze ${esc(isNum(p.squeeze_danger) ? p.squeeze_danger : DASH)}</span>${isNum(obj(p.size).capacity) ? `<span>${esc(sizeLabel(S.size))} cost ${esc(pct(costAt(p), 1))} · max ${esc(money(p.size.capacity))}</span>` : ''}</div>`;
  return `<tr data-sym="${sym}">
    <td class="c-rank">${isNum(p.rank) ? pad2(p.rank) : DASH}</td>
    <td class="c-name"><button type="button" class="tk" data-open="${sym}">${sym}</button><span class="nm">${txt(nm)}</span>${arr(p.flags).length ? flagChips(p.flags, 4) : ''}</td>
    <td class="c-prob"><span class="pv">${esc(pct(p.prob_dump))}</span>${probBar(p.prob_dump, S.base, max)}</td>
    <td class="c-lift">${esc(liftTxt(p.lift))}</td>
    <td class="c-fam">${familyBars(p.families)}</td>
    <td class="c-gap">${esc(gap)}</td>
    <td class="c-borrow">${borrowChip(p.shortability)}<span class="borrow__txt">${esc(borrowText(p.shortability))}</span></td>
    <td class="c-size">${sizeCell(p)}</td>
    <td class="c-sq">${squeezeMeter(p.squeeze_danger)}</td>
    <td class="c-spark">${sparkline(p.chart)}</td>
    <td class="c-meta-m">${meta}</td>
  </tr>`;
}

function renderBoard() {
  const root = $('#board-root');
  if (!root) return;
  const board = arr(S.today && S.today.board);
  const dek = $('#board-dek');
  if (dek && isNum(S.base)) {
    dek.textContent = `Ranked by the model's probability of an open→close drop of 5% or more this session. The tick on each bar is the ${pct(S.base, 1)} base rate for the average eligible name.`;
  }
  if (!S.today) { root.innerHTML = empty('The board could not be loaded.'); return; }
  const chips = FILTERS.map((f) => {
    const n = board.filter(f.test).length;
    return `<button type="button" class="fchip" data-filter="${f.id}" aria-pressed="${S.filters.has(f.id)}" title="${esc(f.title)}">${esc(f.label)}<span class="ct">${n}</span></button>`;
  }).join('');
  const opts = Object.entries(SORTS).map(([k, s]) => `<option value="${k}"${S.sort.key === k ? ' selected' : ''}>${esc(s.label)}</option>`).join('');
  root.innerHTML = `
    <div class="board-tools">
      <div class="filters" role="group" aria-label="Filter the board">${chips}</div>
      <label class="board-sort">Sort <select id="board-sort">${opts}</select></label>
      <label class="board-sort">My size <select id="board-size">${SIZE_OPTS.map((v) => `<option value="${v}"${v === S.size ? ' selected' : ''}>${sizeLabel(v)}</option>`).join('')}</select></label>
      <p class="board-count" id="board-count" aria-live="polite"></p>
    </div>
    <div class="board-scroll"><table class="board" id="board-table"><caption class="sr-only">Today's board, ranked by modeled dump odds. Select a ticker for its dossier.</caption><thead></thead><tbody></tbody></table></div>`;
  drawBoard();
  const bs = $('#board-size');
  if (bs) bs.addEventListener('change', () => {
    S.size = Number(bs.value) || 50000;
    saveSize(S.size);
    safe('board', renderBoard);
    const hs = $('#hero-size');
    if (hs && S.today && S.today.top) hs.innerHTML = sizeLine(S.today.top);
    drawLookup();
    const nb = $('#board-size');
    if (nb) nb.focus();
  });
}

function drawBoard() {
  const board = arr(S.today && S.today.board);
  const table = $('#board-table');
  if (!table) return;
  const max = probScale(board);
  const cols = [
    { cls: 'h-rank', key: 'rank', label: '#' },
    { key: 'symbol', label: 'Ticker' },
    { key: 'prob', label: 'Dump odds', scale: isNum(S.base) ? `tick = ${pct(S.base, 1)} base · 0–${pct(max)}` : `0–${pct(max)} scale` },
    { cls: 'h-lift', key: 'lift', label: 'Lift' },
    { cls: 'h-fam', label: 'Signals', scale: 'dil exh dec flow st news' },
    { key: 'gap', label: 'Pre-mkt' },
    { key: 'fee', label: 'Borrow' },
    { cls: 'h-size', key: 'size', label: `At ${sizeLabel(S.size)}`, scale: 'est. round-trip cost · max size' },
    { key: 'sq', label: 'Squeeze' },
    { cls: 'h-spark', label: '120 sessions' },
  ];
  const th = cols.map((c) => {
    const active = c.key && S.sort.key === c.key;
    const sortAttr = active ? ` aria-sort="${S.sort.dir > 0 ? 'ascending' : 'descending'}"` : '';
    const inner = c.key
      ? `<button type="button" data-sort="${c.key}">${esc(c.label)}<span class="dir" aria-hidden="true">${active ? (S.sort.dir > 0 ? '↑' : '↓') : ''}</span></button>`
      : esc(c.label);
    return `<th scope="col"${c.cls ? ` class="${c.cls}"` : ''}${sortAttr}>${inner}${c.scale ? `<span class="scale">${esc(c.scale)}</span>` : ''}</th>`;
  }).join('');
  setHTML($('thead', table), `<tr>${th}</tr>`);
  const rows = sortedFiltered(board);
  let body;
  if (!board.length) body = '<tr class="board-empty"><td colspan="10">No names made the board this session — see the squeeze zone for what was held back.</td></tr>';
  else if (!rows.length) body = '<tr class="board-empty"><td colspan="10">No names match every active filter. Clear a filter to see more.</td></tr>';
  else body = rows.map((p) => boardRow(p, max)).join('');
  setHTML($('tbody', table), body);
  const cnt = $('#board-count');
  if (cnt) cnt.textContent = board.length ? `Showing ${rows.length} of ${board.length}` : '';
  const sel = $('#board-sort');
  if (sel) sel.value = S.sort.key;
}

function setSort(key, fromSelect = false) {
  if (!SORTS[key]) return;
  if (!fromSelect && S.sort.key === key) S.sort.dir *= -1;
  else S.sort = { key, dir: SORTS[key].dir };
  drawBoard();
  if (!fromSelect) {
    const b = $(`#board-table th [data-sort="${key}"]`);
    if (b) b.focus();
  }
}

/* ── catalyst wire ─────────────────────────────────────────────────────── */
function wireItem(w) {
  const sym = w.symbol ? esc(w.symbol) : DASH;
  const symHTML = w.symbol && S.recs.has(w.symbol)
    ? `<button type="button" data-open="${sym}">${sym}</button>`
    : sym;
  const when = parseTs(w.time) ? fmtStamp(w.time).replace(/ ET$/, '') : fmtDay(w.time, { weekday: false });
  return `<li>
    <span class="t"${parseTs(w.time) ? ` title="${esc(fmtStamp(w.time, { year: true }))}"` : ''}>${esc(when)}</span>
    <span class="sym">${symHTML}</span>
    <span class="k">${txt(w.kind)}</span>
    <span class="h">${extLink(w.url, w.headline || DASH)}<span class="src">${txt(w.source)}</span></span>
    ${severity(w.severity)}
  </li>`;
}

/** Newest first by real instant: feeds can mix UTC and ET offsets, so string order is not time order. */
function sortedWire(t) {
  const when = (w) => { const d = parseTs(w.time) || parseDay(w.time); return d ? d.getTime() : -Infinity; };
  return arr(t && t.catalyst_wire).filter(Boolean).map((w, i) => ({ w, i, k: when(w) }))
    .sort((a, b) => b.k - a.k || a.i - b.i).map((x) => x.w);
}

function renderWire() {
  const root = $('#wire-root');
  if (!root) return;
  const t = S.today;
  const wire = sortedWire(t);
  let html = wire.length
    ? `<ul class="wire" id="wire-list">${wire.slice(0, WIRE_SHOW).map(wireItem).join('')}</ul>${wire.length > WIRE_SHOW ? `<div class="more"><button class="btn btn--quiet" type="button" data-wire-all aria-controls="wire-list">Show all ${wire.length}</button></div>` : ''}`
    : empty(t ? 'Quiet overnight: no fresh supply filings, bearish headlines or 20%+ pre-market moves among eligible names.' : 'The wire could not be loaded.');
  const earn = arr(t && t.earnings).filter((e) => e && e.symbol);
  if (earn.length) {
    const when = (s) => (/pre/.test(s || '') ? 'Before the open' : /after/.test(s || '') ? 'After the close' : 'Time not supplied');
    html += `<div class="earn-block"><p class="eyebrow" style="margin-bottom:16px">Reporting earnings this session</p><ul class="lst earn">${earn.map((e) => {
      const sym = esc(e.symbol);
      const s = S.recs.has(e.symbol) ? `<button type="button" class="tk" data-open="${sym}">${sym}</button>` : `<span class="tk">${sym}</span>`;
      const bits = [when(e.time), isNum(e.eps_forecast) ? `EPS est. ${e.eps_forecast < 0 ? '−' : ''}$${Math.abs(e.eps_forecast).toFixed(2)}` : 'No EPS estimate', isNum(e.n_ests) ? `${e.n_ests} est.` : null, isNum(e.market_cap) ? `cap ${money(e.market_cap)}` : null, e.in_universe ? 'in the scored universe' : 'outside the universe'].filter(Boolean);
      return `<li>${s} <span class="muted">· ${esc(bits.join(' · '))}</span></li>`;
    }).join('')}</ul></div>`;
  }
  root.innerHTML = html;
}

/*__NEW_SECTIONS__*/

/* ── twins ─────────────────────────────────────────────────────────────── */
function renderTwins() {
  const root = $('#twins-root');
  if (!root) return;
  const tw = arr(S.today && S.today.twins).filter((x) => x && x.symbol);
  if (!tw.length) { root.innerHTML = empty(S.today ? 'No lookalikes cleared the similarity bar this run.' : 'Twins could not be loaded.'); return; }
  root.innerHTML = `<div class="grid-cards reveal">${tw.map((t) => {
    const sym = esc(t.symbol);
    const can = S.recs.has(t.symbol);
    const flags = arr(t.flags).length ? t.flags : arr(t.pick && t.pick.flags);
    const simN = isNum(t.similarity) ? Math.round(t.similarity * 100) : null;
    const sub = [t.name, t.country].filter(Boolean).join(' · ');
    return `<article class="card${can ? ' card--link' : ''}">
      <div class="card__top">
        <div>${can ? `<button type="button" class="card__sym stretch" data-open="${sym}">${sym}</button>` : `<p class="card__sym">${sym}</p>`}<p class="card__name">${txt(sub)}</p></div>
        <span class="eyebrow">${isNum(t.rank) ? `Board #${esc(t.rank)}` : 'Off the board'}</span>
      </div>
      <p class="card__big">${simN == null ? DASH : `${simN}<small>% alike</small>`}</p>
      ${arr(t.reasons).length ? `<ul class="why">${arr(t.reasons).map((r) => `<li>${esc(r)}</li>`).join('')}</ul>` : ''}
      ${flags.length ? flagChips(flags, 4) : ''}
      <div class="card__meta"><span>Dump odds <b>${esc(pct(t.prob_dump))}</b></span><span>${esc(price(t.price))} · ${esc(money(t.market_cap))}</span></div>
    </article>`;
  }).join('')}</div>`;
}

/* ── squeeze zone ──────────────────────────────────────────────────────── */
const PARTS = [
  ['model', 'Model', 'Model odds of a +20% intraday spike'],
  ['fee', 'Fee', 'Borrow fee'],
  ['scarcity', 'Supply', 'Few shares available to borrow'],
  ['si', 'SI', 'Short interest as a share of float'],
  ['dtc', 'DTC', 'Days to cover'],
  ['float', 'Float', 'Tiny float'],
  ['short_vol', 'Sh. vol', 'FINRA short-sale share of volume'],
];

function partsBars(parts) {
  const p = obj(parts);
  const label = PARTS.map(([k, , help]) => `${help}: ${isNum(p[k]) ? Math.round(p[k] * 100) : 'unknown'}`).join('; ');
  return `<div class="parts" role="img" aria-label="${esc(`Squeeze danger parts (0–100). ${label}`)}">${PARTS.map(([k, short, help]) => {
    const v = p[k];
    const bar = isNum(v) ? `<i><b style="--h:${Math.round(Math.max(0, Math.min(1, v)) * 100)}%"></b></i>` : '<i class="null"></i>';
    return `<span title="${esc(help)}">${bar}${esc(short)}</span>`;
  }).join('')}</div>`;
}

function renderSqueeze() {
  const root = $('#squeeze-root');
  if (!root) return;
  const z = arr(S.today && S.today.squeeze_zone).filter((p) => p && p.symbol);
  if (!z.length) { root.innerHTML = empty(S.today ? 'Nothing was held back for squeeze danger or missing borrow this session.' : 'The squeeze zone could not be loaded.'); return; }
  root.innerHTML = `<ul class="zone reveal">${z.map((p) => {
    const sym = esc(p.symbol);
    return `<li>
      <div class="z-name"><button type="button" class="tk" data-open="${sym}">${sym}</button><span class="nm">${txt([p.name, p.country].filter(Boolean).join(' · '))}</span>${borrowInline(p.shortability)}</div>
      <div class="z-prob"><b>${esc(pct(p.prob_dump))}</b><small>dump odds</small></div>
      <div class="z-sq">${squeezeMeter(p.squeeze_danger, { large: true })}</div>
      <p class="why">${txt(p.zone_reason)}</p>
      ${partsBars(p.squeeze_parts)}
    </li>`;
  }).join('')}</ul>`;
}

/* ── track record ──────────────────────────────────────────────────────── */
function hbars(items, max, { cls = '', base = null } = {}) {
  const m = max || Math.max(0.01, ...items.map((x) => x.v).filter(isNum));
  const tick = isNum(base) ? `<span class="base" style="left:${(Math.max(0, Math.min(1, base / m)) * 100).toFixed(1)}%"></span>` : '';
  return `<ul class="hbars${cls ? ` ${cls}` : ''}">${items.map((x) => `<li class="${x.hot ? 'hot' : ''}"${x.title ? ` title="${esc(x.title)}"` : ''}><span>${x.html || esc(x.label)}</span><span class="b" aria-hidden="true">${isNum(x.v) ? `<i style="width:${(Math.max(0, Math.min(1, x.v / m)) * 100).toFixed(1)}%"></i>` : ''}${tick}</span><span class="v">${esc(x.fmt ? x.fmt(x.v) : pct(x.v, 1))}</span></li>`).join('')}</ul>`;
}

function liveBlock(sc) {
  const live = obj(sc && sc.live);
  const days = arr(sc && sc.days);
  const head = '<h3 class="rec-sub">Live</h3><p class="note">Every published #1, graded after that session\'s close. Late (after-the-open) runs are excluded.</p>';
  if (!sc) return `<div>${head}${empty(`The scorecard could not be loaded${S.errors['scorecard.json'] ? ` (${esc(S.errors['scorecard.json'])})` : ''}.`)}</div>`;
  if (!live.n_days) return `<div>${head}${empty('<b>No graded sessions yet.</b> The first #1 is graded after its session closes; results appear here that evening. Nothing is back-filled.')}</div>`;
  const first = days.length ? days[days.length - 1].session_date : null;
  return `<div>${head}
    <dl class="tiles">
      ${stat('Sessions', esc(int(live.n_days)), first ? `since ${esc(fmtDay(first, { weekday: false }))}` : '')}
      ${stat('#1 dumped', esc(pct(live.top_dump_rate)), esc(`of ${int(live.top_n)} graded${live.top_halted ? ` (+${int(live.top_halted)} halted, not tradable)` : ''} · all names ${pct(live.universe_dump_rate)}`))}
      ${stat('#1 avg o→c', esc(pct(live.top_mean_oc, 1, true)), esc(`all names ${pct(live.universe_mean_oc, 1, true)}`))}
      ${stat('#1 squeezed', esc(pct(live.top_squeeze_rate)), 'rose 20%+ from the open')}
    </dl>
    ${hbars([
      { label: 'Published #1', v: live.top_dump_rate, hot: true },
      { label: 'Whole board', v: live.board_dump_rate, hot: true },
      { label: 'All eligible names', v: live.universe_dump_rate },
    ])}
    <p class="note" style="margin-top:8px">Share of sessions that fell 5%+ from open to close.${live.top_squeeze_rate != null ? ` The #1 spiked 20%+ above the open on ${pct(live.top_squeeze_rate)} of graded sessions.` : ''}${live.n_unverified ? ` ${int(live.n_unverified)} session(s) excluded because the pick could not be confirmed public before the open.` : ''} ${live.n_days < 60 ? 'Small sample: these rates will move a lot as days accumulate.' : ''}</p>
  </div>`;
}

function backtestBlock(m) {
  const head = '<h3 class="rec-sub">Backtest</h3>';
  if (!m) return `<div>${head}${empty(`No model report yet${S.errors['model.json'] ? ` (${esc(S.errors['model.json'])})` : ''}. It appears after the first training run.`)}</div>`;
  const tab = S.tab;
  const sim = obj(obj(m.sim)[tab]);
  const oos = obj(obj(m.oos)[tab]);
  const base = obj(m.base_rate).dump;
  const cost = isNum(sim.cost_assumption) ? sim.cost_assumption : null;
  const tabs = ['m0', 'm1'].filter((k) => obj(m.sim)[k] || obj(m.oos)[k]);
  const tabHTML = tabs.map((k) => `<button type="button" role="tab" id="tab-${k}" aria-controls="bt-panel" aria-selected="${k === tab}" tabindex="${k === tab ? 0 : -1}" data-tab="${k}">${k === 'm1' ? 'M1 · knows the real open (live uses a pre-market proxy)' : 'M0 · close only — what the morning run can do'}</button>`).join('');
  const nDays = arr(sim.daily).length;
  return `<div>${head}
    <p class="note">Walk-forward, out of sample${m.trained_through ? ` through ${esc(fmtDay(m.trained_through, { weekday: false, year: true }))}` : ''}. Short the #1 at the open, cover at the close${isNum(cost) ? `, ${esc(pct(cost, 0))} round-trip cost` : ''}. The published rule skips names under the Rule 201 short-sale restriction and names trading under $300K a day; it still assumes borrow was always available — often it isn't.${m.publication_rule ? ` <span class="sr-only">${esc(m.publication_rule.text || '')}</span>` : ''}</p>
    <div class="tabs" role="tablist" aria-label="Model version">${tabHTML}</div>
    <div id="bt-panel" role="tabpanel" aria-labelledby="tab-${tab}">
      <dl class="tiles">
        ${stat('Published #1 dumped', esc(pct(isNum(oos.pub1_hit) ? oos.pub1_hit : oos.top1_hit)), esc(`raw #1 ${pct(oos.top1_hit)} · average name ${pct(base, 1)}`))}
        ${stat('…and spiked 20%+', esc(pct(isNum(oos.pub1_squeeze_rate) ? oos.pub1_squeeze_rate : oos.top1_squeeze_rate)), 'above the open, same session — the move that stops shorts out')}
        ${realisticTiles(sim, cost)}
      </dl>
      ${swingTiles()}
      ${nDays ? '<div class="chart" id="eq-chart"></div><div class="legend"><span><i></i>Published rule, net of each pick\'s own est. cost at $10K</span><span><i class="g"></i>Raw #1 (incl. Rule 201 names), flat 1% cost</span></div>' : empty('No simulated days in this report.')}
    </div>
  </div>`;
}

/** Net of each pick's own estimated cost (capacity model) at $10K / $50K, flat-1% beside it. */
function realisticTiles(sim, cost) {
  const re = obj(sim.realistic);
  const r10 = obj(re['10000']);
  const r50 = obj(re['50000']);
  if (!isNum(r10.net_mean)) {
    return `${stat('Win rate', esc(pct(isNum(sim.pub_win_rate) ? sim.pub_win_rate : sim.win_rate)), esc(`after a flat ${pct(cost, 0)} cost`))}
      ${stat('Risking 10% a day', esc(isNum(obj(sim.compounded_pub).final_multiple) ? `${fixed(sim.compounded_pub.final_multiple, 2)}×` : DASH), 'flat-cost compounded multiple')}`;
  }
  return `${stat('Net per trade at $10K', `<span class="${r10.net_mean < 0 ? 'red' : ''}">${esc(pct(r10.net_mean, 1, true))}</span>`, esc(`after each pick's own est. cost (avg ${pct(r10.mean_cost, 1)}); at $50K ${pct(r50.net_mean, 1, true)} (cost ${pct(r50.mean_cost, 1)})`))}
      ${stat('Win rate at $10K', esc(pct(r10.win_rate)), esc(`compounded 10%/day ×${fixed(r10.compounded, 2)}; flat-1% version ×${isNum(obj(sim.compounded_pub).final_multiple) ? fixed(sim.compounded_pub.final_multiple, 2) : DASH}`))}`;
}

function swingTiles() {
  const sw = swingReport();
  if (!isNum(sw.top1_hit) && !isNum(sw.pub1_hit)) return '';
  const sim = obj(sw.sim_pub && Object.keys(obj(sw.sim_pub)).length ? sw.sim_pub : sw.sim);
  const bs = isNum(sw.base_rate) ? sw.base_rate : baseOf('swing');
  return `<p class="eyebrow" style="margin:8px 0 0">5-session swing · short the swing #1 at the next open, cover at the fifth close</p>
    <dl class="tiles">
      ${stat('Fell 15%+', esc(pct(isNum(sw.pub1_hit) ? sw.pub1_hit : sw.top1_hit)), esc(`average name ${pct(bs, 1)}`))}
      ${stat('Average 5-session move', esc(pct(isNum(sw.pub1_mean_c5) ? sw.pub1_mean_c5 : sw.top1_mean_c5, 1, true)), esc(`median ${pct(isNum(sw.pub1_median_c5) ? sw.pub1_median_c5 : sw.top1_median_c5, 1, true)}`))}
      ${stat('Win rate', esc(pct(sim.win_rate)), 'non-overlapping trades, after cost')}
      ${stat('Worst trade', esc(pct(isNum(sim.worst_trade) ? sim.worst_trade : sim.worst, 0, true)), esc(isNum(sim.n_trades) ? `${int(sim.n_trades)} trades` : 'short return'))}
    </dl>`;
}

function modelDetail(m) {
  if (!m) return '';
  const tab = S.tab;
  const oos = obj(m.oos);
  const o = obj(oos[tab]);
  const base = obj(m.base_rate).dump;
  const rowsDef = [
    ['AUC (0.5 = coin flip)', 'auc', (v) => fixed(v, 3)],
    ['Brier score (lower is better)', 'brier', (v) => fixed(v, 3)],
    ['#1 of the day dumped', 'top1_hit', (v) => pct(v, 1)],
    ['Top 10 dumped', 'top10_hit', (v) => pct(v, 1)],
    ['Top decile dumped', 'top_decile_hit', (v) => pct(v, 1)],
    ['#1 avg open→close', 'top1_mean_oc', (v) => pct(v, 1, true)],
    ['Top 10 avg open→close', 'top10_mean_oc', (v) => pct(v, 1, true)],
    ['Published-rule #1 dumped', 'pub1_hit', (v) => pct(v, 1)],
    ['Published-rule #1 avg open→close', 'pub1_mean_oc', (v) => pct(v, 1, true)],
    ['Raw #1 under Rule 201 next day', 'top1_ssr_rate', (v) => pct(v, 0)],
    ['Same score ranks +5% UP days (AUC)', '@pump_auc', (v) => fixed(v, 3)],
    ['Dump vs pump among big moves (0.5 = no direction)', '@dump_vs_pump_auc', (v) => fixed(v, 3)],
    ['Test sessions', 'days', int],
  ];
  const getv = (c, k) => (k[0] === '@' ? obj(obj(m.directional)[c])[k.slice(1)] : obj(oos[c])[k]);
  const cols = ['m0', 'm1'].filter((k) => oos[k]);
  const mt = cols.length ? `<div class="table-scroll"><table class="mtable"><caption class="sr-only">Out-of-sample metrics by model</caption><thead><tr><th scope="col">Metric</th>${cols.map((k) => `<th scope="col">${k.toUpperCase()}</th>`).join('')}</tr></thead><tbody>${rowsDef.map(([l, k, f]) => `<tr><th scope="row">${esc(l)}</th>${cols.map((c) => `<td>${esc(f(getv(c, k)))}</td>`).join('')}</tr>`).join('')}</tbody></table></div>` : '';
  const fam = {};
  arr(tab === 'm1' && arr(m.importance_m1).length ? m.importance_m1 : m.importance).forEach((r) => { if (r && isNum(r.importance)) fam[r.family] = (fam[r.family] || 0) + r.importance; });
  const famItems = Object.entries(fam).sort((a, b) => b[1] - a[1]).map(([k, v]) => ({ label: FAMILY_LABEL[k] || humanize(k), v, fmt: (x) => fixed(x * 1000, 1) }));
  const trainMeta = [
    isNum(m.n_rows) ? `${int(m.n_rows)} name-days` : null,
    isNum(m.n_symbols) ? `${int(m.n_symbols)} symbols` : null,
    isNum(m.n_days) ? `${int(m.n_days)} sessions` : null,
    m.trained_at ? `trained ${fmtStamp(m.trained_at, { year: true })}` : null,
  ].filter(Boolean).join(' · ');
  return `
    <div class="rec-grid rec-block reveal">
      <div>
        <h3 class="rec-sub">Calibration</h3>
        <p class="note">When the model said X%, how often did it happen? Points on the dashed line mean the odds can be taken at face value.</p>
        <div class="chart chart--fixed" id="cal-chart"></div>
      </div>
      <div>
        <h3 class="rec-sub">Top of the list vs everyone</h3>
        <p class="note">Share of test sessions that dumped, ${esc(tab.toUpperCase())}.</p>
        ${hbars([
          { label: '#1 of the day', v: o.top1_hit, hot: true },
          { label: 'Top 10', v: o.top10_hit, hot: true },
          { label: 'Top decile', v: o.top_decile_hit, hot: true },
          { label: 'Base rate (all names)', v: base },
        ])}
        <div style="margin-top:32px">${mt}</div>
      </div>
    </div>
    <div class="rec-grid rec-block reveal">
      <div>
        <h3 class="rec-sub">What it leans on</h3>
        <p class="note">Permutation importance summed by signal family (× 1,000). Bigger = the model's accuracy drops more when that family is scrambled.</p>
        ${famItems.length ? hbars(famItems) : empty('No importance table in this report.')}
      </div>
      <div>
        <h3 class="rec-sub">Caveats</h3>
        ${trainMeta ? `<p class="note">${esc(trainMeta)}</p>` : ''}
        ${arr(m.caveats).length ? `<ul class="caveats">${arr(m.caveats).map((c) => `<li>${esc(c)}</li>`).join('')}</ul>` : empty('None listed in this report.')}
        ${m.targets ? `<ul class="caveats">${Object.entries(obj(m.targets)).map(([k, v]) => `<li><b>${esc(humanize(k))}</b> = ${esc(v)}</li>`).join('')}</ul>` : ''}
      </div>
    </div>
    `;
}

function dayRow(d) {
  const o = obj(d.outcome);
  const top = obj(d.top);
  const ot = obj(o.top);
  let res = DASH;
  let cls = '';
  if (ot.missing) res = 'No data';
  else if (ot.dump === true) { res = 'Dumped'; cls = 'res-dump'; } else if (ot.squeezed === true) { res = 'Squeezed'; cls = 'res-sq'; } else if (ot.dump === false) res = 'No dump';
  return `<tr><td>${esc(fmtDay(d.session_date, { weekday: true }))}</td><td>${txt(top.symbol)}</td><td>${esc(pct(top.prob_dump))}</td><td>${esc(pct(ot.oc, 1, true))}</td><td class="${cls}">${esc(res)}</td><td>${esc(pct(o.board_dump_rate))}</td><td>${esc(pct(o.universe_dump_rate))}</td></tr>`;
}

function daysBlock(sc) {
  const days = arr(sc && sc.days).filter(Boolean);
  if (!days.length) return '';
  const head = '<thead><tr><th scope="col">Session</th><th scope="col">#1</th><th scope="col">Odds</th><th scope="col">Open→close</th><th scope="col">Result</th><th scope="col">Board dumped</th><th scope="col">All names</th></tr></thead>';
  return `<div class="rec-block reveal">
    <h3 class="rec-sub">Every graded session</h3>
    <p class="note" style="margin-bottom:24px">Newest first. “Dumped” = fell 5%+ from open to close. “Squeezed” = rose 20%+ from the open at some point.</p>
    <div class="table-scroll"><table class="days" id="days-table"><caption class="sr-only">Graded sessions</caption>${head}<tbody>${days.slice(0, DAYS_SHOW).map(dayRow).join('')}</tbody></table></div>
    ${days.length > DAYS_SHOW ? `<div class="more"><button class="btn btn--quiet" type="button" data-days-all aria-controls="days-table">Show all ${days.length}</button></div>` : ''}
  </div>`;
}

function renderRecord() {
  const root = $('#record-root');
  if (!root) return;
  dispose(S.disposers.record);
  const m = S.model;
  if (m && !obj(m.sim)[S.tab] && !obj(m.oos)[S.tab]) S.tab = obj(m.sim).m0 ? 'm0' : 'm1';
  root.innerHTML = `<div class="rec-grid rec-block reveal">${liveBlock(S.scorecard)}${backtestBlock(m)}</div>${modelDetail(m)}${daysBlock(S.scorecard)}`;
  mountRecordCharts();
}

function mountRecordCharts() {
  const m = S.model;
  if (!m) return;
  const sim = obj(obj(m.sim)[S.tab]);
  const eq = $('#eq-chart');
  if (eq) {
    const daily = arr(sim.daily).filter((r) => Array.isArray(r) && typeof r[0] === 'string');
    const cost = isNum(sim.cost_assumption) ? sim.cost_assumption : 0;
    let g = 0;
    let n = 0;
    const gross = [];
    const net = [];
    daily.forEach((r) => {
      if (isNum(r[1])) g += -r[1] - cost;               // raw #1, net
      if (isNum(r[4])) n += -r[4] - (isNum(r[5]) ? r[5] : cost);   // published rule, net of its own est. cost at $10K
      else if (r.length < 5 && isNum(r[1])) n += -r[1] - cost;
      gross.push(g);
      net.push(n);
    });
    S.disposers.record.push(equityChart(eq, daily.map((r) => r[0]), [
      { name: 'Raw #1, net', cls: 'l-gross', values: gross },
      { name: 'Published rule, net', cls: 'l-net', values: net },
    ], { label: `Cumulative sum of daily short returns, ${S.tab.toUpperCase()}: the published rule versus the raw #1, both net of cost` }));
  }
  const cal = $('#cal-chart');
  if (cal) {
    const c = obj(m.calibration);
    const bins = arr(c[S.tab]).length ? c[S.tab] : arr(c.m1);
    S.disposers.record.push(calibrationChart(cal, bins));
  }
}

/* ── evidence ──────────────────────────────────────────────────────────── */
function renderEvidence() {
  const root = $('#evidence-root');
  if (!root) return;
  const ev = S.evidence;
  if (!ev) { root.innerHTML = empty(`No event studies yet${S.errors['evidence.json'] ? ` (${esc(S.errors['evidence.json'])})` : ''}. They appear after the first training run.`); return; }
  const base = ev.baseline || arr(ev.studies).find((s) => s && s.id === 'baseline') || null;
  const studies = arr(ev.studies).filter((s) => s && s.id !== 'baseline');
  const all = [base, ...studies].filter(Boolean);
  if (!all.length) { root.innerHTML = empty('The evidence file has no studies.'); return; }
  const vals = all.flatMap((s) => [s.pct_dump, ...arr(s.ci_pct_dump)]).filter(isNum);
  const max = Math.max(0.1, Math.ceil((Math.max(0, ...vals) * 1.08) / 0.05) * 0.05);
  const bp = base ? base.pct_dump : null;
  root.innerHTML = `<p class="note reveal" style="margin-bottom:24px">Red tick = share that fell 5%+ open→close the next session · whisker = 95% interval · dashed = all names (${esc(pct(bp, 1))}).</p>
    <div class="grid-cards reveal">${all.map((s) => {
      const isBase = s === base;
      const ci = arr(s.ci_pct_dump);
      const small = isNum(s.n) && s.n < 200;
      const sr = `${pct(s.pct_dump, 1)} dumped${ci.length === 2 && isNum(ci[0]) && isNum(ci[1]) ? ` (95% interval ${pct(ci[0], 1)} to ${pct(ci[1], 1)})` : ''}${isBase ? '' : `, versus ${pct(bp, 1)} for all names`}.`;
      return `<article class="card ev-card${isBase ? ' is-base' : ''}">
        <p class="eyebrow">${isBase ? 'Baseline · ' : ''}n = ${esc(int(s.n))}${isNum(s.n_symbols) ? ` · ${esc(int(s.n_symbols))} names` : ''}${small ? ' · small sample' : ''}</p>
        <h3>${txt(s.title)}</h3>
        <p class="ev-plain">${txt(s.plain)}</p>
        <div>${whisker(s.pct_dump, ci, isBase ? null : bp, max)}<div class="ev-scale" aria-hidden="true"><span>0%</span><span>${esc(pct(max))}</span></div></div>
        <p class="sr-only">${esc(sr)}</p>
        <div class="ev-nums">
          <span><b>${esc(pct(s.pct_dump, 1))}</b>dumped next session</span>
          <span><b>${esc(isBase ? '1.0×' : liftTxt(s.lift_dump))}</b>vs all names</span>
          <span><b>${esc(pct(s.median_oc, 1, true))}</b>median open→close</span>
          ${isNum(s.pct_up5) ? `<span><b>${esc(pct(s.pct_up5, 1))}</b>ran UP 5%+ instead</span>` : ''}
          ${isNum(s.pct_swing) ? `<span><b>${esc(pct(s.pct_swing, 1))}</b>fell 15%+ within 5 sessions</span>` : ''}
          ${!isBase && s.skew ? `<span><b>${esc(s.skew === 'down' ? 'Down' : s.skew === 'up' ? 'Up' : 'Both ways')}</b>which way it tilts</span>` : ''}
        </div>
      </article>`;
    }).join('')}</div>`;
}

/* ── swing report (model.json) ─────────────────────────────────────────── */
function swingReport() {
  const sw = obj(obj(S.model).swing);
  return obj(sw[S.tab] || sw.m0);
}

/* ── live tape (docs/data/live.json, ~every 15 min in session) ─────────── */
function liveFresh() {
  const L = S.live;
  const t = S.today;
  return L && t && L.session_date && L.session_date === t.session_date ? L : null;
}

function renderLive() {
  const sec = $('#live');
  const root = $('#live-root');
  if (!sec || !root) return;
  S.disposers.live.forEach((d) => { try { d(); } catch { /* ignore */ } });
  S.disposers.live = [];
  const L = liveFresh();
  sec.hidden = !L;
  if (!L) { root.innerHTML = ''; return; }
  const top = L.top;
  const rows = arr(L.board).filter((r) => r && r.symbol)
    .slice().sort((a, b) => (isNum(a.oc_now) ? a.oc_now : 9) - (isNum(b.oc_now) ? b.oc_now : 9));
  const down5 = rows.filter((r) => isNum(r.oc_now) && r.oc_now <= -0.05).length;
  const tk = (s) => (S.recs.has(s) ? `<button type="button" class="tk" data-open="${esc(s)}">${esc(s)}</button>` : esc(s));
  root.innerHTML = `
    <p class="note live-note">Snapshot ${esc(fmtStamp(L.asof))}. ${esc(L.note || 'Live and unofficial.')}</p>
    ${top ? `<dl class="tiles reveal">
      ${stat(`#1 ${esc(top.symbol)} · open → now`, `<span class="${isNum(top.oc_now) && top.oc_now < 0 ? 'red' : ''}">${esc(pct(top.oc_now, 1, true))}</span>`, esc(`${price(top.open)} open → ${price(top.last)} last`))}
      ${stat('High since the open', esc(pct(top.oh_now, 1, true)), 'the rip a short had to sit through')}
      ${stat('Low since the open', esc(pct(top.ol_now, 1, true)), 'the best cover so far')}
      ${stat('Board, open → now', esc(pct(L.board_mean_oc_now, 1, true)), esc(`${down5} of ${rows.length} down 5%+ so far`))}
    </dl>
    <div class="chart" id="live-chart"></div>` : empty('No live quote for the #1 in this snapshot.')}
    ${rows.length ? `<div class="table-scroll" style="margin-top:32px"><table class="mtable"><caption class="sr-only">Board names, open to now</caption>
      <thead><tr><th scope="col">Ticker</th><th scope="col">Open</th><th scope="col">Last</th><th scope="col">Open→now</th><th scope="col">High vs open</th><th scope="col">Low vs open</th><th scope="col">Spread now</th></tr></thead>
      <tbody>${rows.map((r) => `<tr><th scope="row">${tk(r.symbol)}</th><td>${esc(price(r.open))}</td><td>${esc(price(r.last))}</td><td class="${isNum(r.oc_now) && r.oc_now <= -0.05 ? 'red' : ''}">${esc(pct(r.oc_now, 1, true))}</td><td>${esc(pct(isNum(r.high) && r.open ? r.high / r.open - 1 : null, 1, true))}</td><td>${esc(pct(isNum(r.low) && r.open ? r.low / r.open - 1 : null, 1, true))}</td><td>${esc(isNum(r.spread_now) ? pct(r.spread_now, 2) : DASH)}</td></tr>`).join('')}</tbody></table></div>` : ''}`;
  const el = $('#live-chart');
  if (el && top) {
    try { S.disposers.live.push(intradayChart(el, top.points, { open: top.open, label: `${top.symbol} since the open` })); } catch (e) { console.error('[gravity] live chart', e); }
  }
  revealAll(root);  // re-rendered nodes must not stay at opacity 0
}

let livePoll = 0;
function startLivePoll() {
  if (livePoll) return;
  livePoll = setInterval(async () => {
    if (document.hidden || marketPhaseNow() !== 'open') return;
    const L = await getJSON('live.json');
    if (L && (!S.live || L.asof !== S.live.asof)) {
      S.live = L;
      safe('live', renderLive);
    }
  }, LIVE_POLL_MS);
}

/* ── swing board ───────────────────────────────────────────────────────── */
function renderSwing() {
  const root = $('#swing-root');
  if (!root) return;
  const sb = arr(S.today && S.today.swing_board).filter((p) => p && p.symbol);
  if (!sb.length) { root.innerHTML = empty(S.today ? 'The swing model has not published a board yet — it appears after the next retrain.' : 'The swing board could not be loaded.'); return; }
  const bs = baseOf('swing');
  const max = Math.max(0.2, Math.ceil((Math.max(0, ...sb.map((p) => p.prob_swing).filter(isNum)) * 1.1) / 0.1) * 0.1);
  const sw = swingReport();
  const bt = isNum(sw.pub1_hit) || isNum(sw.top1_hit)
    ? `<p class="note" style="margin-bottom:24px">Backtest (out of sample): the top swing pick fell 15%+ by the fifth close on ${esc(pct(isNum(sw.pub1_hit) ? sw.pub1_hit : sw.top1_hit))} of entries${isNum(sw.pub1_mean_c5) ? `, averaging ${esc(pct(sw.pub1_mean_c5, 1, true))}` : ''}${isNum(bs) ? ` — versus ${esc(pct(bs, 1))} for the average name` : ''}. Multi-day shorts also pay borrow every day and ride every overnight gap.</p>` : '';
  root.innerHTML = `${bt}<div class="table-scroll reveal"><table class="mtable swing-table"><caption class="sr-only">Swing board: highest odds of a 15%+ fall over five sessions</caption>
    <thead><tr><th scope="col">#</th><th scope="col">Ticker</th><th scope="col">Swing odds${isNum(bs) ? ` <span class="muted">· base ${esc(pct(bs, 1))}</span>` : ''}</th><th scope="col">Today's skew</th><th scope="col">Today's dump odds</th><th scope="col">Borrow</th><th scope="col">Board</th></tr></thead>
    <tbody>${sb.map((p) => `<tr>
      <td>${isNum(p.rank) ? pad2(p.rank) : DASH}</td>
      <th scope="row" class="sw-name"><button type="button" class="tk" data-open="${esc(p.symbol)}">${esc(p.symbol)}</button><span class="nm">${txt([p.name, p.country].filter(Boolean).join(' · '))}</span>${arr(p.flags).length ? flagChips(p.flags, 3) : ''}</th>
      <td class="c-prob"><span class="pv">${esc(pct(p.prob_swing))}</span>${probBar(p.prob_swing, bs, max)}</td>
      <td>${esc(isNum(p.skew) ? `${pct(p.skew)} down` : DASH)}</td>
      <td>${esc(pct(p.prob_dump))}</td>
      <td>${borrowInline(p.shortability)}</td>
      <td>${isNum(p.board_rank) ? `#${esc(p.board_rank)}` : DASH}</td>
    </tr>`).join('')}</tbody></table></div>`;
}

/* ── market regime & sector heat ───────────────────────────────────────── */
function ordinal(n) {
  const v = n % 100;
  return `${n}${v >= 11 && v <= 13 ? 'th' : ({ 1: 'st', 2: 'nd', 3: 'rd' })[n % 10] || 'th'}`;
}
function renderMarket() {
  const root = $('#market-root');
  if (!root) return;
  const m = S.today && S.today.market;
  const secs = arr(S.today && S.today.sectors);
  if (!m && !secs.length) { root.innerHTML = empty('No market block in this feed yet.'); return; }
  const mm = obj(m);
  const regime = mm.regime ? `${humanize(mm.regime)}${isNum(mm.regime_pct) ? ` · ${ordinal(Math.round(mm.regime_pct))} pct` : ''}` : DASH;
  const tiles = `<dl class="tiles reveal">
    ${stat('Dump-odds regime', esc(regime), 'today\'s average odds vs the last year of sessions')}
    ${stat('Up yesterday', esc(pct(mm.breadth_up)), esc(`median name ${pct(mm.median_r1, 1, true)} · IWM ${pct(mm.iwm_r1, 1, true)}`))}
    ${stat('Big movers', esc(`${int(mm.n_up20)} / ${int(mm.n_down20)}`), 'names up 20%+ / down 20%+ yesterday')}
    ${stat('Realised dump rate', esc(pct(mm.universe_dump_rate_20d, 1)), 'share of names that fell 5%+ open→close, last 20 sessions')}
  </dl>`;
  const maxP = Math.max(0.01, ...secs.map((s) => s.mean_prob).filter(isNum));
  const bars = secs.length ? hbars(secs.map((s, i) => ({
    label: s.sector, v: s.mean_prob, hot: i === 0, fmt: (x) => pct(x, 1),
    html: `${esc(s.sector)} <span class="muted mono">· ${int(s.n)} names${s.n_top100 ? ` · ${int(s.n_top100)} in top 100` : ''}${arr(s.top).length ? ` · ${arr(s.top).map((x) => (S.recs.has(x) ? `<button type="button" class="tk tk--sm" data-open="${esc(x)}">${esc(x)}</button>` : esc(x))).join(' ')}` : ''}</span>`,
  })), maxP * 1.1, { base: S.base }) : empty('No sector breakdown in this feed.');
  root.innerHTML = `${tiles}<div class="rec-grid rec-block reveal"><div><h3 class="rec-sub">Sector heat</h3><p class="note">Average dump odds by sector (tick = the base rate). Sectors with fewer than five names are left out.</p>${bars}</div></div>`;
}

/* ── catalyst calendar ─────────────────────────────────────────────────── */
function renderCalendar() {
  const root = $('#calendar-root');
  if (!root) return;
  const c = obj(S.today && S.today.calendar);
  const earn = arr(c.earnings);
  const locks = arr(c.lockups);
  if (!S.today || (!earn.length && !locks.length && !S.today.calendar)) { root.innerHTML = empty('No calendar in this feed yet.'); return; }
  const tk = (s) => (S.recs.has(s) ? `<button type="button" class="tk" data-open="${esc(s)}">${esc(s)}</button>` : esc(s));
  const TIME = { 'time-pre-market': 'before the open', 'pre-market': 'before the open', 'time-after-hours': 'after the close', 'after-hours': 'after the close', 'time-not-supplied': 'time n/a', 'not-supplied': 'time n/a' };
  const byDay = {};
  earn.forEach((e) => { (byDay[e.date || '?'] = byDay[e.date || '?'] || []).push(e); });
  const earnHTML = earn.length
    ? `<div class="table-scroll"><table class="mtable"><caption class="sr-only">Earnings from names in the universe, next five sessions</caption><thead><tr><th scope="col">Ticker</th><th scope="col">When</th><th scope="col">EPS est.</th><th scope="col">Dump odds</th></tr></thead><tbody>${Object.keys(byDay).sort().map((d) => `<tr class="grp"><th scope="rowgroup" colspan="4">${esc(fmtDay(d))}</th></tr>${byDay[d].map((e) => `<tr><th scope="row">${tk(e.symbol)} <span class="nm">${txt(e.name)}</span></th><td>${esc(TIME[e.time] || humanize(e.time || '') || DASH)}</td><td>${esc(e.eps_forecast || DASH)}</td><td>${esc(pct(e.prob_dump))}</td></tr>`).join('')}`).join('')}</tbody></table></div>`
    : empty('No universe names report in the next five sessions (or the calendar did not load).');
  const lockHTML = locks.length
    ? `<div class="table-scroll"><table class="mtable"><caption class="sr-only">IPO lock-up expiries around now</caption><thead><tr><th scope="col">Ticker</th><th scope="col">IPO</th><th scope="col">Lock-up ends</th><th scope="col">Days</th><th scope="col">Price vs IPO</th><th scope="col">Dump odds</th></tr></thead><tbody>${locks.map((l) => `<tr><th scope="row">${tk(l.symbol)} <span class="nm">${txt(l.name)}</span></th><td>${esc(fmtDay(l.ipo_date, { weekday: false, year: true }))} · ${esc(price(l.ipo_price))}</td><td>${esc(fmtDay(l.lockup_date, { weekday: false }))}</td><td class="${isNum(l.days_to) && l.days_to >= 0 && l.days_to <= 5 ? 'red' : ''}">${esc(isNum(l.days_to) ? (l.days_to < 0 ? `${-l.days_to} ago` : `in ${l.days_to}`) : DASH)}</td><td>${esc(pct(l.vs_ipo, 0, true))}</td><td>${esc(pct(l.prob_dump))}</td></tr>`).join('')}</tbody></table></div><p class="note" style="margin-top:8px">Lock-up dates assume the standard 180 days after pricing; the real date is in each prospectus. Insiders are free to sell after it.${lockupEvidence()}</p>`
    : empty('No IPO lock-ups expire around now among recent listings.');
  root.innerHTML = `<div class="rec-grid rec-block reveal"><div><h3 class="rec-sub">Earnings · next five sessions</h3>${earnHTML}</div><div><h3 class="rec-sub">IPO lock-up expiries</h3>${lockHTML}</div></div>`;
}

/** What the Evidence Lab actually measured around day ~180 — never more. */
function lockupEvidence() {
  const st = arr(obj(S.evidence).studies).find((x) => x && x.id === 'lockup_180');
  if (!st || !isNum(st.pct_dump) || !isNum(st.ref_pct_dump)) return '';
  return ` In this data, names in the day-170–190 window fell 5%+ open→close ${pct(st.pct_dump, 1)} of the time vs ${pct(st.ref_pct_dump, 1)} for comparable names (${liftTxt(st.lift_dump)}, n = ${int(st.n)})${isNum(st.lift_dump) && st.lift_dump < 1.2 ? ' — not a meaningful edge on its own' : ''}.`;
}

/* ── look up any ticker (docs/data/universe.json) ─────────────────────── */
const LK_FILTERS = [
  { id: 'short', label: 'Borrowable', test: (r) => r.borrow_status === 'ETB' || r.borrow_status === 'HTB' },
  { id: 'nossr', label: 'No Rule 201', test: (r) => !r.ssr },
  { id: 'sub1', label: 'Under $1', test: (r) => isNum(r.price) && r.price < 1 },
  { id: 'board', label: 'On the board', test: (r) => isNum(r.board_rank) },
  { id: 'fits500', label: 'Takes $500K', test: (r) => isNum(r.capacity) && r.capacity >= 5e5 },
];
const LK_COLS = [
  ['symbol', 'Ticker', 1], ['price', 'Price', -1], ['market_cap', 'Mkt cap', -1], ['prob', 'Dump odds', -1],
  ['prob_squeeze', 'Squeeze odds', -1], ['prob_swing', 'Swing odds', -1], ['skew', 'Skew', -1],
  ['fee_rate', 'Borrow fee', 1], ['squeeze_danger', 'Sq. danger', 1], ['capacity', 'Max size', -1],
];

async function loadUniverse() {
  if (S.uni.status !== 'idle') return;
  S.uni.status = 'loading';
  drawLookup();
  const u = await getJSON('universe.json');
  if (!u || !Array.isArray(u.rows) || !Array.isArray(u.columns)) {
    S.uni.status = 'error';
  } else {
    const cols = u.columns;
    S.uni.rows = u.rows.map((r) => Object.fromEntries(cols.map((c, i) => [c, r[i]])));
    S.uni.rows.forEach((r) => { r.prob = r.prob_dump; S.uni.bySym.set(r.symbol, r); });
    S.uni.asof = u.asof;
    S.uni.session = u.session_date;
    S.uni.status = 'ready';
  }
  drawLookup();
}

function renderLookup() {
  const root = $('#lookup-root');
  if (!root) return;
  const chips = LK_FILTERS.map((f) => `<button type="button" class="fchip" data-lkfilter="${f.id}" aria-pressed="${S.look.filters.has(f.id)}">${esc(f.label)}</button>`).join('');
  root.innerHTML = `
    <div class="board-tools">
      <label class="lk-search"><span class="sr-only">Search by ticker or company</span><input id="lk-q" type="search" placeholder="Type a ticker or company" autocomplete="off" autocapitalize="characters" spellcheck="false" value="${esc(S.look.q)}"></label>
      <div class="filters" role="group" aria-label="Filter the lookup">${chips}</div>
      <p class="board-count" id="lk-count" aria-live="polite"></p>
    </div>
    <div id="lk-card"></div>
    <div class="table-scroll"><table class="mtable lk-table" id="lk-table"><caption class="sr-only">Every scored name today. Select a ticker to open it.</caption><thead></thead><tbody></tbody></table></div>
    <div class="lk-pager" id="lk-pager"></div>`;
  const q = $('#lk-q');
  q.addEventListener('focus', loadUniverse, { once: true });
  q.addEventListener('input', () => { S.look.q = q.value; S.look.page = 0; drawLookup(); });
  q.addEventListener('keydown', (e) => {
    if (e.key !== 'Enter') return;
    const hit = S.uni.bySym.get(q.value.trim().toUpperCase());
    if (hit) showLookupCard(hit.symbol);
  });
  root.addEventListener('click', (e) => {
    const f = e.target.closest('[data-lkfilter]');
    if (f) {
      const id = f.dataset.lkfilter;
      if (S.look.filters.has(id)) S.look.filters.delete(id); else S.look.filters.add(id);
      f.setAttribute('aria-pressed', String(S.look.filters.has(id)));
      S.look.page = 0; drawLookup(); return;
    }
    const s = e.target.closest('[data-lksort]');
    if (s) {
      const k = s.dataset.lksort;
      const def = (LK_COLS.find((c) => c[0] === k) || [0, 0, -1])[2];
      S.look.dir = S.look.sort === k ? -S.look.dir : def;
      S.look.sort = k; S.look.page = 0; drawLookup(); return;
    }
    const pg = e.target.closest('[data-lkpage]');
    if (pg) { S.look.page = Math.max(0, S.look.page + Number(pg.dataset.lkpage)); drawLookup(); return; }
    const row = e.target.closest('[data-lksym]');
    if (row && !e.target.closest('[data-open]')) showLookupCard(row.dataset.lksym);
  });
  if ('IntersectionObserver' in window) {
    const io = new IntersectionObserver((es) => { if (es.some((x) => x.isIntersecting)) { io.disconnect(); loadUniverse(); } }, { rootMargin: '400px' });
    io.observe(root);
  } else loadUniverse();
  drawLookup();
}

function lookupRows() {
  const q = S.look.q.trim().toLowerCase();
  const tests = LK_FILTERS.filter((f) => S.look.filters.has(f.id));
  let rows = S.uni.rows.filter((r) => tests.every((f) => f.test(r)));
  if (q) {
    rows = rows.filter((r) => String(r.symbol).toLowerCase().startsWith(q) || String(r.name || '').toLowerCase().includes(q));
    rows.sort((a, b) => (String(a.symbol).toLowerCase() === q ? -1 : 0) - (String(b.symbol).toLowerCase() === q ? -1 : 0));
  }
  const k = S.look.sort;
  const d = S.look.dir;
  if (!q) {
    rows = rows.slice().sort((a, b) => {
      const va = a[k];
      const vb = b[k];
      if (va == null && vb == null) return 0;
      if (va == null) return 1;
      if (vb == null) return -1;
      return (typeof va === 'string' ? va.localeCompare(vb) : va - vb) * d;
    });
  }
  return rows;
}

function drawLookup() {
  const table = $('#lk-table');
  const cnt = $('#lk-count');
  const pager = $('#lk-pager');
  if (!table) return;
  if (S.uni.status !== 'ready') {
    setHTML($('tbody', table), `<tr class="board-empty"><td colspan="${LK_COLS.length + 1}">${S.uni.status === 'error' ? `The full-universe file could not be loaded (${esc(S.errors['universe.json'] || 'unknown error')}).` : 'Loading every scored name…'}</td></tr>`);
    return;
  }
  const th = LK_COLS.map(([k, label]) => {
    const active = S.look.sort === k;
    return `<th scope="col"${active ? ` aria-sort="${S.look.dir > 0 ? 'ascending' : 'descending'}"` : ''}><button type="button" data-lksort="${k}">${esc(label)}<span class="dir" aria-hidden="true">${active ? (S.look.dir > 0 ? '↑' : '↓') : ''}</span></button></th>`;
  }).join('');
  setHTML($('thead', table), `<tr>${th}<th scope="col">30 sessions</th></tr>`);
  const rows = lookupRows();
  const pages = Math.max(1, Math.ceil(rows.length / LOOKUP_PAGE));
  S.look.page = Math.min(S.look.page, pages - 1);
  const slice = rows.slice(S.look.page * LOOKUP_PAGE, (S.look.page + 1) * LOOKUP_PAGE);
  setHTML($('tbody', table), slice.length ? slice.map((r) => {
    const sym = esc(r.symbol);
    const tk = S.recs.has(r.symbol) ? `<button type="button" class="tk" data-open="${sym}">${sym}</button>` : `<button type="button" class="tk tk--quiet">${sym}</button>`;
    return `<tr data-lksym="${sym}"><th scope="row">${tk}<span class="nm">${txt(r.name)}</span>${r.ssr ? '<span class="chip chip--red">SSR</span>' : ''}</th>
      <td>${esc(price(r.price))}</td><td>${esc(money(r.market_cap))}</td><td class="${r.board_rank ? 'red' : ''}">${esc(pct(r.prob_dump))}</td>
      <td>${esc(pct(r.prob_squeeze))}</td><td>${esc(pct(r.prob_swing))}</td><td>${esc(isNum(r.skew) ? pct(r.skew) : DASH)}</td>
      <td>${esc(isNum(r.fee_rate) ? `${pctUnits(r.fee_rate, 0)}` : (r.borrow_status === 'NONE' ? 'none' : DASH))}</td><td>${esc(isNum(r.squeeze_danger) ? r.squeeze_danger : DASH)}</td>
      <td>${esc(isNum(r.capacity) ? money(r.capacity) : DASH)}</td>
      <td>${sparkCloses(r.spark)}</td></tr>`;
  }).join('') : `<tr class="board-empty"><td colspan="${LK_COLS.length + 1}">No scored name matches “${esc(S.look.q)}”. Names outside the small/micro-cap universe (or that did not trade last session) are not scored.</td></tr>`);
  if (cnt) cnt.textContent = `${int(rows.length)} of ${int(S.uni.rows.length)} names · as of ${S.uni.asof ? fmtStamp(S.uni.asof) : DASH}`;
  if (pager) {
    pager.innerHTML = pages > 1
      ? `<button type="button" class="btn btn--quiet" data-lkpage="-1"${S.look.page === 0 ? ' disabled' : ''}>Previous</button><span class="mono">Page ${S.look.page + 1} of ${pages}</span><button type="button" class="btn btn--quiet" data-lkpage="1"${S.look.page >= pages - 1 ? ' disabled' : ''}>Next</button>`
      : '';
  }
}

function showLookupCard(sym) {
  if (S.recs.has(sym)) { openDossier(sym, { opener: document.activeElement }); return; }
  const r = S.uni.bySym.get(sym);
  const el = $('#lk-card');
  if (!el || !r) return;
  const links = fallbackLinks(sym);
  el.innerHTML = `<article class="card lk-card">
    <div class="card__top"><div><p class="card__sym">${esc(sym)}</p><p class="card__name">${txt([r.name, r.sector, r.country].filter(Boolean).join(' · '))}</p></div>
      <span class="eyebrow">${isNum(r.board_rank) ? `Board #${esc(r.board_rank)}` : 'Not on today\'s boards'}</span></div>
    <dl class="tiles">
      ${stat('Dump odds', `<span class="red">${esc(pct(r.prob_dump))}</span>`, esc(isNum(S.base) ? `base ${pct(S.base, 1)}${isNum(r.score) ? ` · higher than ${r.score}% of names` : ''}` : ''))}
      ${stat('Squeeze odds', esc(pct(r.prob_squeeze)), 'P(+20% above the open)')}
      ${stat('Swing odds', esc(pct(r.prob_swing)), '15%+ lower five sessions out')}
      ${stat('Borrow', esc(r.borrow_status || DASH), esc(isNum(r.fee_rate) ? `${pctUnits(r.fee_rate, 1)}/yr · ${compact(r.available)} sh` : 'IBKR'))}
    </dl>
    <div class="lk-card__row">${sparkCloses(r.spark, { w: 220, h: 44 })}<span class="mono">${esc(price(r.price))} · ${esc(money(r.market_cap))}</span></div>
    ${arr(r.flags).length ? flagChips(r.flags) : ''}
    <p class="note" style="margin-top:16px">A compact card: full dossiers (filings, news, borrow history, dilution) are built for the board, swing board, squeeze zone and lookalikes. ${r.ssr ? 'Rule 201 short-sale restriction likely in effect today. ' : ''}</p>
    <div class="links">${Object.entries(links).map(([k, u]) => extLink(u, k)).join('')}</div>
  </article>`;
  el.scrollIntoView({ block: 'nearest', behavior: prefersReducedMotion() ? 'auto' : 'smooth' });
}

/* ── dossier add-ons (§14.3) ───────────────────────────────────────────── */
function dilutionHTML(d) {
  if (!d) return '';
  const lo = obj(d.last_offering);
  const rows = [
    ['Shares outstanding', esc(compact(d.shares_now)), d.shares_asof ? `SEC, ${fmtDay(d.shares_asof, { weekday: false, year: true })}` : ''],
    ['Change in a year', esc(isNum(d.shares_growth_1y) ? `${fixed(d.shares_growth_1y, 1)}×` : DASH), isNum(d.shares_1y_ago) ? `from ${compact(d.shares_1y_ago)}` : ''],
    ['Cash', esc(money(d.cash)), d.cash_date ? fmtDay(d.cash_date, { weekday: false, year: true }) : ''],
    ['Quarterly burn', esc(money(d.quarterly_burn)), 'operating cash flow'],
    ['Runway', esc(isNum(d.runway_q) ? `${fixed(d.runway_q, 1)} quarters` : DASH), isNum(d.runway_q) && d.runway_q < 2 ? 'short — a raise is likely' : ''],
  ];
  if (lo.date) {
    rows.push(['Last offering', esc(`${humanize(lo.type || 'offering')} · ${lo.form || ''}`), fmtDay(lo.date, { weekday: false, year: true })]);
    if (isNum(lo.price)) rows.push(['Offering price', esc(price(lo.price)), isNum(lo.shares) ? `${compact(lo.shares)} shares` : '']);
    if (isNum(lo.gross)) rows.push(['Gross proceeds', esc(money(lo.gross))]);
  }
  if (isNum(d.atm_capacity)) rows.push(['At-the-market program', esc(`up to ${money(d.atm_capacity)}`), 'shares can be sold into the market at any time']);
  const w = obj(d.warrants);
  if (isNum(w.shares)) rows.push(['Warrants', esc(`${compact(w.shares)} sh`), isNum(w.exercise_price) ? `exercise ${price(w.exercise_price)}` : '']);
  const src = arr(d.sources).filter(safeUrl).slice(0, 3).map((u, i) => extLink(u, `Filing ${i + 1} ↗`)).join(' ');
  return kv(rows) + (src ? `<p class="note" style="margin-top:8px">${src}</p>` : '');
}

function insiderHTML(ins) {
  if (!ins) return '';
  if (ins.fpi) return '<p class="note">Foreign private issuers don\'t file Form 4, so insider sales aren\'t disclosed this way. Form 144 notices still appear in the filings list when filed.</p>';
  const tr = arr(ins.trades).slice(0, 8);
  return kv([
    ['Open-market sales', esc(int(ins.n_sales)), isNum(ins.value_sold) ? `${money(ins.value_sold)} sold` : ''],
    ['Open-market buys', esc(int(ins.n_buys)), isNum(ins.value_bought) ? `${money(ins.value_bought)} bought` : ''],
    ['Form 144 notices', esc(int(ins.n_144)), 'intent to sell restricted stock'],
  ]) + (tr.length ? `<ul class="lst" style="margin-top:16px">${tr.map((x) => `<li><span class="meta">${esc(fmtDay(x.date, { weekday: false, year: true }))} · ${txt(x.name)}${x.title ? ` · ${esc(x.title)}` : ''}</span>${esc(x.code === 'P' ? 'Bought' : 'Sold')} ${esc(compact(x.shares))} sh${isNum(x.price) ? ` at ${esc(price(x.price))}` : ''}${safeUrl(x.url) ? ` ${extLink(x.url, '↗')}` : ''}</li>`).join('')}</ul>` : '');
}

function chatterHTML(c) {
  if (!c) return '';
  const tot = (c.bull || 0) + (c.bear || 0);
  return kv([
    ['Messages a day', esc(isNum(c.msgs_per_day) ? fixed(c.msgs_per_day, c.msgs_per_day < 10 ? 1 : 0) : DASH), 'Stocktwits, from the latest 30'],
    ['Bullish / bearish tags', esc(tot ? `${c.bull || 0} / ${c.bear || 0}` : DASH), tot ? `${pct((c.bear || 0) / tot)} bearish` : 'few tagged posts'],
    ['Watchers', esc(compact(c.watchers))],
  ]) + (safeUrl(c.url) ? `<p class="note" style="margin-top:8px">${extLink(c.url, 'Open the stream ↗')} · Chatter is crowd noise; spikes often accompany pumps.</p>` : '');
}

function historyHTML(p) {
  const ph = arr(p.prob_history);
  const bh = arr(p.borrow_history);
  const bits = [];
  if (ph.length >= 2) bits.push(`<div class="hist-row"><span class="eyebrow">Dump odds, last ${ph.length} sessions</span>${miniLine(ph, { label: `${p.symbol} dump odds history` })}<span class="mono">${esc(pct(ph[0][1]))} → ${esc(pct(ph[ph.length - 1][1]))}</span></div>`);
  if (bh.length >= 2) {
    bits.push(`<div class="hist-row"><span class="eyebrow">Borrow fee</span>${miniLine(bh.map((r) => [r[0], r[1]]), { label: `${p.symbol} borrow fee history` })}<span class="mono">${esc(isNum(bh[0][1]) ? pctUnits(bh[0][1], 0) : DASH)} → ${esc(isNum(bh[bh.length - 1][1]) ? pctUnits(bh[bh.length - 1][1], 0) : DASH)}</span></div>`);
    bits.push(`<div class="hist-row"><span class="eyebrow">Shares to borrow</span>${miniLine(bh.map((r) => [r[0], r[2]]), { label: `${p.symbol} lendable shares history` })}<span class="mono">${esc(compact(bh[0][2]))} → ${esc(compact(bh[bh.length - 1][2]))}</span></div>`);
  }
  return bits.join('');
}

/* ── trade size (§ size) ───────────────────────────────────────────────── */
const SIZE_KEY = 'gravity:size:v1';
const SIZE_OPTS = [10000, 50000, 100000, 500000, 1000000];
function loadSize() {
  try { const v = Number(localStorage.getItem(SIZE_KEY)); if (SIZE_OPTS.includes(v)) return v; } catch { /* storage blocked */ }
  return 50000;
}
function saveSize(v) { try { localStorage.setItem(SIZE_KEY, String(v)); } catch { /* storage blocked */ } }
const sizeLabel = (v) => (v >= 1e6 ? `$${v / 1e6}M` : `$${Math.round(v / 1e3)}K`);

/** Estimated round-trip cost at the selected size, or null. */
function costAt(p, size = S.size) {
  const c = obj(obj(p && p.size).costs)[String(size)];
  return isNum(c) ? c : null;
}
function fitsSize(p, size = S.size) {
  const cap = obj(p && p.size).capacity;
  return isNum(cap) && cap >= size;
}
const LIMIT_TXT = { volume: '5% of expected volume', impact: '0.5% estimated impact', borrow: 'IBKR lendable shares' };

function sizeCell(p) {
  const sz = obj(p.size);
  if (!isNum(sz.capacity)) return `<span class="muted">${DASH}</span>`;
  const c = costAt(p);
  const fits = fitsSize(p);
  const be = sz.breakeven;
  const tooBig = isNum(be) && S.size > be;
  return `<span class="${!fits || tooBig ? 'red' : ''}" title="${esc(`Estimated round-trip cost at ${sizeLabel(S.size)}: impact on both legs + spread`)}">${esc(pct(c, 1))}</span><small class="sz-cap">max ${esc(money(sz.capacity))}</small>`;
}

function sizeLine(p) {
  const sz = obj(p.size);
  if (!isNum(sz.capacity)) return '';
  const be = isNum(sz.breakeven) ? (sz.breakeven >= 1e4 ? money(sz.breakeven) : 'under $10K') : null;
  return `<p class="hero-sub__line"><span class="eyebrow">Size</span>Absorbs about <b>${esc(money(sz.capacity))}</b> before moving the price (limit: ${esc(LIMIT_TXT[sz.limit] || sz.limit || DASH)})${be ? ` · breakeven ≈ <b>${esc(be)}</b> — above that, estimated costs exceed ${sz.move_basis === '#1' ? "the historical #1's" : "a historical top-10 pick's"} average move` : ''} · at ${esc(sizeLabel(S.size))}: <b>${esc(pct(costAt(p), 1))}</b> round-trip</p>`;
}

function sizeDossier(p) {
  const sz = obj(p.size);
  if (!isNum(sz.capacity)) return '';
  const rows = SIZE_OPTS.map((q) => {
    const c = obj(sz.costs)[String(q)];
    const part = obj(sz.participation)[String(q)];
    const ok = isNum(sz.capacity) && q <= sz.capacity;
    const be = isNum(sz.breakeven) && q <= sz.breakeven;
    return `<tr${q === S.size ? ' class="hot"' : ''}><th scope="row">${esc(sizeLabel(q))}</th><td>${esc(pct(c, 1))}</td><td>${esc(isNum(part) ? pct(part, part < 0.1 ? 1 : 0) : DASH)}</td><td class="${ok ? '' : 'red'}">${ok ? 'Yes' : 'No'}</td><td class="${be ? '' : 'red'}">${isNum(sz.breakeven) ? (be ? 'Yes' : 'No') : DASH}</td></tr>`;
  }).join('');
  return kv([
    ['Max size without moving the price', esc(money(sz.capacity)), `limit: ${LIMIT_TXT[sz.limit] || sz.limit || DASH}`],
    ['Expected session $ volume', esc(money(sz.exp_dvol)), 'conservative estimate'],
    ['IBKR lendable value', esc(isNum(sz.cap_borrow) ? money(sz.cap_borrow) : DASH)],
    ['Spread used in costs', esc(isNum(sz.spread) ? pct(sz.spread, 2) : DASH), sz.spread_estimated === false ? 'estimator could not read it — conservative fallback (≥ 0.5% or one cent)' : 'Abdi–Ranaldo estimate, 20 sessions'],
    ['Breakeven size', esc(isNum(sz.breakeven) ? money(sz.breakeven) : DASH), isNum(sz.avg_move) ? `where costs reach ${sz.move_basis === '#1' ? "the historical #1's" : "a historical top-10 pick's"} ${pct(sz.avg_move, 1)} average move — a point estimate with wide uncertainty` : ''],
  ]) + `<div class="table-scroll" style="margin-top:16px"><table class="mtable"><caption class="sr-only">Estimated cost by position size</caption><thead><tr><th scope="col">Size</th><th scope="col">Round-trip cost</th><th scope="col">Share of volume</th><th scope="col">Within capacity</th><th scope="col">Below breakeven</th></tr></thead><tbody>${rows}</tbody></table></div><p class="note" style="margin-top:8px">Square-root impact model (Y = 0.7 × daily volatility × √(size ÷ volume)) on both legs plus one spread. Planning estimates — real costs depend on how the order is worked.</p>`;
}

/* ── size lab (docs/data/size_research.json) ───────────────────────────── */
function renderSizeLab() {
  const root = $('#sizelab-root');
  if (!root) return;
  const r = S.sizeLab;
  if (!r) { root.innerHTML = empty('The size research file has not been published.'); return; }
  const dc = (o, f) => `${esc(f(obj(o).dev))} <span class="muted">/</span> ${esc(f(obj(o).confirm))}`;
  const sp = (v) => pct(v, 1, true);
  const intr = arr(r.intraday).map((t) => `<tr><th scope="row">${esc(t.tier)} <span class="nm">tested at ${esc(t.size_tested)}</span></th><td>${dc(t.names_per_day, (v) => int(v))}</td><td>${dc(t.base_dump, (v) => pct(v, 1))}</td><td>${dc(t.top1_hit, (v) => pct(v, 0))}</td><td>${dc(t.top1_mean_oc, sp)}</td><td>${dc(t.cost, (v) => pct(v, 1))}</td><td>${dc(t.net, sp)}</td><td>${dc(t.net_tight, sp)}</td></tr>`).join('');
  const md = obj(r.multi_day_20);
  const mdRows = ['$500K', '$1M'].map((t) => {
    const x = obj(md[t]);
    return `<tr><th scope="row">${esc(t)}</th><td>${dc(x.biased_net, sp)}</td><td>${dc(x.biased_tier, sp)}</td><td>${dc(x.pit_net, sp)}</td><td>${dc(x.pit_tier, sp)}</td><td>${dc(x.pit_t, (v) => fixed(v, 1))}</td><td>${dc(x.indep, (v) => int(v))}</td></tr>`;
  }).join('');
  const mp = obj(obj(r.main_pit).pit);
  const g = (k) => ({ dev: obj(mp.dev)[k], confirm: obj(mp.confirm)[k] });
  const ss = obj(S.today && S.today.size_summary);
  const today = isNum(ss.deployable) ? `<dl class="tiles reveal">
      ${stat('Deployable today, top 10', esc(money(ss.deployable)), esc(`spread across ${int(ss.names)} names, each within its capacity and breakeven`))}
      ${stat('#1 capacity', esc(money(ss.top_capacity)), 'before moving the price')}
      ${stat('#1 breakeven size', esc(isNum(ss.top_breakeven) ? money(ss.top_breakeven) : DASH), esc(isNum(ss.avg_move_top1) ? `costs reach the historical #1's ${pct(ss.avg_move_top1, 1)} average move` : ''))}
      ${stat('Avg move, a top-10 pick', esc(pct(-ss.avg_move_top10, 1, true)), 'open→close before costs, walk-forward backtest')}
    </dl>` : '';
  root.innerHTML = `
    ${today}
    <p class="sec-q reveal">${esc(r.question || '')}</p>
    <ul class="notes verdict reveal">${arr(r.verdict).map((v) => `<li>${esc(v)}</li>`).join('')}</ul>
    <div class="rec-block reveal"><h3 class="rec-sub">The daily #1 on a point-in-time universe</h3><p class="note">Development / confirmation period. Net uses each pick's own estimated cost at $10K; "tight" assumes spreads near one tick.</p>
      <div class="table-scroll"><table class="mtable"><thead><tr><th scope="col">Period</th><th scope="col">Base dump</th><th scope="col">#1 dumped</th><th scope="col">#1 avg open→close</th><th scope="col">Median</th><th scope="col">Est. cost $10K</th><th scope="col">Net $10K</th><th scope="col">Net, tight spread</th><th scope="col">AUC</th></tr></thead>
      <tbody><tr><th scope="row">Dev / confirm</th><td>${dc(g('base'), (v) => pct(v, 1))}</td><td>${dc(g('hit'), (v) => pct(v, 0))}</td><td>${dc(g('mean_oc'), sp)}</td><td>${dc(g('median_oc'), sp)}</td><td>${dc(g('cost10k'), (v) => pct(v, 1))}</td><td>${dc(g('net10k'), sp)}</td><td>${dc(g('net10k_tight'), sp)}</td><td>${dc(g('auc'), (v) => fixed(v, 3))}</td></tr></tbody></table></div></div>
    <div class="rec-block reveal"><h3 class="rec-sub">Same-day shorts by size</h3><p class="note">Point-in-time universe, production cost model. Each cell: development / confirmation.</p>
      <div class="table-scroll"><table class="mtable"><thead><tr><th scope="col">Tier</th><th scope="col">Names/day</th><th scope="col">Base dump</th><th scope="col">#1 dumped</th><th scope="col">#1 avg open→close</th><th scope="col">Est. cost</th><th scope="col">Net / trade</th><th scope="col">Net, tight spread</th></tr></thead><tbody>${intr}</tbody></table></div></div>
    <div class="rec-block reveal"><h3 class="rec-sub">20-session shorts in liquid names — today's list vs point-in-time</h3>
      <div class="table-scroll"><table class="mtable"><thead><tr><th scope="col">Tier</th><th scope="col">Net / trade, today's list</th><th scope="col">Tier avg, today's list</th><th scope="col">Net / trade, point-in-time</th><th scope="col">Tier avg, point-in-time</th><th scope="col">t-stat</th><th scope="col">Independent trades</th></tr></thead><tbody>${mdRows}</tbody></table></div>
      <p class="note" style="margin-top:8px">On today's list the tier itself drifted down — the signature of stocks that had already collapsed into small-cap range. Point-in-time, the result is mixed and within noise.</p></div>
    <details class="rec-block reveal"><summary class="rec-sub">How this was tested</summary><ul class="caveats">${arr(r.method).map((m) => `<li>${esc(m)}</li>`).join('')}</ul></details>`;
}

/* ── method & footer ───────────────────────────────────────────────────── */
function renderMethod() {
  const top = S.today && S.today.top;
  const p = top && isNum(top.prob_dump) ? top.prob_dump : null;
  $$('[data-bind="example-prob"]').forEach((el) => { if (p != null) el.textContent = pct(p); });
  $$('[data-bind="example-n"]').forEach((el) => { if (p != null) el.textContent = String(Math.round(p * 100)); });
  $$('[data-bind="base-rate"]').forEach((el) => { el.textContent = isNum(S.base) ? String(Math.round(S.base * 100)) : DASH; });
}

function renderFooter() {
  const t = S.today || {};
  const m = S.model || {};
  const rows = [
    ['today.json', t.generated_at ? `${fmtStamp(t.generated_at, { year: true })} (${ago(t.generated_at)})` : (S.errors['today.json'] ? `not loaded — ${S.errors['today.json']}` : DASH)],
    ['Session', t.session_date ? fmtDay(t.session_date, { year: true }) : DASH],
    ['Features as of', t.features_asof ? `close ${fmtDay(t.features_asof, { year: true })}` : DASH],
    ['Model', [obj(t.model).version ? String(obj(t.model).version).toUpperCase() : null, obj(t.model).trained_through ? `trained through ${fmtDay(obj(t.model).trained_through, { weekday: false, year: true })}` : null, isNum(obj(t.model).m1_names) ? `M1 used for ${obj(t.model).m1_names} names` : null].filter(Boolean).join(' · ') || DASH],
    ['model.json', m.trained_at ? fmtStamp(m.trained_at, { year: true }) : (S.errors['model.json'] ? 'not loaded' : DASH)],
    ['scorecard.json', S.scorecard && S.scorecard.asof ? fmtStamp(S.scorecard.asof, { year: true }) : (S.errors['scorecard.json'] ? 'not loaded' : DASH)],
    ['evidence.json', S.evidence && S.evidence.generated_at ? fmtStamp(S.evidence.generated_at, { year: true }) : (S.errors['evidence.json'] ? 'not loaded' : DASH)],
    ['Universe', t.universe ? `${int(t.universe.listed)} listed · ${int(t.universe.eligible)} eligible · ${int(t.universe.scored)} scored` : DASH],
  ];
  if (S.live && S.live.asof) rows.push(['live.json', `${fmtStamp(S.live.asof, { year: true })}${liveFresh() ? '' : ' (previous session)'}`]);
  if (S.uni.asof) rows.push(['universe.json', fmtStamp(S.uni.asof, { year: true })]);
  if (S.sample) rows.push(['Mode', 'SAMPLE DATA — placeholder numbers']);
  setHTML('#foot-times', rows.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join('')
    + '<dt>Follow</dt><dd><a href="feed.xml">RSS feed of each day\'s #1</a> · on a phone, Share → Add to Home Screen installs GRAVITY as an app</dd>');
  if (t.disclaimer) { const d = $('#foot-disclaimer'); if (d) d.textContent = t.disclaimer; }
}

/* ── dossier ───────────────────────────────────────────────────────────── */
const METRICS = [
  ['r1', 'Last session', (v) => pct(v, 1, true), 'close to close'],
  ['r3', 'Last 3 sessions', (v) => pct(v, 0, true)],
  ['r5', 'Last 5 sessions', (v) => pct(v, 0, true)],
  ['r20', 'Last 20 sessions', (v) => pct(v, 0, true)],
  ['rvol1', 'Volume vs normal', (v) => (isNum(v) ? `${fixed(v, 1)}×` : DASH), 'vs 20-day avg'],
  ['rsi14', 'RSI (14)', (v) => fixed(v, 0), '80+ is stretched'],
  ['dist_ma20', 'vs 20-day average', (v) => pct(v, 0, true)],
  ['dist_ma50', 'vs 50-day average', (v) => pct(v, 0, true)],
  ['dd_52w', 'From 52-week high', (v) => pct(v, 0, true)],
  ['vol20', 'Typical daily swing', (v) => pct(v, 1), '20-day volatility'],
  ['dvol20', 'Dollars traded a day', money, '20-day avg'],
  ['float', 'Float', compact, 'shares'],
  ['shares_out', 'Shares outstanding', compact],
  ['si_pct_float', 'Short interest', (v) => pct(v, 1), '% of float'],
  ['days_to_cover', 'Days to cover', (v) => fixed(v, 1)],
  ['short_ratio_5d', 'Short share of volume', (v) => pct(v, 0), 'FINRA, 5 days'],
  ['rs_count_2y', 'Reverse splits', int, 'last 2 years'],
  ['n_offer_90', 'Offerings filed', int, 'last 90 days'],
  ['n_reg_90', 'Registrations filed', int, 'last 90 days'],
];

function kv(rows) {
  return `<dl class="kv">${rows.map(([k, v, small]) => `<div><dt>${esc(k)}</dt><dd>${v}${small ? ` <small>${esc(small)}</small>` : ''}</dd></div>`).join('')}</dl>`;
}

function dzSec(title, html) {
  return `<section class="dz-sec"><h3>${esc(title)}</h3>${html}</section>`;
}

function dossierHTML(rec) {
  const p = rec.pick;
  const t = S.today || {};
  const sym = esc(p.symbol);
  const base = S.base;
  let where = 'Dossier';
  if (rec.where === 'board') where = `Board #${isNum(p.rank) ? p.rank : DASH}`;
  else if (rec.where === 'zone') where = 'Squeeze zone — held off the board';
  else if (rec.where === 'twin') where = `Lookalike of ${twinsAnchor() || 'the #1'}`;
  else if (rec.where === 'swing') where = `Swing board #${isNum(p.rank) ? p.rank : DASH}`;
  if (rec.twin && rec.where !== 'twin') where += ` · lookalike of ${twinsAnchor() || 'the #1'}`;
  const sub = [p.name, p.exchange, p.country, p.industry || p.sector].filter(Boolean).map(esc).join(' · ');

  const pre = p.premarket;
  const gap = pre && isNum(pre.gap_pct) ? pct(pre.gap_pct, 0, true) : DASH;
  const keys = `<dl class="dz-keys">
    <div><dt class="eyebrow">Dump odds</dt><dd class="${isNum(p.prob_dump) ? 'red' : ''}">${esc(pct(p.prob_dump))}<small>${esc(isNum(base) ? `base ${pct(base, 1)} · ${liftTxt(p.lift)}` : 'base rate not reported')}</small></dd></div>
    <div><dt class="eyebrow">Swing odds</dt><dd>${esc(pct(p.prob_swing))}<small>15%+ lower five sessions out</small></dd></div>
    <div><dt class="eyebrow">Pre-market</dt><dd>${esc(gap)}<small>${esc(pre ? `${price(pre.price)} at ${fmtTimeET(pre.asof)} ET` : 'no print reported')}</small></dd></div>
    <div><dt class="eyebrow">Squeeze danger</dt><dd>${squeezeMeter(p.squeeze_danger)}<small>${esc(sqWord(p.squeeze_danger))}</small></dd></div>
  </dl>`;

  const chartRows = arr(p.chart).filter((r) => Array.isArray(r) && isNum(r[4]));
  const lb = chartRows.length ? chartRows[chartRows.length - 1] : null;
  const chart = dzSec(`Price${chartRows.length ? ` — ${chartRows.length} sessions to ${fmtDay(lb[0], { weekday: false })}` : ''}`,
    `<div class="lwc" id="dz-chart" role="img" aria-label="${esc(`${p.symbol} daily candles and volume${lb ? ` to ${fmtDay(lb[0], { weekday: false, year: true })}` : ''}`)}"></div><p class="note" style="margin-top:8px">Split-adjusted daily bars. Hollow = closed up, red = closed down; volume along the bottom.</p>`);

  const parts = [];
  if (p._noModel) parts.push(`<p class="dz-note">No model reading for ${sym} in this feed${S.sample ? ' (sample mode never scores a real company)' : ''}. Price, borrow, filings and news are shown where available.</p>`);
  if (rec.where === 'zone' && p.zone_reason) parts.push(`<p class="dz-warn"><b>Why it's off the board</b>${esc(p.zone_reason)}</p>`);
  if (arr(p.reasons).length) parts.push(dzSec('Why it is on the radar', reasonsList(p.reasons)));
  const weighed = attributionBars(p.attribution);
  if (weighed) parts.push(dzSec('What the model weighed', `${weighed}<p class="note" style="margin-top:8px">Share of this reading that disappears when each family of inputs is swapped for a typical name's values. It describes the model, not the company.</p>`));
  const hist = historyHTML(p);
  if (hist) parts.push(dzSec('Over time', `${hist}<p class="note" style="margin-top:8px">Recorded by GRAVITY each session since it started; nothing back-filled.</p>`));
  if (p.dilution) parts.push(dzSec('Dilution & cash', dilutionHTML(p.dilution)));
  if (p.insider) parts.push(dzSec('Insider activity · 90 days', insiderHTML(p.insider)));
  if (p.chatter) parts.push(dzSec('Chatter', chatterHTML(p.chatter)));
  parts.push(dzSec('Flags', flagChips(p.flags)));
  if (p.families) {
    parts.push(dzSec('Signal families', `<ul class="famrows">${FAMILIES.map((k) => {
      const v = obj(p.families)[k];
      const has = isNum(v);
      return `<li class="${has ? '' : 'null'}" title="${esc(FAMILY_HELP[k] || '')}"><span>${esc(FAMILY_LABEL[k])}</span><span class="b" aria-hidden="true">${has ? `<i style="width:${Math.max(0, Math.min(100, v))}%"></i>` : ''}</span><span class="v">${has ? Math.round(v) : DASH}</span></li>`;
    }).join('')}</ul><p class="note" style="margin-top:8px">0–100 within today's universe. ${FAMILIES.map((k) => `${esc(FAMILY_LABEL[k])}: ${esc(FAMILY_HELP[k])}.`).join(' ')}</p>`));
  }
  if (p.metrics) {
    const m = obj(p.metrics);
    const rows = METRICS.filter(([k]) => k in m).map(([k, label, f, hint]) => [label, esc(f(m[k])), hint]);
    const si = obj(m.short_interest);
    if (isNum(si.interest)) rows.push(['Shares sold short', esc(compact(si.interest)), si.settlement_date ? `settled ${fmtDay(si.settlement_date, { weekday: false })}` : '']);
    parts.push(dzSec('The numbers, in plain English', kv(rows)));
  }
  const sh = obj(p.shortability);
  parts.push(dzSec('Borrow & squeeze', kv([
    ['Borrow status', esc(STATUS_LABEL[sh.status] || DASH), 'IBKR'],
    ['Borrow fee', esc(isNum(sh.fee_rate) ? `${pctUnits(sh.fee_rate, 1)}/yr` : DASH), isNum(sh.fee_rate) ? `≈ ${pctUnits(sh.fee_rate / 12, 1)} a month` : ''],
    ['Shares available', esc(compact(sh.available))],
    ['As of', esc(sh.asof ? fmtStamp(sh.asof) : DASH)],
    ['Squeeze danger', esc(isNum(p.squeeze_danger) ? `${p.squeeze_danger} / 100` : DASH), sqWord(p.squeeze_danger)],
    ['Odds of a +20% spike', esc(pct(p.prob_squeeze)), 'open→high, model'],
  ]) + (p.squeeze_parts ? `<p class="eyebrow" style="margin:24px 0 8px">What drives the squeeze score</p><div style="max-width:480px">${partsBars(p.squeeze_parts)}</div>` : '')));
  if (p.size) parts.push(dzSec('Size & cost', sizeDossier(p)));
  parts.push(dzSec('SEC filings', filingTable(p.filings)));
  parts.push(dzSec('Headlines', newsList(p.news)));

  const st = obj(p.street);
  const a = st.analyst;
  const dn = st.danelfin;
  let street = '';
  if (a) {
    street += kv([
      ['Consensus', txt(a.mean_rating)],
      ['Analysts', esc(int(a.n_analysts))],
      ['Price target', esc(price(a.price_target))],
    ]);
    if (arr(a.changes).length) street += `<ul class="lst" style="margin-top:16px">${arr(a.changes).slice(0, 6).map((c) => `<li><span class="meta">${esc(fmtDay(c.date, { weekday: false, year: true }))} · ${txt(c.firm)}</span>${txt(c.action)}${c.from || c.to ? ` <span class="muted">${esc([c.from, c.to].filter(Boolean).join(' → '))}</span>` : ''}</li>`).join('')}</ul>`;
    if (safeUrl(a.source_url)) street += `<p class="note" style="margin-top:8px">${extLink(a.source_url, 'Source ↗')}</p>`;
  } else {
    street += '<p class="note">No analyst coverage in this feed.</p>';
  }
  if (dn) {
    street += `<p class="eyebrow" style="margin:24px 0 8px">Danelfin (your API key)${dn.date ? ` · ${esc(fmtDay(dn.date, { weekday: false }))}` : ''}</p>${kv([
      ['AI score', esc(isNum(dn.ai_score) ? `${dn.ai_score} / 10` : DASH)],
      ['Technical', esc(isNum(dn.technical) ? dn.technical : DASH)],
      ['Fundamental', esc(isNum(dn.fundamental) ? dn.fundamental : DASH)],
      ['Sentiment', esc(isNum(dn.sentiment) ? dn.sentiment : DASH)],
      ['Low risk', esc(isNum(dn.low_risk) ? dn.low_risk : DASH)],
    ])}`;
  } else {
    street += '<p class="note" style="margin-top:8px">Danelfin, Zacks and Bloomberg ratings are proprietary: open them with the links below.</p>';
  }
  parts.push(dzSec('Street', street));
  parts.push(dzSec('Research elsewhere', linksRow(p)));
  if (!p._noModel) {
    parts.push(dzSec('About this reading', kv([
      ['Model', esc(p.model_used === 'm1' ? 'M1 · uses pre-market' : p.model_used === 'm0' ? 'M0 · close only' : DASH)],
      ['Odds of a 15%+ drop', esc(pct(p.prob_bigdump, 1)), 'open→close'],
      ['Higher odds than', esc(isNum(p.score) ? (p.score >= 100 ? 'every other name' : `${p.score}% of names`) : DASH), 'scored today'],
      ['Odds of a +5% rise instead', esc(pct(p.prob_pump, 1)), 'open→close'],
      ['Skew', esc(isNum(p.skew) ? `${pct(p.skew)} down` : DASH), 'share of this session\'s ±5% odds pointing down'],
      ['Expected open→close', esc(pct(p.exp_oc, 1, true)), 'weak on its own — see the Record'],
      ['Features as of', esc(fmtDay(t.features_asof, { weekday: false, year: true }))],
    ])));
  }

  return `
    <header class="dz-head">
      <div>
        <p class="eyebrow">${esc(where)}${t.session_date ? ` · ${esc(fmtDay(t.session_date))}` : ''}</p>
        <h2 class="dz-ticker" id="dossier-title">${sym}</h2>
        <p class="dz-name">${sub || DASH}</p>
      </div>
      <button class="dz-close" type="button" data-close>Close <kbd>Esc</kbd></button>
    </header>
    <div class="dz-body">
      ${keys}
      ${chart}
      ${parts.join('')}
      <p class="note" style="margin-top:32px">Research, not advice. Probabilities describe how similar setups behaved; this session can go either way.</p>
    </div>`;
}

const dz = { chart: null, disposers: [], opener: null, sym: null, token: 0 };
let lwcPromise = null;

function loadLWC() {
  if (window.LightweightCharts) return Promise.resolve(window.LightweightCharts);
  if (!lwcPromise) {
    lwcPromise = new Promise((resolve, reject) => {
      const s = document.createElement('script');
      s.src = LWC_URL;
      s.async = true;
      s.crossOrigin = 'anonymous';
      s.integrity = LWC_SRI;
      s.referrerPolicy = 'no-referrer';
      s.onload = () => (window.LightweightCharts ? resolve(window.LightweightCharts) : reject(new Error('chart library missing')));
      s.onerror = () => { lwcPromise = null; s.remove(); reject(new Error('chart library failed to load')); };
      document.head.appendChild(s);
    });
  }
  return lwcPromise;
}

function candleChart(el, LWC, rows, rec) {
  el.innerHTML = '';
  const last = rows[rows.length - 1][4];
  const prec = last >= 1 ? 2 : last >= 0.1 ? 3 : 4;
  const line = 'rgba(214, 222, 235, 0.14)';
  const chart = LWC.createChart(el, {
    autoSize: true,
    layout: { background: { type: 'solid', color: 'transparent' }, textColor: '#858e99', fontFamily: 'JetBrains Mono, ui-monospace, Menlo, monospace', fontSize: 11 },
    grid: { vertLines: { color: 'rgba(214, 222, 235, 0.04)' }, horzLines: { color: 'rgba(214, 222, 235, 0.06)' } },
    rightPriceScale: { borderColor: line },
    timeScale: { borderColor: line, rightOffset: 2, fixLeftEdge: true, fixRightEdge: true },
    crosshair: {
      mode: 0,
      vertLine: { color: 'rgba(214, 222, 235, 0.3)', labelBackgroundColor: '#1d2228' },
      horzLine: { color: 'rgba(214, 222, 235, 0.3)', labelBackgroundColor: '#1d2228' },
    },
    handleScroll: { vertTouchDrag: false },
    handleScale: { axisPressedMouseMove: { time: true, price: false } },
  });
  const candles = chart.addCandlestickSeries({
    upColor: 'rgba(0, 0, 0, 0)', borderUpColor: '#b4bcc6', wickUpColor: '#b4bcc6',
    downColor: '#ff2e4d', borderDownColor: '#ff2e4d', wickDownColor: '#ff2e4d',
    priceLineColor: 'rgba(214, 222, 235, 0.35)',
    priceFormat: { type: 'price', precision: prec, minMove: 10 ** -prec },
  });
  candles.priceScale().applyOptions({ scaleMargins: { top: 0.08, bottom: 0.26 } });
  candles.setData(rows.map((r) => ({ time: r[0], open: r[1], high: r[2], low: r[3], close: r[4] })));
  const vol = chart.addHistogramSeries({ priceFormat: { type: 'volume' }, priceScaleId: 'vol', lastValueVisible: false, priceLineVisible: false });
  chart.priceScale('vol').applyOptions({ scaleMargins: { top: 0.8, bottom: 0 } });
  vol.setData(rows.map((r) => (isNum(r[5])
    ? { time: r[0], value: r[5], color: r[4] >= r[1] ? 'rgba(180, 188, 198, 0.28)' : 'rgba(255, 46, 77, 0.4)' }
    : { time: r[0] })));
  const ref = null;
  if (ref && ref.short_ref_date) {
    let idx = -1;
    for (let i = 0; i < rows.length; i++) if (rows[i][0] <= ref.short_ref_date) idx = i;
    if (idx >= 0) candles.setMarkers([{ time: rows[idx][0], position: 'aboveBar', color: '#ff2e4d', shape: 'arrowDown', text: 'Short ref' }]);
    if (isNum(ref.short_ref_price)) {
      candles.createPriceLine({ price: ref.short_ref_price, color: 'rgba(255, 46, 77, 0.6)', lineWidth: 1, lineStyle: 2, axisLabelVisible: true, title: 'ref' });
    }
  }
  chart.timeScale().fitContent();
  return chart;
}

async function mountDossierChart(rec) {
  const el = $('#dz-chart');
  if (!el) return;
  const rows = arr(rec.pick.chart).filter((r) => Array.isArray(r) && typeof r[0] === 'string' && [1, 2, 3, 4].every((i) => isNum(r[i])));
  if (rows.length < 2) {
    el.classList.add('lwc-fallback');
    el.removeAttribute('role');
    el.innerHTML = empty('No price history in this feed.');
    return;
  }
  const token = ++dz.token;
  const fallback = () => {
    el.classList.add('lwc-fallback', 'chart');
    const ref = rec.ref;
    dz.disposers.push(priceChart(el, rows, {
      marker: ref && ref.short_ref_date ? { date: ref.short_ref_date, label: 'Short ref' } : null,
      ref: ref && isNum(ref.short_ref_price) ? { price: ref.short_ref_price } : null,
      height: 300,
      label: `${rec.pick.symbol} daily closes`,
    }));
  };
  try {
    const LWC = await loadLWC();
    if (token !== dz.token || !el.isConnected) return;
    dz.chart = candleChart(el, LWC, rows, rec);
  } catch (e) {
    if (token !== dz.token || !el.isConnected) return;
    fallback();
  }
}

function teardownDossier() {
  dz.token++;
  if (dz.chart) {
    try { dz.chart.remove(); } catch { /* gone */ }
    dz.chart = null;
  }
  dispose(dz.disposers);
}

function openDossier(sym, { push = true, opener = null } = {}) {
  const dlg = $('#dossier');
  const rec = S.recs.get(sym);
  if (!dlg || !rec) return false;
  if (dz.sym === sym && dlg.open) return true;
  teardownDossier();
  if (!dlg.open) dz.opener = opener || document.activeElement;
  dz.sym = sym;
  dlg.innerHTML = dossierHTML(rec);
  if (!dlg.open) {
    if (typeof dlg.showModal === 'function') dlg.showModal();
    else dlg.setAttribute('open', '');
    document.documentElement.classList.add('dz-lock');
  }
  const body = $('.dz-body', dlg);
  if (body) body.scrollTop = 0;
  const close = $('[data-close]', dlg);
  if (close) close.focus();
  if (push && location.hash !== `#/${encodeURIComponent(sym)}`) {
    try { history.pushState({ dz: sym }, '', `#/${encodeURIComponent(sym)}`); } catch { /* sandboxed */ }
  }
  mountDossierChart(rec);
  return true;
}

function finalizeClose() {
  const dlg = $('#dossier');
  teardownDossier();
  dz.sym = null;
  if (dlg && dlg.open) {
    if (typeof dlg.close === 'function') dlg.close();
    else dlg.removeAttribute('open');
  }
  document.documentElement.classList.remove('dz-lock');
  const o = dz.opener;
  dz.opener = null;
  if (o && o.isConnected && typeof o.focus === 'function') o.focus({ preventScroll: true });
}

/** Leave the #/SYM entry: step back over the entry we pushed, else just strip the hash. */
function leaveDossierUrl() {
  if (history.state && history.state.dz) {
    history.back();
  } else if (location.hash.startsWith('#/')) {
    try { history.replaceState(null, '', location.pathname + location.search); } catch { /* sandboxed */ }
  }
}

/** Close button / backdrop. Esc closes natively; every path ends in onDialogClose(). */
function closeDossier() {
  const dlg = $('#dossier');
  if (dlg && dlg.open && typeof dlg.close === 'function') dlg.close();
  onDialogClose(); // don't wait for the async "close" event (it can be deferred in hidden tabs)
}

function onDialogClose() {
  if (!dz.sym) return; // already finalized (e.g. closed by route())
  finalizeClose();
  leaveDossierUrl();
}

function route() {
  const m = /^#\/([^/?#]+)$/.exec(location.hash);
  let sym = null;
  if (m) {
    try { sym = decodeURIComponent(m[1]).toUpperCase(); } catch { sym = null; }
  }
  if (sym && S.recs.has(sym)) {
    openDossier(sym, { push: false });
    return;
  }
  if (dz.sym) finalizeClose();
  if (m) {
    try { history.replaceState(null, '', location.pathname + location.search); } catch { /* sandboxed */ }
  }
}

function trapFocus(e) {
  if (e.key !== 'Tab') return;
  const dlg = e.currentTarget;
  const f = $$('a[href], button:not([disabled]), input:not([disabled]), select, textarea, [tabindex]:not([tabindex="-1"])', dlg)
    .filter((n) => n.offsetParent !== null || n === document.activeElement);
  if (!f.length) return;
  const first = f[0];
  const last = f[f.length - 1];
  if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); } else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
}

/* ── global wiring ─────────────────────────────────────────────────────── */
function wireGlobal() {
  document.addEventListener('click', (e) => {
    const t = e.target;
    if (!(t instanceof Element)) return;
    const open = t.closest('[data-open]');
    if (open) { e.preventDefault(); openDossier(open.getAttribute('data-open'), { opener: open }); return; }
    if (t.closest('[data-close]')) { closeDossier(); return; }
    const sortB = t.closest('[data-sort]');
    if (sortB) { setSort(sortB.getAttribute('data-sort')); return; }
    const fb = t.closest('[data-filter]');
    if (fb) {
      const id = fb.getAttribute('data-filter');
      if (S.filters.has(id)) S.filters.delete(id); else S.filters.add(id);
      fb.setAttribute('aria-pressed', String(S.filters.has(id)));
      drawBoard();
      return;
    }
    const row = t.closest('#board-table tbody tr[data-sym]');
    if (row && !t.closest('a, button, input, select')) { openDossier(row.getAttribute('data-sym'), { opener: $('.tk', row) }); return; }
    const tab = t.closest('[data-tab]');
    if (tab) { S.tab = tab.getAttribute('data-tab'); renderRecord(); revealAll($('#record-root')); const nt = $(`[data-tab="${S.tab}"]`); if (nt) nt.focus(); return; }
    if (t.closest('[data-wire-all]')) {
      setHTML('#wire-list', sortedWire(S.today).map(wireItem).join(''));
      t.closest('.more').remove();
      return;
    }
    if (t.closest('[data-days-all]')) {
      const days = arr(S.scorecard && S.scorecard.days).filter(Boolean);
      setHTML('#days-table tbody', days.map(dayRow).join(''));
      t.closest('.more').remove();
      return;
    }
    if (t.closest('[data-reload]')) { location.reload(); return; }
    if (t.closest('#feeds-btn')) { toggleFeeds(); return; }
    const pop = $('#feeds-pop');
    if (pop && !pop.hidden && !t.closest('#feeds-pop')) toggleFeeds(false);
  });

  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      const pop = $('#feeds-pop');
      if (pop && !pop.hidden) { toggleFeeds(false); const b = $('#feeds-btn'); if (b) b.focus(); }
    }
    const tab = e.target instanceof Element && e.target.closest('[role="tab"]');
    if (tab && (e.key === 'ArrowRight' || e.key === 'ArrowLeft')) {
      const tabs = $$('[role="tab"]', tab.parentElement);
      const i = tabs.indexOf(tab);
      const next = tabs[(i + (e.key === 'ArrowRight' ? 1 : tabs.length - 1)) % tabs.length];
      if (next && next !== tab) { e.preventDefault(); next.click(); }
    }
  });

  document.addEventListener('change', (e) => {
    if (e.target && e.target.id === 'board-sort') setSort(e.target.value, true);
  });

  const dlg = $('#dossier');
  if (dlg) {
    dlg.addEventListener('keydown', trapFocus);
    dlg.addEventListener('click', (e) => {
      if (e.target !== dlg) return;
      const r = dlg.getBoundingClientRect();
      const inside = e.clientX >= r.left && e.clientX <= r.right && e.clientY >= r.top && e.clientY <= r.bottom;
      if (!inside) closeDossier();
    });
    dlg.addEventListener('close', onDialogClose);
    // Esc closes natively right after "cancel"; finish synchronously instead of waiting for "close".
    dlg.addEventListener('cancel', () => { queueMicrotask(() => { if (!dlg.open) onDialogClose(); }); });
  }

  window.addEventListener('popstate', route);
  window.addEventListener('hashchange', route);
  document.addEventListener('visibilitychange', () => { if (!document.hidden) tickBar(); });
}

/* ── reveal on scroll ──────────────────────────────────────────────────── */
let revealIO = null;
function revealAll(root = document) {
  $$('.reveal:not(.is-in)', root).forEach((el) => el.classList.add('is-in'));
}
function observeReveal(root = document) {
  const els = $$('.reveal:not(.is-in)', root);
  if (!('IntersectionObserver' in window) || prefersReducedMotion()) { els.forEach((el) => el.classList.add('is-in')); return; }
  if (!revealIO) {
    revealIO = new IntersectionObserver((entries) => {
      entries.forEach((en) => {
        if (en.isIntersecting) { en.target.classList.add('is-in'); revealIO.unobserve(en.target); }
      });
    }, { rootMargin: '0px 0px -6% 0px', threshold: 0.01 });
  }
  els.forEach((el) => revealIO.observe(el));
}

/* ── freshness ─────────────────────────────────────────────────────────── */
function watchFreshness() {
  setInterval(tickBar, 30e3);
  setInterval(async () => {
    if (document.hidden || S.newer || !S.lastModified) return;
    try {
      const r = await fetch('data/today.json', { method: 'HEAD', cache: 'no-cache', credentials: 'omit' });
      const lm = r.ok ? r.headers.get('last-modified') : null;
      if (lm && lm !== S.lastModified) { S.newer = true; renderBanners(); }
    } catch { /* offline — try again later */ }
  }, 5 * 60e3);
}

/* ── boot ──────────────────────────────────────────────────────────────── */
async function init() {
  wireGlobal();
  S.size = loadSize();
  const [today, model, evidence, scorecard, live, sizeLab] = await Promise.all(
    ['today.json', 'model.json', 'evidence.json', 'scorecard.json', 'live.json', 'size_research.json'].map(getJSON),
  );
  delete S.errors['live.json'];  // optional feed: absent outside the session
  S.sizeLab = sizeLab && typeof sizeLab === 'object' ? sizeLab : null;
  S.live = live && typeof live === 'object' ? live : null;
  S.today = today && typeof today === 'object' ? today : null;
  S.model = model && typeof model === 'object' ? model : null;
  S.evidence = evidence && typeof evidence === 'object' ? evidence : null;
  S.scorecard = scorecard && typeof scorecard === 'object' ? scorecard : null;
  S.sample = [S.today, S.model, S.evidence, S.scorecard].some((f) => f && f.sample === true);
  const b1 = S.today && S.today.model ? S.today.model.base_rate_dump : null;
  const b2 = S.model && S.model.base_rate ? S.model.base_rate.dump : null;
  S.base = isNum(b1) ? b1 : isNum(b2) ? b2 : null;
  document.documentElement.classList.toggle('is-sample', S.sample);

  safe('index', indexRecords);
  safe('bar', renderBar);
  safe('banners', renderBanners);
  safe('hero', renderHero);
  safe('board', renderBoard);
  safe('live', renderLive);
  safe('wire', renderWire);
  safe('swing', renderSwing);
  safe('twins', renderTwins);
  safe('squeeze', renderSqueeze);
  safe('sizelab', renderSizeLab);
  safe('market', renderMarket);
  safe('calendar', renderCalendar);
  safe('lookup', renderLookup);
  safe('record', renderRecord);
  safe('evidence', renderEvidence);
  safe('method', renderMethod);
  safe('footer', renderFooter);
  observeReveal();
  route();
  watchFreshness();
  startLivePoll();
  document.documentElement.classList.add('is-ready');
}

if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init, { once: true });
else init();
