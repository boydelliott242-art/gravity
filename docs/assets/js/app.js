/**
 * GRAVITY — site controller.
 *
 * Reads the four static feeds (today, model, evidence, scorecard) written by
 * the pipeline (CONTRACTS.md §11–13), renders every section, and wires the
 * board, the dossier drawer and the private INHD calculator.
 *
 * Honesty: every missing value renders as "—"; nothing is imputed, rounded
 * into a different claim, or invented here. Probabilities always sit next to
 * the base rate they should be read against.
 */

import {
  DASH, isNum, esc, safeUrl, extLink, pct, pctUnits, price, compact, money, fixed, signedMoney,
  parseDay, fmtDay, parseTs, fmtTimeET, fmtStamp, etDate, ago, daysBetween, marketPhaseNow,
  CAT_LABEL, SUPPLY_CATS, humanize, ASIA, $, $$, prefersReducedMotion, fallbackLinks, store,
} from './util.js';
import {
  sparkline, probBar, familyBars, squeezeMeter, severity, priceChart, equityChart,
  calibrationChart, whisker, FAMILIES, FAMILY_LABEL, FAMILY_HELP,
} from './charts.js';
import { mountField } from './field.js';

const LWC_URL = 'https://unpkg.com/lightweight-charts@4.2.0/dist/lightweight-charts.standalone.production.js';
const LWC_SRI = 'sha384-OK7vELvjHdhUFi31JYioPIcRHTROLdcDa6ZsNWgvgLaKj+9JqhU0Ad8g4wz3CXjA';
const CALC_KEY = 'gravity:inhd-calc:v1';
const WIRE_SHOW = 14;
const DAYS_SHOW = 20;

const S = {
  today: null,
  model: null,
  evidence: null,
  scorecard: null,
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
  disposers: { record: [], position: [] },
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
  return ` Backtest of this exact rule (no Rule-201 names, ≥$300K daily volume): the #1 fell 5%+ open→close on ${Math.round(hit * 100)}% of sessions${isNum(sqr) ? `, spiked 20%+ above the open on ${Math.round(sqr * 100)}%` : ''}${isNum(avg) ? `, averaging ${pct(avg, 1, true)} open→close` : ''} — a tilt in the odds, never a sure thing.`;
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

function refRecord(ref) {
  const p = ref.pick ? { ...ref.pick } : {
    symbol: ref.symbol, name: ref.name, price: ref.price, prev_close: ref.prev_close,
    shortability: ref.borrow, flags: [], reasons: [], families: null, metrics: null,
    links: null, premarket: null, street: null,
  };
  if (!arr(p.chart).length) p.chart = arr(ref.chart);
  if (!arr(p.filings).length) p.filings = arr(ref.filings);
  if (!arr(p.news).length) p.news = arr(ref.news);
  if (!p.shortability) p.shortability = ref.borrow;
  p._noModel = !ref.pick;
  return p;
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
  arr(t.twins).forEach((tw) => { if (tw && tw.pick) put(tw.pick, { where: 'twin' }); });
  arr(t.twins).forEach((tw) => {
    const r = tw && S.recs.get(tw.symbol);
    if (r) r.twin = tw;
  });
  const ref = t.reference;
  if (ref && ref.symbol) {
    const r = S.recs.get(ref.symbol);
    if (r) {
      r.ref = ref;
      if (!arr(r.pick.chart).length) r.pick = { ...r.pick, chart: arr(ref.chart) };
    } else {
      S.recs.set(ref.symbol, { pick: refRecord(ref), where: 'ref', ref });
    }
  }
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

  setHTML(body, `
    <dl class="hstats reveal">
      ${stat('Squeeze odds', esc(pct(top.prob_squeeze, 0)), esc(isNum(obj(obj(S.model).base_rate).squeeze) ? `P(+20% above the open) · ${fixed(top.prob_squeeze / S.model.base_rate.squeeze, 1)}× normal — dump odds and squeeze odds rise together` : 'P(+20% above the open), same session'))}
      ${stat('Pre-market gap', esc(gap), preNote)}
      ${stat('Borrow', esc(sh.status === 'NONE' ? 'None' : sh.status === 'ETB' || sh.status === 'HTB' ? sh.status : DASH), shNote)}
      ${stat('Squeeze danger', squeezeMeter(top.squeeze_danger, { large: true }), esc(sqNote))}
    </dl>
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
        <div class="side-block"><p class="eyebrow">Research ${sym} elsewhere</p>${linksRow(top)}</div>
        <div class="side-block">
          <p class="eyebrow">About this reading</p>
          <p class="note">${esc(modelLine)}. Features as of the close on ${esc(fmtDay(t.features_asof, { year: true }))}.${isNum(top.score) ? (top.score >= 100 ? (t.top_tie_count > 1 ? ` Tied with ${esc(int(t.top_tie_count - 1))} other name${t.top_tie_count > 2 ? 's' : ''} for the highest odds of the ${esc(int(scored))} scored — ties are ordered by the model's raw score.` : ` The highest odds of the ${esc(int(scored))} names scored today.`) : ` Higher odds than ${esc(top.score)}% of the ${esc(int(scored))} names scored today.`) : ''}${isNum(top.inhd_similarity) ? ` ${esc(pct(top.inhd_similarity))} similar to INHD.` : ''}${backtestLine(top.model_used)}${isNum(top.rank) && top.rank > 1 ? ` The ${top.rank - 1} higher-ranked board name${top.rank > 2 ? 's are' : ' is'} skipped because ${top.rank > 2 ? 'they are' : 'it is'} under the Rule 201 short-sale restriction today or trade under $300K a day — the rule the backtest measured.` : ''}</p>
        </div>
      </aside>
    </div>`);
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
];

const SORTS = {
  rank: { label: 'Rank', get: (p) => p.rank, dir: 1 },
  prob: { label: 'Dump odds', get: (p) => p.prob_dump, dir: -1 },
  lift: { label: 'Lift', get: (p) => p.lift, dir: -1 },
  gap: { label: 'Pre-market gap', get: (p) => (p.premarket ? p.premarket.gap_pct : null), dir: -1 },
  fee: { label: 'Borrow fee', get: (p) => obj(p.shortability).fee_rate, dir: 1 },
  sq: { label: 'Squeeze danger', get: (p) => p.squeeze_danger, dir: 1 },
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
  const meta = `<div class="mcard-meta">${probBar(p.prob_dump, S.base, max)}<span>Lift ${esc(liftTxt(p.lift))}</span><span>Gap ${esc(gap)}</span>${borrowInline(p.shortability)}<span>Squeeze ${esc(isNum(p.squeeze_danger) ? p.squeeze_danger : DASH)}</span></div>`;
  return `<tr data-sym="${sym}">
    <td class="c-rank">${isNum(p.rank) ? pad2(p.rank) : DASH}</td>
    <td class="c-name"><button type="button" class="tk" data-open="${sym}">${sym}</button><span class="nm">${txt(nm)}</span>${arr(p.flags).length ? flagChips(p.flags, 4) : ''}</td>
    <td class="c-prob"><span class="pv">${esc(pct(p.prob_dump))}</span>${probBar(p.prob_dump, S.base, max)}</td>
    <td class="c-lift">${esc(liftTxt(p.lift))}</td>
    <td class="c-fam">${familyBars(p.families)}</td>
    <td class="c-gap">${esc(gap)}</td>
    <td class="c-borrow">${borrowChip(p.shortability)}<span class="borrow__txt">${esc(borrowText(p.shortability))}</span></td>
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
      <p class="board-count" id="board-count" aria-live="polite"></p>
    </div>
    <div class="board-scroll"><table class="board" id="board-table"><caption class="sr-only">Today's board, ranked by modeled dump odds. Select a ticker for its dossier.</caption><thead></thead><tbody></tbody></table></div>`;
  drawBoard();
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
  if (!board.length) body = '<tr class="board-empty"><td colspan="9">No names made the board this session — see the squeeze zone for what was held back.</td></tr>';
  else if (!rows.length) body = '<tr class="board-empty"><td colspan="9">No names match every active filter. Clear a filter to see more.</td></tr>';
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

/* ── INHD position + calculator ────────────────────────────────────────── */
function renderPosition() {
  const root = $('#position-root');
  if (!root) return;
  dispose(S.disposers.position);
  const t = S.today;
  const ref = t && t.reference;
  if (!ref) { root.innerHTML = empty('No reference position in this feed.'); return; }
  const sym = esc(ref.symbol || 'INHD');
  const pick = ref.pick;
  const sh = obj(ref.borrow);
  const lb = lastBar(ref.chart);
  const chg = ref.change_since_ref;
  const refDay = fmtDay(ref.short_ref_date, { weekday: false, year: true });
  const odds = pick
    ? stat('Dump odds today', esc(pct(pick.prob_dump)), esc(isNum(ref.rank) ? `Board #${ref.rank}` : 'Not on the board'))
    : stat('Dump odds today', DASH, 'No model reading in this feed');
  root.innerHTML = `
    <div class="pos-grid">
      <div class="reveal">
        ${S.sample ? '<p class="placeholder-tag">Placeholder values — sample mode</p>' : ''}
        <p class="pos-ticker"><button type="button" data-open="${sym}">${sym}<span class="sr-only">, open the dossier</span></button></p>
        <p class="pos-name">${txt(ref.name)}</p>
        <p class="pos-change">${esc(pct(chg, 0, true))}</p>
        <p>Price change since your short reference on ${esc(refDay)} (${esc(price(ref.short_ref_price))} → ${esc(price(ref.price))}${lb ? `, close ${esc(fmtDay(lb[0], { weekday: false }))}` : ''}). A fall is a gain for a short.</p>
      </div>
      <div class="chart-box reveal">
        <div class="chart" id="pos-chart"></div>
        ${S.sample ? '<div class="ph-over" aria-hidden="true">Placeholder</div>' : ''}
      </div>
    </div>
    <dl class="stats-row reveal">
      ${stat('Last close', esc(price(ref.price)), esc(lb ? fmtDay(lb[0], { weekday: false }) : ''))}
      ${stat('Short reference', esc(price(ref.short_ref_price)), esc(fmtDay(ref.short_ref_date, { weekday: false })))}
      ${stat('Short gain', esc(isNum(chg) ? pct(-chg, 1, true) : DASH), 'Per share, before fees')}
      ${stat('Borrow fee', esc(isNum(sh.fee_rate) ? `${pctUnits(sh.fee_rate, 1)}/yr` : DASH), esc(STATUS_LABEL[sh.status] || 'Status unknown'))}
      ${stat('Available', esc(compact(sh.available)), esc(`Shares at IBKR${sh.asof ? ` · ${fmtStamp(sh.asof)}` : ''}`))}
      ${odds}
    </dl>
    <div class="pos-lower">
      <div class="reveal">
        <div class="side-block"><p class="eyebrow">Notes</p>${arr(ref.notes).length ? `<ul class="notes">${arr(ref.notes).map((n) => `<li>${esc(n)}</li>`).join('')}</ul>` : empty('No notes this run.')}</div>
        <div class="side-block"><p class="eyebrow">Filings</p>${filingTable(ref.filings)}</div>
      </div>
      <div class="reveal">
        <div class="side-block"><p class="eyebrow">Headlines</p>${newsList(ref.news)}</div>
        <div class="side-block"><p class="eyebrow">Research ${sym}</p>${linksRow(pick || { symbol: ref.symbol || 'INHD' })}</div>
        <div class="side-block"><button class="btn" type="button" data-open="${sym}">Open the ${sym} dossier</button></div>
      </div>
    </div>
    ${calcHTML(ref)}`;
  const el = $('#pos-chart');
  if (el) {
    S.disposers.position.push(priceChart(el, ref.chart, {
      marker: ref.short_ref_date ? { date: ref.short_ref_date, label: `Short ref ${fmtDay(ref.short_ref_date, { weekday: false })}` } : null,
      ref: isNum(ref.short_ref_price) ? { price: ref.short_ref_price } : null,
      height: 320,
      label: `${ref.symbol || 'INHD'} daily closes with the short reference marked`,
    }));
  }
  wireCalc(ref);
}

function calcHTML(ref) {
  const hint = isNum(ref.short_ref_price) && ref.short_ref_date
    ? `<button type="button" class="btn btn--quiet" data-calc="ref">Use the reference (${esc(price(ref.short_ref_price))}, ${esc(fmtDay(ref.short_ref_date, { weekday: false }))})</button>`
    : '';
  return `
    <form class="calc reveal" id="calc" novalidate autocomplete="off" aria-labelledby="calc-title">
      <div>
        <h3 id="calc-title">Your P&amp;L</h3>
        <p class="lede">Private. What you type stays in this browser's local storage and is never sent anywhere. Marked at the last close; the borrow cost is an estimate.</p>
        <div class="field-row">
          <div class="field"><label class="eyebrow" for="calc-entry">Entry price ($)</label><input id="calc-entry" name="entry" type="number" inputmode="decimal" min="0" step="any" placeholder="${esc(isNum(ref.short_ref_price) ? ref.short_ref_price.toFixed(2) : '0.00')}" aria-describedby="calc-msg"></div>
          <div class="field"><label class="eyebrow" for="calc-shares">Shares short</label><input id="calc-shares" name="shares" type="number" inputmode="numeric" min="1" step="1" placeholder="1000" aria-describedby="calc-msg"></div>
          <div class="field"><label class="eyebrow" for="calc-date">Entry date</label><input id="calc-date" name="date" type="date" max="${esc(etDate())}" aria-describedby="calc-msg"></div>
        </div>
        <div class="calc-actions">${hint}<button type="button" class="btn btn--quiet" data-calc="clear">Clear</button><p class="note" id="calc-msg" aria-live="polite"></p></div>
      </div>
      <dl class="calc-out" id="calc-out" aria-live="polite"></dl>
    </form>`;
}

function wireCalc(ref) {
  const form = $('#calc');
  if (!form) return;
  const f = { entry: $('#calc-entry'), shares: $('#calc-shares'), date: $('#calc-date') };
  const saved = store.get(CALC_KEY);
  if (saved && typeof saved === 'object') {
    if (saved.entry != null) f.entry.value = String(saved.entry);
    if (saved.shares != null) f.shares.value = String(saved.shares);
    if (saved.date) f.date.value = String(saved.date);
  }
  const canStore = store.available();
  const update = (persist) => {
    const vals = { entry: f.entry.value.trim(), shares: f.shares.value.trim(), date: f.date.value };
    if (persist && canStore) {
      if (vals.entry || vals.shares || vals.date) store.set(CALC_KEY, vals);
      else store.remove(CALC_KEY);
    }
    calcCompute(ref, f, canStore);
  };
  form.addEventListener('input', () => update(true));
  form.addEventListener('submit', (e) => e.preventDefault());
  form.addEventListener('click', (e) => {
    const b = e.target.closest('[data-calc]');
    if (!b) return;
    if (b.dataset.calc === 'clear') {
      f.entry.value = '';
      f.shares.value = '';
      f.date.value = '';
      store.remove(CALC_KEY);
      update(false);
      f.entry.focus();
    } else if (b.dataset.calc === 'ref') {
      if (isNum(ref.short_ref_price)) f.entry.value = String(ref.short_ref_price);
      if (ref.short_ref_date) f.date.value = ref.short_ref_date;
      update(true);
      f.shares.focus();
    }
  });
  update(false);
}

function calcCompute(ref, f, canStore) {
  const out = $('#calc-out');
  const msg = $('#calc-msg');
  if (!out) return;
  const entry = f.entry.value === '' ? null : Number(f.entry.value);
  const shares = f.shares.value === '' ? null : Number(f.shares.value);
  const d = f.date.value || null;
  const today = etDate();
  const bad = {
    entry: entry != null && !(Number.isFinite(entry) && entry > 0),
    shares: shares != null && !(Number.isFinite(shares) && shares > 0),
    date: d != null && (!parseDay(d) || d > today),
  };
  Object.entries(bad).forEach(([k, v]) => f[k].setAttribute('aria-invalid', String(v)));
  const problems = [];
  if (bad.entry) problems.push('entry price must be above 0');
  if (bad.shares) problems.push('shares must be above 0');
  if (bad.date) problems.push('entry date can’t be in the future');
  const notes = [];
  if (problems.length) notes.push(`Check: ${problems.join('; ')}.`);
  if (!canStore) notes.push('Storage is blocked in this browser, so these numbers won’t be remembered.');
  if (S.sample) notes.push('Sample mode: marked to a placeholder price, not a quote.');
  if (msg) msg.textContent = notes.join(' ');

  const mark = isNum(ref.price) ? ref.price : null;
  const lb = lastBar(ref.chart);
  const fee = obj(ref.borrow).fee_rate;
  if (entry == null || shares == null || bad.entry || bad.shares) {
    out.innerHTML = `<div class="wide"><dt class="eyebrow">Result</dt><dd class="calc-hint">Enter your entry price and share count to see the P&amp;L.</dd></div>`;
    return;
  }
  if (mark == null) {
    out.innerHTML = `<div class="wide"><dt class="eyebrow">Result</dt><dd class="calc-hint">No ${esc(ref.symbol || 'INHD')} price in this feed, so nothing can be marked.</dd></div>`;
    return;
  }
  const pnl = (entry - mark) * shares;
  const ret = (entry - mark) / entry;
  const days = d && !bad.date ? daysBetween(d, today) : null;
  const cost = isNum(fee) && isNum(days) ? (fee / 100) * mark * shares * (days / 360) : null;
  const net = isNum(cost) ? pnl - cost : null;
  const costNote = !d || !isNum(days)
    ? 'Add a valid entry date to estimate borrow'
    : !isNum(fee)
      ? 'Fee not reported this run'
      : `${pctUnits(fee, 1)}/yr × ${days} day${days === 1 ? '' : 's'} ÷ 360 on today's value. Estimate only: IBKR charges daily and the fee moves.`;
  out.innerHTML = `
    <div><dt class="eyebrow">P&amp;L before fees</dt><dd>${esc(signedMoney(pnl))}</dd><small>Marked at ${esc(price(mark))}${lb ? `, close ${esc(fmtDay(lb[0], { weekday: false }))}` : ''}</small></div>
    <div><dt class="eyebrow">Return</dt><dd>${esc(pct(ret, 1, true))}</dd><small>On the ${esc(money(entry * shares))} you shorted</small></div>
    <div><dt class="eyebrow">Est. borrow cost</dt><dd>${isNum(cost) ? esc(signedMoney(-cost)) : DASH}</dd><small>${esc(costNote)}</small></div>
    <div><dt class="eyebrow">Net, estimated</dt><dd>${isNum(net) ? esc(signedMoney(net)) : DASH}</dd><small>P&amp;L minus estimated borrow; excludes commissions</small></div>`;
}

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
function hbars(items, max) {
  const m = max || Math.max(0.01, ...items.map((x) => x.v).filter(isNum));
  return `<ul class="hbars">${items.map((x) => `<li class="${x.hot ? 'hot' : ''}"><span>${esc(x.label)}</span><span class="b" aria-hidden="true">${isNum(x.v) ? `<i style="width:${(Math.max(0, Math.min(1, x.v / m)) * 100).toFixed(1)}%"></i>` : ''}</span><span class="v">${esc(x.fmt ? x.fmt(x.v) : pct(x.v, 1))}</span></li>`).join('')}</ul>`;
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
        ${stat('Win rate', esc(pct(isNum(sim.pub_win_rate) ? sim.pub_win_rate : sim.win_rate)), esc(`days the short made money, after ${pct(cost, 0)} cost`))}
        ${stat('Risking 10% a day', esc(isNum(obj(sim.compounded_pub).final_multiple) ? `${fixed(sim.compounded_pub.final_multiple, 2)}×` : DASH), esc(isNum(obj(sim.compounded_pub).max_drawdown_pct) ? `compounded; worst drawdown −${pct(sim.compounded_pub.max_drawdown_pct, 0)}; worst day ${pct(sim.pub_worst_day, 0, true)}` : 'compounded equity multiple'))}
      </dl>
      ${nDays ? '<div class="chart" id="eq-chart"></div><div class="legend"><span><i></i>Published rule, net</span><span><i class="g"></i>Raw #1 (incl. Rule 201 names), net</span></div>' : empty('No simulated days in this report.')}
    </div>
  </div>`;
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
      if (isNum(r[4])) n += -r[4] - cost;               // published rule, net
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
          ${!isBase && s.skew ? `<span><b>${esc(s.skew === 'down' ? 'Down' : s.skew === 'up' ? 'Up' : 'Both ways')}</b>which way it tilts</span>` : ''}
        </div>
      </article>`;
    }).join('')}</div>`;
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
  if (S.sample) rows.push(['Mode', 'SAMPLE DATA — placeholder numbers']);
  setHTML('#foot-times', rows.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join(''));
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
  else if (rec.where === 'twin') where = 'INHD twin';
  else if (rec.where === 'ref') where = 'Your reference short';
  if (rec.twin && rec.where !== 'twin') where += ` · INHD twin`;
  const sub = [p.name, p.exchange, p.country, p.industry || p.sector].filter(Boolean).map(esc).join(' · ');

  const pre = p.premarket;
  const gap = pre && isNum(pre.gap_pct) ? pct(pre.gap_pct, 0, true) : DASH;
  const keys = `<dl class="dz-keys">
    <div><dt class="eyebrow">Dump odds</dt><dd class="${isNum(p.prob_dump) ? 'red' : ''}">${esc(pct(p.prob_dump))}<small>${esc(isNum(base) ? `base ${pct(base, 1)} · ${liftTxt(p.lift)}` : 'base rate not reported')}</small></dd></div>
    <div><dt class="eyebrow">Exp. open→close</dt><dd>${esc(pct(p.exp_oc, 1, true))}<small>model average</small></dd></div>
    <div><dt class="eyebrow">Pre-market</dt><dd>${esc(gap)}<small>${esc(pre ? `${price(pre.price)} at ${fmtTimeET(pre.asof)} ET` : 'no print reported')}</small></dd></div>
    <div><dt class="eyebrow">Squeeze danger</dt><dd>${squeezeMeter(p.squeeze_danger)}<small>${esc(sqWord(p.squeeze_danger))}</small></dd></div>
  </dl>`;

  const chartRows = arr(p.chart).filter((r) => Array.isArray(r) && isNum(r[4]));
  const lb = chartRows.length ? chartRows[chartRows.length - 1] : null;
  const chart = dzSec(`Price${chartRows.length ? ` — ${chartRows.length} sessions to ${fmtDay(lb[0], { weekday: false })}` : ''}`,
    `<div class="lwc" id="dz-chart" role="img" aria-label="${esc(`${p.symbol} daily candles and volume${lb ? ` to ${fmtDay(lb[0], { weekday: false, year: true })}` : ''}`)}"></div><p class="note" style="margin-top:8px">Split-adjusted daily bars. Hollow = closed up, red = closed down; volume along the bottom.${rec.ref ? ' Arrow = your short reference date.' : ''}</p>`);

  const parts = [];
  if (p._noModel) parts.push(`<p class="dz-note">No model reading for ${sym} in this feed${S.sample ? ' (sample mode never scores a real company)' : ''}. Price, borrow, filings and news are shown where available.</p>`);
  if (rec.where === 'zone' && p.zone_reason) parts.push(`<p class="dz-warn"><b>Why it's off the board</b>${esc(p.zone_reason)}</p>`);
  if (rec.ref) {
    const r = rec.ref;
    parts.push(dzSec('Your reference', kv([
      ['Short reference', esc(price(r.short_ref_price)), fmtDay(r.short_ref_date, { weekday: false, year: true })],
      ['Last close', esc(price(r.price))],
      ['Change since reference', esc(pct(r.change_since_ref, 1, true)), 'a fall is a short gain'],
    ]) + (arr(r.notes).length ? `<ul class="notes" style="margin-top:16px">${arr(r.notes).map((n) => `<li>${esc(n)}</li>`).join('')}</ul>` : '')));
  }
  if (arr(p.reasons).length) parts.push(dzSec('Why it is on the radar', reasonsList(p.reasons)));
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
      ['Similarity to INHD', esc(pct(p.inhd_similarity))],
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
  const ref = rec.ref;
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
  const [today, model, evidence, scorecard] = await Promise.all(
    ['today.json', 'model.json', 'evidence.json', 'scorecard.json'].map(getJSON),
  );
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
  safe('wire', renderWire);
  safe('position', renderPosition);
  safe('twins', renderTwins);
  safe('squeeze', renderSqueeze);
  safe('record', renderRecord);
  safe('evidence', renderEvidence);
  safe('method', renderMethod);
  safe('footer', renderFooter);
  observeReveal();
  route();
  watchFreshness();
  document.documentElement.classList.add('is-ready');
}

if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init, { once: true });
else init();
