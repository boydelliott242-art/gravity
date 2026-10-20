/**
 * Formatting + DOM helpers shared by every module.
 *
 * Honesty rule: every formatter returns "—" for missing data. Nothing here
 * ever substitutes a default number for a missing one.
 */

export const DASH = '—';
export const MINUS = '−';
export const ET_TZ = 'America/New_York';

export const isNum = (x) => typeof x === 'number' && Number.isFinite(x);

const ESC = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };
/** Escape any value for safe interpolation into HTML text or attributes. */
export const esc = (s) => (s == null ? '' : String(s).replace(/[&<>"']/g, (c) => ESC[c]));

/** Only http(s) URLs survive; anything else (javascript:, data:, junk) → null. */
export function safeUrl(u) {
  if (typeof u !== 'string' || !u.trim()) return null;
  try {
    const x = new URL(u, location.href);
    return x.protocol === 'https:' || x.protocol === 'http:' ? x.href : null;
  } catch {
    return null;
  }
}

/** <a> for an external source, or plain text when the URL is unusable. */
export function extLink(url, text, cls = '') {
  const u = safeUrl(url);
  const c = cls ? ` class="${esc(cls)}"` : '';
  return u
    ? `<a${c} href="${esc(u)}" target="_blank" rel="noopener noreferrer">${esc(text)}</a>`
    : `<span${c}>${esc(text)}</span>`;
}

function signOf(v, rounded) {
  if (Number(rounded) === 0) return '';
  return v > 0 ? '+' : v < 0 ? MINUS : '';
}

/** Fraction → percent string. pct(0.312) → "31%"; signed → "+31%" / "−4.1%". */
export function pct(x, digits = 0, signed = false) {
  if (!isNum(x)) return DASH;
  const v = x * 100;
  const s = Math.abs(v).toFixed(digits);
  if (signed) return signOf(v, s) + s + '%';
  return (v < 0 && Number(s) !== 0 ? MINUS : '') + s + '%';
}

/** Value already in percent units (IBKR fee 98.81 → "98.8%"). */
export function pctUnits(x, digits = 1) {
  if (!isNum(x)) return DASH;
  return (x < 0 ? MINUS : '') + Math.abs(x).toFixed(digits) + '%';
}

/** Share price with precision that suits micro-caps. */
export function price(x) {
  if (!isNum(x)) return DASH;
  const a = Math.abs(x);
  const d = a >= 1 ? 2 : a >= 0.1 ? 3 : 4;
  return (x < 0 ? MINUS : '') + '$' + a.toFixed(d);
}

/** 35000 → "35K", 12_400_000 → "12.4M". */
export function compact(x, digits = 1) {
  if (!isNum(x)) return DASH;
  const a = Math.abs(x);
  const sign = x < 0 ? MINUS : '';
  const fmt = (v, suf) => {
    const s = v.toFixed(v >= 100 ? 0 : digits).replace(/\.0+$/, '');
    return sign + s + suf;
  };
  if (a >= 1e12) return fmt(a / 1e12, 'T');
  if (a >= 1e9) return fmt(a / 1e9, 'B');
  if (a >= 1e6) return fmt(a / 1e6, 'M');
  if (a >= 1e3) return fmt(a / 1e3, 'K');
  return sign + a.toFixed(a >= 100 || Number.isInteger(a) ? 0 : 2);
}

export const money = (x) => (isNum(x) ? (x < 0 ? MINUS : '') + '$' + compact(Math.abs(x)) : DASH);

export function fixed(x, d = 1, suffix = '') {
  if (!isNum(x)) return DASH;
  return (x < 0 ? MINUS : '') + Math.abs(x).toFixed(d) + suffix;
}

export function signedMoney(x) {
  if (!isNum(x)) return DASH;
  const a = Math.abs(x);
  const s = a >= 1e4 ? a.toLocaleString('en-US', { maximumFractionDigits: 0 }) : a.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  return (x > 0 ? '+' : x < 0 ? MINUS : '') + '$' + s;
}

/* ── dates ─────────────────────────────────────────────────────────────── */
const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
const WDAYS = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];

/** 'YYYY-MM-DD…' → UTC-midnight Date (date-only strings must not drift by timezone). */
export function parseDay(s) {
  if (typeof s !== 'string') return null;
  const m = /^(\d{4})-(\d{2})-(\d{2})/.exec(s);
  if (!m) return null;
  const d = new Date(Date.UTC(+m[1], +m[2] - 1, +m[3]));
  return Number.isNaN(d.getTime()) ? null : d;
}

/** "2026-09-30" → "Tue 30 Sep" (options: year, weekday). */
export function fmtDay(s, { year = false, weekday = true } = {}) {
  const d = parseDay(s);
  if (!d) return DASH;
  const parts = [];
  if (weekday) parts.push(WDAYS[d.getUTCDay()]);
  parts.push(String(d.getUTCDate()), MONTHS[d.getUTCMonth()]);
  if (year) parts.push(String(d.getUTCFullYear()));
  return parts.join(' ');
}

export function parseTs(s) {
  if (typeof s !== 'string' || !s) return null;
  if (/^\d{4}-\d{2}-\d{2}$/.test(s)) return null; // date-only: not a timestamp
  const d = new Date(s);
  return Number.isNaN(d.getTime()) ? null : d;
}

let _etFmt;
function etParts(d) {
  _etFmt = _etFmt || new Intl.DateTimeFormat('en-US', {
    timeZone: ET_TZ, year: 'numeric', month: '2-digit', day: '2-digit',
    hour: '2-digit', minute: '2-digit', weekday: 'short', hour12: false,
  });
  const o = {};
  for (const p of _etFmt.formatToParts(d)) o[p.type] = p.value;
  if (o.hour === '24') o.hour = '00';
  return o;
}

/** ISO timestamp → "07:42" in US Eastern. Date-only strings → "—". */
export function fmtTimeET(s) {
  const d = parseTs(s);
  if (!d) return DASH;
  const p = etParts(d);
  return `${p.hour}:${p.minute}`;
}

/** ISO timestamp or date → "29 Sep 18:02 ET" / "29 Sep". */
export function fmtStamp(s, { year = false } = {}) {
  const d = parseTs(s);
  if (!d) return parseDay(s) ? fmtDay(s, { year, weekday: false }) : DASH;
  const p = etParts(d);
  const day = `${+p.day} ${MONTHS[+p.month - 1]}${year ? ' ' + p.year : ''}`;
  return `${day} ${p.hour}:${p.minute} ET`;
}

/** YYYY-MM-DD of a timestamp in US Eastern. */
export function etDate(d = new Date()) {
  const p = etParts(d);
  return `${p.year}-${p.month}-${p.day}`;
}

export function ago(s, now = Date.now()) {
  const d = parseTs(s);
  if (!d) return DASH;
  const sec = Math.max(0, (now - d.getTime()) / 1000);
  if (sec < 90) return 'just now';
  const min = sec / 60;
  if (min < 60) return `${Math.round(min)} min ago`;
  const h = min / 60;
  if (h < 36) return `${Math.round(h)} h ago`;
  return `${Math.round(h / 24)} days ago`;
}

/** Whole calendar days from a to b (date-only strings or Dates). */
export function daysBetween(a, b) {
  const da = a instanceof Date ? a : parseDay(a);
  const db = b instanceof Date ? b : parseDay(b);
  if (!da || !db) return null;
  return Math.round((db - da) / 864e5);
}

/* ── market clock (display only) ───────────────────────────────────────── */
const HOLIDAYS = new Set([
  '2025-01-01', '2025-01-09', '2025-01-20', '2025-02-17', '2025-04-18', '2025-05-26', '2025-06-19',
  '2025-07-04', '2025-09-01', '2025-11-27', '2025-12-25',
  '2026-01-01', '2026-01-19', '2026-02-16', '2026-04-03', '2026-05-25', '2026-06-19', '2026-07-03',
  '2026-09-07', '2026-11-26', '2026-12-25',
  '2027-01-01', '2027-01-18', '2027-02-15', '2027-03-26', '2027-05-31', '2027-06-18', '2027-07-05',
  '2027-09-06', '2027-11-25', '2027-12-24',
]);

/** Live US market phase from the viewer's clock: pre-market | open | after-hours | closed. */
export function marketPhaseNow(now = new Date()) {
  const p = etParts(now);
  const iso = `${p.year}-${p.month}-${p.day}`;
  if (p.weekday === 'Sat' || p.weekday === 'Sun' || HOLIDAYS.has(iso)) return 'closed';
  const m = +p.hour * 60 + +p.minute;
  if (m >= 240 && m < 570) return 'pre-market';
  if (m >= 570 && m < 960) return 'open';
  if (m >= 960 && m < 1200) return 'after-hours';
  return 'closed';
}

/* ── labels ────────────────────────────────────────────────────────────── */
export const CAT_LABEL = {
  offering: 'Offering',
  atm: 'At-the-market program',
  toxic_financing: 'Toxic financing',
  unregistered_sale: 'Unregistered sale',
  registration: 'Registration',
  resale: 'Resale registration',
  effective: 'Declared effective',
  delisting_notice: 'Deficiency / delisting notice',
  going_concern: 'Going-concern doubt',
  late_filing: 'Late filing',
  reverse_split: 'Reverse split',
  charter_amendment: 'Charter amendment',
  insider_sale_notice: 'Form 144 sale notice',
  insider: 'Insider filing',
  material_agreement: 'Material agreement',
  other: 'Other',
};
export const SUPPLY_CATS = new Set(['offering', 'atm', 'toxic_financing', 'unregistered_sale']);

export const humanize = (s) => (s == null ? '' : String(s).replace(/_/g, ' ').replace(/^\w/, (c) => c.toUpperCase()));

export const ASIA = new Set([
  'China', 'Hong Kong', 'Singapore', 'Malaysia', 'Taiwan', 'Japan', 'Cayman Islands',
  'British Virgin Islands', 'Macau', 'Thailand', 'Vietnam', 'Indonesia', 'Philippines',
]);

/* ── DOM ───────────────────────────────────────────────────────────────── */
export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

export function prefersReducedMotion() {
  try {
    return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  } catch {
    return false;
  }
}

export function onIdle(fn, timeout = 400) {
  if ('requestIdleCallback' in window) window.requestIdleCallback(fn, { timeout });
  else setTimeout(fn, 32);
}

export function rafThrottle(fn) {
  let queued = false;
  let lastArgs;
  return (...args) => {
    lastArgs = args;
    if (queued) return;
    queued = true;
    requestAnimationFrame(() => {
      queued = false;
      fn(...lastArgs);
    });
  };
}

/** Research links fallback, used only when a record has none of its own. */
export function fallbackLinks(sym) {
  const s = encodeURIComponent(sym);
  const lo = s.toLowerCase();
  return {
    Zacks: `https://www.zacks.com/stock/quote/${s}`,
    Danelfin: `https://danelfin.com/stock/${s}`,
    Bloomberg: `https://www.bloomberg.com/quote/${s}:US`,
    WSJ: `https://www.wsj.com/market-data/quotes/${s}`,
    Finviz: `https://finviz.com/quote.ashx?t=${s}`,
    TradingView: `https://www.tradingview.com/symbols/${s}/`,
    Stocktwits: `https://stocktwits.com/symbol/${s}`,
    Yahoo: `https://finance.yahoo.com/quote/${s}`,
    Nasdaq: `https://www.nasdaq.com/market-activity/stocks/${lo}`,
    'SEC EDGAR': `https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=${s}&type=&dateb=&owner=include&count=40`,
    Fintel: `https://fintel.io/ss/us/${lo}`,
    iBorrowDesk: `https://iborrowdesk.com/report/${s}`,
  };
}

/** Local storage that never throws (private windows, blocked site data). */
export const store = {
  get(key) {
    try {
      const v = window.localStorage.getItem(key);
      return v == null ? null : JSON.parse(v);
    } catch {
      return null;
    }
  },
  set(key, val) {
    try {
      window.localStorage.setItem(key, JSON.stringify(val));
      return true;
    } catch {
      return false;
    }
  },
  remove(key) {
    try {
      window.localStorage.removeItem(key);
      return true;
    } catch {
      return false;
    }
  },
  available() {
    try {
      const k = '__gravity_probe__';
      window.localStorage.setItem(k, '1');
      window.localStorage.removeItem(k);
      return true;
    } catch {
      return false;
    }
  },
};
