/* Mara v1 — browser inference engine.
 * Pure-JS port of literary_lm.c's forward pass (dim 256, 8 heads, 6 layers,
 * ff 1056, rotary, vocab 133). Weights: mara_weights.f32 (fp32, flat).
 * Protocol: tokens 128=<CHANNEL: 129=<SUMMARY> 130=<END> 131=[Z>A 132=[A
 * Byte tokens 0..127 are ASCII byte values.
 */
const DIM = 256, HEADS = 8, LAYERS = 6, FF = 1056, VOCAB = 133, CTX = 512;
const HEAD_W = DIM / HEADS, RMS_EPS = 1e-5;
const ATOM = { CHANNEL: 128, SUMMARY: 129, END: 130, ZA: 131, A: 132 };
const ATOM_STR = { 128: "<CHANNEL:", 129: "<SUMMARY>", 130: "<END>", 131: "[Z>A", 132: "[A" };

let W = null; // name -> Float32Array

export async function loadWeights(url, onProgress) {
  let f32;
  if (url instanceof Uint8Array) {
    f32 = new Float32Array(url.buffer, url.byteOffset, url.byteLength / 4);
  } else {
    const resp = await fetch(url);
    if (!resp.ok) throw new Error("weights fetch " + resp.status);
    const total = +resp.headers.get("Content-Length") || 19417088;
    const reader = resp.body.getReader();
    const chunks = [];
    let got = 0;
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value); got += value.length;
      if (onProgress) onProgress(got / total);
    }
    const buf = new Uint8Array(got);
    let o = 0; for (const c of chunks) { buf.set(c, o); o += c.length; }
    f32 = new Float32Array(buf.buffer);
  }
  // slice parameters in fixed order
  const order = [["token_embedding", VOCAB * DIM]];
  for (let l = 0; l < LAYERS; ++l) order.push(
    [`l${l}norm1`, DIM], [`l${l}wq`, DIM * DIM], [`l${l}wk`, DIM * DIM],
    [`l${l}wv`, DIM * DIM], [`l${l}wo`, DIM * DIM], [`l${l}norm2`, DIM],
    [`l${l}w1`, DIM * FF], [`l${l}w2`, FF * DIM]);
  order.push(["final_norm", DIM]);
  W = {};
  let off = 0;
  for (const [name, n] of order) { W[name] = f32.subarray(off, off + n); off += n; }
  return off;
}

function rmsnorm(x, gamma, out) {
  let ms = 0;
  for (let i = 0; i < DIM; ++i) ms += x[i] * x[i];
  const inv = 1 / Math.sqrt(ms / DIM + RMS_EPS);
  for (let i = 0; i < DIM; ++i) out[i] = x[i] * inv * gamma[i];
}
function linear(x, w, inDim, outDim, out) {
  for (let j = 0; j < outDim; ++j) {
    const row = j * inDim;
    let sum = 0;
    for (let i = 0; i < inDim; ++i) sum += x[i] * w[row + i];
    out[j] = sum;
  }
}
function gelu(x) {
  return 0.5 * x * (1 + Math.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)));
}
function rope(v, pos, inverse) {
  for (let h = 0; h < HEADS; ++h) {
    const ho = h * HEAD_W;
    for (let p = 0; p < HEAD_W / 2; ++p) {
      const freq = Math.pow(10000, -(2 * p) / HEAD_W);
      const ang = pos * freq;
      const c = Math.cos(ang), s = inverse ? -Math.sin(ang) : Math.sin(ang);
      const o = ho + 2 * p, a = v[o], b = v[o + 1];
      v[o] = a * c - b * s; v[o + 1] = a * s + b * c;
    }
  }
}

// KV cache: kc[l][pos*DIM + i], vc likewise
const kc = [], vc = [];
for (let l = 0; l < LAYERS; ++l) { kc.push(new Float32Array(CTX * DIM)); vc.push(new Float32Array(CTX * DIM)); }
let seqLen = 0;

const tmp = new Float32Array(Math.max(DIM, FF));
const n1 = new Float32Array(DIM), n2 = new Float32Array(DIM);
const q = new Float32Array(DIM), k = new Float32Array(DIM), v = new Float32Array(DIM);
const att = new Float32Array(DIM), probs = new Float32Array(CTX);
const xbuf = new Float32Array(DIM);

export function reset() { seqLen = 0; }

function step(token) {
  const pos = seqLen;
  if (pos >= CTX) throw new Error("context full");
  const emb = W.token_embedding;
  for (let i = 0; i < DIM; ++i) xbuf[i] = emb[token * DIM + i];
  seqLen = pos + 1;
  for (let l = 0; l < LAYERS; ++l) {
    rmsnorm(xbuf, W[`l${l}norm1`], n1);
    linear(n1, W[`l${l}wq`], DIM, DIM, q);
    linear(n1, W[`l${l}wk`], DIM, DIM, k);
    linear(n1, W[`l${l}wv`], DIM, DIM, v);
    rope(q, pos, false); rope(k, pos, false);
    kc[l].set(k, pos * DIM); vc[l].set(v, pos * DIM);
    // causal attention over 0..pos
    for (let h = 0; h < HEADS; ++h) {
      const ho = h * HEAD_W;
      let max = -Infinity;
      for (let s = 0; s <= pos; ++s) {
        let dot = 0;
        const kro = s * DIM + ho;
        for (let i = 0; i < HEAD_W; ++i) dot += q[ho + i] * kc[l][kro + i];
        probs[s] = dot / Math.sqrt(HEAD_W);
        if (probs[s] > max) max = probs[s];
      }
      let total = 0;
      for (let s = 0; s <= pos; ++s) { probs[s] = Math.exp(probs[s] - max); total += probs[s]; }
      for (let i = 0; i < HEAD_W; ++i) att[ho + i] = 0;
      for (let s = 0; s <= pos; ++s) {
        const wgt = probs[s] / total, vro = s * DIM + ho;
        for (let i = 0; i < HEAD_W; ++i) att[ho + i] += wgt * vc[l][vro + i];
      }
    }
    linear(att, W[`l${l}wo`], DIM, DIM, tmp);
    for (let i = 0; i < DIM; ++i) xbuf[i] += tmp[i];
    rmsnorm(xbuf, W[`l${l}norm2`], n2);
    linear(n2, W[`l${l}w1`], DIM, FF, tmp);
    for (let i = 0; i < FF; ++i) tmp[i] = gelu(tmp[i]);
    const out = new Float32Array(DIM);
    linear(tmp, W[`l${l}w2`], FF, DIM, out);
    for (let i = 0; i < DIM; ++i) xbuf[i] = xbuf[i] + out[i];
  }
  rmsnorm(xbuf, W.final_norm, n1);
  const logits = new Float32Array(VOCAB);
  linear(n1, W.token_embedding, DIM, VOCAB, logits);
  return logits;
}

export function sample(logits, temperature, topK, rng) {
  const scaled = new Float32Array(VOCAB);
  let max = -Infinity;
  for (let i = 0; i < VOCAB; ++i) { scaled[i] = logits[i] / temperature; if (scaled[i] > max) max = scaled[i]; }
  let sum = 0;
  for (let i = 0; i < VOCAB; ++i) { scaled[i] = Math.exp(scaled[i] - max); sum += scaled[i]; }
  for (let i = 0; i < VOCAB; ++i) scaled[i] /= sum;
  const idx = [...Array(VOCAB).keys()].sort((a, b) => scaled[b] - scaled[a]).slice(0, topK);
  let total = 0; for (const i of idx) total += scaled[i];
  let r = rng() * total;
  for (const i of idx) { r -= scaled[i]; if (r <= 0) return i; }
  return idx[0];
}

/* Generation loop */
export function generate(promptTokens, opts) {
  const o = Object.assign({ temperature: 0.7, topK: 40, maxTokens: 180 }, opts);
  reset();
  const p = promptTokens.slice(0, CTX - o.maxTokens - 1);
  let logits = null;
  for (const t of p) logits = step(t);
  let out = [];
  for (let n = 0; n < o.maxTokens; ++n) {
    const t = sample(logits, o.temperature, o.topK, Math.random);
    // stop atoms: record end, channel restart, or a new record starting
    if (t === ATOM.END || t === ATOM.CHANNEL || t === ATOM.A || t === ATOM.SUMMARY) break;
    out.push(t);
    logits = step(t);
  }
  return out;
}

export function decode(tokens) {
  let s = "";
  for (const t of tokens) {
    if (t < 128) { if (t >= 32 || t === 10) s += String.fromCharCode(t); }
    else s += ATOM_STR[t];
  }
  return s;
}

// --- Retrieval: the ledger is memory outside the weights ---------------
const STOP = new Set("what who where when why how are is the and you your for does do can was were this that with".split(" "));
function tokenize(q) {
  return q.toLowerCase().replace(/[^a-z0-9 ]/g, " ").split(/\s+/).filter(w => w.length > 2 && !STOP.has(w));
}
export function retrieve(ledger, question, k = 1) {
  const qw = tokenize(question);
  if (!qw.length) return [];
  const scored = ledger.map(rec => {
    const rw = tokenize(rec.q);
    let hit = 0;
    for (const w of rw) if (qw.includes(w)) ++hit;
    return [hit / Math.max(1, rw.length), rec];
  }).sort((a, b) => b[0] - a[0]);
  return scored.slice(0, k).filter(s => s[0] >= 0.34).map(s => s[1]);
}

// RAG turn: lay the retrieved records before her in EXACT channel-record
// format (protocol tokens), then ask. She continues the pattern from
// sources she never memorized.
export function askMaraRAG(question, ledger, opts) {
  const qn = question.trim().replace(/^./, c => c.toUpperCase());
  const hits = retrieve(ledger, qn, 1);
  let ctx = [];
  for (const rec of hits) {
    ctx = ctx.concat([ATOM.CHANNEL]);
    for (const ch of "W") ctx.push(ch.charCodeAt(0));
    ctx.push(ATOM.SUMMARY);
    for (const ch of "seeker asks the warren of Mara") ctx.push(ch.charCodeAt(0));
    ctx.push(4);
    ctx.push(2);
    for (const ch of "A" + rec.q) if (ch.charCodeAt(0) < 128) ctx.push(ch.charCodeAt(0));
    ctx.push(4);
    ctx.push(2);
    for (const ch of "Z") ctx.push(ch.charCodeAt(0));
    ctx.push(3);
    for (const ch of "A") ctx.push(ch.charCodeAt(0));
    ctx.push(6);
    for (const ch of rec.a) if (ch.charCodeAt(0) < 128) ctx.push(ch.charCodeAt(0));
    ctx.push(4);
    ctx.push(5);
  }
  // the live question
  ctx.push(ATOM.CHANNEL);
  for (const ch of "warrenmind ") ctx.push(ch.charCodeAt(0));
  ctx.push(ATOM.A);
  for (const ch of qn) if (ch.charCodeAt(0) < 128) ctx.push(ch.charCodeAt(0));
  const rawTokens = generate(ctx, opts);
  let s = decode(rawTokens);
  let mood = "record", emoji = "🕯️";
  if (s.includes("~?")) { mood = "doubt"; emoji = "🌫️"; }
  const i = s.indexOf("[Z>A");
  if (i < 0) { return { text: "", mood: "unrecorded", emoji: "🌑", source: hits[0] || null }; }
  s = s.slice(i + 4).replace(/^\s*\]?\s*=>\s*/, "");
  const j = s.indexOf("]");
  if (j >= 0) s = s.slice(0, j);
  s = s.trim();
  if (!s || s.startsWith("[A") || (s.endsWith("?") && s.split(/\s+/).length <= 8)) {
    mood = "seeking"; emoji = "🌱";
  }
  s = s.replace(/\s*~\?/g, " " + emoji);
  return { text: s, mood, emoji, source: hits[0] || null };
}

// Mood: read from the raw record before slicing. Doubt marks, silence,
// question-backs become emoji on the page.
export function askMara(question, opts) {
  const toks = [ATOM.CHANNEL];
  for (const ch of "warrenmind ") toks.push(ch.charCodeAt(0));
  toks.push(ATOM.A);
  for (const ch of question) if (ch.charCodeAt(0) < 128) toks.push(ch.charCodeAt(0));
  const rawTokens = generate(toks, opts);
  let s = decode(rawTokens);
  // mood first, from the untouched record
  let mood = "record", emoji = "🕯️";       // clean record, lantern lit
  if (s.includes("~?")) { mood = "doubt"; emoji = "🌫️"; }   // doubt mark
  const i = s.indexOf("[Z>A");
  if (i < 0) { mood = "unrecorded"; emoji = "🌑"; return { text: "", mood, emoji }; }
  s = s.slice(i + 4);
  s = s.replace(/^\s*\]?\s*=>\s*/, "");
  const j = s.indexOf("]");
  if (j >= 0) s = s.slice(0, j);
  s = s.trim();
  // question-back: the reply slot holds another question instead of an answer
  if (!s || s.startsWith("[A") || s.endsWith("?") && s.split(/\s+/).length <= 8) {
    mood = "seeking"; emoji = "🌱";
  }
  // doubt on the page: show fog instead of the raw ~?
  s = s.replace(/\s*~\?/g, " " + emoji);
  return { text: s, mood, emoji };
}
