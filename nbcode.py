"""
Shared code for the CSE120 music-generation project.

Two symbolic tasks built on the Nottingham folk-tune database:
  Task 1 (unconditioned): learn p(melody) and sample new melodies.
  Task 2 (conditioned):    harmonization -- melody bars -> chord labels.

This module is developed/tested as a script and then mirrored into the
narrated workbook. Functions are grouped: DATA -> MODELS -> EVAL -> RENDER.
"""

import os, glob, math, random, warnings
from collections import Counter, defaultdict

import numpy as np
import pretty_midi

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# DATA: parsing & tokenization
# ----------------------------------------------------------------------------

GRID = 4                      # grid steps per quarter note (16th-note resolution)
MIDI_DIR = "nottingham-dataset/MIDI"

# Reserved vocabulary symbols (pitch onsets are stored as plain ints)
BOS, EOS, REST, HOLD, PAD = "BOS", "EOS", "REST", "HOLD", "PAD"


def list_files(midi_dir=MIDI_DIR):
    return sorted(glob.glob(os.path.join(midi_dir, "*.mid")))


def _sixteenth_seconds(pm):
    """Seconds per 16th note from the file's (constant) tempo."""
    bpm = float(pm.get_tempo_changes()[1][0])
    return (60.0 / bpm) / GRID


def bar_steps(pm):
    """Length of one bar in grid-steps, from the first time signature (default 4/4)."""
    if pm.time_signature_changes:
        ts = pm.time_signature_changes[0]
        num, den = ts.numerator, ts.denominator
    else:
        num, den = 4, 4
    return int(round(num * (4.0 / den) * GRID))


# Krumhansl-Schmuckler key profiles
_KS_MAJOR = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
_KS_MINOR = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])


def _pc_histogram(pm):
    """Duration-weighted pitch-class histogram over melody + chord notes."""
    h = np.zeros(12)
    for inst in pm.instruments:
        for n in inst.notes:
            h[n.pitch % 12] += (n.end - n.start)
    return h


def estimate_key(pm):
    """Krumhansl-Schmuckler key finding. Returns (tonic_pc, mode, shift).

    `shift` is the semitone transposition (in [-6, 5]) that moves a major tune's
    tonic to C, or a minor tune's tonic to A (both => C-major pitch collection).
    """
    h = _pc_histogram(pm)
    if h.sum() == 0:
        return 0, "major", 0
    best = (-2, 0, "major")
    for t in range(12):
        for prof, mode in ((_KS_MAJOR, "major"), (_KS_MINOR, "minor")):
            r = np.corrcoef(h, np.roll(prof, t))[0, 1]
            if r > best[0]:
                best = (r, t, mode)
    _, tonic, mode = best
    target = 0 if mode == "major" else 9          # C for major, A for minor
    shift = ((target - tonic) + 6) % 12 - 6       # wrap into [-6, 5]
    return tonic, mode, shift


def transpose_pm(pm, shift):
    """Shift every note in place by `shift` semitones (clamped to MIDI range)."""
    for inst in pm.instruments:
        for n in inst.notes:
            n.pitch = int(np.clip(n.pitch + shift, 0, 127))


def _trim_rests(seq):
    """Drop leading/trailing REST symbols."""
    i, j = 0, len(seq)
    while i < j and seq[i] == REST:
        i += 1
    while j > i and seq[j - 1] == REST:
        j -= 1
    return seq[i:j]


def melody_to_steps(pm):
    """Monophonic step-sequence for the melody track (instrument 0).

    Returns a list of symbols, one per 16th-note step:
      REST  -> silence, HOLD -> sustain previous pitch, int -> note onset (MIDI pitch).
    Overlaps are resolved by keeping the highest pitch (monophonic reduction).
    """
    if not pm.instruments:
        return []
    six = _sixteenth_seconds(pm)
    notes = sorted(pm.instruments[0].notes, key=lambda n: (n.start, -n.pitch))
    if not notes:
        return []
    T = int(round(max(n.end for n in notes) / six))
    seq = [REST] * max(T, 1)
    occupied = [False] * len(seq)            # True where a note already sounds
    for n in notes:
        s = int(round(n.start / six))
        e = max(int(round(n.end / six)), s + 1)
        if s >= len(seq):
            continue
        if occupied[s]:                      # lower voice of a chord / overlap: drop
            continue
        seq[s] = n.pitch
        occupied[s] = True
        for k in range(s + 1, min(e, len(seq))):
            if not occupied[k]:
                seq[k] = HOLD
                occupied[k] = True
    return _trim_rests(seq)


_PC_NAME = {0: "C", 1: "C#", 2: "D", 3: "Eb", 4: "E", 5: "F",
            6: "F#", 7: "G", 8: "Ab", 9: "A", 10: "Bb", 11: "B"}
_QUALITIES = [("", {0, 4, 7}), ("m", {0, 3, 7}), ("7", {0, 4, 7, 10}),
              ("m7", {0, 3, 7, 10}), ("dim", {0, 3, 6}), ("aug", {0, 4, 8})]


def _chord_label(pitches):
    """Name a pitch set as a compact chord label by best pitch-class template fit.

    Pure-numpy (no music21): fast enough for the whole corpus. Returns e.g.
    'C', 'Am', 'G7'. Ties broken toward simpler (smaller) templates.
    """
    pcs = {int(p) % 12 for p in pitches}
    if not pcs:
        return None
    best, best_score, best_size = None, -1.0, 99
    for root in range(12):
        for suf, tmpl in _QUALITIES:
            t = {(root + iv) % 12 for iv in tmpl}
            score = len(pcs & t) / len(pcs | t)               # Jaccard
            if score > best_score or (score == best_score and len(t) < best_size):
                best, best_score, best_size = (root, suf), score, len(t)
    root, suf = best
    return _PC_NAME[root] + suf


def melody_chords_by_bar(pm):
    """Per-bar (melody pitch-class vector, chord label) for harmonization.

    Melody = instrument 0, chords = instrument 1. For each bar we build a
    duration-weighted 12-dim pitch-class histogram of the melody and take the
    chord whose onset is nearest the bar start as the target label.
    """
    if len(pm.instruments) < 2:
        return []
    six = _sixteenth_seconds(pm)
    bs = bar_steps(pm)
    mel = sorted(pm.instruments[0].notes, key=lambda n: n.start)
    chd = pm.instruments[1].notes
    if not mel or not chd:
        return []

    # group chord notes by onset step -> label
    groups = defaultdict(list)
    for n in chd:
        groups[int(round(n.start / six))].append(n.pitch)
    chord_events = sorted((step, _chord_label(ps)) for step, ps in groups.items())
    chord_events = [(s, l) for s, l in chord_events if l]
    if not chord_events:
        return []

    total_steps = int(round(max(n.end for n in mel) / six))
    n_bars = max(1, math.ceil(total_steps / bs))
    out = []
    for b in range(n_bars):
        lo, hi = b * bs, (b + 1) * bs
        pcv = np.zeros(12, dtype=np.float32)
        for n in mel:
            s = int(round(n.start / six)); e = int(round(n.end / six))
            dur = min(e, hi) - max(s, lo)
            if dur > 0:
                pcv[n.pitch % 12] += dur
        if pcv.sum() == 0:                  # empty bar: skip
            continue
        # nearest chord onset at/just before bar start
        cands = [(s, l) for s, l in chord_events if s <= lo + bs // 2]
        label = (cands[-1][1] if cands else chord_events[0][1])
        out.append((pcv, label))
    return out


def build_dataset(files=None, limit=None, normalize_key=True):
    """Parse all tunes into aligned per-tune records.

    Each record is {name, melody (step list or None), bars (per-bar
    (pcv,label) list or None), bar_steps, key}. Keeping melody and bars in the
    same record means one train/test split serves both tasks, and a held-out
    tune can be used to demo harmonization against its ground-truth chords.
    Returns (records, info).
    """
    files = files or list_files()
    if limit:
        files = files[:limit]
    records, keys_before, skipped = [], [], 0
    for f in files:
        try:
            pm = pretty_midi.PrettyMIDI(f)
        except Exception:
            skipped += 1
            continue
        tonic, mode, shift = estimate_key(pm)
        keys_before.append((tonic, mode))
        if normalize_key:
            transpose_pm(pm, shift)
        steps = melody_to_steps(pm)
        bars = melody_chords_by_bar(pm)
        records.append({
            "name": os.path.basename(f),
            "melody": steps if len(steps) >= 8 else None,
            "bars": bars if len(bars) >= 4 else None,
            "bar_steps": bar_steps(pm),
            "key": (tonic, mode),
        })
    info = {"skipped": skipped, "keys_before": keys_before,
            "n_files": len(files), "normalized": normalize_key}
    return records, info


def melodies_of(records):
    return [r["melody"] for r in records if r["melody"] is not None]


def harmonies_of(records):
    return [r["bars"] for r in records if r["bars"] is not None]


def split_indices(n, seed=0, fracs=(0.8, 0.1, 0.1)):
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    a = int(fracs[0] * n); b = a + int(fracs[1] * n)
    return idx[:a], idx[a:b], idx[b:]


# ----------------------------------------------------------------------------
# MODELS -- Task 1: vocabulary, n-gram baseline, LSTM language model
# ----------------------------------------------------------------------------

import torch
import torch.nn as nn

SEED = 0


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_melody_vocab(melodies):
    syms = set()
    for m in melodies:
        syms.update(m)
    pitches = sorted(s for s in syms if isinstance(s, int))
    itos = [PAD, BOS, EOS, REST, HOLD] + pitches
    stoi = {s: i for i, s in enumerate(itos)}
    return stoi, itos


def encode_melody(m, stoi):
    return [stoi[BOS]] + [stoi[s] for s in m] + [stoi[EOS]]


class MarkovModel:
    """Order-n token Markov chain with add-k smoothing and backoff.

    Baseline for Task 1. Trained on encoded melody id-sequences (with BOS/EOS).
    """

    def __init__(self, order=3, k=0.05, vocab_size=None):
        self.order = order
        self.k = k
        self.V = vocab_size
        self.ctx = [defaultdict(Counter) for _ in range(order + 1)]

    def fit(self, sequences):
        for seq in sequences:
            for i in range(1, len(seq)):
                nxt = seq[i]
                for o in range(self.order + 1):
                    if i - o < 0:
                        break
                    context = tuple(seq[i - o:i])
                    self.ctx[o][context][nxt] += 1
        return self

    def _dist(self, context):
        """p(next | context) as a numpy vector, backing off to shorter contexts."""
        for o in range(min(self.order, len(context)), -1, -1):
            c = tuple(context[len(context) - o:]) if o else tuple()
            counter = self.ctx[o].get(c)
            if counter and sum(counter.values()) > 0:
                total = sum(counter.values())
                p = np.full(self.V, self.k)
                for tok, cnt in counter.items():
                    p[tok] += cnt
                return p / (total + self.k * self.V)
        return np.full(self.V, 1.0 / self.V)

    def logprob_seq(self, seq):
        """Total log-prob and token count for perplexity (predicts seq[1:])."""
        lp, n = 0.0, 0
        for i in range(1, len(seq)):
            context = seq[max(0, i - self.order):i]
            p = self._dist(context)
            lp += math.log(max(p[seq[i]], 1e-12))
            n += 1
        return lp, n

    def generate(self, stoi, itos, max_len=400, temperature=1.0, seed=0):
        rng = np.random.default_rng(seed)
        seq = [stoi[BOS]]
        for _ in range(max_len):
            p = self._dist(seq[-self.order:])
            if temperature != 1.0:
                p = np.power(p, 1.0 / temperature)
            p = p / p.sum()
            nxt = int(rng.choice(self.V, p=p))
            if nxt == stoi[EOS]:
                break
            seq.append(nxt)
        return [itos[i] for i in seq[1:]]            # strip BOS


class MelodyLSTM(nn.Module):
    """LSTM language model over melody step-tokens (Task 1 main model)."""

    def __init__(self, vocab_size, emb=128, hidden=256, layers=2, dropout=0.3):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.lstm = nn.LSTM(emb, hidden, layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, vocab_size)

    def forward(self, x, state=None):
        x = self.emb(x)
        out, state = self.lstm(x, state)
        return self.head(self.drop(out)), state


def make_windows(sequences, seq_len=64, stride=32):
    """Concatenate encoded sequences and cut overlapping (input, target) windows."""
    stream = []
    for s in sequences:
        stream.extend(s)
    stream = torch.tensor(stream, dtype=torch.long)
    xs, ys = [], []
    for i in range(0, len(stream) - seq_len - 1, stride):
        xs.append(stream[i:i + seq_len])
        ys.append(stream[i + 1:i + seq_len + 1])
    return torch.stack(xs), torch.stack(ys)


def train_lstm(model, train_seqs, val_seqs, stoi, device, epochs=18,
               seq_len=64, batch=64, lr=2e-3, log=print):
    """Train the melody LSTM; returns history of (epoch, train_ppl, val_ppl)."""
    Xtr, Ytr = make_windows(train_seqs, seq_len)
    Xva, Yva = make_windows(val_seqs, seq_len)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=6, gamma=0.5)
    lossf = nn.CrossEntropyLoss(ignore_index=0)
    history, best_val, best_state = [], float("inf"), None
    n = len(Xtr)
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        tot, cnt = 0.0, 0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb, yb = Xtr[idx].to(device), Ytr[idx].to(device)
            opt.zero_grad()
            logits, _ = model(xb)
            loss = lossf(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item() * len(idx); cnt += len(idx)
        sched.step()
        val = eval_lstm_ppl(model, Xva, Yva, device, batch)
        tr_ppl = math.exp(tot / cnt)
        history.append((ep, tr_ppl, val))
        if val < best_val:
            best_val, best_state = val, {k: v.detach().cpu().clone()
                                         for k, v in model.state_dict().items()}
        log(f"epoch {ep:2d}  train_ppl={tr_ppl:6.2f}  val_ppl={val:6.2f}")
    if best_state:
        model.load_state_dict(best_state)
    return history


@torch.no_grad()
def eval_lstm_ppl(model, X, Y, device, batch=128):
    model.eval()
    lossf = nn.CrossEntropyLoss(ignore_index=0, reduction="sum")
    tot, cnt = 0.0, 0
    for i in range(0, len(X), batch):
        xb, yb = X[i:i + batch].to(device), Y[i:i + batch].to(device)
        logits, _ = model(xb)
        tot += lossf(logits.reshape(-1, logits.size(-1)), yb.reshape(-1)).item()
        cnt += (yb != 0).sum().item()
    return math.exp(tot / max(cnt, 1))


@torch.no_grad()
def sample_lstm(model, stoi, itos, device, max_len=400, temperature=1.0, seed=0):
    model.eval()
    torch.manual_seed(seed)
    x = torch.tensor([[stoi[BOS]]], device=device)
    state, out = None, []
    for _ in range(max_len):
        logits, state = model(x, state)
        logits = logits[0, -1] / temperature
        p = torch.softmax(logits, dim=-1)
        nxt = int(torch.multinomial(p, 1).item())
        if nxt == stoi[EOS]:
            break
        out.append(itos[nxt])
        x = torch.tensor([[nxt]], device=device)
    return out


# ----------------------------------------------------------------------------
# MODELS -- Task 2: harmonization (baselines + BiLSTM)
# ----------------------------------------------------------------------------

UNK = "UNK"
_NOTE_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def chord_pitch_classes(label):
    """Pitch-class set implied by a chord label, e.g. 'G7' -> {7,11,2,5}.

    Non-chord labels (PAD/UNK) return the empty set.
    """
    if not label or label[0] not in _NOTE_PC:
        return set()
    pc = _NOTE_PC[label[0]]
    i = 1
    if len(label) > 1 and label[1] in "b#":
        pc += (-1 if label[1] == "b" else 1)
        i = 2
    suf = label[i:]
    intervals = {"": [0, 4, 7], "m": [0, 3, 7], "7": [0, 4, 7, 10],
                 "m7": [0, 3, 7, 10], "dim": [0, 3, 6], "aug": [0, 4, 8]}.get(suf, [0, 4, 7])
    return {(pc + iv) % 12 for iv in intervals}


def chord_root_pc(label):
    """Root pitch class of a chord label; -1 for non-chord labels (PAD/UNK)."""
    if not label or label[0] not in _NOTE_PC:
        return -1
    pc = _NOTE_PC[label[0]]
    if len(label) > 1 and label[1] in "b#":
        pc += (-1 if label[1] == "b" else 1)
    return pc % 12


def build_chord_vocab(harmonies, min_count=5):
    cnt = Counter(l for h in harmonies for _, l in h)
    labels = [l for l, c in cnt.most_common() if c >= min_count]
    itos = [PAD, UNK] + labels
    stoi = {l: i for i, l in enumerate(itos)}
    return stoi, itos


class MajorityBaseline:
    """Always predict the most frequent training chord."""
    def fit(self, harmonies):
        self.label = Counter(l for h in harmonies for _, l in h).most_common(1)[0][0]
        return self
    def predict_tune(self, pcvs):
        return [self.label] * len(pcvs)


class TemplateBaseline:
    """Pick the chord whose pitch-class template best matches each bar (rule-based)."""
    def fit(self, harmonies):
        self.labels = sorted({l for h in harmonies for _, l in h})
        self.templates = {}
        for l in self.labels:
            v = np.zeros(12)
            for p in chord_pitch_classes(l):
                v[p] = 1.0
            self.templates[l] = v / np.linalg.norm(v)
        return self
    def predict_tune(self, pcvs):
        out = []
        for v in pcvs:
            v = v / (np.linalg.norm(v) + 1e-9)
            out.append(max(self.labels, key=lambda l: float(v @ self.templates[l])))
        return out


class HarmonizerBiLSTM(nn.Module):
    """BiLSTM sequence tagger: per-bar melody PC vector -> chord label."""
    def __init__(self, n_chords, hidden=128, layers=1, dropout=0.3):
        super().__init__()
        self.proj = nn.Linear(12, 64)
        self.lstm = nn.LSTM(64, hidden, layers, batch_first=True,
                            bidirectional=True, dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(2 * hidden, n_chords)

    def forward(self, x):
        h = torch.relu(self.proj(x))
        out, _ = self.lstm(h)
        return self.head(self.drop(out))


def harmonies_to_tensors(harmonies, stoi, device, max_bars=64):
    """Pad tunes to (B, max_bars, 12) features and (B, max_bars) chord-id targets."""
    X, Y = [], []
    for h in harmonies:
        h = h[:max_bars]
        feats = np.zeros((max_bars, 12), dtype=np.float32)
        labs = np.zeros(max_bars, dtype=np.int64)
        for i, (pcv, lab) in enumerate(h):
            s = pcv.sum()
            feats[i] = pcv / s if s > 0 else pcv
            labs[i] = stoi.get(lab, stoi[UNK])
        X.append(feats); Y.append(labs)
    return (torch.tensor(np.stack(X), device=device),
            torch.tensor(np.stack(Y), device=device))


def train_harmonizer(model, train_h, val_h, stoi, device, epochs=40, batch=32,
                     lr=3e-3, log=print):
    Xtr, Ytr = harmonies_to_tensors(train_h, stoi, device)
    Xva, Yva = harmonies_to_tensors(val_h, stoi, device)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss(ignore_index=0)
    best_acc, best_state, history = 0.0, None, []
    n = len(Xtr)
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            logits = model(Xtr[idx])
            loss = lossf(logits.reshape(-1, logits.size(-1)), Ytr[idx].reshape(-1))
            loss.backward(); opt.step()
        acc = harmonizer_accuracy(model, Xva, Yva)
        history.append((ep, acc))
        if acc > best_acc:
            best_acc, best_state = acc, {k: v.detach().cpu().clone()
                                        for k, v in model.state_dict().items()}
        if ep % 5 == 0 or ep == 1:
            log(f"epoch {ep:2d}  val_acc={acc:.3f}")
    if best_state:
        model.load_state_dict(best_state)
    return history


@torch.no_grad()
def harmonizer_accuracy(model, X, Y):
    model.eval()
    pred = model(X).argmax(-1)
    mask = Y != 0
    return float((pred[mask] == Y[mask]).float().mean().item())


@torch.no_grad()
def harmonizer_predict(model, pcvs, itos, device, exclude_special=False):
    """Predict a chord label per bar. If exclude_special, never emit PAD/UNK
    (indices 0,1) -- used when rendering music so every bar gets a real chord."""
    model.eval()
    feats = np.zeros((1, len(pcvs), 12), dtype=np.float32)
    for i, v in enumerate(pcvs):
        s = v.sum(); feats[0, i] = v / s if s > 0 else v
    logits = model(torch.tensor(feats, device=device))[0]
    if exclude_special:
        logits[:, :2] = float("-inf")
    return [itos[i] for i in logits.argmax(-1).tolist()]


# ----------------------------------------------------------------------------
# EVALUATION -- musical metrics & helpers
# ----------------------------------------------------------------------------

C_MAJOR = {0, 2, 4, 5, 7, 9, 11}


def steps_to_notes(steps):
    """Convert a step-sequence to [(pitch, start_step, duration_steps), ...]."""
    notes, cur, start = [], None, None
    for t, s in enumerate(steps):
        if s == HOLD:
            continue
        if cur is not None:
            notes.append((cur, start, t - start))
        if isinstance(s, int):
            cur, start = s, t
        else:                      # REST
            cur, start = None, None
    if cur is not None:
        notes.append((cur, start, len(steps) - start))
    return notes


def pitch_class_hist(step_lists):
    h = np.zeros(12)
    for steps in step_lists:
        for p, _, d in steps_to_notes(steps):
            h[p % 12] += d
    return h / h.sum() if h.sum() else h


def duration_hist(step_lists, maxd=16):
    h = np.zeros(maxd + 1)
    for steps in step_lists:
        for _, _, d in steps_to_notes(steps):
            h[min(d, maxd)] += 1
    return h / h.sum() if h.sum() else h


def in_key_fraction(steps, scale=C_MAJOR):
    notes = steps_to_notes(steps)
    if not notes:
        return 0.0
    return np.mean([1.0 if (p % 12) in scale else 0.0 for p, _, _ in notes])


def ngram_diversity(step_lists, n=4):
    seen, total = set(), 0
    for steps in step_lists:
        toks = [str(s) for s in steps]
        for i in range(len(toks) - n + 1):
            seen.add(tuple(toks[i:i + n])); total += 1
    return len(seen) / total if total else 0.0


def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def melody_bar_pcvs(steps, bs):
    """Per-bar duration-weighted pitch-class vectors for an arbitrary melody.

    Lets us harmonize a *generated* melody (which has no chord track) by
    segmenting it into bars of `bs` grid-steps. Used for the end-to-end demo.
    """
    notes = steps_to_notes(steps)
    n_bars = max(1, math.ceil(len(steps) / bs))
    pcvs = []
    for b in range(n_bars):
        lo, hi = b * bs, (b + 1) * bs
        v = np.zeros(12, dtype=np.float32)
        for p, s, d in notes:
            ov = min(s + d, hi) - max(s, lo)
            if ov > 0:
                v[p % 12] += ov
        pcvs.append(v)
    return pcvs


# ----------------------------------------------------------------------------
# RENDER -- step-sequences / chords back to MIDI
# ----------------------------------------------------------------------------

def melody_to_midi(steps, path, bpm=120, program=0):
    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    inst = pretty_midi.Instrument(program=program)
    spb = (60.0 / bpm) / GRID
    for p, s, d in steps_to_notes(steps):
        inst.notes.append(pretty_midi.Note(velocity=90, pitch=int(p),
                                            start=s * spb, end=(s + d) * spb))
    pm.instruments.append(inst)
    pm.write(path)
    return path


def harmonization_to_midi(steps, chord_labels, bs, path, bpm=120, bass_octave=4):
    """Render melody + a chord accompaniment (one chord per bar) to MIDI."""
    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    mel = pretty_midi.Instrument(program=0, name="melody")
    chords = pretty_midi.Instrument(program=24, name="chords")
    spb = (60.0 / bpm) / GRID
    for p, s, d in steps_to_notes(steps):
        mel.notes.append(pretty_midi.Note(90, int(p), s * spb, (s + d) * spb))
    for b, lab in enumerate(chord_labels):
        if lab in (PAD, UNK):
            continue
        start, end = b * bs * spb, (b + 1) * bs * spb
        for pc in sorted(chord_pitch_classes(lab)):
            pitch = 12 * bass_octave + pc
            chords.notes.append(pretty_midi.Note(70, int(pitch), start, end))
    pm.instruments += [mel, chords]
    pm.write(path)
    return path


# ----------------------------------------------------------------------------
# quick self-test when run directly
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    files = list_files()
    print("files:", len(files))
    pm = pretty_midi.PrettyMIDI(files[0])
    steps = melody_to_steps(pm)
    print("melody steps (file0):", len(steps))
    print("first 32:", steps[:32])
    bars = melody_chords_by_bar(pm)
    print("bars (file0):", len(bars), "first chords:", [b[1] for b in bars[:8]])

    records, info = build_dataset(limit=120, normalize_key=True)
    mel, harm = melodies_of(records), harmonies_of(records)
    print(f"\nparsed: {len(mel)} melodies, {len(harm)} harmonized, "
          f"skipped {info['skipped']}")
    lens = [len(m) for m in mel]
    print("melody length steps: min/med/max = %d/%d/%d" %
          (min(lens), int(np.median(lens)), max(lens)))
    pitches = [s for m in mel for s in m if isinstance(s, int)]
    print("pitch range:", min(pitches), "-", max(pitches), "unique:", len(set(pitches)))
    chords = [lbl for h in harm for _, lbl in h]
    print("chord vocab (normalized):", len(set(chords)),
          "top:", Counter(chords).most_common(10))
