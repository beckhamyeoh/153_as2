# Presentation outline (~20 minutes)

Format: slides + live code walk-through of `workbook.html`. Graders watch this while
following the notebook, so each section below points at the matching notebook section.
Target ~18 min of content + play music at the end (uncounted). Times are budgets.

> Numbers in **[brackets]** are filled from the executed notebook — see the final
> "metrics to quote" section at the bottom; update if you re-run.

---

## 0. Intro (1.5 min) — notebook §title
- One sentence on the goal: *"make beautiful music with ML."*
- State the two tasks and why they're paired:
  - Task 1 — **symbolic, unconditioned**: learn p(melody), sample new folk tunes.
  - Task 2 — **symbolic, conditioned**: harmonization, p(chords | melody).
- Why this pair: shared data pipeline + one split; and the output of Task 1 feeds Task 2
  ("compose, then harmonise"). Both train from scratch on a laptop in minutes.

## 1. Data — shared (3 min) — notebook §1
- **Context**: Nottingham folk corpus (~1000 tunes), Jukedeck cleaned MIDI; each file =
  melody track + chord track. Standard benchmark since Boulanger-Lewandowski 2012.
- **Pre-processing** (walk the 4 steps):
  1. quantise to a 16th-note grid (constant 120 BPM, perfectly aligned).
  2. melody → step tokens (`REST` / `HOLD` / pitch), monophonic reduction.
  3. **key normalisation** via Krumhansl–Schmuckler → transpose all to C major / A minor.
  4. chords → per-bar labels via pitch-class template naming.
- **Show the EDA plots**: collection counts; keys *before* normalisation (cluster in D/G/A);
  pitch-class histogram *after* (**97.5%** on the C-major scale → key-finding worked);
  duration distribution (8ths/16ths dominate); chord frequency (C, G7, F, Am).
- Punchline: the corpus is small but extremely regular/tonal, so a compact model is viable.

## 2. Task 1 modelling (2.5 min) — notebook §2
- **ML framing**: autoregressive language model over step tokens; minimise per-token NLL;
  metric = perplexity = exp(NLL); generate by temperature sampling to an end token.
- **Two models** = baseline + learned:
  - Markov n-gram (the Module-3 model) — transparent counts, but no long memory and tables
    blow up with order. *Used only as the baseline to beat.*
  - **LSTM LM** (embedding + 2-layer LSTM + head) — learned hidden state captures
    phrase-length structure at low parameter cost. Trade-off: needs training, less interpretable.
- Show the **training curve** (val perplexity dropping).

## 3. Task 1 evaluation (3 min) — notebook §2 eval
- **Context**: perplexity ≠ musicality, so pair it with distributional metrics
  (pitch-class & duration cosine vs real, in-key %, 4-gram diversity to catch copying).
- **Results to show**:
  - Perplexity table: LSTM **2.42** beats Markov orders 1/2/3 (**4.27 / 3.35 / 3.18**; n-gram plateaus).
  - Musical-metric table + overlaid PC/duration plots: both match local stats; the key nuance is
    **4-gram diversity** — real tunes are repetitive (~2% unique), LSTM (~4%) matches that far
    better than Markov (~8%, too random); LSTM also has the highest in-key rate (0.989).
- Show the **generated piano-roll** and play `symbolic_unconditioned.mid`.

## 4. Task 2 modelling (2.5 min) — notebook §3
- **ML framing**: conditional sequence labelling — bar → chord; each bar = 12-d melody
  pitch-class histogram; predict one chord/bar.
- **Baselines + model**: majority (tonic); template matching (instantaneous pitch, no context);
  **BiLSTM tagger** (bidirectional context → learns progressions).
- Show the **accuracy training curve**.

## 5. Task 2 evaluation (3 min) — notebook §3 eval
- **Context**: chord accuracy + root accuracy; caveat that multiple chords fit a bar.
- **Results**:
  - Accuracy table: BiLSTM **0.655** chord / **0.716** root ≈ doubles majority **0.394 / 0.420**;
    **template is *worse* than majority (0.238)** — the key lesson: *harmonization is contextual.*
  - Confusion matrix: BiLSTM's "errors" are musical substitutions (C↔Am, G↔G7).
  - Ground-truth comparison table for one held-out tune.
- **End-to-end demo**: harmonise the Task-1 melody; play `symbolic_conditioned.mid`.

## 6. Related work (1.5 min) — notebook §4
- Dataset lineage: Boulanger-Lewandowski 2012; folk-rnn (Sturm 2016).
- Task 1 ≈ Magenta Melody RNN; Music Transformer as the modern attention successor.
- Task 2 ≈ DeepBach / Coconet (chorale harmonization), ours is a lightweight label tagger.
- Our findings echo the literature: learned > baselines; local stats easy, long-range form hard;
  perplexity/accuracy correlate with — but don't fully capture — musical quality.

## 7. Play the music (uncounted) — notebook §5
- Play both files again in full so graders hear them even if their copy has issues.

---

### Recording tips
- Screen-record the notebook scrolling alongside slides (or just narrate the notebook).
- Keep to ~20 min (graders allot 20–30 min); don't exceed by >10% excluding the music.
- Export to **mp4** (Zoom/QuickTime/OBS) with a common codec (H.264); upload to Google Drive,
  set "anyone with the link", paste the link into `video_url.txt`.

### Metrics to quote (final, from the executed notebook)
- Corpus: 1034 tunes parsed (1021 harmonised); melody vocab **46** tokens; chord vocab **29** labels (>=5x).
- Split: 827 / 103 / 104 train / val / test tunes.
- % melody notes in C-major scale: **97.5%**.
- Model sizes: LSTM **939K** params; BiLSTM harmonizer **207K** params.
- Task 1 test perplexity: Markov-1 **4.27**, Markov-2 **3.35**, Markov-3 **3.18**, LSTM **2.42**.
- Task 1 metrics: PC-hist cosine LSTM **0.994** / Markov **0.996**; in-key LSTM **0.989** (real 0.975);
  4-gram diversity real **0.020**, LSTM **0.043**, Markov **0.080**.
- Task 2 accuracy: majority **0.394**, template **0.238**, BiLSTM **0.655** (root acc: 0.420 / 0.329 / **0.716**).
- Held-out harmonization demo (reelsa-c60): **78%** bars exactly match the original chords.
- Deliverables: `symbolic_unconditioned.mid` (50s, 184 notes), `symbolic_conditioned.mid` (melody+chords).
