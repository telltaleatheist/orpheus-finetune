# The Voice Training Pipeline — Canonical, Reproducible

**Status: CURRENT (2026-07-25).** This is the dialed-in recipe from the July
2026 testing campaign. Every constant in here was chosen by measurement or by
Owen's ear against a controlled A/B — change nothing casually, and when you do
change something, change ONE variable and re-audition. The failure-modes table
at the bottom exists because we hit every one of them; read it before
"improving" anything.

> **What changed on 2026-07-24 (read this if you last used the 07-20 version):**
> the **bed is no longer the default** — the raw-verbatim pivot (07-21) replaced
> it, and the bed is now the documented FALLBACK for sources whose noise floor is
> dead (§2). EOS margin now comes from an inference-side **logit boost**, not from
> corpus tricks (§6a). A **loop/completeness gate** was added (§7.2) — its absence
> is how a broken thirdreich shipped and survived two separate audits. And the
> "clean at assembly, never in training" rule is now stated explicitly (principle
> 2). The thirdreich post-mortem in §Case study is the worked example.

> **What changed on 2026-07-25:** a new **Step 5b — match the tail distribution**
> to the reference corpus. A narrator's long end-of-sentence pause is trained in
> verbatim and re-emitted on every chunk, and because generation is KV-limited that
> costs throughput quadratically — it is most of the measured 30 vs 70 chunks/min
> gap between thirdreich and deathstalker. Trim by SCALING (`--median-to`), never by
> clamping, and only ever cut in true silence so a breath cannot be halved.
> Corpus size is also no longer hardcoded: `build_2h_corpus.py` takes `--hours` /
> `--clips` and errors if given neither. **2.0 h is a floor, not a target.**

All pipeline scripts live in `pipeline/` in this repo. WSL paths assume the
standard layout (`/home/telltale/...`, corpora under `E:\training\<voice>\build\`
since the 07-21 consolidation, beds under `/home/telltale/beds/`). Python: WSL
conda envs `orpheus_tts` (analysis/inference) and `orpheus_train` (training).


### The order, in one place

| # | step | needs | cost |
|---|---|---|---|
| 0 | pick the master — MEASURE its pause floor, cut from the LIVE one | — | minutes |
| 0b | **TEXT QC** — endnote markers, heading bleed, drop asr-fallback cues, number convention | CPU | minutes |
| 1 | cut verbatim (no trail-cap, no gain, long tails kept) | CPU | ~5 min / 20 h |
| 2 | *(fallback only)* harvest a room-tone bed | CPU | minutes |
| 3 | brightness + cleanliness census | CPU | ~10 min |
| 4-5 | curate to the target hours and build the corpus | CPU | minutes |
| 5b | **match the tail distribution** to the reference corpus (scale, breath-safe) | CPU | ~2 min |
| 5c | **corpus gates** — SNAC pause conc, floor, breath/lead, text/audio coverage | GPU (brief) | ~5 min |
| 6 | train (8 epochs, patience 2) + merge + arm the boost per epoch | GPU | ~30 min |
| 7 | **gate battery on EVERY epoch**, keeper = the winner | GPU | ~6 min/epoch |
| 8 | deploy local -> ear-check in the app -> Mac + HF | — | minutes |

Everything up to step 5c is CPU, so a cut/census for one voice can run while another
voice trains. Only ONE GPU job at a time (principle 8).

---

## Principles (each one paid for)

1. **Speech is never filtered.** No low-pass (muffles — energy-rolloff metrics
   LIE about perceptual bandwidth; the top 1% of energy is the sibilance), no
   notching, no RVC/enhance on training audio. RVC-washed speech trains up
   rough: training amplifies what listening forgives.
2. **Clean at assembly, never in training.** If the source has a steady
   artifact (hum, whine, SNAC comb), carry it through training and remove it in
   the voice's assembly post-chain via its notch map. Cleaning the *training*
   audio is what killed thirdreich: broadband dehum removed the hum AND the
   noise floor, and a dead floor is unrecoverable downstream. A steady tone is
   trivially notched at output; a dead pause can never be un-deadened.
3. **Curate, don't accumulate.** More data made models WORSE: an 8 h corpus
   with an 8 dB per-clip brightness spread taught the model to oscillate
   muffled/sharp per chunk. ~2 h of the brightest, liveliest clips (spread
   ≤~3 dB) beat the full corpus on brightness, consistency, AND prosody.
   Voice cloning wants tight overfit to the target voice, not generalization.
4. **Pauses must be spectrally alive.** Dead digital silence SNAC-encodes into
   a handful of identical frames → a silence attractor → runaways (the
   CUDA-vs-Metal EOS cliff). Achieve this by CUTTING FROM A SOURCE WHOSE FLOOR
   IS ALIVE (§1). Only if no such source exists, lay a bed (§2).
5. **Hums are learnable; dead silence is not recoverable.** The asymmetry drives
   principle 2. A model trained on hummy audio reproduces the hum — annoying,
   and removable at assembly. A model trained on dead pauses loses EOS margin —
   and no downstream step can give it back.
6. **Clip edges must be breath-safe.** 95-100% of raw cuts end inside the kept
   trailing pause where the narrator's next-sentence inhale lives; trained
   with tails, that becomes whispery ghost-breaths in every rendered pause.
   The builder trims edges to 200 ms past last speech (see §5).
7. **The GATE BATTERY picks the keeper — not a fixed offset from the loss
   minimum.** Run §7's full battery on EVERY kept epoch and take the winner
   (Owen 2026-07-24). Do not default to "settled" or "settled+1": both are
   guesses about where quality lives, and the battery measures it directly.
   *Why the fixed-offset rules died:* settle+1 came from the 2026-07-13
   pause-consolidation finding; eval-min replaced it on 07-18; then tr_rv2 on
   07-24 measured **dropped-text 2 / 1 / 0 across ep220 / ep330 / ep440** — the
   eval-min epoch was the WORST of the three and the deepest was flawless. Any
   fixed offset would have shipped a lesser model. eval_loss is
   speech-frame-dominated and cannot see EOS margin, looping, or pause
   behaviour, so it selects candidates and nothing more.
   *Corollary:* whether quality rises or falls with depth is a property of the
   CORPUS. On a dead-floor corpus tr_rv1 eroded (20/20 → 18 → 15); on the
   live-floor re-cut tr_rv2 improved with depth. So the battery must be re-run
   per corpus — you cannot carry a keeper offset across runs.
   On 2 h corpora the loss runs away hard past the bottom (no late dip — tested
   to +4 epochs).
8. **One GPU job at a time.** Stacked vLLM/training jobs blew commit memory
   through vmmemWSL and OOM'd the host. Sequencers own the GPU; nothing else
   touches it while they run. Every GPU chain script creates and removes
   `%APPDATA%\BookForge\external-gpu-job.lock` so BookForge's process sweeps
   skip it.

9. **NARRATION ONLY — exclude character voices (Owen, 2026-07-24).** Orpheus has
   no speaker map. It can tell that *someone* is speaking but not *who*, so
   character voices in the corpus become a capability it fires AT RANDOM: "at its
   best, good character voices at complete random; at its worst, the character
   voices make a mess of the audio." Narration-only reads dialogue in the
   narrator's own voice, which is the wanted behaviour. Owen sent fiction through
   the narration-only deathstalker and **scrapped the dialogue model entirely**.
   ⚠ The old "with ~0 dialogue clips the model reads dialogue turns awkwardly"
   note was an **e2a BUG, since fixed** — it was never a data-coverage finding.
   Do not resurrect it as an argument for keeping character voices.

---

## Step 0 — Source audio and transcript

Per voice you need: the full-book master audio (FLAC/WAV, the cleanest
available generation — NEVER a lossy re-encode when a master exists) and its
aligned VTT. Existing builds: `E:\training\deathstalker\build\` (Marked Man),
`E:\training\mistborn\build\` (Alloy of Law), `E:\training\thirdreich\source\`
(merged FLAC + VTT). For a NEW book: align with `align_from_epub.py` /
`correct_vtt.py` per `AUDIBLE_BOOK_CLEANING.md` in this repo.

**Choosing WHICH master to cut from — the decision that matters most.**
Measure the candidate masters' pause floor before choosing:

```
# per-clip / per-region floor: is the quiet part alive or dead?
python - <<'EOF'   # or reuse pause_frame_conc.py on a trial cut
# want: quiet-region RMS around -50..-60 dBFS, ~0% exactly-zero samples
EOF
```

| master | floor | verdict |
|---|---|---|
| raw / gain-only-leveled | −50..−60 dBFS, ~0% zero samples | **CUT FROM THIS** |
| broadband-denoised / dehummed | −100 dBFS or lower, high zero-sample % | do not cut from this |

A raw master with a hum is still the right choice (principles 2 and 5) — take
the hum, notch it at assembly.

## Step 0b — TEXT QC (MANDATORY; added 2026-07-24 after it broke a model)

**The training text must match the spoken audio IDENTICALLY.** A token in the text
that the narrator never speaks teaches the model that the token means STOP. This is
not theoretical: 8% of thirdreich's clips carried endnote reference markers, and the
resulting model truncated at exactly those points. It was diagnosed for two days as
an audio/EOS problem. See §Case study.

Run these BEFORE cutting. Every one is cheap; a corpus is not allowed past here dirty.

1. **Endnote / footnote reference markers.** Academic titles mark notes with a
   digits-only superscript (`<sup class="calibre11">55</sup>` — 1,864 in one Evans
   volume). BookForge strips these at extraction as of 2026-07-24
   (`epub-processor.ts`, digits-only `<sup>`; lettered superscripts like `25<sup>th</sup>`
   are KEPT). Verify anyway:
   ```
   # in the VTT/corpus text: a bare integer between sentence punctuation and a capital
   grep -cE '[.!?]\s+[0-9]{1,3}\s+[A-Z]' <text>     # want 0
   ```
   FICTION is usually clean (Alloy of Law: 0). NONFICTION is the risk case.
2. **Heading / TOC bleed.** ALLCAPS runs (`GOSPELS OF HATE I .`) and `Part 3 - …`
   lines. Chapter titles the narrator DOES announce are fine; a table of contents is
   not. The aligner's fit tests drop unspoken front matter for you (below) — confirm
   they did.
3. **ASR-fallback cues must be EXCLUDED from a training corpus.** `align_audiobook.py`
   fills unmatched audio holes (>= `--hole-min-s`, default 30 s) with rough-transcript
   text and tags them `NOTE asr-fallback`. Those cues ARE the intro music, the Audible
   credits and the outro — which also means the "trim the first/last 30 seconds"
   problem is solved STRUCTURALLY, not by guessing a timestamp. Drop them.
   (An older VTT with 0 `asr-fallback` tags predates tagging — re-align to get them.)
4. **Bare ASR is NEVER training text (LAW).** Ebook text is ground truth; ASR supplies
   timings only. ASR mangles proper nouns, which a voice model then learns. The aligner
   enforces this — "ebook cues always win" — and it does NOT substitute ASR when the
   narration merely paraphrases. So:
5. **Paraphrase drift needs its own check** — nothing upstream catches it. Whisper a
   sample of the finished corpus and compare coverage against the clip text (the
   `loop_gate.py check` logic). Do it BEFORE training, not after.
6. **Speaker contamination sweep** (from the ds_nr1 campaign; make it standard):
   WeSpeaker centroid over the corpus, list clips below 0.40 similarity with their text
   and duration. Catches a different-announcer promo spliced into the source (found in
   `deathstalker_rv2h`) and, incidentally, surfaces TEXT junk — the low-similarity
   thirdreich clips were the ones with page numbers and heading bleed.

### Number convention: keep ARABIC NUMERALS (decided 2026-07-24)

Leave digits as the book writes them. Do NOT pre-expand to words.
- e2a's `expandNumbersEn` DELIBERATELY refuses ambiguous shapes — year ranges
  (`1914-1918`), colons (`5:30`), fractions, digits-in-words. This book has **701 year
  ranges**, so digits reach the model at inference NO MATTER WHAT is trained. Training
  on digits covers exactly those; training on words leaves them undefined.
- The audio supplies the pronunciation, so `1933` -> "nineteen thirty-three" is learned
  directly, and word-form numbers remain trivial for the Llama-3.2 base. Digits are
  therefore strictly MORE capability, not a trade.
- Verified against the source: Pratt reads "1933" as "nineteen thirty-three" (Owen, by
  ear). Whisper CANNOT verify this — it normalizes spoken numbers back to digits, so a
  transcript comparison is worthless here. Use ears.
- Historic note: `mm_narr2h` and `mistborn_rv2h` are ~99% number-WORDS, but that is
  because FICTION spells numbers out, not because anything expanded them.
- Watch in the gate: register ambiguity (`1933` = a year vs `1,933` = a quantity) is
  learned from context — check number-bearing chunks explicitly.

## Step 1 — Cut clips (deterministic, VERBATIM)

`cut_audiobook.py` — silence-snapped, quote-safe, breath-aware boundaries.
Same source + same VTT ⇒ identical clips; re-running is always safe. Output:
a corpus dir with `wavs/` + `metadata_train.csv` + `metadata_eval.csv`
(pipe-delimited: `audio_file|text|speaker_name`), 24 kHz mono.

**Run it in WSL `orpheus_tts`** — `cut_audiobook.py` imports librosa, which the bare
Windows `python` does not have (hit 07-25). Use
`/home/telltale/anaconda3/envs/orpheus_tts/bin/python` with `/mnt/e/...` paths.

**Prerequisite — the silence map** (`--silence-map`), or cuts snap to librosa's
detection instead of real silences and can land mid-word:
```
python autoeditor_silences.py <master>.flac <out>_silences.json
# fps is pinned at EXACTLY 30.0 with a hard guard; never let it be derived as
# frames/duration (auto-editor drops the trailing partial chunk, which would smear
# a ~1.5s deficit across the whole book). HoA: 39,334 silences, 29.7% of the book.
```
Also confirm the decoded master's duration matches the VTT's to the microsecond
(`ffprobe` vs the align's `audioDurationSeconds`) — the VTT is only valid if it does.

**The locked verbatim cut (rv recipe, 07-21):**
```
python cut_audiobook.py \
  --audio "<RAW master>.flac" --vtt "<aligned>.vtt" \
  --out-dir <corpus> --source-name <token> \
  --target-minutes 0 --max-clip 20 --mix 2:3-10,3:10-20 \
  --dialogue-aware
# NO --trail-cap, NO --gain-db: pauses and tails are kept EXACTLY as the
# narrator performed them. --max-clip 20 respects the 2048-token ceiling.
```
Keep the natural trailing pause — the clip ends at the far edge of the
sentence-end silence. This is deliberate: the model learns to end utterances
with a pause. (The kept tails still contain breaths — expected; §5 handles the
edges.)

A VTT aligned against a *dehummed* render applies unchanged to the *raw*
master when the two are sample-aligned (durations match to the microsecond).
Verify before relying on it.

## Step 1b — Split narration from character voices, and VERIFY the speaker

**TEXT SELECTS, EMBEDDINGS VERIFY.** Two different failures, two different tools;
they are not interchangeable.

```
# 1. narration vs character voice — by the BOOK'S OWN quote marks. Exact.
node cli/clipforge-process.js narration --corpus <corpus>
#    -> metadata_narration.csv / metadata_dialogue.csv (build_2h_corpus.py eats either)

# 2. is every clip actually the narrator? Centroid from the narration set.
node cli/clipforge-process.js verify --corpus <corpus>      --reference metadata_narration.csv --compare metadata_dialogue.csv
```

**Why text and not embeddings for the split — MEASURED on Alloy of Law** (centroid
built from narration): narration median **0.9316** / min **0.5655**; dialogue median
**0.8644** / min **0.5901**. Character voices DO pull similarity down, but the
distributions **OVERLAP** — narration dips below dialogue's floor — so no threshold
separates them. Any embedding threshold leaks partial character voices, which is the
exact failure being eliminated. A quote mark is a fact about the text.

**What `verify` is for:** a FOREIGN voice — publisher promo, staff announcer,
co-narrator. The `deathstalker_rv2h` HarperAudio promo scored **0.051**. Nothing
text-based can see it, because the words are fine.

⚠ **The 0.40 flag / 0.28 floor are WESPEAKER-SCALE.** The `speakers` verb uses
resemblyzer, where a genuinely different speaker still scores **~0.79** — passing it
0.40 flags NOTHING and reports a clean corpus regardless of contents. Never carry a
threshold between embedders.

**Known residual:** unquoted direct speech ("...had said X") is invisible to quote
flagging — 4.8% of clips on Marked Man, 65 on Alloy of Law. `narration` COUNTS and
reports them and leaves them in: over-filtering costs more narration than the
residual costs quality.

**Nonfiction:** there are no characters, so quoted material (documents, speeches) is
read in the narrator's own voice and the split buys nothing. Measure before
discarding a third of the book on a fiction-shaped assumption.

## Step 2 — (FALLBACK) Harvest a room-tone bed

**Only when no master with a live floor exists.** Superseded as the default by
the raw-verbatim pivot — but it remains the proven rescue for a dead-floor
corpus, and it beat even a naturally-live corpus in the twins (see §Evidence).
Spec (implemented in `pipeline/build_rebuild_corpora.py`, `harvest_bed` +
`filter_bed`):

- Collect quiet regions: 50 ms windows with RMS in **-58..-45 dBFS**, merged
  runs ≥ 200 ms, from the first ~40 min of the source master.
- Join with **equal-power overlap crossfades** (10 ms). Never concat ramped
  pieces — the V-dips read as dead air.
- **HP 120 Hz** (kills the LF rumble users previously had to de-hum —
  never HP the speech, the narrator f0 lives down there).
- **Iteratively notch** any narrowband line ≥ 6 dB above local median
  (300 Hz–11.5 kHz, Q 30). Raw beds carry real whines (thirdreich: +11 dB
  at 4430/4786 Hz).
- **NO low-pass. NO renormalization to unit RMS** — renorm packs removed-band
  energy into the audible midband (+6 dB perceived hiss, ear-rejected).
  Scale the finished bed file to **RMS -6 dBFS** exactly.
- Verify with a LOCAL median (not a global one — noise slopes, so a global
  median flags the low end on every clean bed): no residual line > ~6 dB over
  a ~1 kHz-wide local median in 300 Hz–11.5 kHz.

Existing beds: `tr_bed_v2.wav` (+6.0 dB worst line, 1 bin — cleanest),
`aol_bed_v2.wav` (+6.9 dB), `mm_bed_v2.wav` (+7.6 dB, still produced a 20/20
model).

## Step 3 — Brightness + cleanliness census

```
python pipeline/brightness_census.py <corpus_dir> /home/telltale/<voice>_bright
```
Scores every clip: brightness ratio (3.5–8 kHz minus 0.3–2 kHz, dB) and RMS
dynamics (prosody proxy). Prints the histogram, the p90-p10 spread, and
chronological drift. Raw books measured 5.5–8.0 dB spreads.

When cutting from a RAW master, also score **cleanliness** per clip — the raw
book is not uniformly clean, and the curated 2 h should avoid its worst
stretches:
- quiet-region noise floor (dBFS) — reject the noisiest decile
- narrowband tone excess over a local median — reject clips with a strong hum
  or whine relative to the book's own median
- clipping / dropout fraction

## Step 4 — Curated selection

Rank by brightness, cap outliers at p99.5, drop the bottom dynamics decile
(monotone), drop the worst cleanliness decile, take the **top 480 clips (~2 h)**,
hold out 40 for eval. Expected result: spread narrows to ~2–3 dB, mean rises
several dB. **The selection IS the quality lever.**

## Step 5 — Build the training corpus

**Default (raw-verbatim):** selection only — clips are copied bit-identical,
natural pauses and tails intact, no bed, no tails synthesized.
```
python pipeline/build_2h_corpus.py <src_corpus> <dst_corpus> --verbatim \
       /home/telltale/<voice>_bright
```

**Fallback (dead-floor source only):** the bed path, which additionally applies
```
python pipeline/build_2h_corpus.py <src> <dst> <bed.wav> /home/telltale/<voice>_bright
```
1. **Breath-safe edges**: trim to 200 ms past the last >-30 dB speech window,
   50 ms fades. Lead-in: 150 ms before the first speech window — UNLESS the
   pre-onset region carries breath-band (-55..-30 dB) energy, then cut to
   **40 ms before onset with a 15 ms fade** (leading-inhale rule).
   *Why the lead rule (mb_2hb postmortem, 2026-07-20): the trailing fix alone
   left AoL clips 94% breathy at the START — an audible inhale before every
   phrase, which the model learned to reproduce as an utterance-initial breath.*

   **NOT the ghost whisper — do not conflate these (Owen, 07-25).** An earlier
   version of this line blamed breathy leads for the ghost-syllable-in-pauses
   failure. Wrong. The 6-arm sprint + full trains (07-21) proved the ghost came
   **ONLY from the ADDED hiss bed**: a 60 s bed sliced across 480 clips is
   memorizable, so the model emitted it in pauses and SNAC renders low-level hiss
   codes as syllable babble (w2/w3/mb_2hc all ghosted; pink bed rendered as hiss).
   The model DROPS true random noise — w4, verbatim Marked Man with an atrocious
   −60 dBFS floor, rendered near-silent clean pauses. The law is **learnable vs
   unlearnable, not natural vs synthetic.** Breathy leads are a separate and much
   smaller cosmetic issue, and they are irrelevant under raw-verbatim, which adds
   no bed at all.
2. **Bed under the whole clip** at **-65 dB**, RANDOM offset per clip (random
   per-tile offsets for clips longer than the bed — periodicity re-creates
   the silence attractor; measured: hum bed = as bad as dead silence).
3. **Punctuation-scaled pause tail** of the same bed: **0.40 s** after
   terminal punctuation (`.!?"'`), **0.20 s** after continuations (`,;:—-`).
   Tail shrinks/skips to keep total ≤ **19.9 s**.

**Gate the corpus before burning GPU — this is the cheap gate, always run it:**
```
python pipeline/pause_frame_conc.py <dst_corpus>/wavs 20   # want ~0.3-0.5%; dead silence = 27-44%
python pipeline/breath_edge_census.py <dst_corpus> 300     # ENDS gate: dead >= 95%, breathy ~0%
python pipeline/lead_probe.py <dst_corpus> 300             # STARTS gate (the real one):
#   breathy leads (>=60ms, peak -50..-32dB) ~0%; lead p90 <= ~130ms
#   CAUTION: the census's "start" bucket measures the first 300ms RMS and
#   cannot tell an inhale from a soft speech onset — mb passed the lead probe
#   at 0% while the census still said "93% breathy" (Kramer starts phrases
#   quietly). Gate STARTS on lead_probe, ENDS on the census.
```
**Corpus floor check (added 07-24 — the one that would have caught thirdreich):**
a healthy corpus measures **~0% exactly-zero samples** and a pause floor around
**−50..−60 dBFS**. Measured: `mm_narr2h` 0.23% zeros / −55.4 dBFS (ships
flawlessly) vs `thirdreich_rv2h` **19.18% zeros / −107.5 dBFS** (produced the
looping model). If zeros > ~2%, STOP — cut from a different master or lay a bed.

## Step 5b — Match the TAIL DISTRIBUTION to the reference corpus (added 07-25)

**Why this exists.** The narrator's natural end-of-sentence pause is trained in
verbatim and reproduced on *every generated chunk*. Pratt's median tail is 1.63 s
where Rohan's is 0.98 s, so thirdreich emitted ~0.6 s of extra silence per chunk.
Generation is KV-limited, so clip length costs throughput **quadratically** (longer
chunks cost more tokens each AND fewer fit concurrently, and the overflow shows up
as `RECOMPUTE` preemption — work discarded and re-run). Trimming buys back that
silence for free: it is pure overhead, identical on every chunk, and the assembly
stage injects the production gap anyway.

**Honest bound on the win (corrected 07-25).** An earlier version of this section
claimed the tail was "most of" the measured **30 vs 70 chunks/min** gap. That is
NOT supported. Measured on the tr_tp1 → tr_tt1 pair: rendered trailing pause fell
1.21–1.43 s → 0.62–0.75 s, but clip median only fell ~10% (16.30/15.06 s → 14.08 s),
which under 1/L² predicts roughly 30.7 → ~38 chunks/min. A real lever, not the
whole gap. Two hypotheses for the remainder are already **ruled out**:

- *Speaking rate* — Pratt is FASTER than Rohan per second of speech (29.8 vs 27.2
  chars/sec measured over the two corpora), so it points the wrong way.
- *Trained clip length* — the ds corpus clips are LONGER (15.94 s median vs
  12.03 s), which would make ds slower, not 2.3× faster.

So the remaining gap is still **unexplained** and must be settled by measuring an
actual render, not by inference. A strong candidate not yet tested: chunks/min is
not comparable across books if the packer emits different chunk sizes — long
nonfiction sentences pack into fewer, larger chunks, which lowers chunks/min at
identical words/min. Measure words/min, or the same book on both voices, before
believing any voice-to-voice throughput comparison.

```
python pipeline/trim_tails.py <src_corpus> <dst_corpus> --median-to 0.98
```

**SCALE, never CLAMP.** `--median-to` multiplies every tail by one constant factor
derived from the corpus's own median, so natural variation survives. `--target`
(clamp) flattens every tail to the same length — p90 collapses onto the median, and
a periodic, identical pause is exactly the condition that re-creates the silence
attractor. Clamp exists only for diagnosis; the run-book uses scale.

**The breath rule is structural, not statistical.** The cut point is the first
frame at/after the goal that is **true silence (≤ −55 dBFS)**, so it *cannot* land
in a breath (−55..−30 dBFS). A clip whose tail holds no silence frame at/after the
goal is **left untouched and reported** — never trimmed on a guess. Measured on
thirdreich before trimming: 763 breath runs across 574 clips, onset median 0.00 s /
p90 0.16 s (Pratt's breaths are the trailing exhale), duration median 0.10 s, and
**0 runs straddling the cut**.

Verify the *distribution*, not just the median — and re-run the SNAC gate, because
trimming changes what the codec sees at the end of every clip:
```
python pipeline/pause_frame_conc.py <dst_corpus>/wavs 20     # must stay ~0.3-0.5%
```
thirdreich, 609 clips (2.135 h → 2.027 h, 6.5 min removed):

| corpus | tail p10 | median | p90 | max | ends mid-breath | SNAC conc |
|---|---|---|---|---|---|---|
| pretrim | 1.06 | 1.63 | 2.17 | 2.73 | 14 | 0.7% |
| **trimmed ×0.601** | 0.65 | **0.99** | 1.31 | 1.65 | **0** | **0.5%** |
| mm_narr2h (reference) | 0.64 | 0.98 | 1.43 | 2.49 | 139 | 0.4% |

Also check the cut cannot click: final-5 ms peak should sit at the floor
(measured −90.3 dBFS median, worst −69.5 — a 20 ms fade is applied at the cut).

**Do NOT trim the whole source corpus** — trim the curated 2 h that will actually
train, after selection. Selection reads brightness/cleanliness, which tail length
does not affect, and trimming 30 h to throw away 28 of them is wasted work.

## Step 6 — Train

```
bash pipeline/run_rv_voice.sh <prefix> <voice_token> <dst_corpus> [gate_epochs]
```
which wraps `pipeline/run_2h_retrain.sh` and additionally arms the EOS boost on
every merged epoch and gates the last N. The training call itself (env
`orpheus_train`):
```
orpheus_owen.py --source-name <token> --recut-dir <corpus> --out-base <out> \
  --mask-prompt-loss --no-dedup --lr-schedule constant_with_warmup \
  --epochs 8 --stop-overtrain --overtrain-patience 2 train
```
~5 min/epoch at 440 clips on the 3090 Ti; settles around epoch 3; whole run
≈ 25–40 min. Every epoch checkpoint is kept, then merged + registered as
`<prefix>_ep<N>` (local models.json) at **repPenalty 1.10** — 1.15 audibly
wobbles, 1.10 doesn't (ear-tested twice).

### 6a — EOS margin comes from the boost, not from corpus tricks

`backends.vllm.eosBoost: 8, eosBoostStart: 2.0` biases the EOS logit ONLY after
generation passes 2.0× the chunk's expected token count, ramping with the
overrun. It cannot truncate unspoken speech. **vLLM only** — MLX/Metal resolves
the EOS tie differently and does not need it.

Arm it on every epoch before gating, and stamp it on the deployed entry. A voice
deployed WITHOUT it runs unguarded — that is exactly the thirdreich bug.

**Caveat (measured 07-24):** the boost ends a silence loop; it does not prevent
one. A margin-poor model still generates 12–25 s of silence before the boost
fires, and e2a deliberately does NOT trim that (no-fallback: abnormal silence
must be VISIBLE). The boost is a safety net, not a fix for a bad corpus.

## Step 7 — Gates (all mandatory before deploy)

1. **Greedy EOS gate** — `eos_gate.py <model_dir> <token> <corpus_dir> 20 1.10 8 2.0`:
   20 real 200–360-char chunks, greedy and deterministic. Require **20/20**.
   eval_loss cannot see EOS margin; this can.
2. **Loop / completeness gate (added 07-24 — MANDATORY).**
   ```
   # stage 1 (WSL, env orpheus_tts — holds the GPU)
   python pipeline/loop_gate.py render <model_dir> <token> <corpus> <out_dir> 20 1.10 8 2.0
   # stage 2 (any env with faster_whisper — BookForge's e2a-env has it)
   python pipeline/loop_gate.py check <out_dir> medium.en
   ```
   The chars-per-second truncation guard in e2a only catches audio that is too
   SHORT for its text. A model that loops back and re-speaks a phrase produces
   audio that is too LONG, so it sails past every guard in the system and ships
   duplicated sentences. Fails on a repeated word 6-gram (LOOP) or transcript
   coverage below 85% of the source text (DROPPED). Runs SAMPLED at production
   settings, not greedy — looping is a sampled-path failure a greedy probe does
   not reproduce. **This gate did not exist before 07-24; its absence let a
   broken thirdreich ship in July and survive an audit on 07-24 that passed it
   20/20 on termination alone.**

   **BLOATED, the third verdict (added 07-25).** LOOP and DROPPED both missed a
   clip that ran 38.9 s for a text its own sibling epoch read in 17.4 s — coverage
   was 93.1% (above the floor) and 15.6 s of dead air repeats no n-gram, so it was
   called clean. Now flagged on seconds-per-character vs the probe set's own median,
   ceiling **1.5×** (measured over 80 clips: clean all under 1.3×, defects 1.80×
   and 1.94×). Relative to the probe set, so it travels across speaking rates.
3. **Pause gate — TWO instruments, and the second one is the important one.**
   ```
   python pipeline/pause_meter.py <gap-0 render>.wav          # INTERNAL pauses
   python pipeline/render_pause_report.py <render_dir> [...]   # + TRAILING pause
   ```
   `pause_meter.py` measures the gaps the model puts BETWEEN sentences — it
   deliberately excludes the run at the very end of a clip. That excluded run is
   the one that matters most: it is emitted on every chunk and it is where the
   "never-ending pause" lives. `render_pause_report.py` reports both.

   **PAUSE CONSOLIDATION gates the keeper (proven 07-25).** Pauses are chaotic for
   the first epochs and then settle, and *the settling epoch varies per model.*
   On tr_tt1 the eval_loss MINIMUM (ep280, 3.611 — the best loss in the run) was
   PRE-consolidation and emitted **18.6 s** of trailing dead air; ep420 emitted
   1.12 s. Two cheap detectors:
   - **trailing max** collapses ~16 s → ~1 s
   - **speech fraction** of the clip jumps ~65% → ~75%

   So eval-min SELECTS the candidate and consolidation GATES it. If eval-min is
   pre-consolidation, take the first consolidated epoch — that is not an ear
   override, the eval-min epoch is simply defective. Also still true: a keeper that
   squishes INTERNAL pauses to ~0.2 s (narrator's own ~0.5 s) is the opposite
   failure, and why the ear decides between consolidated candidates.

   **Consolidation is a VETO, not a RANKING (Owen, 07-25).** *"The epoch with the
   best pausing might not sound the best. If loss regresses too much it might sound
   awful. Pausing isn't the only consideration."* An epoch emitting 18 s of dead air
   is out regardless of timbre — but among the SURVIVORS, prefer the one closest to
   eval-min, and let the ear rank. Do NOT chase the shortest tail down the epochs.
   Cross-check the survivors on **brightness spread and level consistency** (the
   wobble proxy) before auditioning; measured on tr_tt1's four epochs:

   | epoch | brightness spread | level sd | eval_loss | |
   |---|---|---|---|---|
   | ep140 | 8.1 dB | 1.07 | 3.668 | pre-consolidation |
   | ep280 | 6.4 dB | 0.85 | 3.611 (min) | pre-consolidation |
   | ep420 | **2.6 dB** | **0.40** | 3.616 | keeper |
   | ep560 | 3.2 dB | 0.51 | 3.682 | looser — over-training starting |

   Note what this run does NOT show: any trade-off between pauses and consistency.
   They improved TOGETHER (pre-consolidation epochs wobble 6–8 dB), and the drift
   back up at ep560 is the over-training Owen warned about, appearing mildly. So do
   not ASSUME a pause-vs-quality tension exists — measure it. These numbers cannot
   see graininess, mispronunciation or prosody, so they narrow the audition, never
   replace it.
4. **Ring gate** — `pipeline/ring_probe.py <model_dir> <token> <state.json>`:
   greedy-decodes 6 chunks to audio, scans 5–12 kHz for the SNAC comb. The
   comb is codec-intrinsic (~8.4 kHz on full-band corpora) — the gate's
   OUTPUT is the voice's **notch map**: the frequencies to notch at assembly.
   Fixing it in the corpus is impossible without muffling; don't try.
5. **Hum check (raw-cut voices) — judge on ABSOLUTE level, never on excess over
   a local median.** A raw-cut voice DOES learn the source hum (measured
   2026-07-24: thirdreich training clips carried 120 Hz at +21 dB over local
   median, the render reproduced it at +26 dB — slightly louder than trained).
   That relative figure is worthless on its own: the same tone measured
   **−88.3 dBFS absolute, 69 dB below speech**, and Owen's verdict on the render
   was "sounded flawless." **Only notch a tone that is audible.** Rule of thumb:
   a line more than ~50 dB below speech level needs no treatment; notching it
   costs real voice body (the trial notch here removed 1.2 dB of 80–300 Hz
   speech energy to suppress something inaudible) and 120 Hz sits inside a male
   narrator's fundamental. Report absolute dBFS and dB-below-speech, then let
   the ear decide.
6. **Ear gate** — render the battery's top candidates via the BookForge CLI:
   ```
   python cli\bookforge-tts.py --tts --engine=orpheus --voice=<prefix>_ep<N> \
     --input <3-paragraph sample.txt> --out <out.wav> --sentence-gap 0.6
   ```
   Judge with production gaps (0.6 s). Owen picks the keeper by ear from the
   candidates the battery ranked (principle 7).

## Step 7b — Measuring THROUGHPUT honestly (added 07-25, after three bad estimates)

Throughput is the one metric that has repeatedly been reported wrong here. Every
shortcut has a bias, and they point in different directions:

| method | bias | verdict |
|---|---|---|
| the app's Speed readout mid-render | INFLATED by the initial batch burst (reported 69 when the real figure was 30.7) | never quote it |
| `chunks / totalElapsedSeconds` | DEFLATED — folds in vLLM model load and prep | never quote it |
| a 3-minute window on the flac count | ±33% QUANTIZATION — the worker writes sentence flacs in **batches of 64**, so a short window resolves to the nearest batch. Three consecutive 3-min windows on ONE unchanged run read 64.0 / 64.0 / 85.3 | never quote it |
| **cumulative delta over ≥ 8 min, generation only** | small | **use this** |

The measurement recipe:
```
# real BookForge path, GPU, fast tier (pipeline/../scratch bench.sh does this)
python cli/bookforge-tts.py --audiobook --project <book> --voice <id> --tier fast --fresh
# progress signal = COUNT OF .flac in <session>/<hash>/chapters/sentences/
#   NOT .wav (the worker writes flac), and NOT session-state.json, which is the
#   hyphenated PREP state file and carries no progress fields at all.
# wait until >=128 flacs exist (past warmup), then delta over >= 8 minutes.
```

**chunks/min is NOT comparable across books.** The packer emits different chunk sizes
for different prose, so a voice can look slow purely because a book packs into fewer,
larger chunks. Always benchmark the candidate and the reference voice on the SAME book,
in the same session, back to back. `--device CPU` in the process list is the `--prep_only`
chunking phase and does not mean generation is on CPU — check `nvidia-smi` instead.

## Step 8 — Deploy the keeper

- Register/deploy as the live voice id (`deploy_voice.sh` for HF + Mac), with
  `maxCharsPerSec` per the rate-ratchet recipe, `repPenalty 1.10`, and
  **`eosBoost 8 @ 2.0`**.
- Set `maxCharsPerSec` from the voice's MEASURED rate, don't copy another
  voice's. The gate's healthy chunks give it: thirdreich measured 15–20.5 ch/s,
  so the stock 19.0 guard sits inside its natural range and false-trips → 21.5.
- Attach the voice's **notch map** (ring gate + hum check) to its assembly
  post-chain, plus the standard denoise option.
- Production pauses are **deterministic gaps at assembly** (`sentence_gap`
  0.6 s, per-voice in `electron/data/orpheus-models.json`). Model tails exist
  for TRAINING (EOS safety), not for pacing — don't render books at gap 0
  unless the voice's measured tail already matches its narrator.
- **Tune `sentenceGap` from a real render — see Step 8b. It is NOT optional and
  it does NOT survive a retrain.**
- **Record the deploy.** A trained-but-undeployed keeper is worse than no
  keeper: thirdreich was retrained four times (tr_v2, tr_2h, tr_2hb, tr_rv1)
  and none shipped, while the Jul-18 model stayed live for six days.

---

## Step 8b — Tune the assembly gap (MANDATORY; added 07-25 after both new models shipped wrong)

Chunks are separate generations. At assembly e2a concatenates them, so **every
chunk boundary is a sentence boundary that the model did not itself perform.**
`sentenceGap` is the silence injected there. Get it wrong and the error repeats
thousands of times per book.

### The three numbers (all from ONE command)

```bash
python pipeline/measure_sentence_gaps.py <project>/stages/03-tts/sessions/<lang>/ebook-*/*/chapters/sentences "<voice>"
```

| number | what it is | why it matters |
|---|---|---|
| `modelSelfTailS` | trailing silence the model emits on its own | **this is what the inject stacks onto** |
| `modelInternalGapS` | gap the model leaves between sentences *inside* one packed chunk | the model's own sentence rhythm — what a join must imitate |
| join | `modelSelfTailS + sentenceGap` | what the listener actually hears at a chunk boundary |

### You do NOT need a book render — Step 7's gate clips are enough

`loop_gate.py` already renders 20 clips per checkpoint into
`C:/tmp/loopgate/<voice>_<epoch>/`. Those are **raw** renders (no e2a `_save_audio`),
so pass `baked_gap = 0` and the tail you measure IS `modelSelfTailS` with nothing to
subtract:

```bash
python pipeline/measure_sentence_gaps.py C:/tmp/loopgate/<voice>_<epoch> "<voice>" 0
```

That means the gap can be tuned **at gate time, before the voice ever renders a book** —
which is how mistborn was closed out instead of waiting for its first title.

It also gives a free cross-check on any voice you later measure from a real cache. For
tr_tt1 ep420 the two agree within **0.03 s** — gate render 0.755 / 0.565 vs the 400-clip
e2a cache's 0.740 / 0.590 — which validates the method *and* the 0.6 s baked-gap
assumption the cache figures depend on. If they ever disagree by more than ~0.1 s,
distrust the cache number: something injected a non-default gap.

The cached sentence files are `[lead] + [model audio, tail INTACT] + [trail_gap]`
— the trailing-silence trim was removed from `_save_audio` on 2026-07-11 under the
no-fallback rule, so the model's real tail survives. `trail_gap` is 0.6 s unless
`ORPHEUS_SENTENCE_GAP` is exported; subtract it. Internal gaps are untouched by
anything, which is what makes them ground truth.

### The target

```
modelSelfTailS + sentenceGap  ≈  0.75 × modelInternalGapS
```

0.75 because that is where **deathstalker** sits (tail 0.31 s, internal 0.42 s,
`sentenceGap` 0.0 → join 0.31 s) and it is the voice approved by ear across many
books. A join slightly TIGHTER than a within-chunk boundary reads as natural;
longer reads as a stall. If the bare tail already exceeds the target, the answer
is `sentenceGap: 0.0` — you cannot subtract by injecting.

### FIRST sanity-check the denominator — a model can have a BROKEN rhythm

The target imitates `modelInternalGapS`, which only works if the model's own rhythm
is healthy. **Validate it against the narrator before you trust it**: compare against
`measuredSentenceGapS` (the source), or against a sibling model trained on the same
narrator. If the model's internal gap is far below the narrator's, the model is
under-pausing and imitating it will produce breathless audio.

Worked example — the same Deathstalker narrator, two models:

| model | internal gap p25 / median / p75 |
|---|---|
| deathstalker (full) | 0.25 / **0.42** / 0.58 s |
| deathstalker_narration | 0.14 / **0.18** / 0.26 s |

Every percentile roughly HALVED. Cause: a narration-only corpus is built by cutting
dialogue OUT and splicing the narration either side back together, so wherever a
dialogue turn was removed the two narration halves butt against each other. The model
sees thousands of "sentence boundaries" with near-zero pause and learns to under-pause.
That is a **Step 1b corpus-construction artifact, and no assembly gap can repair it** —
`sentenceGap` only touches chunk JOINS, while these bad pauses are *inside* the chunk.

Two consequences:

- **Do not chase a high join/internal ratio when `internal` is the broken number.**
  deathstalker_narration at the 0.6 default reads as "4× over-paused", but 0.6 gives
  0.72 s joins, which is much closer to the narrator's real 0.42 s rhythm than the
  model's own 0.18 s is. The ratio was inflated by its denominator.
- **When cutting a narration-only corpus, insert the narrator's measured gap at every
  dialogue excision** rather than butt-splicing. Otherwise the model learns the splice.

### The mistake this step exists to prevent

thirdreich's `sentenceGap` was set to 0.3 from a 07-23 probe reporting "the model
self-produces ~0.5 s". That 0.5 s was the **internal** gap. The inject stacks on
the **tail**, which measured **0.74 s** — so joins landed at **1.04 s**, 1.76× the
model's own 0.59 s rhythm, when the stated intent had been ~0.8 s. Two quantities,
one name, 0.24 s of extra dead air at every chunk boundary of every book.

### Re-measure after EVERY retrain

`modelSelfTailS` is a direct product of the corpus's tail distribution, so
**Step 5b changes it by construction.** A `sentenceGap` inherited across a retrain
is stale by definition. Likewise a voice slot re-pointed at a model trained on a
different book (mistborn: The Final Empire → Hero of Ages) invalidates
`measuredSentenceGapS`, which is a property of the *narrator's source*, not the model.

### Worked numbers (2026-07-25)

| voice | selfTail | internal | `sentenceGap` | join | ratio |
|---|---|---|---|---|---|
| **deathstalker** *(reference)* | 0.310 | 0.420 | 0.0 | 0.310 | **0.74×** |
| thirdreich tr_tt1 ep420 | 0.740 | 0.590 | 0.0 | 0.740 | 1.25× |
| mistborn mb_hoa1 ep344 | 0.975 | 0.540 | 0.0 | 0.975 | 1.81× |

Both new voices sit above the reference ratio and **cannot be brought down by inject** —
`0.0` is already the floor. That is a Step 5b signal, not a Step 8b one: thirdreich got
tail scaling (×0.601) and landed at 1.25×; mistborn never did and sits at 1.81× with a
wide spread (p10 0.46 / p90 1.58). **If a voice measures above ~1.3×, the fix is to scale
its corpus tails on the next retrain, not to fiddle with the gap.**

### Checklist before flipping a voice live

- [ ] a real render exists for **this** checkpoint (the Step 7 gate clips count)
- [ ] `measure_sentence_gaps.py` run on it, ≥ 300 clips
- [ ] `modelInternalGapS` sanity-checked against the narrator — if it is far under,
      STOP: the model's rhythm is broken and this is a corpus bug, not a gap bug
- [ ] `modelSelfTailS`, `modelInternalGapS`, `gapMeasuredAt` written to the catalog
- [ ] `sentenceGap` satisfies the 0.75× target, or `0.0` with a note saying why
- [ ] `_gapNote` states which checkpoint and which book the numbers came from

---

## Evidence — why the defaults are the defaults

**Corpus floor decides EOS margin** (twins: same 120 clips, same config, only
the audio differs):

| corpus arm | greedy runaway | sampled |
|---|---|---|
| untouched (broadband-cleaned) | 60% | 70% — broken |
| pause-capped | 45% | 10% |
| **noise-injected bed** | **5%** | **0/20 — best** |
| naturally-live "novels" source | 15% | 0% |

The bed beat even a naturally-live source — which is why it stays as the
fallback rather than being deleted.

**Live corpora vs the dead one** (measured 07-24, 60 clips each):

| corpus | zero samples | pause floor | outcome |
|---|---|---|---|
| `mm_narr2h` (live deathstalker) | 0.23% | −55.4 dBFS | flawless full book |
| `thirdreich_rv2h` | 19.18% | −107.5 dBFS | looping, dead air |

## Case study — thirdreich (the worked example of every failure above)

The Jul-18 thirdreich shipped **without a pre-deploy gate**, was later measured
at 70% greedy runaway, and stayed live while four retrains sat undeployed. On a
19-chapter nonfiction book it produced, per 945 rendered clips: 1.8% of clips
with 12–25 s of dead air, 3 completely empty clips, and phrase-level looping
("It listed Ten Commandments… It listed Ten Commandments…"). Random 80-clip
sample: 0 defects — the damage is rare and concentrated, which is exactly why
spot-checking missed it.

Arming the EOS boost took the greedy gate from 7/20 FAIL to 20/20 PASS with
**identical token counts on the 13 already-healthy chunks** — surgical, and a
genuine fix for the runaways. But it did not fix the looping, because the loop
gate did not exist. Root cause upstream of all of it: the corpus was cut from a
**dehummed** master (chosen in July to avoid a learnable 120 Hz hum), which left
19% exactly-zero samples in the pauses.

Lessons, all now encoded above: cut from the raw master and notch the hum at
assembly (principles 2 and 5); check the corpus floor before training (§5); gate
on loops as well as termination (§7.2); and deploy the keeper you trained (§8).

---

## Failure modes — signature → cause → fix

| You hear/see | Cause | Fix |
|---|---|---|
| Runaway generation, worker exit 1, CUDA-only | Dead-silence pauses → SNAC silence attractor | Cut from a live-floor master (§1); bed as fallback (§2); pause-frame conc. gate |
| **Duplicated sentences in the audiobook** | **Model loops back and re-speaks; invisible to the chars/sec guard (audio is too LONG, not short)** | **Loop gate (§7.2); retrain on a live-floor corpus** |
| **12–25 s of dead air mid-chapter** | **Margin-poor model emits silence until the EOS boost fires; e2a deliberately does not trim it** | **Retrain (§1); the boost is a net, not a fix** |
| "Audio too short for text" guard trips | Genuine early-EOS truncation | Lower `maxChars` for the voice; retrain |
| Muffled voice | Someone low-passed the speech | Never filter speech; rebuild corpus |
| Muffled↔sharp oscillation per chunk | Wide per-clip brightness spread learned | Curated selection (§3–4) |
| Flat prosody + dull at the loss bottom | Undertrained checkpoint | Audition the next epoch and let the ear override eval-min |
| Squished inter-sentence pauses (~0.2s) | Keeper picked on eval_loss alone | Run pause_meter.py on a gap-0 render and compare to the narrator baseline |
| Vocal wobble | repPenalty 1.15 | Use 1.10 |
| Faint steady ringing over voice (~8.4 kHz) | SNAC combs low-level hiss (codec-intrinsic) | Notch map at assembly (ring gate output) |
| Model truncates mid-chunk at the SAME text position every time | A token in the training text the narrator never speaks (endnote marker, TOC bleed) taught it STOP | Step 0b text QC; strip at the epub source |
| Corpus text and audio disagree but nothing flags it | Aligner keeps ebook text on a PARAPHRASE — it only substitutes ASR for absent text | Whisper-coverage a sample BEFORE training (step 0b.5) |
| A voice that is not the narrator appears in the corpus | Publisher promo / different announcer spliced into the source | `clipforge verify` — wespeaker, flag < 0.40 (step 1b) |
| Random character voices in rendered output | Character voices trained in; Orpheus has no speaker map so it fires them at random | NARRATION-ONLY corpus (principle 9, step 1b) |
| Embedding sweep reports a clean corpus that is not clean | resemblyzer thresholds confused with wespeaker's — a different speaker scores 0.79 there, not 0.17 | Never carry a threshold between embedders (step 1b) |
| AUDIBLE steady hum in output | Source hum learned (expected when cutting raw) | Notch at assembly (§7.5) — NOT by cleaning the corpus |
| Hum shows in a spectrum but nobody hears it | Measured as excess over a local median, not absolute | Judge on absolute dBFS / dB-below-speech; >~50 dB down needs no notch |
| Quiet ghost whispers / syllable babble in pauses | The ADDED BED — a short bed repeated across clips is memorizable, so the model emits it in pauses and SNAC renders low-level hiss codes as syllables. **NOT breath residue** (Owen, 07-25) | Add NO bed: raw-verbatim. The model DROPS true random noise (w4, −60 dBFS floor, rendered clean) |
| Audible inhale before every phrase | Breath residue at clip STARTS (94% on AoL) — a separate, cosmetic failure, not the ghost | Breath-safe edges (§5.1); moot under raw-verbatim |
| ~99 Hz hum in output | Bed harvested with LF rumble | HP 120 the BED (never the speech) |
| Hiss louder than expected | Bed renormalized after filtering | Scale bed to -6 dBFS, never unit |
| Sentences run together at gap 0 | Model tails don't land at boundaries | Render with sentence_gap 0.6 |
| Host OOM during renders/training | Stacked GPU jobs (vmmemWSL commit) | One GPU job at a time, always |
| Renders slower than another voice | Often NOT a defect — a slower narrator is more audio per sentence | Compare audio-seconds/min, not sentences/min |

## Constants (locked)

| Constant | Value | Locked by |
|---|---|---|
| Corpus source | RAW / gain-only master; ~0% zero samples, floor −50..−60 dBFS | 07-24 floor A/B |
| Cut | verbatim: no trail-cap, no gain, --max-clip 20, --mix 2:3-10,3:10-20 | rv recipe |
| Bed (fallback only) | -65 dBFS under clip (file at -6 dBFS RMS) | EOS twins + ear |
| Bed filtering | HP 120 Hz + notch ≥6 dB lines (LOCAL median); no LP, no renorm | five-arm sprint |
| Tails (bed path) | 0.40 s terminal / 0.20 s continuation | tails experiment |
| Clip ceiling | 19.9 s | 20 s/2048-token training limit |
| Curated size | 480 clips ≈ 2 h, top-brightness + cleanliness, spread ≤~3 dB | curated-2h A/B |
| Edge trim | 200 ms after last speech; 150 ms lead OR 40 ms+15 ms fade if lead is breathy | breath census |
| Train | epochs ≤8, patience 2, constant_with_warmup, mask-prompt-loss | late-dip null |
| Keeper epoch | **the epoch that WINS the gate battery** (no fixed offset) | Owen 2026-07-24; tr_rv2 drops 2/1/0 across ep220/330/440 |
| Corpus content | **NARRATION ONLY** — quote-mark split, character voices excluded | Owen 2026-07-24; dialogue model scrapped |
| Numbers | **keep ARABIC NUMERALS**, do NOT pre-expand | 701 year ranges reach the model regardless |
| Dropout gate | zero-RUNS >= **100ms** (5ms rejected 58% of a good corpus) | AoL threshold sweep |
| Speaker verify | wespeaker, centroid from narration, flag < **0.40** | bake-off margin ~0.78 |
| repPenalty | 1.10 | wobble A/B |
| eosBoost | 8 @ 2.0× expected (vLLM only) | ghost-whisper campaign + 07-24 A/B |
| Production gaps | sentence_gap 0.6 at assembly (per-voice override in the catalog) | pause placement |

*Full forensic history: memory files `orpheus-corpus-e-recipe`,
`orpheus-ringing-snac-comb`, `orpheus-eos-capability-failure`,
`thirdreich-eos-boost-fix`, and `AUDIBLE_BOOK_CLEANING.md` in this repo.
Results ledger: `TESTING_BIBLE.md` — append every experiment (chronological).
Per-voice index of runs and settings tried: `MODEL_LEDGER.md` — add a row for
every train and every settings change, or the next person re-runs it.*
