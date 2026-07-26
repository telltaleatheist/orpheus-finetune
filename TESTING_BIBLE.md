# TESTING BIBLE — measured results ledger (append-only)

Every entry: what was tested, how it was measured, the numbers, the verdict.
No theory without a measurement attached. Newest campaign at top. Older
forensic history lives in SAMPLING_AND_VOICE_QUALITY_FINDINGS.md and the
run-books (VOICE_TRAINING_PIPELINE.md, RVC_TRAINING_PIPELINE.md).

---

## 2026-07-25 (later) — throughput measured; the tail trim helps, and does NOT close the gap

**Measured properly at last.** Same book (God's People), same engine, --tier fast, GPU,
1 worker, gpu_memory_utilization 0.54, 9-minute cumulative windows through the real
BookForge CLI audiobook path:

| voice | chunks/min | mean chunk | chunks/batch cycle | cycle | guard trips |
|---|---|---|---|---|---|
| deathstalker | **64.0** | 16.14 s | 60.1 | 54.2 s | 0 |
| mb_hoa1_ep344 | **49.8** | 18.69 s | 58.7 | 69.5 s | 0 |
| tr_tt1_ep420 | **42.7** | 17.42 s | **36.3** | 56.3 s | **9** |

Owen's success criterion was "match deathstalker". **Neither does.** mistborn reaches
78%, thirdreich 67%. The tail trim was real — thirdreich went 30.5 -> 42.7 chunks/min
on this same book, +40% — but it does not close the gap.

**Three measurement biases, all of which I hit before getting a usable number:**
the app's Speed readout is INFLATED by the initial batch burst; chunks/totalElapsed is
DEFLATED by model load; and a 3-minute window carries +/-33% quantization because the
worker flushes flacs in batches of 64 (three consecutive 3-min windows on ONE unchanged
run read 64.0 / 64.0 / 85.3). Even at 9 minutes ds read 64.0 and 71.1 on two windows of
the same run, so treat every figure as +/-10%.

**RULED OUT for the residual gap:** mean audio length (tr is 1.08x ds, predicts 59);
length superlinearly (1/L^2 predicts 1.17x vs 1.50x measured); length ORDERING at all
(mb is LONGER than tr yet FASTER — this kills the family); KV preemption (ds had MORE:
9 events vs tr's 3); token-cap runaways (zero in all three); worker count and vLLM
config (identical); speaking rate (Pratt is FASTER per speech-second, 29.8 vs 27.2
chars/sec); trained clip length (ds corpus clips are LONGER, 15.94 vs 12.03 s).

**The surviving lead.** Batch CADENCE is the same for all three (54-70 s); what differs
is chunks cleared per cycle — 36 for tr against ~59 for the other two. thirdreich is
also the only voice that trips the completeness guard: 9 times in the window, about one
per three cycles, where ds and mb never trip. Each trip logs "audio too short for text
(28.9 ch/s > 21.5) - re-rendering split at sentence boundaries", i.e. the chunk is
generated, rejected, then re-rendered SERIALLY as separate sentences, stalling its
batch. Those are genuine early-EOS failures (one read 89.3 ch/s), not false trips, so
raising maxCharsPerSec would hide them rather than fix them.

**NOT PROVEN.** Confirming it needs per-chunk wall-time and token counts, which nothing
currently logs. Test battery designed in THROUGHPUT_INVESTIGATION.md — including the
hypotheses this data already kills, so they are not re-chased.

## 2026-07-25 — tail trimming works; eval-min was PRE-consolidation and defective

**Corpus.** `thirdreich_trim2h` — the curated 2 h re-cut with `trim_tails.py
--median-to 0.98`, scaling every tail by one factor (x0.601) derived from the
corpus's own median so natural variation survives. Cut point is the first true
silence frame (<= -55 dBFS) at/after the goal, so it cannot land in a breath.
Tail median 1.63 -> 0.99 s, p90 2.17 -> 1.31, clips ending mid-breath 14 -> 0,
SNAC pause concentration 0.7% -> 0.5%. 2.135 -> 2.027 h.

**The model reproduces the tail it was trained on.** Two voices, independent:

| training corpus | trained tail median | model | rendered trailing median |
|---|---|---|---|
| thirdreich_trip2h | 1.64 s | tr_tp1 (3 epochs) | 1.21-1.43 s |
| mistborn_narr2h | 1.03 s | mb_aol1_ep348 | 0.87 s |
| thirdreich_trim2h | 0.99 s | tr_tt1_ep420/560 | 0.75 / 0.62 s |

**eval_loss picked a broken epoch.** tr_tt1 self-stopped at epoch 4. Battery on
20 chunks, identical texts to tr_tp1, production sampled settings:

| epoch | trailing max | speech % | eval_loss | verdict |
|---|---|---|---|---|
| ep140 | 15.56 s | 65.3% | 3.668 | BLOAT 1.94x |
| ep280 (eval-min) | 18.58 s | 65.3% | 3.611 BEST | BLOAT 1.80x |
| ep420 | 1.12 s | 74.8% | 3.616 | PASS 20/20 |
| ep560 | 0.94 s | 73.5% | 3.682 | PASS 20/20 |

Owen, from the ear and before seeing any of this: *"the problem with some of these
is the never ending pause at the end... there's a certain epoch at which pausing is
finally resolved and settled and it varies from model to model."* Exactly right.
Consolidation lands between ep280 and ep420 here, and **eval-min is on the wrong
side of it.** Rule updated: eval-min SELECTS, consolidation GATES.

**Two detectors for consolidation** (cheap, no ear needed): trailing-pause max
collapses ~16 s -> ~1 s, and speech fraction jumps ~65% -> ~75%.

**A gate hole, now closed.** ep140 #013 rendered 38.9 s / 3193 tokens for a text
ep420 read in 17.4 s / 1429. `loop_gate.py` called it CLEAN: coverage 93.1% is
above the 85% floor (not DROPPED) and dead air repeats no n-gram (not LOOP). A clip
can therefore balloon 2.2x and pass. Added a **BLOATED** verdict on
seconds-per-character vs the probe set's own median, ceiling 1.5x — measured over
80 clips, clean all under 1.3x, the two defects 1.80x and 1.94x. Also wrote
`render_pause_report.py`, because `pause_meter.py` structurally cannot see this:
it excludes the run at the very end of a clip, which is the run that matters.

**CORRECTION to an earlier claim in this campaign.** I wrote that the trained tail
was "most of" the 30 vs 70 chunks/min gap. NOT supported. Clip median fell only
~10% (16.30/15.06 -> 14.08 s), which under 1/L^2 predicts ~30.7 -> ~38 chunks/min.
Two candidate explanations for the remainder are now RULED OUT by measurement:
speaking rate (Pratt is FASTER than Rohan per second of speech, 29.8 vs 27.2
chars/sec) and trained clip length (the ds corpus clips are LONGER, 15.94 vs
12.03 s median, which would make ds slower). Untested candidate: chunks/min is not
comparable across books if the packer emits different chunk sizes. Throughput gain
from this trim remains UNMEASURED — it needs a real render.

## 2026-07-24 — thirdreich: the truncations were a TEXT defect, not an audio one

**Symptom.** Production render of *And the Witnesses Were Silent* (2406 chunks)
with the Jul-18 thirdreich: ~30 chunks/min, repeated `hit the audio-token cap`,
and on audit 1.8% of clips with 12-25 s of dead air, 3 empty clips, and
**phrase-level LOOPING** ("It listed Ten Commandments... It listed Ten
Commandments..."). A random 80-clip sample was 0/80 clean — the damage is rare and
concentrated, which is exactly why spot-checking never found it.

**Finding 1 — the EOS boost was never armed on this voice.** thirdreich declared no
tuning at all, so e2a ran `ORPHEUS_EOS_BOOST=0`. A/B on the LIVE model, eos_gate.py,
20 greedy chunks, identical texts:

| arm | result |
|---|---|
| pen 1.10, boost OFF | **FAIL 7/20** — every failure at exactly 3700 tokens |
| pen 1.10, boost 8 @ 2.0 | **PASS 20/20** |

The boost is surgical: the 13 already-healthy chunks returned IDENTICAL token
counts in both arms. But it ENDS a silence loop rather than preventing one, and it
does nothing about repetition — a full render with the boost armed still looped.

**Finding 2 — the corpus was cut from a DEHUMMED master and its pauses were dead.**

| master / corpus | exactly-zero samples in quiet regions | SNAC pause-frame top-3 share |
|---|---|---|
| dehummed master | **99.87%** | — |
| `thirdreich_rv2h` (cut from it) | 19.18% | **29.7%** (dead band = 27-44%) |
| RAW master | 0.23% | — |
| `mm_narr2h` (live deathstalker) | 0.23% | 0.4% |
| `thirdreich_raw2h` (the re-cut) | 6.06%* | **0.4-0.5%** |

\* scattered PCM_16 quantization of an -83 dBFS floor, NOT dead air: zero-RUNS >=5 ms
measure 0.00%, and SNAC still resolves 766 distinct pause frames out of 767.

Re-cut verbatim from the RAW master (hum intact) using the dehummed-aligned VTT —
durations match to **0.014 ms**, so the VTT transfers. Curated 480 clips / 1.77 h,
brightness mean -17.0 -> -14.5 dB, spread 5.6 -> 2.3 dB.

**tr_rv2 gates.** All of ep220/ep330/ep440 PASS greedy 20/20 — the depth erosion of
tr_rv1 on the dehummed corpus (20/20 -> 18 -> 15) is GONE. Loop gate (new, sampled):
**0 loops on every epoch**; dropped-text 2 / 1 / 0 across ep220 / ep330 / ep440.
Drops DECREASE with depth on a live-floor corpus — the inverse of the dehummed run.
=> **the keeper is chosen by the gate battery, not by a fixed offset from eval-min**
(the eval-min epoch was the WORST of the three here).

**Finding 3 (THE ROOT CAUSE of dropped text) — page numbers and chapter headings are
embedded in the training TEXT.** Measured on the corpora:

| corpus | clips | page-number artifact | ALLCAPS heading bleed | either |
|---|---|---|---|---|
| `thirdreich_raw2h` | 480 | 39 (8.1%) | 1 | **40 (8.3%)** |
| `thirdreich_raw_full` | 3362 | 258 (7.7%) | 16 | **268 (8.0%)** |

Examples straight from `metadata_train.csv`:
```
...socialists followed. twenty five The consequences of the Anti-Socialist Law...
...stormtroopers of the SA. one hundred forty seven The paramilitary groups...
Cambridge, July two thousand three one THE LEGACY OF THE PAST GERMAN PECULIARITIES I .
```
The narrator never speaks these, so 8% of clips trained text tokens with NO
corresponding audio, and the model learned that the junk means STOP. Every
dropped-text failure in the loop gate lands **exactly at a page number**:
ep330 #003 stopped at "twenty five" (46% coverage), ep220 #019 stopped before
"one hundred forty seven" (79%).

**This was found on 2026-07-22** during the WeSpeaker contamination sweep and logged
as "FINDING (not actioned)". It went un-actioned for two days while the failure it
causes was diagnosed as an audio/EOS problem. **Text QC is now a pre-training gate.**

**Consequence for source choice:** a fresh book (e.g. *The Third Reich in Power*)
inherits this defect unchanged — the contamination comes from the epub->VTT text
pipeline, not the audio. Fix the text prep first or the retrain reproduces it.

**Gate tooling added:** `pipeline/loop_gate.py` (sampled render -> whisper -> reject
repeated 6-gram or <85% coverage) and `pipeline/cleanliness_census.py`. Two metric
bugs found and fixed in the process, both of which had produced confident wrong
answers: (1) coverage counted number-word expansion ("one thousand nine hundred
twenty eight" vs Whisper's "1928") as ~15% missing, false-flagging complete readings
identically across all three epochs; (2) the dropout gate counted scattered PCM_16
quantization zeros as dead air and rejected all 3362 clips. **A metric that fails
identically across different models is measuring the text, not the model.**

**Hum: measure ABSOLUTE, not excess-over-local-median.** The raw-cut voice DOES learn
the 120 Hz hum (+21 dB over local median in training clips, +26 dB in the render —
louder than trained). But absolute level is **-88.3 dBFS, 69 dB below speech**, and
Owen's verdict on the render was "sounded flawless". The relative figure nearly
triggered a notch that would have removed 1.2 dB of 80-300 Hz speech body at a
frequency inside the narrator's fundamental. Only notch what is audible.

**Throughput (OPEN).** ep440 renders at 30.7 chunks/min vs deathstalker's 69.8 on the
SAME book, same day, same 2406 chunks (analytics `sentencesPerMinute` == chunks/min
for both — verified, not assumed). Clip length is only ~1 s different, so length alone
cannot explain 2.27x. Preemption is active (KV-limited), where throughput scales as
KV/L^2 — untested. NOT yet explained; do not accept "slower narrator" without the
measurement.

---

## 2026-07-22 — Owen Morgan (God's People) twin: Adobe source is the ceiling

**Setup:** only surviving God's People audio is Adobe-resynthesized (raws
deleted at record time; raw had room echo + background noise, Adobe was
required for the Audible master). Source = `E:\training\owen\source`
(_owen_dataset: 1734 clips 24k mono, corrected book-truth text).

**ow_rv1** (brightest-480, locked recipe, early-stop ep4, eval bottom ep2):
Owen verdict — "wobbly and sharp, no softness." Same artifact class as
en_qa1's quiver → second independent confirmation that **Adobe texture is
learnable and reproduced**.

**ow_rv2** (anti-Adobe selection: lowest-flutter within brightness p25–p75;
flutter = mean |Δ dB/frame| of the 1–8 kHz envelope in speech frames):
corpus flutter spread was NARROW (p10 3.10 / mean 3.40 / p90 3.69 dB/frame;
selection only reached 3.20) ⇒ Adobe damage is UNIFORM, not concentrated.
Verdict: "does sound better" than rv1 (softer — rv1's sharpness was partly
the brightest-selection concentrating Adobe sharpening) but a doubled/
chorused quality remains.

**Tinny diagnosis (LTAS, speech-active):** render vs corpus vs source vs
Audible master are within 1.4 dB of the same balance — pipeline is honest.
Deficit vs the approved Audible master: mostly +2.5 dB body needed @150 Hz.
Measured correction curve (firequalizer points, saved in
`owen_postrender_eq.txt`) applied post-render = "a bit better" (Owen).

**Doubling diagnosis:** stereo sources are digitally dual-mono (corr 1.0000,
L−R −216 dB) — downmix comb REFUTED. Cepstral "echo" at 8.71 ms REFUTED —
that is Owen's pitch period (~115 Hz); inverse-comb made it worse at every
gain (lesson: cepstral doubling peaks in the 8–12 ms range are F0 for male
narrators, not echoes). Ear test on the SOURCE itself settled it — Owen:
"kind of a tinny sound in the original... manifesting as a doubling in the
Orpheus render. This is an original clip problem. Not an Orpheus problem."

**VERDICT:** Adobe resynthesis chorus is uniform and source-baked; no
selection/filter/retrain removes it. ow_rv2 + the measured post-render EQ =
ceiling for this source. Real fix = re-record (~3 h of reading suffices for
the 2 h curated corpus; fix echo/noise at CAPTURE: closer mic, soft
treatment; record raw, keep raw — no Adobe on training audio ever).
Keepers: ow_rv2_ep220/ep330 (registered, pen 1.10 / 21.5). ow_rv1_* deleted.

**Addendum (2026-07-22 ~02:30):** older Owen narrations examined as raw-source
candidates — UJW (192 kHz/24-bit masters, pre-Adobe) and WHAA (corrected
44.1k mp3, best-sounding). Owen's verdict after sampling: "give up on the
other books narrations. They aren't suited to Orpheus work" — early
narration method (audiobook-fast read; WHAA additionally YouTube-edited:
all pauses >1 s cut, commas ignored). WHAA verbatim arm (ow_wh1) canceled
mid-chain. Owen voice = pending his own re-record.

### Source decisions (measured, not guessed)
- **MM (deathstalker)**: `E:\training\deathstalker\build\markedman_raw_leveled.flac` — raw
  Audible, gain-only chapter leveling (ch4 +3.35 / ch8 +3.5 dB). No epub
  exists → whisper-medium VTT (`--generate-sentences`), so the cut uses
  `--snap-window 0.7` (whisper timestamps are rough; default 0.15 makes
  sentence-ends fail to snap and clips merge past --max-clip and get dropped).
- **AoL (mistborn)**: `aol_book.flac` + epub-aligned `aol.vtt` +
  `aol_silences.json` (autoeditor map, fps 30.0). Cut verbatim, no gain.
- **TR (thirdreich)**: probed the RAW merged master's floor: consistent
  **120 Hz line at +12.3..+14.1 dB over local median** in all 3 windows
  (floor p5 ~-90 dBFS). Hums are LEARNABLE (the law) → raw would train the
  hum into pauses. The dehummed master's floor is digital silence (-240 —
  same class as the AoL master, which trained clean in w1/mb_2hd), durations
  match raw to the microsecond so the VTT applies. **Cut source = dehummed.**
  Dehum's old cost (EOS margin erosion) is covered by the boost.

### Cut results (cut_audiobook.py, verbatim: NO --trail-cap, no gain,
### --target-minutes 0, --max-clip 20, --mix 2:3-10,3:10-20, dialogue-aware)
| Voice | Source hours | Clips | Minutes |
|---|---|---|---|
| mistborn (AoL) | 10.8 | 2659 | 644.0 |
| thirdreich (dehummed) | 21.2 | 3388 | 742.7 |
| deathstalker (MM leveled) | 12.15 | (cutting after whisper VTT) | |

Curation: brightness census -> `build_2h_corpus.py --verbatim` (NEW mode,
c3abb12): selection ONLY — top ~480 bright/dynamic clips copied bit-identical,
no breath edges, no bed, no tails. Corpora: `<voice>_rv2h` on WSL.

Trains: `run_2h_retrain.sh` ds_rv1 -> mb_rv1 -> tr_rv1 (stop-overtrain
patience 2, every epoch merged + registered at 21.5/1.10), boost 8 @ 2.0
armed on every epoch, greedy gate (pen 1.10 + boost) on last 3 epochs,
ear renders of last 2 at gap 0.6. Orchestrated by C:\tmp\rv_master.ps1 +
/home/telltale/rv_train_chain.sh.

### Greedy EOS gates — raw-verbatim full trains (pen 1.10, boost 8 @ 2.0)
Each train ~27 min on the 3090 Ti (880 steps, ~2.2s/it, batch 4).
| Voice | Epochs kept | Gate results |
|---|---|---|
| ds_rv1 (MM leveled raw) | 5 (ep110-550) | ep330 **20/20**, ep440 **20/20**, ep550 **20/20** |
| mb_rv1 (AoL raw) | 4 (ep110-440) | ep220 **20/20**, ep330 **20/20**, ep440 **20/20** |
| tr_rv1 (TR dehummed) | 4 (ep110-440) | ep220 **20/20**, ep330 18/20, ep440 15/20 |

Verdicts:
- **The raw-verbatim + boost recipe holds at full-train scale.** ds (loud
  natural floor, w4-class source that gated 8/20 as an unboosted probe) and
  mb (the recipe that needed the boost to reach 20/20 as mb_2hd) are
  PERFECT on every late epoch. Natural pauses/tails verbatim, zero
  acoustics, EOS entirely from the boost.
- **tr erodes with depth** (20/20 -> 18 -> 15): the dehummed master's pauses
  are DIGITAL SILENCE (gated), and margin decays as pause behavior
  consolidates in late epochs — the silence-mass mechanism from the
  EOS-capability postmortem, now visible EPOCH-RESOLVED on one corpus. The
  settle epoch is gate-clean; keeper candidates (settle/settle+1 = ep220/
  ep330) are fine. If a later tr epoch wins by ear, tune the boost up
  per-voice (+10-12) and re-gate. AoL is also gated-silent but mb shows no
  erosion through ep440 — TR's failure cues are long scholarly sentences
  (277-302 chars), suggesting silence mass x sentence length compounds.
- tr fails are DEEP loops (3700-token ceiling), not near-misses.

### Narrator pause baselines (auto-editor -40dB, 3x10 min per master)
The target the rendered voices should roughly match (Owen's ask):
| Narrator | pauses/min | med | p90 | max | >2s per 10 min |
|---|---|---|---|---|---|
| MM / Rohan | 15-20 | 0.30-0.37s | 0.67-0.83s | ~2.7s | ~1 |
| AoL / Kramer | 14-17 | 0.37-0.40s | 0.70-0.93s | 1.3-1.6s | 0 |
| TR / Pratt | 13-14 | 0.40-0.47s | 1.4-1.5s | up to 3.7s | 3-7 |
Kramer never exceeds 1.6s; Pratt regularly holds 2-3s (scholarly pacing) —
per-voice pacing identity is real and measurable.

### Ear verdict — KEEPERS (Owen, 2026-07-21 ~13:45)
ds_rv1_ep440 + mb_rv1_ep330 (settle+1, pen 1.10, gap 0.6): **"it sounds
good... very good. these are the keepers."** Note: "might need some post
tts brightening" — that's the doctrine's high-shelf AT ASSEMBLY (post-SNAC),
never on training audio. Ghost-free pauses on fully raw corpora at
production settings — the raw-verbatim pivot is CONFIRMED end to end.

### Rep-penalty 1.0 vs 1.10 — SETTLED (ear + census agree, 2026-07-21)
Hypothesis: with EOS solved by the boost, pen 1.10 is an unneeded crutch.
**REFUTED.** The repetition penalty taxes repeating silence frames, so it is
load-bearing for PAUSE LENGTH, independent of EOS. Owen's ear on the ds A/B:
"pauses are too long in 1.0." Census (gap-0 renders, every silence
model-generated, auto-editor -40dB) vs narrator baselines:
| Arm | med | p90 | max | >2s |
|---|---|---|---|---|
| Rohan book | 0.30-0.37 | 0.67-0.83 | ~2.7 | ~1/10min |
| ds_rv1_ep440 @ 1.10 | **0.30** | 0.61 | 1.07 | 0 |
| ds_rv1_ep440 @ 1.0 | 0.52 | 1.00 | 1.47 | 0 |
| Kramer book | 0.37-0.40 | 0.70-0.93 | 1.3-1.6 | 0 |
| mb_rv1_ep330 @ 1.10 | **0.37** | **0.84** | **1.20** | **0** |
| mb_rv1_ep330 @ 1.0 | 0.68 | 2.28 | **42.5** | 6 |
At 1.10 the models match their narrators' pause distributions almost
exactly (mb inside the book range on every metric). At 1.0, ds runs ~45%
long and mb is broken (a 42.5s silence; renders took ~13x longer).
**pen 1.10 SHIPS. Never lower it without re-running this census.**
Operational corollary: slow renders (minutes per sentence-batch instead of
~90s/sample) = silence bloat in generation — a live diagnostic.

### DEPLOYED (2026-07-21): the rv keepers are the live voices
- deathstalker <- ds_rv1_ep440 (Owen: "very good. these are the keepers")
- mistborn <- mb_rv1_ep330
Via deploy_voice.sh (local WSL + HF owenmorgan/<voice>-orpheus-3b + Mac
pull), live entries stamped maxCharsPerSec 21.5, repPenalty 1.10,
eosBoost 8 @ 2.0. Owen's brightening request handled at ASSEMBLY
(high-shelf post-SNAC), not in the models. thirdreich NOT redeployed —
tr_rv1 ep220 (gate-clean) / ep330 kept as candidates, decision later;
current live tr unchanged. Post-deploy cleanup: all other rv epochs, rv
out-dirs, superseded 2h corpora + old-keeper models, old live adapters,
and E:\Shared\orpheus_adapter_backup deleted (Owen: "no need to keep
backups of the epochs" — recovery = HF merged models + deterministic
re-cut/retrain pipeline).

### Disk cleanup before trains (Owen: "so we have room")
Deleted 33 xtts_ft *_merged staging duplicates, 28 superseded orpheus-models
serving copies, 21 concluded out-dirs (adapters backed up first to
E:\Shared\orpheus_adapter_backup — now 16+10 out-dirs), 32 superseded
datasets. WSL 652G -> 213G used; fstrim released 790 GiB to the host
(C: 90 -> 511 GB free). Keepers: live voices (owen/deathstalker/thirdreich/
mistborn), mb_2hd_ep330+440 (boost-validated best mb), ds_2hb_ep440 (Owen
"sounds best"), wt_w4_probe (guard-gap repro), live lora dirs, base_models,
keeper corpora mistborn_aol2h_nobed + deathstalker_mm2h_tails. models.json
pruned to exactly the surviving dirs (verified).

---

## 2026-07-21 — Ghost-whisper campaign (overnight sprint, mistborn testbed)

### The law that fell out: LEARNABLE vs UNLEARNABLE
The model reproduces PREDICTABLE audio (narrowband tones, hums, any texture
repeated across clips) and DISCARDS unpredictable audio (true random room
noise). Our -65 dB "genuine hiss bed" was 480 slices of ONE 60-second file —
memorizable — so the model emitted it in pauses and SNAC rendered the
near-silent hiss codes as ghostly syllable-babble. Marked Man's atrocious
natural floor (-60 dBFS, never repeating) was simply dropped: near-silent
rendered pauses. **Orpheus is its own denoiser. Never add textures.**

### Ghost verdicts (Owen's ear, same text per voice, vLLM renders)
| Arm | Recipe | Ghosts? |
|---|---|---|
| mb_2hc (full) | genuine bed everywhere + bed tails | GHOSTS (syllable babble in pauses) |
| w2 (2-ep probe) | genuine bed under speech, SILENT tails | GHOSTS → bed-under-speech is the source |
| w3 (2-ep probe) | synthetic pink bed everywhere | audible HISS in pauses (bed reproduced) |
| w1 (2-ep probe) | nothing added (AoL) | CLEAN, "sounds best" |
| mb_2hd (full) | nothing added, breath-safe edges (AoL) | CLEAN, "sounds good" |
| w4 (2-ep probe) | Marked Man VERBATIM (loud natural floor, breaths, no trims) | CLEAN, "nearly flawless", floor NOT reproduced |
| w5 (2-ep probe) | MM natural + breath-safe edges only | CLEAN, no ghosts |

### Greedy EOS gate (20 real chunks, greedy decode, penalty 1.10, stop=128258)
Gate = eos_gate.py; a run counts as FAIL if it hits the 3700-token ceiling.
| Model | Recipe | Stops |
|---|---|---|
| mb_2hc ep440 | bed everywhere | **20/20** |
| mb_2hd ep330 | nothing added | 15/20 |
| mb_2he ep330 | hiss TAILS only | 14/20 → tails do NOT carry the margin; bed-under-speech did |
| w4 probe | MM verbatim (loud natural floor) | 8/20 → natural floor loudness ≠ margin either |
| w5 probe | MM + breath-safe edges | 10/20 |
| (mb_2hd, NO penalty) | — | 0/20 — greedy gate REQUIRES penalty 1.10 to be comparable |
Caveat: 2-epoch probes may under-read margin vs full trains — compare like
with like. Production sampling masks thin margins (all renders completed
normally); greedy is the deterministic canary.

### EOS logit boost (the chosen lever) — VALIDATED
e2a 4700c4b5: ORPHEUS_EOS_BOOST biases token 128258 only past
ORPHEUS_EOS_BOOST_START x expected tokens (chars/18.4 x 84 tok/s), ramps
with overrun, cap 4x, vLLM only, default OFF. Per-voice knob: models.json
backends.vllm.eosBoost (bookforge 476623e).

Ladder + calibration (2026-07-21 04:00-04:45):
- +1/+2/+3 @ start 1.2x: NO effect (identical fails) — NOT a processor bug:
  +50 @ 0.5x force-stopped 20/20 at the threshold (mechanism proven).
- Root cause: the 18.4 ch/s anchor is wrong for narration — measured honest
  reads run 4.6-7.6 tokens/char (median ~6 ≈ 14 ch/s with natural pauses),
  so 1.2x "expected" lands INSIDE honest speech; and greedy runaways are
  DEEP loops (EOS deficit can exceed +14 logits), not razor ties.
- Data-derived tuning: start ~9 tokens/char = START 2.0, boost +8.
  Honest reads (max 7.6 tok/char) never reach the threshold.
- **w4 stress dummy: 8/20 -> 15/20** (5 deep attractors survive even the
  ramp — acceptable; w4-class models never ship).
- **mb_2hd ep330: 15/20 -> 20/20 PASS** — the clean no-bed recipe now
  matches the bed model's perfect greedy score with zero acoustics.
- DEPLOYED: mb_2hd_ep330/440 registered with eosBoost 8 / eosBoostStart 2.0
  (WSL models.json; WSL e2a pulled 4700c4b5). Endings ear-check + a
  128-sentence production-sampled run remain before any book ships on it.

### Production-scale stress test — worst model + boost (2026-07-21 05:00)
wt_w4_probe (greedy 8/20 unboosted — worst measured) rendered 950 MM book
cues (48k chars) through the FULL production path, sampled decode, boost
+8 @ 2.0x via env: **969/969 sentences converted, 0 failed, 0 retries, in
303s** (54 min audio, ~10x realtime, 15 vLLM batches — 7.5x the historical
2-batch failure surface). Acceptance: faster-whisper small.en transcript
matched **94.0% of 8967 ref words with ONE missing run >=8 words** (20
words); auto-editor pause census: 1117 pauses, med 0.33s, p90 0.50s, max
1.50s, **zero >2s**. The one miss is CONFIRMED REAL: the cue `"So now,
Victor," said Quick, "for us to do what we need to do, you have to tell
us: Who is your client?"` rendered to ~0.6s (early-EOS on a quote+colon+
question pileup; neighbors fine). OPEN BUG: ~110 chars in <1s should
have tripped the 21.5 ch/s truncation guard + force_split and did NOT —
reproduce this sentence solo with guard logging before books ship.
VERDICT: the boost holds at production scale on the worst case (968/969). Green light for the final
verbatim trains (MM leveled-raw, AoL raw, TR raw — tail pauses and silences
preserved, out of the box, boost armed per-voice).

### Pause-length ladder (auto-editor, threshold -40dB, same text throughout)
| Render | pauses/min | med | p90 | max | >2s |
|---|---|---|---|---|---|
| mb_2hd ep110 | 11.1 | 0.35 | 0.80 | 0.87 | 0 |
| mb_2hd ep220 | 12.3 | 0.30 | 0.61 | 1.07 | 0 |
| mb_2hd ep330 | 9.9 | 0.33 | 0.67 | 0.77 | 0 |
| mb_2hd ep440 | 12.0 | 0.32 | 0.62 | 0.73 | 0 |
| mb_2hc ep440 (bed) | 15.0 | 0.37 | 0.72 | 1.13 | 0 |
| w4 2-ep probe | 13.4 | 0.33 | 0.51 | 1.03 | 0 |
| w5 2-ep probe | 12.7 | 0.30 | 0.76 | 2.17 | 1 |
Verdict: on curated-2h corpora, pause profile is mature by ~ep220 and stable
through ep440 — no giant pauses at any full-train epoch. 2-epoch probes have
short/unstable pauses (w4 smallest p90; w5 one 2.17s giant) — matches Owen's
ear. Pause quality is NOT a reason to chase late epochs on these corpora.

### Source floor census (floor = p5 of 20ms frame RMS, 3x60s windows)
| Source | floor dBFS | speech p95 | SNR | note |
|---|---|---|---|---|
| Alloy of Law master (aol_book.flac, 48k, non-Audible) | -145 | -24 | ~121 dB | pauses GATED to digital silence at production; hiss rides under speech only |
| Final Empire (original Audible m4a) | -77 | -20 | ~58 dB | audible floor; known narrowband birdies (tonal scan) — the REAL objection to FE remains the birdies, which ARE learnable |
| Marked Man (raw Audible m4b, 22.05kHz native!) | -60 | -15 | ~45 dB | atrocious floor; model drops it (w4) |

### Marked Man chapter loudness census (raw m4bs, integrated LUFS)
Ch1-3,5-7,9-12: -17.0..-17.5. **Ch4: -20.8. Ch8: -20.9.** Fix: gain-only
+3.35/+3.5 dB on those two chapters; all others untouched. Leveled master:
E:\training\deathstalker\build\markedman_raw_leveled.flac (12.15 h, FLAC, mono 22.05k).

### Operational lessons (same night)
- C: at 0 bytes → WSL Errno-5 crash-loop (see wsl-vhdx-balloon memory);
  545 GB of duplicate audition models deleted (adapters backed up to E:).
- PowerShell `*>>` redirects hold the log EXCLUSIVELY → concurrent
  Add-Content markers vanish silently. One log per writer, or use separate
  marker files.
- GPU chains must serialize on COMPLETION MARKERS, not GPU-memory-idle
  polling (two chains raced into a CUDA crash).
- BookForge global process sweeps nearly killed a training chain →
  %APPDATA%\BookForge\external-gpu-job.lock now gates all global sweeps
  (bookforge 8632197); every chain script creates/removes it.

### The go-forward recipe (Owen, 2026-07-21 ~04:00)
1. RAW sources, verbatim clips: no cleaning, no beds, no tails, no breath
   surgery, natural pauses kept at book length. Loudness-match chapters
   (gain only) first.
2. Screen sources for TONAL defects only (ring scan) — tones are learnable;
   noise is not.
3. Curate brightest/clearest ~2h (existing census + curation).
4. EOS margin from the logit boost, tuned per voice via the greedy gate,
   endings ear-checked.
5. Order: deathstalker (MM leveled raw) → mistborn (AoL raw) → thirdreich
   last (current tr model acceptable).
6. Brightening, if wanted: high-shelf at ASSEMBLY (post-SNAC) only.

## 2026-07-21 evening — Adobe Podcast pre-cleaning: REFUTED for training audio (ds)

Round-trip machinery (ClipForge merge/split, gap 0.5, mergemaps) worked
perfectly: Adobe esv2 75%speech/10%bg/10%music returned every piece
sample-exact in length; split drift mean 12-81ms (max ~0.48s = plateau
centering inside wide join silences, signed drifts straddle zero). Adobe's
acoustic effect on the corpus: floor -55.5 -> -75.1dBFS, ~7dB quieter
overall, bandwidth unchanged; F0 contour r=0.991 vs original (prosody of
the AUDIO essentially untouched).

**ds_ad1** (399 Adobe-cleaned rv2h clips, 1.76h, ASR texts inherited 1:1,
8 epochs, boost 8@2.0): EOS spotless — 20/20 greedy on ep180/270/360.
Ear A/B same text/settings: **ds_rv1_ep440 (raw) "sounds much better. by
far" (Owen). Adobe arm loses decisively.**

Confounds (honest): corpus 399 vs 480 clips (eval 40 stranded in the
un-uploaded piece3 + 41 dullest trimmed); ~7dB level drop not renormalized
(verbatim doctrine); fresh eval carve. But the effect size ("by far") and
the pattern — RVC muffle, corpus-LP muffle, hiss-bed ghost, now this —
all point the same way: EVERY pre-training "cleanup" of the audio has
lost to raw. The model drops noise itself; cleaning removes voice texture
it needs. Consistent with mb_2hd/w4 raw-verbatim findings.

**Doctrine: Adobe Podcast (any strength) joins RVC on the BANNED list for
training audio. Post-TTS enhancement at assembly remains the brightening
lever.** Quicktrain caveat: en_qt1 (Adobe, 4.3min) had sounded "good" solo
— small-corpus solo listening is NOT a substitute for a full-corpus A/B.

Still useful from tonight: merge/split round-trip (proven, drift-logged),
speakers fine-clustering for similar narrators (0.20/0.10 + ground-truth
verification), sentence-generation/epub text doctrine (ASR proper nouns),
MM epub found -> markedman_corrected.vtt (correct_vtt running) for the
ds re-run with book-truth text ON RAW AUDIO.

## 2026-07-21 late — Ender twin quicktrains: Adobe WINS for a weak source

en_qa1 (Adobe esv2 75/10/10) vs en_qr1 (raw 44.1k), IDENTICAL 121 clips +
book-truth texts (embedded epub-aligned VTT, silence-verified boundary
mapping: 121/294 kept, 90 speech-before-start / 80 speech-after-end / 3
overhang>6s dropped), identical settings, both ep135, both 20/20 greedy.
**Owen: Adobe arm "significantly clearer and cleaner" — opposite of the ds
verdict.** Combined law: RAW wins on decent sources (MM 22k Audible raw beat
its Adobe arm "by far"); ADOBE wins when the source itself is weak (Ender
44.1k library encode is hissy/dull raw). Per-source A/B is cheap (~25min/arm)
and is now the standard gate before committing a corpus.

Known cost: en_qa1 has a VOICE QUIVER. Hypothesis (Owen + measurement): esv2
blend character varies per clip (at 100% the treble/bass jumps are plainly
audible; subtle at 75%) → small corpus = smeared timbre distribution → model
reproduces variance as tremolo. Expect the full corpus (~2.2h vs 0.62h) to
average it out; verify at full-build audition.

## 2026-07-22 — KA ring v2: notch the measured residual SHOULDERS, then stop

Owen still heard light ringing in v1-notched Killing America (the locked
10-notch chain). Full rescan of the NOTCHED file: density 25 -> 9 hits/min,
the dominant 8.4k cluster fully dead — but survivors sat AT THE SHOULDERS
of the v1 notches (7415 beside 7364:w220; 7680 between 7617/7730; 4485 just
outside 4323:w140) plus unnotched 5121 and 2434/2597. Precision pass
(ring_precise.py: exemplar windows, parabolic interpolation) gave centers
tight to +/-30Hz. v2 = v1 + 6 measured notches, re-encoded from the ORIGINAL
(one lossy pass). **Owen: "it sounds good. im not hearing a ringing" —
v2 chain LOCKED as the permanent deathstalker cleaning step**; stamped as
`postRenderFilter` on deathstalker in WSL+Mac models.json (owen stamped with
the measured warm EQ at the same time). v2-of-v2 rescan showed whack-a-mole
(new shoulder flags at 7.5-7.6k/7.7-7.8k, density flat ~9/min) = the scanner
chasing its own notch edges; ear said done, so DONE. Law: notch from
measured residuals on the notched output, one added round max — if the ear
still hears ring after round two, it's burst-class (birdie gate territory),
not filterable.

1.2-1.7kHz scanner clusters are vowel harmonics (identical pre/post chain);
never notch the formant band. Chain text: C:\tmp\ds_notch_map.txt (v2).

## 2026-07-22 — Narration-only corpus (ds_nr1): quote-mapping beats voice-matching

Owen's complaint: the deathstalker model randomly drifts into CHARACTER voices
when narrating. His insight — Marked Man is one reader, so any acoustically
distinct "voice" in it IS a character voice — turned out to be measurable, but
TEXT was the better selector than embeddings.

**Method (repeatable).** The aligned VTT has NO quote marks (alignment strips
them), so dialogue was recovered by mapping all 12,505 cues back to the epub
with a forward-only fuzzy pointer: **97.8% word match**. Quote spans (non-nesting
rule: open -> first of close / next open / paragraph break; a naive depth counter
LEAKS and reports 73% dialogue vs the true 51%) flagged 9,739 dialogue vs 2,766
narration cues. Keep-mask = narration runs >=15s, edges snapped to real >=0.30s
silences from the silence map (NOT blind-trimmed — blind trims fight
cut_audiobook's boundary contract and kill every run-boundary clip). Result:
645 clips / 2.56h -> curated 480 / 1.99h, QC mean WER 0.012, first/last 40/40.
Bleed audit: of 645 clips only ONE has >20ms of non-silent overlap with a
dialogue cue (27ms).

**WeSpeaker verification (the control is the point).** mm_narr2h: mean 0.908,
min 0.575, ZERO clips below 0.40. The dialogue-inclusive deathstalker_rv2h run
through the identical pipeline: mean 0.868, min **0.051**, p5 0.731 — and its
low tail is literally the character voices (Mrs. Kalakos's Greek accent lines,
0.56-0.66). Premise confirmed.

**CONTAMINATION FOUND in the SHIPPED model's corpus:** deathstalker_rv2h holds
`e_deathstalker_00002457.wav` at sim 0.051 — the HarperAudio children's-audiobook
promo read by a COMPLETELY DIFFERENT announcer. That is in the data behind the
deployed deathstalker voice and could alone explain some drift. Lesson: run a
WeSpeaker outlier sweep on EVERY corpus before training; promos/credits/outros
slip past text-based curation because their text looks like normal prose.

**Residual failure mode (honest):** quote-based flagging cannot catch UNQUOTED
direct speech ("...had said Theodore Purcell" — the epub prints it bare). 23/480
clips (4.8%) show an attribution verb with no quotes; their mean similarity
(0.911) is indistinguishable from the corpus mean (0.908), so the narrator
generally does NOT perform remembered speech in character. One clip stands out.

**Train:** early-stopped at 4 epochs (overtrain+2). eval_loss 4.2853 / **4.2345
(ep2, best)** / 4.2414 / 4.2909. Epoch ladder ep220 + ep330 rendered on held-out
eval text for Owen's ear (pause consolidation lags the loss bottom by ~1 epoch,
so ep330 is a real contender despite the worse number). Gates NOT yet run —
they follow the ear verdict.

**Trap found:** `mm_silences.json` was built for mm_book.flac (42,803s), NOT
markedman_raw_leveled.flac (43,724s) — a 921s offset that would have mis-snapped
every cut. Correct map now at `mm_leveled_silences.json` (fps 30.0 verified).
Also `cut_audiobook.py --target-minutes` DEFAULTS TO 180 and subsets BEFORE
dropping over-length spans; pass `--target-minutes 0` when cutting a restricted
region or you silently lose most of it.

## 2026-07-23 — Inter-sentence gap: the model already trained tail pauses; the 0.6 pad double-counts

Owen: KA (deathstalker) inter-sentence gaps drag. Measured the REAL speech-to-speech
gap (auto-editor on the full 12h Marked Man human source; per-sentence-FLAC trailing/
leading analysis on KA), not just the artificial pad:

- **Marked Man (human narrator)**: natural pause median 0.567s (ALL pauses, chapter
  gaps excluded); sentence-range (0.4–1.5s) median **0.70s**, mean 0.73. That's the
  target — "what we're shooting for."
- **Killing America (TTS, deathstalker Story)**: real gap median **1.48s**. Breakdown:
  the MODEL emits ~**0.83s** of trailing silence on its own (learned tail, before/around
  EOS) + the e2a **0.6s artificial pad** appended on top = ~1.43s trailing + 0.04 lead.
  So KA overshoots the human gap by ~0.8s (>2x).

**Key insight (Owen):** the model was trained on real narration WITH its pauses, so it
already produces a proper tail (0.83s ≈ MM's 0.70–0.87s sentence range). The 0.6 pad is
redundant. The fix is NOT to lower the pad (0.4 pad still => ~1.28s) but to STRIP the pad
and let the model's own tail stand => ~0.83s gap, right in the human range.

**Mechanism**: the pad is EXACTLY 0.0 samples (torch.zeros, orpheus.py ~1359), the model
audio/tail never is — so trailing-exact-zero trim removes only the pad, keeps the tail.
Must run BEFORE denoise (denoise makes the zeros non-exact). Built as an assembly-time
`normalizeSentenceGaps` step + per-voice `sentenceGap` default in models.json (deathstalker
& deathstalker_narration = 0; `measuredSentenceGapS` = 0.7 recorded as the target),
resolution CLI/BookForge > model default > untouched. CLI `--sentence-gap`.

Tools: `C:\tmp\gap_wordtoword.py` (real speech-to-speech: KA per-FLAC trail+lead, MM
auto-editor levels re-thresholded — the cached `mm_leveled_silences.json.levels.npy` lets
you re-threshold with no re-run). Speech threshold -35 dB RMS / 20ms frames.

---

## 2026-07-25 — Adobe Enhance on the TRAINING CORPUS: WINS, but only with EQ restoration

**Result: `mistborn` = `mb_ae2_ep327`, deployed to WSL + HuggingFace + Mac.** Clearly better
by ear on a 96 s mixed narration+dialogue render, and the lowest eval_loss of any
Hero-of-Ages run: **3.8214 vs 4.0640** for the untreated `mistborn_hoa3p0h`.

This REFINES the 07-21 verdict ("Adobe pre-cleaning: refuted for a good source"). The
earlier test was Adobe *alone*. Adobe alone de-brightens hard — measured, not guessed:

| band | dB removed by Enhance v2 @75p |
|---|---|
| 80–1000 Hz | ~1.5 (mostly level) |
| 2–3 kHz | 4.8 |
| **3–4 kHz** | **8.1** |
| 6–8 kHz | **8.8** |
| 8–11.5 kHz | ~7 |

**Adobe + a matched inverse EQ is the thing that wins.** A plain HF shelf is a guess;
`pipeline/eq_restore.py` measures the processed-vs-source speech spectrum and inverts it
at a chosen strength (0.8 keeps some of the denoise benefit instead of restoring the HF
noise with the HF signal).

### The round trip is SAMPLE-EXACT — but verify, never assume

Enhance returns 48 kHz (stereo PCM_24 if the user EQs in a DAW). Files come back **26–32 ms
short**, which looks alarming and is not drift: cross-correlation showed **0.0 ms lag at
both start and end**. It is a dropped final buffer, so only the last clip of each part is
touched, inside its trailing silence. **Check by cross-correlation, not by duration delta.**
`build_full_corpus.py` hard-fails if the timeline actually moved.

### Three prep traps, each of which cost a rebuild

1. **Sort clips by NUMERIC id, not filename.** Eval clips are `e_`-prefixed, so a plain
   sort sweeps every one into part 1 and it stops being a book slice. Hit mistborn (all 62
   eval clips in part1) before it was fixed for thirdreich.
2. **Level-match after the round trip, WITH headroom.** The returned corpus was 7–11 dB
   quiet; the source corpus is loudness-normalised, so training on it as-is makes level an
   uncontrolled variable. Use ONE corpus-wide gain (per-clip flattens the narrator's
   dynamics) and **cap it so the corpus's loudest sample lands at −0.3 dBFS** — the naive
   gain clipped 1 clip of 576, i.e. distortion straight into the training data.
3. **The background slider does NOT restore room tone.** 100% vs 75% moved SNR only 1.2 dB.

### Suppression is SOURCE-dependent — check SNR per voice

SNR = speech level minus its own noise floor, gain-independent:

| corpus | source SNR | after Adobe | verdict |
|---|---|---|---|
| mistborn HoA | 49.3 | **54.6** | 5.3 dB more suppressed - no measured harm |
| thirdreich trim2h | 56.5 | **56.5** | unchanged — source was already clean |

**The EOS concern was NOT substantiated - I flagged it without a baseline.** Mistborn's
Adobe epochs run token overrun 1.36x median / 1.46x p90, and I reported that as an
Adobe-caused risk. But **no overrun measurement was ever taken on an UNTREATED model**, so
that number has nothing to compare against - 1.36x may simply be what this voice does. The
one hard datum, a 10.4 s seam runaway, came from a **27-minute** quicktrain and did NOT
reappear at 2 h (max silence 2.05 s), which points at corpus size, not Adobe. The
`mb_qt_control` twin was built and never run, so the attribution stayed a hypothesis and
should not have been written up as a caveat on the result.

**VERDICT: Adobe does not damage the audio and is SAFE TO TRAIN WITH.** It produced the
best mistborn to date on every axis actually measured - ear, eval-min (3.8214 vs 4.0640),
and a clean greedy gate 20/20 - with a cleaner noise floor and, after EQ restoration, more
brightness than the source. Zero clipped clips, timeline sample-exact.
OWED (~5 min GPU): render 20 chunks through the untreated `mistborn_pre_ae2` and compare
overrun. That settles it either way.

## Measurement lessons (these cost real time tonight)

**`epoch_battery.py` exists because two gates measured models against THEMSELVES.**
The EOS gate passes on `stopped=True`, so a model emitting 10 s of silence and then
stopping is a pass — while the silence mass sits in the token counts it already prints at
1.3–1.5× expected. `loop_gate`'s BLOAT verdict is relative to the probe set's OWN median,
so a uniformly slow model drags the median with it and reads clean; that same model
rendered dialogue at **8.9 ch/s against a normal ~14**. The battery scores every epoch as a
**distance from the training corpus**, which is the ground truth the model was asked to
imitate.

**GREEDY and SAMPLED disagree, and both are real.** The battery samples (temp 0.6, seeded);
the chain's gate is greedy. `mb_ae2_ep545` ranked CLOSEST to the narrator on measurable
fidelity (exact 13.6 ch/s match, shortest tail) and is the **only** epoch that FAILED the
greedy gate — a chunk hit the 3700-token cap. Safety gates must precede fidelity ranking.

**Probe TEXT changes the pause verdict.** ep327 measured 41% silence on a 3-chunk dialogue
passage vs ep545's 35%, which looked like the epoch's character. On a 96 s mixed passage
they were **31% vs 27%, with ep327 TIGHTER at the seams**. Short quote-heavy dialogue
elicits far more pausing than narration. Never generalise pause behaviour from one short probe.

**eval_loss and measurable fidelity picked different epochs.** eval-min was ep327
(3.8214); minimum corpus-distance was ep545 (worst eval_loss, 4.0213). Owen's ear picked
ep327 — agreeing with eval_loss, not with the distance score. Both are filters; the ear ranks last.

## 2026-07-26 — Trailing-silence runaway: the cause chain, and TWO fixes (tr_ae2 campaign)

**The defect.** ~2.4% of sentences ≥200 chars finish the words, then sit MUTE — up to
28.7 s — before stopping (or hitting the 3700 cap, which is the same event logged).
Witnesses render, 1664 sentences: 43 min of excess audio; on every large case
`maxSil == trail` EXACTLY — the silence is trailing, never mid-speech. Two chunks also
DROPPED text (ASR similarity 0.78/0.79 vs 0.99/0.97 re-rendered). Distribution is
bimodal (24 clips ≥1.5×, 23 of them ≥1.8×): a discrete attractor, not slow pacing.
Nothing catches it in production: the ch/s guard fires only on SHORT audio, the cap only
on total runaway.

**FIX 1 — inference (deployed on tr_ae2_ep264): eosBoost 16 @ start 1.5** (was 8 @ 2.0).
Root arithmetic: boost start scales with chunk length while the vLLM cap is FLAT 3700,
so rescue runway shrinks 970→285 tokens exactly where failures live; AND the hardcoded
18.4 ch/s anchor fires rescue LATER for faster voices (tr 15.9 → wall at 1.73× real
speech; mb 13.6 → 1.48×). 600-sentence paired A/B (same texts/seeds, SML-stripped like
e2a): bloat ≥1.5× **2.50% → 0.00%**, cap 0.67% → 0%, p99 2.16× → 1.30×, median/p90
UNCHANGED, clips <0.90× went 35 → 27 (no truncation), ASR clean on the 10 most-shortened.
Do NOT go to start 1.2 — only rung that ate real length (median −2.9%, worst −15%).
The keep-twin isolated the too-late signature in one column: at 8@2.0 cap-hits fell
5→2 while ≥1.5× bloat did NOT move — the production boost converts cap-hits into
sub-cap dead air. **e2a code fix owed: per-voice pace anchor + cap-aware start.**

**FIX 2 — training (Owen's hypothesis, twin-proven): CUT TAILS TO ~ZERO.**
Twin 30-min quicktrains, SAME 144 clips, same config/steps (ckpt-165 both), only tails
differ (median 0.82 s vs 0.02 s). Same 300 sentences, **boost OFF**:
tails-kept **2.00% ≥1.5× / 1.67% cap-hits (max 2.45×)**; tails-cut **0.00% / 0.00%
(max 1.30×)**. The learned tail IS the exposure window: the model emits its trained
~0.8 s of silence frames before EOS, and every frame is an entry lottery for the
attractor. No window → no entry, no boost needed. NOTE the dose distinction:
SCALING tails to 0.82 s (trim_tails --median-to, the tt1/ae2h recipe) does NOT fix it —
tr_ae2h has exactly that and fails; cutting to ~0.05 s does. OWED before recipe
adoption: ear/ASR on cut-arm endings (median 0.957× is consistent with tail loss, not
word loss — but unheard), and pause-prior side effects at full 2 h scale.
Synergy: a near-zero self-tail voice hands gap control entirely to assembly, which is
the architecture Owen wants anyway.

**ELIMINATED tonight, each by measurement — do not re-chase:**
- *Adobe Enhance*: dose runs BACKWARDS (mb SNR moved 5.3 dB → clean 20/20; tr moved
  0.0 dB → fails). Third arm (raw, no-Adobe, same clips) launched to close it fully.
- *Spectrally-dead pauses*: pause_frame_conc on BOTH corpora = 0.2% top-3, every frame
  distinct — healthier than the novels reference. The July predictor is real for
  dead-silence corpora (MM 44%) but CANNOT explain tr-vs-mb.
- *Dehum lineage*: provenance corrected by Owen — tr_ae2h is **The Third Reich in
  POWER**, a brand-new book, NO dehum, NO cleaning, Adobe voice-enhancement only with
  background kept. (Bible previously implied trim2h descended from the dehummed master.)
- *Damaged/specific texts*: the 28 worst long-pause sentences re-render CLEAN at
  production settings (median 1.08×) — it is a per-render dice roll, not a text property.
- *Corpus size*: a 30-min quicktrain reproduces the 2 h model's rate (2.0% vs 2.5%).
- *Chunk length as entry driver (cross-voice)*: mb generates MORE tokens per 350-char
  chunk than tr (slower pace) and stays clean.

**What still separates the voices:** thin greedy EOS margin has now followed Sean Pratt
across TWO unrelated books and processing chains (Coming/dehummed ~70% greedy fail;
Power/Adobe-only: 6/8 offenders + 8/12 controls greedy fail). With every processing
hypothesis dead, the margin looks narrator-inherent (delivery/endings) — plus the anchor
arithmetic above. tr_twcut suggests it doesn't matter: remove the window and the thin
tie never gets rolled.

**Measurement lessons:** greedy vs sampled disagree ~25× on this defect (greedy cannot
escape the loop; sampling escapes ~97%) — greedy is a MARGIN gauge, never a rate
predictor; 8/12 chunks that shipped CLEAN in the book fail greedy. Per-sentence probing
cannot see a stochastic defect — the null result on the 28 worst sentences was the pivot
to volume A/B (rate_probe.py, the design that worked). Cost tables must baseline against
clips that STOPPED honestly (first drift table compared against 3700-token runaways and
read −12.8% when the truth was 0%). And probe harnesses MUST strip SML the way
orpheus.py:1194 does — an unstripped probe SPEAKS "break" (Owen caught this; it was
reported as a production bug and retracted).

**Tools (C:\tmp, promote if kept):** rate_probe.py, pause_scan.py, boost_probe.py,
render_indices.py (+ asr_text_qc.py), build_twin_corpora.py, twin_tail_chain.sh.

**ADDENDUM (01:46) — the raw arm flips Adobe to PROTECTIVE.** Third twin arm: same 144
clips from the pre-Adobe `thirdreich_trim2h`, same config/steps. Boost OFF, same 300
sentences: **raw 6.67% ≥1.5× / 6.33% cap (max 2.99×)** vs Adobe 2.00%/1.67% vs
Adobe+cut-tails 0.00%. 20 vs 6 events, p≈0.008. So the full ordering is
raw+tails ≫ adobe+tails ≫ adobe+cut = clean: TWO independent levers (Adobe ~3×,
tail-cut →0), and the deployed-voice pattern fits (mistborn, the healthy voice, is also
an Adobe corpus). Adobe stays the default; tail-cut joins it. The boost-converts-caps-
to-bloat signature appeared a third time (raw at 8@2.0: caps 19→2, bloat 6.67→5.33%).
`tr_ct1` (Adobe + cut tails, full 1.78 h) is the right full-scale corpus and started
training at 01:46. Mechanism for WHY Adobe protects: not established — do not guess it
into the doctrine; the twin grid (raw+cut arm) would complete the 2×2 if it ever matters.

**ADDENDUM 2 (02:30) — THE TAIL-CUT DID NOT REPLICATE AT FULL SCALE. Do NOT carve it
into doctrine.** tr_ct1 (Adobe + tails 0.05 s, the FULL 1.78 h corpus, 8 epochs) at
boost OFF, 600 sentences: **2.83% ≥1.5× / 2.50% cap (p99 2.36×)** — the same rate as
tails-kept tr_ae2. The twin's 0/300 was real (P(0/300)≈0.05% if the true rate were
2.5%) but it was a property of the 165-step QUICKTRAIN, not of the tail cut: with 4×
the optimization on 4× the clips, the runaway propensity RETURNS even with no trained
tail — the silence mass in verbatim INTERNAL pauses is evidently sufficient for the
end-of-utterance tie once training consolidates. Exposure-window story = incomplete.
Owen's instinct ("only a real full-scale test will tell") was exactly right; screens
that touch silence behavior are scale-dependent — quicktrains under-train pause/EOS
consolidation and CANNOT clear a corpus lever, only kill one.
**The boost is the fix that scales: 16@1.5 = 0/600 on BOTH 2 h models** (tr_ae2_ep264
and tr_ct1_ep264), p99 1.30×, medians untouched.
**Gate blindness, quantified by tonight's own numbers:** tr_ct1_ep264 passed greedy
20/20 while carrying a true sampled boost-off rate of 2.5% — a 20-chunk gate passes a
2.5%-defective model 60% of the time ((1−.025)^20). A 20/20 is NOT evidence of health
at production-relevant rates; it only catches margin collapse. Rate verdicts need the
600-sentence probe (rate_probe.py).
Witnesses recommendation: tr_ae2_ep264 @ 16@1.5 (already registered, voice
ear-approved); tr_ct1 offers no safety advantage, unheard voice, and needs gap retune.

## 2026-07-26 (later) — Silence-attractor MECHANISM measured: EOS is edge-frame-cued, not duration-cued; the voices differ in OVERSHOOT HAZARD

Campaign goal: replace "narrator-inherent by exclusion" with a measured mechanism.
Instruments/logs in C:\tmp (instr_a/b/b2/b3/c/d_*.py + .jsonl/.json/.log outputs).
Models probed with HF teacher-forcing (bf16, WSL orpheus_tts), exact training layout
(mask-prompt irrelevant at inference; --no-dedup interleave; prompt "{token}: {text}").

**H3 (raw floor = dead/concentrated frames; Adobe diversifies) — REFUTED.**
instr_c_floor.py on thirdreich_trim2h / tr_ae2h / mb_ae2h: SNAC top-3 frame share in
PAUSES 0.37/0.25/0.13%, in TAILS 0.57/0.55/0.44% — every frame essentially distinct in
ALL corpora (dead-silence signature is 27-44%). Zero digitally-silent frames anywhere;
raw pause floor p10 -91 dBFS (deeper, wider spread), Adobe -82, mb -84. The production
attractor cycle = SNAC's canonical DIGITAL-ZERO encoding (instr_b3: 2s of zeros -> ONE
repeated frame; -80 dB noise -> all distinct) and exists in NO training corpus.

**H2 (per-clip trained-end margin varies with clip property) — REFUTED.**
instr_b_margin.py, full corpora: teacher-forced margin at the TRUE final frame is
uniformly healthy on BOTH voices (tr median +24.4, min +1.1, 100% rank-1; mb median
+24.4, min -0.2, 99.8% rank-1). No clip property correlates (all |Spearman| <= 0.12).
Pause EOS leakage tiny and equal (median p ~2e-6 both). NOT the discriminant.

**H1 (Pratt's ends acoustically identical to his pauses) — REFUTED, direction reversed.**
instr_a_prosody.py, END vs PAUSE separability (F0 slope/relative terminal F0/energy
slope/onset ratio, 5-fold LDA AUC): adobe-tr 0.712, raw-tr 0.700, mb 0.659. Pratt's
endings are MORE separable than Kramer's. The gate did not erase finality cues.

**Instrument D (end-text misalignment) — REFUTED as discriminant.** ASR final-3s
last-word check, 250/corpus: real mismatches 1.2% vs 1.2% (raw rates 14.4% vs 6.0% are
number-orthography + fantasy-word ASR artifacts).

**THE POSITIVE MECHANISM (instr_b2_overshoot.py — the new gauge).**
1) EOS is cued by the CLIP-SPECIFIC SNAC end-of-file edge frames (last 1-2 frames of an
   unextended encode differ from a continuing-silence encode 24/24 clips, instr_b3), not
   by silence duration: extend the tail with the clip's own floor silence and the +24
   margin NEVER appears — margin stays -6..-10 through +2 s of overshoot.
2) In free-run there is no file edge; stopping relies on the model's residual per-frame
   EOS hazard during self-overshoot. THIS is what separates every model measured.
   P(no EOS through +2 s overshoot), 120 clips each, per-clip product of hazards:
     twraw  ep165  surv mean 0.925 (ever-margin>0: 12/120)   prod rate 6.67%
     tr_ae2 ep264  surv mean 0.732 (37/120)                  prod rate 2.50%
     twkeep ep165  surv mean 0.647 (38/120)                  prod rate 2.00%
     mb_ae2 ep327  surv mean 0.346 (86/120)                  prod rate ~0%
   MONOTONIC with all four production boost-off rates, across scale (165-step quicktrains
   AND 2h/8-epoch) — unlike the greedy 20-chunk gate, this probe SEES the 2.5% defect.
3) tr's hazard also DECAYS with absolute silence duration (p90 p_eos 3.2e-2 at 0.8-1.1s
   -> 4e-3 at 2.4-3s) while mb holds p90 0.10-0.29 through 1.1-2.0s. mb's tail-length
   spread is wider (std 0.36 s, p90 1.54 vs tr 0.26/1.11 — tr Step-5b-uniformized), the
   right shape to teach a duration-spread hazard; correlate, NOT proven cause (twins
   share identical tails yet Adobe-vs-raw still moves hazard 0.925->0.647, so the Adobe
   lever is floor/texture-mediated, mediator still open).
Defect rate = P(drift past own tail) x P(no rescue | overshoot); B2 measures factor 2.
Limitation: B2 teacher-forces corpus-floor silence, not self-generated silence; tiling
introduces a 3-frame phase artifact (tile 6000 smp ~ 2.93 frames) — read hazard at the
favorable phase.

**GATE RECOMMENDATION:** run instr_b2 (120 clips, ~8 min GPU) on every candidate; require
mean overshoot survival ≲ 0.5 (mb 0.35 clean / tr 0.73 defective / raw twin 0.93). This
is the first instrument that rank-orders the sampled production rate from the model alone.
**Corpus lever suggested by the data (untested):** stop uniformizing tails (trim_tails
--median-to); keep/restore natural tail-length spread so EOS is seen at many silence
durations. A floor-consistency arm (raw audio + synthetic -80 dB floor) would isolate the
Adobe mediator. Both need full-scale twins per the scale-dependence law above.
