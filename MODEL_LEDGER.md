# MODEL LEDGER — what was trained, on what, with which settings, and did it ship

**Created 2026-07-25.** Owen asked for this while chasing the mistborn wobble:
*"Do we have a ledger for settings we've tried? We should probably be keeping a
record of settings we configured for each model."* We didn't. We had three
partial records and no index:

| record | what it holds | what it does NOT hold |
|---|---|---|
| `bookforge/electron/data/orpheus-models.json` | the **current winner's** runtime tuning per voice, with rationale | anything that was tried and rejected; which train run produced the model |
| `TESTING_BIBLE.md` | every experiment, **chronologically**, in full detail | any way to ask "what has this voice been through?" |
| `/mnt/c/tmp/<prefix>_train.log` | the ground truth of what each run actually did | outcomes, gate results, whether it shipped |

This file is the **index by voice**. One row per train run, one row per settings
change, and the verdict. It is append-only like the bible: never rewrite a row to
match a later belief — add the correction as its own row, because the wrong belief
is usually the useful part.

**Rule: a run that is not in this table did not happen.** If you train and don't
add a row, the next person re-runs it. Adding the row costs 30 seconds.

---

## 1. Currently deployed tuning

Do **not** duplicate the tuning here — `orpheus-models.json` is authoritative and
propagates to every machine on push. This is the pointer, as of 2026-07-25:

| voice | added | maxCharsPerSec | repPenalty | eosBoost | sentenceGap (inject / measured) | post-filter |
|---|---|---|---|---|---|---|
| owen | 07-15 | — | — | — | — / — | EQ (13-band firequalizer) |
| thirdreich | 07-18 | 21.5 | 1.10 | 8 @ 2.0× | 0.3 / 1.4 | — |
| mistborn | 07-21 | 21.5 | 1.10 | 8 @ 2.0× | — / 0.8 | — |
| ender | 07-22 | 21.5 | 1.10 | 8 @ 2.0× | — / — | — |
| deathstalker | 07-22 | 21.5 | 1.10 | 8 @ 2.0× | 0.0 / 0.7 | — |

`maxCharsPerSec 21.5` and `repPenalty 1.10` are the campaign-wide defaults, not
per-voice discoveries — 21.5 because the stock 19.0 guard sits *inside* Pratt's
natural 15–20.5 ch/s range and false-trips. `eosBoost 8 @ 2.0×` is inference-side
and vLLM-only (e2a defaults to 0.0 = off; the Mac MLX backend ignores it).

---

## 2. Train-run inventory

Every run with a log under `/mnt/c/tmp/<prefix>_train.log`. **Corpus is quoted from
the log's own first line** — not from memory. `chain=1` means the merge chain
finished; `chain=0` means it was killed, crashed, or superseded mid-flight.

Outcome columns marked **—** are genuinely unrecorded: those runs predate this
ledger and their logs don't carry a verdict. Don't guess them; if you need one,
the detail is in `TESTING_BIBLE.md` under that date.

### thirdreich (Sean Pratt, nonfiction)

| date | run | corpus | chain | outcome |
|---|---|---|---|---|
| 07-19 | `tr_v2` | `thirdreich_np8h_v2_tails` | 1 | — |
| 07-20 | `tr_2h` | *(census only, `thirdreich_np_8h`)* | 1 | n/a |
| 07-20 | `tr_2hb` | `thirdreich_2h_tails` (breath-safe edges) | 1 | — |
| 07-21 | `tr_rv1` | `thirdreich_rv2h` | 1 | superseded — corpus was cut from a **dehummed** master, 19.18% exactly-zero samples, SNAC pause conc 29.7% (dead band) |
| 07-24 | `tr_rv2` | `thirdreich_raw2h` | 1 | re-cut from the raw master; floor healthy. Still truncated — the defect was in the TEXT, not the audio |
| 07-25 | `tr_tp1` | `thirdreich_trip2h` (574 clips) | 1 | **ep132 registered.** Text-clean (endnote markers stripped at source). Greedy gate 20/20. Measured **30.5–30.7 chunks/min** on three different books — the throughput baseline |
| 07-25 | `tr_tt1` | `thirdreich_trim2h` (609 clips, 2.03 h, tails scaled ×0.601) | 1 | **ep420 = keeper candidate.** Self-stopped at epoch 4 (overtrain+2). Trim worked: rendered trailing 1.21–1.43 s → 0.62–0.75 s. **eval-min (ep280) is DEFECTIVE** — pre-consolidation, 18.6 s of trailing dead air, BLOAT 1.80×; ep140 likewise (15.6 s, 1.94×). ep420/ep560 PASS everything. **Throughput MEASURED: 42.7 chunks/min** (was 30.5 untrimmed on the same book, +40%) vs deathstalker 64.0 — a real gain that does NOT close the gap |

`tr_tt1` epoch battery, on identical texts to `tr_tp1` (20 chunks, production
sampled settings). This is the table that shows why a keeper cannot be picked on
eval_loss:

| epoch | loop gate | trailing med / p90 / **max** | internal /min, med | clip med | speech % | eval_loss |
|---|---|---|---|---|---|---|
| ep140 | BLOAT 1.94× | 0.94 / 1.32 / **15.56 s** | 20.2, 540 ms | 15.66 s | 65.3% | 3.668 |
| ep280 *(eval-min)* | BLOAT 1.80× | 0.85 / 1.03 / **18.58 s** | 21.6, 560 ms | 16.04 s | 65.3% | **3.611** |
| **ep420** | **PASS 20/20** | 0.75 / 0.90 / **1.12 s** | 20.2, 560 ms | 14.08 s | **74.8%** | 3.616 |
| ep560 | PASS 20/20 | 0.62 / 0.89 / **0.94 s** | 22.0, 520 ms | 14.29 s | 73.5% | 3.682 |
| tr_tp1 (untrimmed) | — | 1.21–1.43 / 1.70–1.99 / 2.36 s | 19–21, 480–520 ms | 14.89–16.30 s | 69–71% | — |

ep420 over ep560: both consolidated and clean, but ep420 sits one epoch from
eval-min (3.616 vs best 3.611) where ep560 is two past it (3.682), and ep560's
0.62 s tail is already shorter than the 0.99 s it trained on. Ear decides.

**The shipped `thirdreich` is none of these.** It is a pre-pivot 07-18 checkpoint
that was deployed before the loop/completeness gate existed, with **zero native EOS
margin** — it only behaves because `eosBoost 8` rescues it at inference. That is
recorded in the catalog note and is why the rebuild exists.

### deathstalker (the reference voice — everything is measured against it)

| date | run | corpus | chain | outcome |
|---|---|---|---|---|
| 07-19 | `ds_v7` | `deathstalker_mm8_v7_tails` | 1 | — |
| 07-20 | `ds_2h` | *(corpus build only)* | 1 | n/a |
| 07-20 | `ds_2hb` | `deathstalker_mm2h_tails` (breath-safe edges) | 1 | — |
| 07-20 | `ds_v8` | `deathstalker_mm8_v8_tails` | 0 | abandoned |
| 07-21 | `ds_rv1` | `deathstalker_rv2h` | 1 | raw-verbatim pivot arm |
| 07-21 | `ds_ad1` | `deathstalker_adobe2h` (399 clips) | 1 | **Adobe pre-cleaning REFUTED for training audio** (bible 07-21) |
| 07-22 | `ds_nr1` | `mm_narr2h` (narration-only, quote-mapped) | 1 | **the keeper.** 69.8 chunks/min; tail median 0.98 s / p90 1.43 s — the distribution Step 5b now targets |

### mistborn (Michael Kramer)

| date | run | corpus | chain | outcome |
|---|---|---|---|---|
| 07-20 | `mb_2h` | *(census only, `mistborn_aol8`)* | 1 | n/a |
| 07-20 | `mb_2hb` | `mistborn_aol2h_tails` | 0 | killed — 94% breathy STARTS (audible inhale before every phrase); produced the leading-inhale rule. **NOT the ghost whisper** — Owen, 07-25: that was the added synthetic hiss bed, unrelated |
| 07-20 | `mb_2hc` | `mistborn_aol2h_tails` | 1 | — |
| 07-21 | `mb_2hd` | `mistborn_aol2h_nobed` | 1 | ghost-whisper campaign arm (bible 07-21) |
| 07-21 | `mb_2he` | `mistborn_aol2h_bedtails` | 1 | ghost-whisper campaign arm |
| 07-21 | `mb_2hf` | `mistborn_aol2h_silenttails` | 2 | ghost-whisper campaign arm |
| 07-21 | `mb_rv1` | `mistborn_rv2h` | 1 | raw-verbatim pivot arm; **this is what shipped** |
| 07-25 | `mb_aol1` | `mistborn_narr2h` (narration-only, Alloy of Law) | 1 | **wobble — rougher than shipped.** Brightness spread regressed to 4.0 dB vs shipped ~3.1: the narration-only pool was only 2.86 h, too shallow to be selective at 2.0 h |
| 07-25 | `mb_hoa1` | `mistborn_hoa3p0h` (750 clips, 3.00 h, Hero of Ages narration-only) | 1 | 5 epochs (stopped on overtrain+2), eval-min **ep516** (4.0640) — but ep344 is 4.0648, a 0.0008 tie, i.e. noise. Greedy EOS gate 20/20 on the last 4. **Throughput 49.8 chunks/min** = 78% of deathstalker's 64.0. Battery below |

**Why Hero of Ages replaced Alloy of Law.** The wobble was a POOL-DEPTH problem, and
selectivity is the whole lever:

| | AoL (`mb_aol1`, wobbled) | HoA (`mb_hoa1`) |
|---|---|---|
| source audio | 10.8 h | **27.4 h** |
| narration-only pool | **2.86 h** | **14.95 h** |
| corpus taken | 2.0 h = ~70% of pool | 3.0 h = ~28% of pool |
| brightness spread | **4.0 dB** | **2.9 dB** (shipped ~3.1) |

HoA was chosen on measured drift across the book (1.8 dB vs 5.7–8.4 for Final Empire /
Well of Ascension / AoL), and the corpus census confirmed it: unimodal, 2.1 dB decile
drift. Align quality: 23,813 of 24,290 sentences narrated (98%), **0 unmatched audio**
(so no ASR-fallback text to exclude), residual drift p95 0.85 s.

**NOT tail-trimmed, deliberately.** Kramer's tails measure 1.08 s median against the
deathstalker reference 0.98 — normal variation, not thirdreich's 1.63 s pathology.
Trimming would risk a healthy distribution to win ~0.1 s per chunk. Step 5b applies to a
defect, not to every corpus.

**Open and NOT confirmed:** that brightness spread causes the wobble. The
discriminating test — render the *shipped* model on the same passage — was
deliberately deferred (*"let's not touch the shipped model until we nail it
perfectly"*), so we do not actually know whether the wobble predates `mb_aol1`.
Hero of Ages was selected as the replacement source on drift (1.8 dB across the
book vs 5.7–8.4 for the others) and its align is running.

`mb_hoa1` epoch battery, all 5 epochs on identical texts (20 chunks, production
sampled settings), plus the SHIPPED model on the same texts — the discriminating
comparison that had been deferred:

| epoch | pauses/min | median | trailing med | brightness | spread | sampled loop gate | eval_loss |
|---|---|---|---|---|---|---|---|
| ep172 | 21.1 | 580 ms | 1.10 s | -22.1 | 3.2 dB | **FAIL** (LOOP #003) | 4.117 |
| **ep344** | **24.0** | **430 ms** | 1.01 s | -21.9 | **2.0 dB** | **PASS 20/20** | 4.0648 |
| ep516 *(eval-min)* | 27.1 | **270 ms** | 1.24 s | -21.3 | 3.4 dB | **FAIL** (LOOP #003) | **4.0640** |
| ep688 | 25.2 | 290 ms | 1.02 s | -21.6 | 2.7 dB | PASS 20/20 | 4.111 |
| ep860 | 23.5 | 420 ms | 1.14 s | -21.7 | 3.5 dB | **FAIL** (LOOP #005) | 4.252 |
| **CORPUS (truth)** | **24.5** | **360 ms** | 1.08 s | **-20.8** | 3.0 dB | — | — |
| shipped mistborn | 26.1 | 540 ms | 0.76 s | **-7.0** | **6.4 dB** | PASS 20/20 | — |

**Keeper = ep344**, deployed as a separate app entry; shipped model untouched pending
Owen's ear. It wins on every axis and is 0.0008 off the loss minimum — a tie, i.e. noise.

**Two findings worth keeping:**
1. **The shipped model wobbles MORE than anything trained here** — 6.4 dB rendered
   brightness spread vs ep344's 2.0. So the roughness Owen reported very likely predates
   this campaign. This is the test that was deliberately deferred ("don't touch the
   shipped model until we nail it"); rendering it is read-only, so it cost nothing.
2. **The shipped model is 13.8 dB BRIGHTER than its own narrator** (-7.0 vs corpus
   -20.8) while every mb_hoa1 epoch sits within ~1 dB of the corpus. The new model is
   faithful and will therefore sound notably DARKER. Faithful is not automatically
   better — that is an ear call. Cause unknown (brighter AoL source vs the model
   itself); both AoL corpora were deleted in the 07-25 cleanup, regenerable from the
   m4b in ~30 min if it ever matters.

**eval-min lost twice in one night, by two different mechanisms** — thirdreich ep280
(18.6 s dead air) and mistborn ep516 (pause squish to 270 ms + loop-gate FAIL). The
greedy EOS gate passed ep516 20/20; only the SAMPLED gate caught the loop. Compare
rendered pauses to the TRAINING CORPUS, never to the previous model.


### ender (Ender's Game)

| date | run | corpus | chain | outcome |
|---|---|---|---|---|
| 07-21 | `en_qt0` / `ender_qt0` | `ender_qt0` (33 clips) | 1 / 0 | twin quick-train |
| 07-21 | `en_qt1` / `ender_qt` | `ender_qt` (32 clips) | 1 / 0 | twin quick-train |
| 07-21 | `en_qr1` | `ender_raw_qt` (121 clips) | 1 | raw arm |
| 07-21 | `en_qa1` | `ender_adobe_qt` (121 clips) | 1 | Adobe arm — **Adobe WINS for a weak source** (bible 07-21 late) |
| 07-21 | `en_qa2` | `ender_adobe_qt2` (257 clips) | 1 | Adobe arm, larger |
| 07-21 | `en_ds1` | `ender_ds1_corpus` (555 clips) | 0 | abandoned |
| 07-22 | `en_sh1` | `ender_sh2h` | 1 | shipped tuning added 07-22 |

The 32/33-clip runs are the **twin quick-train** pattern: two corpora differing in
exactly one variable, trained tiny, compared. That is how causation gets
established here instead of inferred — it's the pattern to reach for before
spending a full 8-epoch chain on a hypothesis.

### owen (Owen Morgan)

| date | run | corpus | chain | outcome |
|---|---|---|---|---|
| 07-11 | `rohan`, `rohan_v2` | *(pre-dating the corpus convention)* | 0 | early XTTS-era runs |
| 07-22 | `ow_rv1` | `owen_rv2h` | 1 | raw-verbatim |
| 07-22 | `ow_rv2` | `owen_sel2h` | 1 | selected; **God's People twin — the Adobe source is the ceiling** (bible 07-22) |

---

## 3. Settings tried, and the verdict

Only entries with a measurement behind them. "Rejected" here means measured worse,
not merely unused.

| setting | tried | verdict | where |
|---|---|---|---|
| `eosBoost 8 @ 2.0×` vs none | 20 greedy chunks, identical texts | **KEEP.** pen 1.10 alone failed 7/20 (all looping to the 3700-token cap); + boost passed 20/20. Surgical: the 13 already-healthy chunks returned *identical* token counts in both arms | catalog note, 07-24 |
| `maxCharsPerSec` 19.0 (stock) → 21.5 | Pratt measured at 15–20.5 ch/s | **21.5.** 19.0 sits inside his natural range, false-trips, and a first trip splices before the ratchet can learn (audible seam) | catalog note |
| `repPenalty` 1.15 → 1.10 | vLLM runaway split | 1.15 + maxCharsPerSec 21.5 gave 0/234 runaways at 16× | memory `orpheus-vllm-runaway-backend-split` |
| `sentenceGap` 0.6 universal default | measured against trained tails | **double-counts.** The model already renders the trained tail; per-voice inject now 0.0–0.3 against a measured 0.7–1.4 | bible 07-23 |
| Adobe Podcast pre-cleaning, ALONE | ds (strong source) vs ender (weak source) | **source-dependent.** Refuted for a good source; wins for a weak one. Adobe alone de-brightens 7-9 dB above 3 kHz | bible 07-21 |
| Adobe Enhance **+ matched inverse EQ** | mb_ae2 (2 h HoA) vs untreated hoa3p0h | **SAFE AND BETTER - now the default.** Best mistborn to date: Owen's ear, eval-min 3.8214 vs 4.0640, greedy gate 20/20. Does NOT damage audio. The EQ restore is what makes it work; Adobe alone is too dark | bible 07-25 |
| RVC on training audio | ds corpus | **BANNED** — muffles to 1.7 kHz | memory `orpheus-audio-cleaning-recipe-and-deploy` |
| room-tone bed under clips | mb 2hd/2he/2hf | superseded by the raw-verbatim pivot; bed is now the documented FALLBACK for dead-floor sources only | bible 07-21 |
| corpus-side EOS tricks (silence mass) | ds twins | **breaks production.** EOS margin comes from the inference-side boost instead | memory `orpheus-eos-capability-failure` |
| narration-only corpus (quote-mapped) | ds_nr1 vs voice-matching | **quote-mapping wins** — text is exact/structural, embedding similarity is an overlapping gradient | bible 07-22 |
| tail scaling to a reference distribution | tr_tt1 ep420 | **KEEP.** 30.5 -> 42.7 chunks/min (+40%) on the same book; best-behaved voice measured (tail 0.76 s, maxSil 1.23 s) | bible 07-25 |

---

## 4. Template — copy this when you train

```
| <date> | <prefix> | <corpus path as the log prints it> (<n> clips, <h> h) | <chain> | <outcome> |
```

The outcome cell must answer three things, or it isn't worth writing:
1. **Which epoch won**, and on which gate — never a fixed offset (settle+1 and
   eval-min were both tried and both superseded; the keeper is whichever epoch
   passes the whole battery).
2. **A number.** chunks/min, greedy pass rate, brightness spread — something the
   next run can be compared against.
3. **Shipped or not**, and if not, why not.

If a run changed a *setting* rather than a corpus, add a §3 row too, and say what
it was measured against. A setting with no comparison arm is a preference, and
preferences belong in the catalog note, not here.
