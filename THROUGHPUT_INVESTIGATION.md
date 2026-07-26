# Why is thirdreich slower? — a test battery to reach ground truth

**Status: DESIGNED, NOT RUN (2026-07-25).** Owen asked for a battery rather than
another round of inference, because I produced three wrong explanations in one night
and each one was plausible before it was measured. Nothing below is a conclusion.

## The observation

Same book (God's People), same engine, same tier (`fast`), same GPU, same
`gpu_memory_utilization=0.54`, 1 worker each, 9-minute cumulative windows:

| voice | chunks/min | mean chunk | chunks/batch cycle | cycle length | guard trips |
|---|---|---|---|---|---|
| deathstalker | **64.0** | 16.14 s | 60.1 | 54.2 s | 0 |
| mb_hoa1_ep344 | **49.8** | 18.69 s | 58.7 | 69.5 s | 0 |
| tr_tt1_ep420 | **42.7** | 17.42 s | **36.3** | 56.3 s | **9** |

The batch CADENCE is the same for all three; what differs is how many chunks clear
per cycle. thirdreich clears 36 where the others clear ~59.

## Already RULED OUT by measurement (do not re-test)

| hypothesis | why it died |
|---|---|
| mean audio length | tr is only 1.08× ds; predicts 59 chunks/min, measured 42.7 |
| length, superlinear (1/L²) | predicts 1.17×, measured 1.50× |
| length ordering at all | **mb is LONGER than tr (18.69 vs 17.42) yet FASTER (49.8 vs 42.7)** |
| KV preemption | ds had MORE preemption events (9) than tr (3) |
| token-cap runaways | zero cap hits in all three runs |
| worker count / vLLM config | identical: 1 worker, util 0.54, 12.02× concurrency |
| speaking rate | Pratt is FASTER per speech-second than Rohan (29.8 vs 27.2 ch/s) |
| trained clip length | ds corpus clips are LONGER (15.94 s vs 12.03 s) |

## Prerequisite — T0: instrument the worker (do this FIRST)

Everything below is guesswork without per-chunk facts. Add a CSV line per generation
in `worker_core.py` / `orpheus.py`:

```
chunk_index, batch_id, prompt_tokens, generated_tokens, wall_ms,
  accepted(bool), reject_reason, chars, audio_seconds
```

That single change makes A, B, C, D and F directly measurable instead of inferred,
and it is the difference between "the numbers are consistent with X" and knowing.
Cost: one small patch, one run. **If only one thing gets done, do this.**

---

## Hypotheses, most plausible first

### A — Guard-trip serialization *(leading)*
**Mechanism.** thirdreich early-EOSes on some chunks. e2a's completeness guard rejects
them (`audio too short for text (28.9 ch/s > 21.5)`) and re-renders **split at sentence
boundaries** — several serial generations that stall the batch they belong to. 9 trips
over 28 cycles ≈ one per three cycles; ds and mb trip zero times.
**Predicts.** Batch cycles containing a trip clear far fewer chunks than clean cycles.
**T1 — guard-off run.** Set `maxCharsPerSec` absurdly high (999) for tr only, re-measure
9 min. If tr jumps toward ~59 chunks/min, A is confirmed. *The audio will be defective —
throw it away, this measures cost only.* Cost: 15 min.
**T1b.** With T0 data, correlate trip timestamps against per-cycle chunk counts. Free
once T0 exists.

### B — Batch tail-blocking by long sequences
**Mechanism.** A batch finishes when its LONGEST sequence does, so slots idle. tr has
16/350 chunks >25 s vs ds 2/350.
**Problem.** mb has **40**/350 >25 s and is faster — so this cannot be the whole story,
though it may explain mb's longer 69.5 s cycle.
**T2 — length-matched subset.** Render the same 100 chunks whose durations are within
±5% across all three voices. If the gap survives on length-matched text, B is dead.
Cost: 20 min.

### C — Token-budgeted batch formation with an inflated estimate
**Mechanism.** If the worker forms batches against a token budget using an *estimated*
token count (the same estimate `eosBoostStart` uses), and tr's estimate runs high, tr
packs fewer chunks per batch — exactly the observed symptom.
**Predicts.** tokens-per-batch is roughly constant across voices; chunks-per-batch is not.
Measured (mean × chunks/cycle): ds ~79k, tr ~51k, mb ~90k — **not** constant, so this is
already weakened, but the estimate itself is worth reading.
**T3.** From T0, plot estimated vs actual tokens per chunk per voice; read the batching
code path for what actually sets batch size.

### D — eosBoost logit-processor overhead
**Mechanism.** The boost adds a per-step logit processor. If tr's sequences routinely pass
`2.0× expected` (where ds's don't), the processor does real work on many more steps.
**T4.** Run tr with `eosBoost 0`, 9 min. Expect runaways — measure speed only. If speed
recovers, D contributes. Cost: 15 min. (Also isolates whether the *guard* or the *boost*
is the cost, together with T1.)

### E — CUDA graphs not captured for tr
**Mechanism.** A stray difference in tr's `config.json` / dtype could force
`enforce_eager`, which is ~6× slower on Windows and still costly on Linux. A 1.5× gap is
much smaller than 6×, so this is unlikely alone — but trivially checkable.
**T5.** `diff` the three model dirs' `config.json` + `generation_config.json`, and grep
each run's log for `Graph capturing finished`. Cost: 2 minutes. **Do it anyway** — it is
nearly free and would be embarrassing to miss.

### F — The merged weights are genuinely slower
**Mechanism.** Shouldn't happen: identical architecture and dtype means identical FLOPs
per token. Included because "shouldn't happen" has been wrong twice tonight.
**T6 — batch-1 latency.** Generate 20 fixed chunks at concurrency 1 for each voice and
compare **ms per generated token**. If those are equal, F is dead and the cause is
scheduling, not the model. If they differ, everything above is downstream of it.
Cost: 15 min. **Highly discriminating — worth doing early.**

### G — Ordering, thermal or clock effects
**Mechanism.** tr ran after two other runs; a hot GPU downclocks. Memory has prior art
here (a dust-jammed fan once cost real throughput).
**T7.** Log `nvidia-smi --query-gpu=clocks.sm,temperature.gpu,power.draw` every 10 s
during every run, and run the three voices in **randomized order, twice**. If the ranking
flips or clocks sag, G contributes. Cost: 1 hour, but it also gives error bars — and we
need those: ds measured 64.0 and 71.1 on two windows of the *same* run (±10%).

### H — The chunk TEXT differs per voice
**Mechanism.** Per-voice `maxChars` / the rate ratchet could split text differently, so
"chunk 412" isn't the same text for both voices and the comparison is invalid.
**T8.** For common indices, compare character counts across voices. Must be identical.
Cost: 2 minutes. **Cheap and it invalidates everything if it fails.**

---

## Silly hypotheses (cheap, and one of them will surprise us)

| # | idea | test | cost |
|---|---|---|---|
| S1 | The voice token string length differs (`thirdreich` = 10 chars vs `deathstalker` = 12), changing prompt length and thus prefill | log prompt_tokens per voice (T0) | free |
| S2 | tr's model dir is cold in the page cache / on a slower part of the disk, so weight load and first batches drag | `vmtouch` the dir before running; compare first-cycle vs later-cycle rates | 10 min |
| S3 | The 64-chunk flush is a **fixed timer** (watchdog is 30 s), not a batch, so we've been measuring a checkpoint interval and the real difference is elsewhere | read the flush code path; check whether flush size tracks time or count | 15 min |
| S4 | The `.flac` encode is the bottleneck for one voice — longer audio = more encode CPU, and encode is serialized with generation | time encode separately; try `.wav` output | 20 min |
| S5 | tr's audio has more high-frequency content, so FLAC compresses worse and writes are bigger/slower | compare bytes-per-second-of-audio across voices | 2 min |
| S6 | Something is *still running* from a previous job and stealing GPU during tr's window (we had orphaned workers all night) | `nvidia-smi` process list + `pgrep` snapshot every 10 s during each run | free with T7 |
| S7 | The eosBoost `2.0×` threshold makes tr generate *longer* before the boost fires, and the extra tokens are then discarded by the guard — so we pay for tokens that never reach the output | T0's generated_tokens vs audio_seconds ratio per voice | free with T0 |

S6 deserves a real mention: orphaned workers were a live problem tonight, and a stray
process during tr's window would produce exactly this result. The randomized repeat (T7)
is the cheapest insurance against it.

---

## Suggested running order

1. **T5** (config/CUDA-graph diff) and **T8** (text identity) — 5 minutes, either could
   invalidate everything.
2. **T0** (instrumentation) — the prerequisite.
3. **T6** (batch-1 ms/token) — splits "model" from "scheduling" in one shot.
4. **T1** (guard off for tr) — direct test of the leading hypothesis.
5. **T7** (randomized repeat with clock logging) — error bars + kills G and S6.
6. Then whichever of B/C/D the T0 data points at.

## What "ground truth" would look like

A per-chunk table for each voice showing wall_ms, generated_tokens and accepted/rejected,
from which we can state: *of thirdreich's N seconds per cycle, X% went to accepted output,
Y% to rejected generations, Z% to idle slots waiting on long sequences.* Until we can fill
in X, Y and Z, every explanation here is a story that happens to fit.
