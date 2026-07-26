# CoD pt05-08 Combined Retrain Plan — deathstalker voice

**Written 2026-07-13 (evening), for execution tonight when the Adobe-cleaned pt05-08 parts land.**
Open this file in a fresh Claude session and work it top to bottom. Every path below was
verified on disk today — but re-verify anything marked ⚠ before acting on it.

---

## ⚡ LATE-NIGHT UPDATE (2026-07-13 ~23:30) — READ BEFORE THE STEPS

The evening test session REWROTE the diagnosis. Findings, in order of importance:

1. **The runaway is BACKEND-SPECIFIC, not (only) a training defect.** Identical 234-chunk
   corpus (43 chunks ≥300 chars): **ep4 on vLLM/WSL = 39 token-cap runaways (~17%)**;
   **ep5 on MLX/Mac = 0 runaways.** ep4≈ep5 on vLLM (so NOT overtraining depth); training
   clip tails measured capped (max 0.16s — NOT tail silence); sampling VALUES identical
   (0.6/0.8/1.1/min_p 0). The verified mechanical difference: vLLM 0.7.3 applies
   repetition_penalty over the ENTIRE prompt+output; MLX uses a 20-token sliding window
   (mlx_lm sample_utils.py:279). Hypothesis: whole-sequence penalty suppresses the
   much-repeated pause/wind-down frames that precede EOS in --no-dedup models.
2. **NEXT STEP — the probe (GPU must be free):** `probe_runaway.py` in this repo
   (chunk texts in `probe_chunks.json`). Run in WSL:
   `wsl -d Ubuntu bash -ic "conda run -n orpheus_tts python /mnt/c/Users/tellt/Projects/orpheus-finetune/probe_runaway.py"`
   It renders the 6 longest chunks on ep5 at rep 1.1/1.0/1.05/1.3 with a 6000-token
   ceiling, saves FULL runaway wavs to /home/telltale/probe_runaway/ + tail-frame loop
   stats + trailing-silence measurements (tests Owen's "it's generating silence, never
   EOS" hypothesis, and the rep-penalty mechanism in one run).
   If rep 1.05/1.0 kills runaways → per-voice sampling fix, maybe NO retrain urgency.
3. **The GPU's second fan is DEAD** (commanded 100%, 0 RPM — hardware). All of today's
   training ran thermally throttled (855-900MHz). Fix before ANY training. Suspect part:
   FD10015M12D ~98-100mm (the 4090-TUF-OG/3090Ti-TUF cooler family) — VERIFY against the
   dead fan's hub sticker before ordering. The Jul-11 training crashes were probably this.
4. The e2a truncation-guard RATCHET went live and validated (1 recalibration to 22.9 ch/s,
   then zero false re-renders across ~230 chunks; vs 15 wasted re-renders on the
   ratchet-less Mac). Voice's natural rate ≈ 22.4 ch/s.
5. Baselines to beat, all same-corpus: vLLM ep4 17% caps / MLX ep5 0% / this morning's
   real book (350-char packing) vLLM ep5 ~11%.

## Why we're retraining (today's evidence — read this first)

The **ep5 overtrained checkpoint** (`/home/telltale/xtts_ft/orpheus_deathstalker_merged_ep5`)
was deployed to `/home/telltale/orpheus-models/deathstalker` today at 20:04
(`logs\deploy_deathstalker_ep5.log`). Its maiden run (Hellworld, BookForge job, 350-char
packing, WSL vLLM) produced, in ~90 real chunks:

- **10× "hit the audio-token cap"** — runaway/no-EOS on ~11% of chunks
- **8× "audio too short for text"** guard trips at 19.1–19.6 ch/s, with force-split
  re-renders measuring 20.0–20.6 ch/s and accepted — i.e. this checkpoint **naturally reads
  ~20 ch/s**, faster than the 15–17 ch/s voices the 19.0 guard default was calibrated on.

The reference number to beat: **deathstalker_v3 = 0 runaway / 126 at the same packing.**

**Key open question:** the pt01-04 dataset is CLEAN by the proven rules (985 clips, 202.3 min,
median 12.7s, max 19.5s ≤ 20s cap, 24 kHz mono, trail-capped — see `CUT_REPORT.md`). So the
11% runaway is NOT a long-clip violation. Two suspects, decided by Step 5's A/B:
- (a) **overtraining to ep5 degraded EOS** (eval-best was checkpoint-418, eval_loss 3.6617), or
- (b) the pt01-04 dataset is too small / this narrator needs more data (fixed by adding pt05-08).

Related code shipped today (independent of training, already live):
- **e2a `feat/orpheus-rate-ratchet`** (commit 75d4ecf): the truncation guard now self-calibrates
  upward when a force-split re-render proves the voice's natural rate exceeds the threshold —
  the 8 false-positive re-renders class is fixed. ⚠ **The WSL live copy at
  `/home/telltale/ebook2audiobook/.../orpheus.py` was synced from this branch** — if the Windows
  e2a checkout switches branches, re-sync deliberately.
- **BookForge `feat/orpheus-voice-caps`**: optional per-voice `maxChars` / `maxCharsPerSec` in the
  voice definition (stopgap lever if a checkpoint still runs away: set `maxChars: 300` for
  deathstalker instead of lowering the global 350 default).

---

## Step 0 — GPU preflight (do NOT skip: the card was sick today)

During today's TTS job the 3090 Ti was in **SW Thermal Slowdown**: pinned at 855 MHz
(max 2115) at only 216W draw, 83°C @ 94% fan — and still 64°C at *idle* (38W). Cooling is
compromised (hot room / heat-soaked case / possibly degraded paste). An overnight train on a
throttling card is slow at best and a crash risk at worst (this card has known load
instability — the `nvidia-smi -pl 400` rule).

1. Physical check: room temp, case intakes, dust filters.
2. `nvidia-smi -q -d PERFORMANCE` → confirm `SW Thermal Slowdown : Not Active` at idle and
   idle temp is sane (≤ ~45°C at ~38W). If it still idles above 55°C, fix cooling before training.
3. Set the power limit for the session (admin): `nvidia-smi -pl 400` (or 350 if temps are
   still marginal — training tolerates it fine).
4. No competing GPU work: no BookForge TTS job, no Ollama models loaded, no RVC.
   `session.py train` runs its own preflight (`train_preflight`, requires ≥16000 MiB free,
   dies loudly) — but it can't check temperature, so do #2 yourself.
5. During the first 10 minutes of training, watch
   `nvidia-smi --query-gpu=temperature.gpu,clocks.sm,power.draw --format=csv -l 30` —
   if SM clocks sag below ~1500 MHz sustained, stop and fix cooling; don't run the night throttled.

## Step 1 — Ingest pt05-08

Expected to land in: `E:\training\deathstalker\source\celebration of discipline\adobe cleaned\`
Tonight pt01-04 are there (`discipline_pt0N-esv2-100p-bg-m-music-m.wav`); raw pt05-08 chunks are
already staged in `..\Celebration_chunks_for_adobe\` (pt08 is short, ~185 MB — that's real, it's
the book's tail, not a truncated file).

1. Confirm the four new cleaned files exist and follow the SAME naming pattern as pt01-04.
2. Sanity-listen to ~30s of one new part (same cleaning chain, no music/bg residue).

## Step 2 — Extend the session config and rebuild the dataset

Authoritative procedure: **`README.md` lines ~116-135** ("Adding pt05-08 later" under the CoD
worked example). Follow it, not memory. Outline:

1. Edit `sessions\deathstalker_cod.json`: append a second `part_sets[]` entry for pt05-08
   (mirror the pt01-04 entry: mix `2:3-10,3:10-20`, `max_clip 20`, −20 LUFS).
2. Re-run the session pipeline for the new part set: merge → transcript/align → coverage
   verify → cut. (pt01-04 artifacts are already done and are NOT redone — the union is staged.)
3. Hard rules the pipeline already enforces, verify they held in the new CUT_REPORT:
   - clip cap **≤ 20s** (`session.py` hard-blocks >20 — PROVEN-BROKEN regime)
   - trailing pause trim **~0.1s** (`--trail-cap 0.1`) — uncapped tails train runaway silence
   - 24000 Hz mono, −20 LUFS
4. Resolve the pt01-04 bookkeeping discrepancy while you're here: **993 wavs on disk vs
   985 metadata rows vs 981 ingested by the last build**. Run the `audit` subcommand; identify
   the ~8 orphan wavs and the 4 build-dropped rows. Don't let unexplained rows ride into the
   combined train.
5. **Intra-clip pause audit — RESOLVED 2026-07-14, verdict: DATA IS FINE, DO NOTHING.**
   The ep5 voice's occasional mid-phrase pause ("after ... all", reproduces on both
   backends) traces to the source material's style, not defects: the audit
   (`audit_intra_clip_pauses.py`, 63% of clips over a strict sentence-terminator pause
   budget) flagged clips that Owen EAR-CHECKED AS NATURAL — Rohan's contemplative CoD
   delivery genuinely pauses mid-sentence, and it sounds good. Do NOT normalize these
   pauses away (would flatten the pause quality that makes the voice). The quirk is
   style transfer (meditative pacing on action prose) — accept as voice character. If
   it ever needs fixing: add PACING DIVERSITY (Rohan reading action prose — the
   Deathstalker novels track), not data surgery. Audit-script caveat for reuse: it
   doesn't discount structural boundaries (subheading gaps inside clips legitimately
   pause long — e.g. clip 91).
5. Expected combined size: ~985 + (pt05-08 yield) ≈ 1800-2000 clips / ~6-7 hrs. Eval split
   comes from the pipeline as before.

## Step 3 — Stage to WSL and build

Per the same README procedure: stage the union to **`/home/telltale/xtts_ft/deathstalker_cod`**
and run `build` (re-encodes SNAC, re-measures tok/s — last run measured 82.4 tok/s, budget
1792 @ 2048−256, longest-safe ≈ 21.8s; expect similar). Confirm `kept N, dropped 0`.

## Step 4 — Train (overnight)

Launch: `python session.py --config sessions/deathstalker_cod.json train`

Flags to match the last run's recipe (check README law table first):
`--mask-prompt-loss --no-dedup --lr-schedule constant_with_warmup`
plus **`--no-early-stop`** — this is deliberate and important:

> **PROVEN (memory: pause-consolidation is a LATE epoch):** Orpheus learns inter-sentence
> pauses ~1 epoch AFTER eval_loss bottoms. Early-stopping on eval_loss ships a
> pause-squishing checkpoint. We keep EVERY epoch checkpoint and pick the keeper in Step 5.

Do NOT pass `--merge` on the train (we don't know the keeper epoch yet; merge it in Step 6).
Outputs: LoRA epochs → `/home/telltale/xtts_ft/orpheus_deathstalker_lora`.
`max_seq_length` stays at the cfg default **2048** — do not override (the 4096 experiment is
the proven-broken one; `orpheus_owen.py` has it reverted with a "Do NOT raise again" comment).

## Step 5 — Checkpoint selection: quantified EOS test + pause ear-check

No runaway smoke-test script exists yet — build the measurement from the BookForge CLI
(`cli\bookforge-tts.py`, run-validated) and the worker log, since the guards print countable lines.

For each candidate checkpoint — **at minimum: eval-best, +1 epoch, +2 epochs** — and also
the pt01-04 A/B that answers the open question. **The pt01-04 candidates are ALREADY MERGED
on disk** (verified 2026-07-13 late): `/home/telltale/xtts_ft/orpheus_deathstalker_merged`
(= eval-best checkpoint-418), `…_merged_ep3`, `…_merged_ep4`, `…_merged_ep5` (= deployed).
LoRA epochs 1-4 also survive as `orpheus_deathstalker_lora/checkpoint-{209,418,627,836}`
(~209 steps/epoch), ep5's LoRA in `deathstalker_overtrain/`. Run all four pt01-04 merged
models through the smoke test first — four points give the SHAPE of EOS degradation vs
epoch before the combined train even starts.

1. For combined-train candidates: merge to 16-bit, deploy to a scratch name (or point
   `--orpheus_model_dir` at it — don't clobber `/home/telltale/orpheus-models/deathstalker`
   until the keeper is chosen). pt01-04 candidates: use the existing merged dirs above.
2. Render the SAME ~100-chunk sample at default 350-char packing via
   `bookforge-tts.py --tts --engine=orpheus` (the `audition` subcommand in session.py wraps this).
3. Count in the worker log:
   - `hit the audio-token cap` → **runaway rate. Keeper target: 0, tolerance ≤1/100.**
     (Today's ep5 baseline: 10/90. deathstalker_v3: 0/126.)
   - `too short for text` trips and any `recalibrating threshold` line → note the voice's
     measured natural ch/s (the new ratchet prints it — free calibration data; feed it into
     the per-voice `maxCharsPerSec` if you set one).
4. Ear-check the survivors for **pause reproduction** (the late-epoch property) and prosody.
5. **Decision rule: keeper = latest epoch that holds 0-runaway AND full pauses.** eval_loss is
   a tiebreaker only. If EVERY combined-dataset epoch runs away: the dataset/narrator is the
   cause → stopgap = per-voice `maxChars: 300` (feat/orpheus-voice-caps) and investigate the
   dataset (rep-penalty findings in `SAMPLING_AND_VOICE_QUALITY_FINDINGS.md` — `--no-dedup`
   models need rep 1.1; verify inference uses it).

## Step 6 — Merge keeper + deploy

1. Merge the keeper epoch to 16-bit (orpheus_owen.py `--merge` path / README).
2. `bash deploy_voice.sh deathstalker "Deathstalker (Richard Rohan)" <merged_dir>` (in WSL) —
   rsyncs into `/home/telltale/orpheus-models/deathstalker`, upserts `models.json`, uploads to
   HF `owenmorgan/deathstalker-orpheus-3b`, prints the Mac-pull prompt.
3. Deploy is a human step by design — session.py will not do it for you.

## Step 7 — Acceptance: the Hellworld re-run

1. Start the Hellworld TTS job **FRESH** (never resume the 2aa4bb62 session: new model, and
   packing-affecting changes invalidate old sessions anyway).
2. Watch the first ~150 chunks in `%APPDATA%\bookforge\logs\worker-output.log`:
   - token-cap hits: expect ~0 (vs 10/90 today)
   - short-trips: expect ≤1 then a single `recalibrating` line, then silence from the guard
   - throughput: back at ~90 sent/min — **but only if Step 0's thermal issue is fixed;**
     a healthy model on a throttled card will still crawl.
3. If acceptance passes, this model supersedes ep5; note the keeper epoch + eval_loss in
   README's law table for the record.

---

## Quick reference (verified paths, 2026-07-13)

| What | Where |
|---|---|
| Session config | `C:\Users\tellt\Projects\orpheus-finetune\sessions\deathstalker_cod.json` |
| Runbook w/ pt05-08 procedure | `C:\Users\tellt\Projects\orpheus-finetune\README.md` (~116-135) |
| pt01-04 cut dataset | `E:\training\deathstalker\source\celebration of discipline\clips\deathstalker\pt01-04\` |
| Adobe-cleaned parts | `E:\Shared\...\celebration of discipline\adobe cleaned\` (pt01-04 present; pt05-08 land here) |
| Raw parts staged for Adobe | `E:\Shared\...\celebration of discipline\Celebration_chunks_for_adobe\` (pt01-08 present) |
| WSL dataset dir | `/home/telltale/xtts_ft/deathstalker_cod` |
| LoRA checkpoints | `/home/telltale/xtts_ft/orpheus_deathstalker_lora` (pt01-04 best = checkpoint-418) |
| Merged models | `/home/telltale/xtts_ft/orpheus_deathstalker_merged` (+ `_ep5` = deployed today) |
| Deploy script | `C:\Users\tellt\Projects\orpheus-finetune\deploy_voice.sh` |
| Live voice dir (WSL) | `/home/telltale/orpheus-models/deathstalker` |
| Worker log (guard lines) | `C:\Users\tellt\AppData\Roaming\bookforge\logs\worker-output.log` |
| Ratchet branch (e2a) | `feat/orpheus-rate-ratchet` @ 75d4ecf — WSL copy already synced |
| Per-voice caps branch (BookForge) | `feat/orpheus-voice-caps` |
