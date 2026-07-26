# Cloning a voice into Orpheus-3B — the BookForge method

**Proven on Owen Morgan (2026-06-28): the result was indistinguishable from the real
voice.** This is the repeatable recipe. The single source of truth for every step is
[`orpheus_owen.py`](orpheus_owen.py) (its module docstring holds the full spec); this
README is the runbook that ties it together.

> TL;DR: clean, book-aligned, breath-safe dataset → SNAC-tokenize → Unsloth LoRA on
> `orpheus-3b-0.1-ft` with **held-out eval + early-stopping** (keep the best epoch) →
> merge to 16-bit. ~1 hour of training got a flawless clone. **Curation, not data
> volume, is what makes it flawless.**

**New here? Start with [SETUP.md](SETUP.md)** — environment, the `.env` roots every script
reads, and a five-step smoke test before you commit to a long run.

> **About the paths in this document.** Concrete paths like `/home/telltale/xtts_ft/...`
> and `E:\training\...` are the author's machine, kept because a worked example with real
> values is easier to follow than one full of `<placeholders>`. They are illustrations, not
> defaults — the scripts read their roots from `.env` (see `.env.example`) and will tell you
> exactly which variable to set if it is missing. The shell scripts under `pipeline/` still
> carry hardcoded paths and will need editing before they run anywhere else.
>
> **No training data ships with this repo** — no audio, no datasets, no model weights, and
> no book text. `probe_chunks.json` is a length-matched fixture drawn from public-domain
> works purely to exercise the generation path.

---

## Result & cost (Owen run)

- **Quality:** flawless / indistinguishable, on a 42s clip *and* a 3-min/500-word render
  of novel text (generalizes, not memorizes).
- **Data:** 7.0 h / 1,516 clips (24 kHz) — **overkill** (see "How little data" below).
- **Training:** epoch 3 was best (eval_loss 3.6348); early-stopped at epoch 5. Training
  loop 1h51m, but the keeper epoch was ~1h06m → **~1h16m to a usable model**, ~2h total
  including SNAC-encode + merge.
- **Hardware:** RTX 3090 Ti (24 GB), WSL2 Ubuntu. 16-bit LoRA (not 4-bit); r=64.

---

## THE proven recipe — settings, values, and WHY (locked 2026-07-12)

Every value below is either A/B-proven or the survivor of a failed experiment. The
reference run is **deathstalker_v3** (full-book data, native token, eval_loss 3.784,
0/30 runaway caps on production-shaped chunks). Do not change these casually — the
"why" column is the scar tissue.

| Setting | Value | Why (evidence) |
|---|---|---|
| **Clip length cap** | **≤ 20 s** (`cut_audiobook.py --max-clip 20`, `MAX_CLIP_SECONDS = 20`) | **The big one.** Training on 38s clips broke EOS: the model learns "audio can run 30s+ before stopping" and stalls into silence at internal sentence boundaries at inference. rohan-v2 (38s/4096): **19% runaway caps**; every ≤20s/2048 voice: **0/126**; same data recut ≤20s: **~0**. |
| **max_seq_length** | **2048** (`TrainingConfig`; `--max-seq-length` exists for experiments only) | Pairs with the 20s cap (20s ≈ 1650 audio tokens + text fits 2048). The 4096 raise existed only to fit 38s clips — see above for how that ended. |
| **Trailing pause** | **trim to ~0.1 s** (`--trail-cap 0.1`) | Uncapped natural trailing pauses teach "silence can continue" → runaway dead air with no EOS. Controlled test: trimmed = 0/29 single-sentence runaways; uncapped = 3/29. Structural ends capped 0.5s. |
| **Internal pauses** | **keep verbatim ≤ 2 s** (`--max-pause 2.0`; longer = structural, trimmed) | Natural mid-clip pauses are what the model reproduces as prosody. Dedup used to compress them → 5s-pause bug in the other direction. |
| **Dedup** | **OFF** (`--no-dedup`) | Keeps duplicate SNAC frames so real pause durations survive in the targets. Was suspected in the runaway saga and **exonerated by A/B** — do not re-enable to "fix" silence issues. |
| **Prompt loss** | **masked** (`--mask-prompt-loss`) | Loss on audio tokens only. NOTE: makes eval_loss audio-only — not comparable with unmasked runs. |
| **LR schedule** | `constant_with_warmup` (`--lr-schedule`) | Early-stopping on eval_loss interacts badly with a decaying schedule (can't tell "converged" from "LR died"). |
| **Data volume** | **use the whole book** (`--target-minutes 0`) | 1,488 clips (v3) beat 540 clips (v2) on eval_loss 3.784 vs 3.836 with zero safety cost. More clips, never longer clips. |
| **Duration mix** | `--mix "2:3-10,3:10-20"` | EOS must be learned at every clip scale under the cap; weighted toward the 10-20s band. |
| **Epochs** | cap 10, **early-stop patience 2 on eval_loss** | Best epoch is consistently **3** (v2, short, v3 all peaked there); eval rises by 4-5. `load_best_model_at_end` ships epoch 3 automatically. |
| **LoRA / precision** | r=64, alpha=64, 16-bit (`load_in_4bit=False`), lr 2e-4, batch 1 × grad-accum 4 | The Owen-run recipe; unchanged through every experiment. |
| **Prompt token** | **name the voice what it will be served as** (`--source-name deathstalker`) | The token is baked into training (`token: text`). A mismatch (trained `rohan`, served `deathstalker`) forces permanent overrides in models.json / HF card / every install. |
| **Sample rate** | 24 kHz mono (SNAC) | Codec requirement. |

**Coupled inference-side settings** (e2a `core.py` / `orpheus.py`, current as of
2026-07-13): packing cap **350 chars, NO sentence cap** (`ORPHEUS_MAX_CHARS` env
overrides), sentence gap **0.6s** deterministic, temp 0.6 / top_p 0.8 / **rep 1.1
(REQUIRED for --no-dedup models** — 1.0 measured 21.7s runaway silence). The earlier
200-char/2-sentence cap was a mitigation for the packed-runaway later PROVEN to be the
long-clip training recipe (old-deathstalker: 0/126 on the very chunks that "broke" it);
350/uncapped is ear-validated on EOS-safe ≤20s/2048 voices. **450 fails everywhere.**

---

## Session automation (`session.py`)

The recipe above is done by hand every time. `session.py` is a single driver that runs a
whole training **session** — one voice whose data **accumulates** over time — from a
per-voice JSON config, with the hard rules baked in (NO FALLBACKS, idempotent/resumable,
verification is a gate). It orchestrates the proven single-file tools; it does not
reimplement them.

A **part-set** is one batch of contiguous source chunks (today `pt01-04`; next week you add
`pt05-08` as a second part-set). Each part-set is merged, aligned, verified and cut on its
own; `stage` then unions every part-set's clips into ONE WSL dataset whose combined CSVs
train together. To extend a session, add another object to `part_sets[]` and re-run the
stages — completed part-sets report "already done".

### The flow

```
config ─▶ merge ─▶ align ─▶ verify ─(GATE)▶ cut ─▶ stage ─▶ train
                                                                └─▶ (human) ear-check + deploy_voice.sh
```

| subcommand | what it does | drives |
|---|---|---|
| `status` | table of which stages are complete per part-set + overall (WSL-aware) | — |
| `merge` | ffmpeg concat (`-f concat -safe 0 -c copy`) the N chunks → merged wav; **verifies merged duration == Σ parts** to the ms or fails | ffmpeg/ffprobe |
| `align` | epub-as-truth sentence VTT + coverage JSON | `bookforge-tts --generate-sentences … --report` |
| `verify` | **drift GATE**: coverage `driftSelfCheck` + dense independent whisper probe + waveform near-silence spot-check → persists `<vtt>.verify.json`. Fails if independent `MAXOFF > 1.5s` (warns at 0.75s) | `verify_vtt.py` (whisperx-env python) |
| `cut` | **refuses unless verify passed.** Measures merged LUFS → **static** gain to −20 LUFS (never per-clip loudnorm), cuts mixed 3–20s clips (`--trail-cap 0.1`), validates | `cut_audiobook.py` + `validate_dataset.py` |
| `stage` | copies each part-set clip dir onto WSL ext4 (copy runs *inside* WSL) + writes COMBINED `metadata_train/eval.csv` referencing `<partset>/wavs/…` | `wsl cp` |
| `train` | **preflight-gated** launch (see below): `orpheus_owen.py build` + `train` on the combined dataset with the v4 recipe, cwd OUTSIDE any git repo; tees the log to Windows; prints the per-epoch eval_loss table + best/merged paths | `orpheus_owen.py` (env `orpheus_train`) |
| `audition` | prints (or `--run`s) the BookForge A/B render for the trained voice | `bookforge-tts --tts` |

**`train` preflight (all fail loudly, no fallbacks):** refuses to launch if (a) `nvidia-smi`
shows a foreign GPU consumer or less than `min_free_vram_mib` free, (b) another
python training/align job is running (Win32 + WSL `pgrep`), (c) any included part-set's
verify gate has not passed, or (d) the WSL dataset is missing or **stale** vs the Windows
clips (per-part-set wav-count check). Deploy is intentionally NOT wrapped — `deploy_voice.sh`
exists and the ear-check is a human step.

> **Loudness deviation (deliberate):** the flow uses **static per-source gain** to −20 LUFS
> (measure the merged wav's integrated loudness once, pass `--gain-db` to `cut_audiobook.py`),
> NOT per-clip loudnorm. This honors THE recipe's rule #4 — single-pass loudnorm on clips
> bakes ±0.5 dB amplitude wobble ("wavy" voice) into the data. Same −20 LUFS target, no wobble.

### Config schema (`sessions/<name>.json`)

Top level: `voice`, `token` (prompt token; defaults to `voice`), `epub`, `mix`, `max_clip`
(rejected if >20), `target_lufs`, `target_minutes` (0 = whole book), `clips_root`,
`wsl_dataset_dir` (the combined dataset), `out_base`, `log_dir`, tool paths
(`bookforge_repo`, `windows_python` = e2a python_env for cutting, `whisperx_python` =
faster-whisper for verify, `align_python`), WSL settings (`wsl_distro`, `wsl_conda_prefix`,
`wsl_train_env`), an optional `train` block (`min_free_vram_mib`, `gpu_consumer_mib`), and
`part_sets[]`. Each part-set: `name`, `chunks_glob`, `merged_wav`, `vtt` (+ optional
`clip_dir`). Derived per part-set: `<vtt>.coverage.json`, `<vtt>.verify.json`, clip dir
`<clips_root>/<voice>/<name>`, WSL subdir `<wsl_dataset_dir>/<name>`.

### Worked example — Celebration of Discipline → voice `deathstalker`

Config: [`sessions/deathstalker_cod.json`](sessions/deathstalker_cod.json). Part-set
`pt01-04` merges `…/adobe cleaned/discipline_pt0[1-4]-*.wav` (4 chunks, Σ 14166.222 s)
against the Foster epub, mix `2:3-10,3:10-20`, into `/home/telltale/xtts_ft/deathstalker_cod`.

```bash
python session.py --config sessions/deathstalker_cod.json status   # see the board
python session.py --config sessions/deathstalker_cod.json merge     # detects the existing merged wav (Σ==merged) as done
python session.py --config sessions/deathstalker_cod.json align     # bookforge-tts epub-align (writes …_merged.vtt[.coverage.json])
python session.py --config sessions/deathstalker_cod.json verify    # GATE: drift check → …_merged.vtt.verify.json
python session.py --config sessions/deathstalker_cod.json cut        # blocked unless verify passed
python session.py --config sessions/deathstalker_cod.json stage      # → WSL /home/telltale/xtts_ft/deathstalker_cod
python session.py --config sessions/deathstalker_cod.json train      # preflight → build + train (v4 recipe), log tee'd to log_dir
# then, by ear + hand:
bash deploy_voice.sh deathstalker "Deathstalker (Celebration of Discipline)"
```

Adding pt05-08 later: append a second `part_sets[]` entry, re-run `merge…cut` for it
(the pt01-04 stages stay "already done"), then `stage` + `train` retrain on the union.

---

## The pipeline (commands)

All data steps run in WSL. The clip-cutting/build/train/speak steps are subcommands of
`orpheus_owen.py`. **argparse quirk:** global flags go BEFORE the subcommand
(`orpheus_owen.py --merge train`, not `train --merge`).

### 0. Environments (WSL conda)

- **`orpheus_ft`** (data prep): `torch==2.5.1+cu121`, `snac`, `librosa`, `soundfile`,
  `datasets`, `numpy==1.26.4`. Used for `audit`/`measure`/`recut`/`build`.
- **`orpheus_train`** (training+inference): created via Unsloth's installer —
  ```bash
  conda create -n orpheus_train python=3.11 -y
  pip install unsloth
  pip install transformers==4.56.2
  pip install --no-deps trl==0.22.2
  pip install snac "datasets>=3.4.1,<4.0.0" librosa soundfile
  ```
  (Unsloth pins transformers 4.56.2 / datasets<4 / trl 0.22.2 — keep it separate from
  `orpheus_ft`.) Pulls torch 2.10+cu128 + xformers + bitsandbytes.
- **`xttstrainer`** (alignment): has `faster_whisper`, `bs4`, `pandas`.
- **WSL gotcha:** run conda via `bash -c` with an explicit clean Linux PATH
  (`export PATH=/home/<user>/anaconda3/bin:/usr/bin:/bin`). The inherited Windows PATH
  has a `C:\Program Files` entry that breaks conda. And **shell variables get eaten**
  inside `wsl.exe bash -c '...$var...'` — use literal paths.

### 1. Build the dataset (the part that actually matters)

> **DEFINITIVE DATASETS (rebuilt 2026-07-07).** All voices re-curated to one standard —
> book-corrected transcripts + full natural pauses + long chunks, zero chapter gaps:
> | voice | dataset | book (narrator) | size |
> |---|---|---|---|
> | owen | `owen_v2/raw` | *God's People* (Owen Morgan) | 389 clips / 1.74 h |
> | mistborn | `mistborn_v2/raw` | *A Prince's Errand* — Tales of the Amulet #1 (Michael Kramer) | 406 / 1.80 h |
> | thirdreich | `thirdreich_v2/raw` | *The Coming of the Third Reich*, Evans (Sean Pratt) | 460 / 1.97 h |
> | deathstalker | `deathstalker_new_orpheus_raw2` | Deathstalker (Richard Rohan, YouTube) | 315 / 1.37 h |
>
> Two lessons baked in: (1) **the "model drops pauses in weird spots" bug was the training
> `_dedup_frames` flag collapsing pause frames — NOT the clip pauses.** Fixed with `--no-dedup`;
> `_normalize_pauses` now keeps every natural pause ≤2 s **verbatim** and only trims structural
> chapter/scene gaps (>2 s). (2) For a single audio segment (not per-chapter), use
> **`align_from_epub.py`** (below): book-as-truth difflib alignment that **auto-anchors** the
> Whisper stream into the book (handles front matter + mid-book starts like a "part 2") and
> **drops Whisper hallucination loops** — this is what fixed Mistborn's garbled transcript.

Source = chapter audio + the book's text (for an audiobook). Reusable scripts live in
`xtts-finetune/scripts/` (+ `align_from_epub.py` in this repo for single-file segments):

> **Source audio from mixed recordings?** `eq-tools/` has deterministic LTAS spectral
> matching (speech-gated, ffmpeg firequalizer) to EQ-match all sources to one reference —
> the no-neural-enhancement alternative to Adobe/resemble. See `eq-tools/README.md`.

```bash
# epub text per chapter  (env: xttstrainer, bs4)
python extract_epub_text.py
# e2a number/date/roman normalization -> spoken form (Windows e2a python_env, stanza)
python_env\python.exe normalize_metadata.py booktext
# Whisper word timestamps  (env: xttstrainer; set LD_LIBRARY_PATH to the env's
#   nvidia/{cudnn,cublas}/lib for faster-whisper)
python batch_transcribe.py
# align book-truth text to Whisper timing -> *.aligned.json  (difflib)
python batch_align.py                  # per-chapter (many files, names match audio stems)
# OR for a single continuous audio segment (whole-book epub, auto-anchored window):
#   align_from_epub.py <epub> <seg.whisper.json> <seg.aligned.json> -1
#   (-1 = auto-anchor: finds where this segment starts in the book; drops Whisper loops)
# (optional QA) fix regnal/roman names e2a mangles ("Pius XII"->"Pius the Twelfth")
python orpheus-finetune/fix_regnal_names.py --apply
# cut LONGER, pause-preserving clips at 24 kHz from the original audio
python orpheus_owen.py recut          # target 15s, cap 20s, keeps internal pauses
python orpheus_owen.py measure        # confirm real SNAC tok/s fits max_seq_length
python orpheus_owen.py --dataset-dir <recut_dir> build   # -> manifest dataset
```

**Curation rules that make or break it** (see also the `curate-tts-dataset` skill):
- Transcripts must match what was **actually spoken** (book text > raw Whisper; fix
  number/date/name normalization to spoken form; verify audio==text version — we caught
  a wrong/older chapter cut via a Whisper-words / book-words ratio check).
- Cuts: **breath-safe** (never split a word/breath), at silence troughs, **keep internal
  pauses** (Orpheus models prosody — do NOT compress them like you would for XTTS), trim
  only the outer edges to a consistent ~0.15s.
- **CUT THE TRAILING SILENCE off every clip (trim tails to ~0.1s). This is not optional.**
  Orpheus learns *when to stop* (the end-of-speech token) directly from where your clips
  end. If clips end with full, open-ended natural pauses, the model learns "silence can
  keep going" and on ~10% of sentences it FAILS to emit end-of-speech — it speaks the
  sentence correctly, then generates runaway silence until the token cap (~35s of dead
  air per hit, wasting GPU and tripping re-render guards). PROVEN empirically (2026-07-11):
  same 29 sentences, same sampling — **owen (tails trimmed to ~0.1s) = 0 runaways; rohan
  (uncapped natural tails) = 3/29 (10%)**. The ONLY difference was the trailing pauses;
  `--no-dedup` was used by both and is fine. The end-of-sentence pause you want at
  playback is added DETERMINISTICALLY at generation (e2a `_classify_gap` → a fixed 0.75s
  `sentence_gap`), so it must NOT be baked into the training data as open-ended silence.
  `cut_audiobook.py --trail-cap 0.1` enforces this. See also `_save_audio`'s removed
  trailing-trim fallback — trimming at *save* time only masked this bug; the fix is at the
  *data* level.
- Clip length: 24 kHz mono; ~85 SNAC tok/s ⇒ ≤~21s fits `max_seq_length=2048`. Longer
  (~15s) pause-preserving clips read more naturally than sentence fragments.

### 2. Train

```bash
orpheus_owen.py --merge train      # eval+early-stopping, keep best epoch, merge to 16bit
```
- Base `unsloth/orpheus-3b-0.1-ft`, LoRA r=64/α=64, lr 2e-4, adamw_8bit, batch1×grad4.
- **Eval each epoch on the 15% held-out split; EarlyStopping(patience=2) on eval_loss;
  `load_best_model_at_end` → final model is the BEST epoch, not the last.** This removes
  epoch-count guessing and prevents the overfit you'd get from too many epochs.
- Outputs: best LoRA `…/orpheus_owen_lora` (+ per-epoch checkpoints for A/B), merged
  16-bit `…/orpheus_owen_merged`.

**Recommended flags for NEW training runs (added 2026-07-07; EAR-VALIDATED on deathstalker
v4, which shipped with all three — "very good quality"):**

```bash
orpheus_owen.py --source-name <voice> --mask-prompt-loss --no-dedup \
    --lr-schedule constant_with_warmup --merge train
```

- `--mask-prompt-loss` — label the text prompt `-100` so ALL gradient goes to
  p(audio | text) instead of re-learning to predict the transcript; eval_loss becomes a
  pure-audio model-selection metric. **eval_loss is NOT comparable to unmasked runs.**
- `--no-dedup` — keep duplicate SNAC frames. The notebook-verbatim `_dedup_frames`
  collapses repeated coarse codes, which is mostly SILENCE (~0.086 s/frame), so dedup
  time-compresses natural pauses in the training targets (a 0.5 s pause → ~0.1–0.2 s)
  and the model learns rushed, crammed pacing. Keeping duplicates preserves real pause
  durations. Token budget still fits: 20 s max clip × 82.5 tok/s + 256 reserve < 2048.
- `--lr-schedule constant_with_warmup` — the default linear decay is computed over the
  10-epoch CAP, but early stopping lands ~epoch 3–5, so the best epoch trains at ~75%
  peak LR and never anneals. Constant+warmup decouples the schedule from early stopping.

**Hard-learned rules (each of these bit us):**

1. **The prompt token comes from `--source-name`** — since 2026-07-07. Before that,
   `_encode_clips` silently used the CSV `speaker_name` column, so a token rename on an
   existing recut dir trained the OLD token (the "rohan" model was actually trained on
   `deathstalker:` prompts). The script now prints a NOTE when CSV ≠ `--source-name`.
2. **Inference must feed token IDs, not a decoded string.** e2a decoded the prompt IDs
   back to a string and let vLLM re-tokenize it, which prepended a stray second BOS —
   an out-of-distribution prompt that made models SPEAK their own voice token at
   sentence starts. Fixed in e2a `orpheus.py` (`TokensPrompt(prompt_token_ids=…)`,
   e2a commit 96fee353). If a voice ever vocalizes its token name, check the framing
   FIRST — it is not a token/content-word collision.
3. **Audition with production sampling** (temp 0.6 / top_p 0.8 / rep 1.1 / 0.65 s gap —
   the e2a defaults). `speak` defaults now match; overriding them means you're
   auditioning a model you won't ship.
4. **Normalize source loudness with STATIC gain** (ffmpeg `volume=<dB>` to −20 LUFS per
   source), never single-pass loudnorm on clips — its dynamic gain bakes ±0.5 dB
   amplitude wobble into the training data ("wavy" voice).
5. **`repetition_penalty=1.1` is REQUIRED with `--no-dedup` models** — measured on
   deathstalker v4: identical text rendered 53.9 s at rep 1.1 (max pause 1.3 s, natural)
   vs 95.0 s at rep 1.0 (max pause **21.7 s** — silence-frame generation runs away with
   nothing to bound it). The penalty is the stabilizer, not an enemy of natural pauses.

### Generation / inference notes (fresh-eyes review, 2026-07-07)

Production sampling = **temp 0.6 / top_p 0.8 / rep 1.1 / 0.65 s inter-sentence gap** —
these are the e2a env defaults (`ORPHEUS_TEMPERATURE`/`TOP_P`/`REP_PENALTY`) and `speak`'s
defaults; audition with exactly these.

- **Multi-sentence chunks — DONE & SHIPPED (e2a f423fe9a, ear-validated on v4 2026-07-07).**
  `lib/core.py` now packs 2–3 sentences per Orpheus generation up to ~450 chars (1.8× the
  base, sized to `MAX_AUDIO_TOKENS`=3700). The old `core.py` comment blamed ~450-char packing
  for "glitched internal sentence transitions — stray syllable/gibberish at sentence starts,"
  but that experiment ran with the stray-BOS corrupted framing (the exact artifact, fixed in
  e2a 96fee353). With the framing correct, packing is clearly BETTER by ear: a short passage
  went 10 chunks → 2, ~9 s shorter, and the model's own inter-sentence prosody replaces the
  hard inserted gaps. **450 was NOT over-length** (the earlier ≤350 guess was wrong — the
  fine-tune handles it fine). Packing is boundary-aware: sentences carrying an SML token
  ([break]/[pause]) are never merged, so paragraph pauses survive. Audition packing with
  `orpheus_owen.py speak --pack --max-chars 450`. Cost: coarser VTT cues + resume granularity
  (acceptable). Training clips are 8–20 s multi-sentence spans, so single sentences were the
  SHORT edge of the distribution anyway — multi-sentence is the model's home turf.

Worth trying next time generation quality is on the table (verified available in
vLLM 0.7.3, not implemented yet):

- **`allowed_token_ids` = SNAC code range [128266, 128266+7·4096) + EOS 128258** — makes
  it impossible to sample text tokens; kills the entire token-vocalization bug class at
  the sampler.
- **`min_tokens=7`** — bans instant-EOS empty generations.
- **`MAX_AUDIO_TOKENS` 3700 → ~1900, `max_model_len` 4096 → 2048** — the fine-tune never
  saw >2048 positions; a normal sentence needs ≤~1500 tokens, so the big caps only let
  runaway loops burn 2× longer before the safe-split catches them, and halving
  max_model_len halves KV cache per slot (more batch headroom). `ORPHEUS_MAX_TOKENS` env
  already overrides.

### 3. Audition

> **PREFERRED RENDER PATH (2026-07-11): the BookForge TTS CLI, not `speak`.** Auditions
> and any real render now go through `bookforge-tts` (in the BookForge repo,
> `cli/bookforge-tts.py`), which drives BookForge's compiled Orpheus worker pool so it
> inherits the full guarded pipeline — WSL wedge-proofing (TERM→`wsl -t` kill ladder,
> never-SIGKILL a guest GPU proc), vLLM `gpu_memory_utilization` memory tiers +
> safe GPU sizing (auto-picks 'fast' etc.), and custom-model resolution. BookForge must be
> BUILT (`npm run build:electron`) but need NOT be running. Install the merged voice into
> the models dir first (see "Deploy" below), then:
> ```bash
> python cli/bookforge-tts.py --tts --engine=orpheus --voice=rohan \
>     --input passage.txt --out sample.wav [--tier fast]
> ```
> The `speak` command below is **debug-only** now: it's a standalone HF-transformers
> `.generate` with **no VRAM cap** (it OOM'd the whole desktop at batch 8) and none of the
> WSL guards. Keep it for poking at this finetune pipeline; do not use it to render.

```bash
orpheus_owen.py --text-file passage.txt --batch 8 --out sample.wav speak
```
Batched (left-padded) generation; reverses the 7-per-frame SNAC layout → `snac.decode`.
Prompt format is `owen: <text>` (the `source` name = the voice).

---

## How little data can you get away with? (for cloning OTHER voices)

The 7 h was overkill. Orpheus's base already speaks English; LoRA only imprints timbre,
which transfers fast. **Curation quality >> quantity.**

| Data (clean, varied) | Result |
|---|---|
| ~15–30 min | recognizable, often already good |
| **~1 hour** | reliable ship-quality sweet spot |
| 2–3+ h | diminishing returns (robustness on rare sounds, not identity) |

With less data, lean **harder** on eval+early-stopping (small sets overfit faster). The
three levers that beat "more hours": accurate transcripts, clean breath-safe cuts,
variety+consistency (same mic/room/voice).

To raise quality further (not via more epochs — epoch 3 already peaked): more/varied
data, try the `-pretrained` base instead of `-ft`, higher LoRA rank (r=128), LR tuning.

---

## Deploy / inference

- The merged 16-bit model is a standard Llama-arch model → load in **vLLM** (CUDA graphs)
  for fast generation. BookForge's Orpheus path already runs vLLM in WSL; point its
  model id at this model and use voice/source name `owen`.
- HF: `owenmorgan/owen-morgan-orpheus-3b` (see upload command in chat / `upload_to_hf.py`).
- Token layout (verbatim) and all constants live in `orpheus_owen.py`.

---

## Cloning ANOTHER voice — incl. voices with NO source text (e.g. YouTube narration)

`orpheus_owen.py` is now **voice-parameterized**: `--source-name`, `--src-chapters`,
`--out-base`, plus an LQ-fill cap (`--lq-prefix` / `--lq-cap-minutes`). Owen's defaults
are unchanged, so his run still reproduces with no flags. The Owen recipe assumes
book-aligned text; for narrations where we DON'T have the matching ebook, use
`transcribe_whisper.py` to make Whisper the text truth (the curate-tts-dataset skill's
ASR fallback). **The cut points are still silence-snapped by `recut` — never at Whisper's
word timestamps** (those are ±100–300 ms off and would halve words), so ASR-grade timing
is fine: it only labels clips, it never places cuts.

Worked example — the **Deathstalker** narrator (4 HQ YouTube readings ≈ 33 min + 1 muffled
2 h source as capped fill). All in WSL; use each env's `bin/python` directly (sidesteps the
conda-PATH gotcha). Stage trimmed sources into ext4 first, tagged `hq_`/`lq_` so the quality
provenance lands in every clip filename (lets a later retrain drop the muffled clips with a
filename glob, no re-curation).

```bash
cd /mnt/c/Users/tellt/Projects/orpheus-finetune
DS=/home/telltale/xtts_ft/deathstalker          # src/  align/  (staged wavs + whisper json)
XT=/home/telltale/anaconda3/envs/xttstrainer/lib/python3.10/site-packages

# 1. transcribe -> *.aligned.json   (env xttstrainer; faster-whisper needs its cuDNN/cuBLAS)
export LD_LIBRARY_PATH=$XT/nvidia/cudnn/lib:$XT/nvidia/cublas/lib:$LD_LIBRARY_PATH
/home/telltale/anaconda3/envs/xttstrainer/bin/python transcribe_whisper.py \
    --in-dir $DS/src --out-dir $DS/align

# 2. recut -> silence-snapped, pause-preserving 24k clips + metadata   (env orpheus_ft)
#    HQ cut in full; LQ ("lq_*") capped at 45 min of fill.
/home/telltale/anaconda3/envs/orpheus_ft/bin/python orpheus_owen.py \
    --source-name deathstalker --src-chapters $DS/src --align-dir $DS/align \
    --recut-dir /home/telltale/xtts_ft/deathstalker_orpheus_raw \
    --lq-prefix lq_ --lq-cap-minutes 45  recut

# 3. build manifest dataset + MEASURE real SNAC tok/s   (env orpheus_ft)
/home/telltale/anaconda3/envs/orpheus_ft/bin/python orpheus_owen.py \
    --source-name deathstalker \
    --dataset-dir /home/telltale/xtts_ft/deathstalker_orpheus_raw \
    --output-dir  /home/telltale/xtts_ft/deathstalker_orpheus_ds  build

# 4. TRAIN — DO NOT RUN until the dataset is eyeball/ear-checked   (env orpheus_train)
/home/telltale/anaconda3/envs/orpheus_train/bin/python orpheus_owen.py \
    --source-name deathstalker \
    --recut-dir /home/telltale/xtts_ft/deathstalker_orpheus_raw \
    --out-base  /home/telltale/xtts_ft  --merge train
#   -> best LoRA  /home/telltale/xtts_ft/orpheus_deathstalker_lora
#      merged 16b  /home/telltale/xtts_ft/orpheus_deathstalker_merged   (vLLM; voice "deathstalker")
```

Mistborn is identical with `--source-name mistborn` and its own `src/`/`align/` dirs
(no LQ source, so drop the `--lq-*` flags). Its source is a single 4 h read — past the
diminishing-returns point — so trim it to a representative ~1.5 h with
`--target-minutes 90`, which keeps an EVENLY-SPREAD clip subset (variety across the whole
book) and deletes the rest:

```bash
/home/telltale/anaconda3/envs/orpheus_ft/bin/python orpheus_owen.py \
    --source-name mistborn --src-chapters $MB/src --align-dir $MB/align \
    --recut-dir /home/telltale/xtts_ft/mistborn_orpheus_raw \
    --target-minutes 90  recut
```

---

## Deploy a trained voice into BookForge

A merged voice (`orpheus_<voice>_merged`) becomes usable in BookForge two ways, and we do
both: **install it locally** (works immediately on this machine) and **publish it to HF**
(makes it appear in BookForge's download catalog + pullable on the Mac). How BookForge finds
voices: `electron/orpheus-models.ts` (local manifest) + `electron/orpheus-hf-catalog.ts`
(HF catalog). Settings hold `orpheusModelsDir`, `orpheusHfUser`, `huggingFaceToken`.

### Quickstart — one command does all of it

`deploy_voice.sh` runs steps 1 (local) + 2 (HF) below and prints the step-3 Mac prompt.
Run it in WSL (env `orpheus_train`):

```bash
bash deploy_voice.sh mistborn   "Mistborn (Michael Kramer)"
bash deploy_voice.sh thirdreich "Third Reich (Sean Pratt)"
bash deploy_voice.sh owen       "Owen Morgan"
```

It rsyncs `<voice>_v2/orpheus_<voice>_merged` → the local models dir (upserting
`models.json`), uploads to `owenmorgan/<voice>-orpheus-3b` (private, tagged), then prints a
paste-ready prompt for Claude-on-Mac. Pass a third arg to override the merged-dir path for
old (`_v2`-less) layouts. The sections below are the manual equivalents, kept for reference.

> **HF token** (needed for step 2) is looked up in this order: `$HF_TOKEN` →
> `~/.cache/huggingface/token` → the Windows-side HF cache when running under WSL
> (`/mnt/c/Users/<you>/.cache/huggingface/token`). Never commit the token; `hf auth login`
> writes it to the cache for you, and `hf auth whoami` confirms which account you are on.

**Local models dir (this machine):** `\\wsl$\Ubuntu\home\telltale\orpheus-models`
(= WSL `/home/telltale/orpheus-models`), with a `models.json` manifest. Each voice lives in
`<dir>/<id>/` (config.json + *.safetensors). The manifest's **`token`** is the prompt token
(`<voice>: text`) — the one thing the filesystem can't tell BookForge, so it MUST be right.

### 1. Install locally

```bash
V=thirdreich                                  # the voice / source-name
rsync -a --exclude='.cache' --exclude='*.lock' \
    /home/telltale/xtts_ft/orpheus_${V}_merged/ /home/telltale/orpheus-models/$V/
# then add a models.json entry (preserve existing ones):
#   {"id":"<V>","label":"<Display>","token":"<V>","dir":"<V>","format":"hf",
#    "sampleRate":24000,"source":{"type":"hf","ref":"owenmorgan/<V>-orpheus-3b"}}
```
BookForge then lists it (manifest-first; valid folders missing from the manifest are also
auto-imported with the folder name guessed as the token).

### 2. Publish to HuggingFace (→ catalog + Mac)

`upload_to_hf.py` writes a model card tagged **`bookforge-orpheus-voice`** with flat
`orpheus_token`/`label`/`sample_rate` keys — that tag is what makes it show up in
**BookForge → Settings → Orpheus voices** under the `owenmorgan` account. Private by default
(real-person clones); the owner's BookForge lists/installs private repos with its token.

```bash
/home/telltale/anaconda3/envs/orpheus_train/bin/python upload_to_hf.py \
    --model-dir /home/telltale/orpheus-models/$V \
    --repo owenmorgan/${V}-orpheus-3b \
    --voice-token $V --label 'Third Reich (Sean Pratt)'
```

### 3. Pull on the Mac

Easiest: **BookForge → Settings → Orpheus voices → Install** (the tagged repo appears under
your account). Or hand Claude-on-Mac a prompt: download `owenmorgan/<V>-orpheus-3b` into the
Mac's `orpheusModelsDir`, upsert a `models.json` entry, and — since the Mac has no NVIDIA —
check whether the Mac Orpheus path needs an MLX conversion (`format:"mlx"`) vs HF safetensors.

> Owen=`owenmorgan/owen-morgan-orpheus-3b`, Deathstalker=`owenmorgan/deathstalker-orpheus-3b`,
> Third Reich=`owenmorgan/thirdreich-orpheus-3b` (all private, tagged).

