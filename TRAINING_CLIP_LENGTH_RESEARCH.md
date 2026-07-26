# Orpheus Fine-Tune Training Clip Length — Research Findings

_2026-07-11. Compiled from a web research sweep (official Canopy Labs / Unsloth / Axolotl
docs, GitHub issues, and TTS length-generalization papers). The sweep's search+fetch
phases completed; the final adversarial-verify pass did NOT run, so treat quotes as
fetched-but-single-sourced. Primary sources are linked throughout._

## Why this document exists

The Black Sun render (2026-07-11, deathstalker voice) exposed **silent early-EOS
truncation**: on long multi-sentence chunks (~400-460 chars ≈ 28-30s), the fine-tuned
voice emits `end_of_speech` (128258) early — cleanly, at a plausible sentence boundary,
with natural falling prosody — and silently drops the trailing text. Measured: healthy
narration ≈ 15.2 chars/sec; the 400-449 char bucket's p90 blew up to 21 ch/s with a tail
to 83 ch/s (= spoke ~1/5 of the text). Whisper-confirmed on multiple chunks.

**Root cause is in OUR training config.** `orpheus_owen.py` trains at
`max_seq_length = 2048` with `MAX_CLIP_SECONDS = 20.0`. The Unsloth/Canopy training row is

```
[SOH] text [EOH][SOA][SOS] snac_codes [EOS(128258)][EOA]   (labels = input_ids)
```

so **EOS placement is learned directly from the clip-duration distribution**. A dataset
whose every example ends by 20s teaches the model "utterances end by ~20s" — on a 30s
prompt it wraps up early at the first plausible boundary. This is not speculation; the
literature shows the same mechanism everywhere (see below).

Interim mitigation already shipped in e2a (`bookforge` 15ce3d41+b0fff267): packing
shrunk 450→~300 chars + a chars/sec truncation guard (`ORPHEUS_MAX_CHARS_PER_SEC`,
default 19.0) that force-splits and re-renders suspect chunks. **The retrain goal is to
push the EOS prior out far enough to restore ~450-char packing** (better prosody, ~35%
fewer generations).

## What the official sources say (and don't)

- **Canopy Labs gives NO clip-length guidance anywhere** — not the
  [repo README](https://github.com/canopyai/Orpheus-TTS), not the
  [model card](https://huggingface.co/canopylabs/orpheus-3b-0.1-ft), not the
  [Axolotl Orpheus docs](https://docs.axolotl.ai/docs/models/orpheus.html). Official
  guidance is dataset SIZE only: ~50 examples for first results, **~300/speaker for best
  quality**.
- **The 3B base model was trained on sequences of length 8192** ("We train the 3b model
  on sequences of length 8192") ≈ **90+ seconds of SNAC audio**. The backbone is NOT the
  ceiling — our fine-tune data and inference `max_model_len` are.
- The [Unsloth notebook/guide](https://docs.unsloth.ai/basics/text-to-speech-tts-fine-tuning)
  defaults `max_seq_length = 2048` ("Choose any for long context!") and applies **no
  duration filtering at all** — its reference dataset (Elise) averages **~9s/clip**. The
  default recipe is built around short clips; nobody documents the consequence.
- Community fine-tunes have run **`max_seq_length = 4096` successfully** (full fine-tune,
  [issue #10 thread](https://github.com/canopyai/Orpheus-TTS/issues/10)). Over-length
  samples at 2048 either hard-error or get truncated
  ([#271](https://github.com/canopyai/Orpheus-TTS/issues/271): "Input IDs of length 2334 >
  ... 2048", mid-run AssertionError) — **truncated targets lose their EOS entirely**, a
  second way short seq-lengths poison EOS calibration.
- Canopy's deepest fine-tune guidance is the
  [multilingual blog post](https://canopylabs.ai/releases/orpheus_can_speak_any_language#training)
  (per maintainer amuvarma13) — worth reading before the retrain.

## What the length-generalization literature says

- **Models cannot reliably generate past their training-clip duration.** VALL-E trained
  on ≤28s clips fails beyond 28s ([HALL-E](https://arxiv.org/pdf/2410.04380)); AR codec
  TTS develops "length attractors" that halt generation near training-length boundaries
  ([VoiceStar](https://arxiv.org/html/2505.19462)). Competing models' WER explodes 2-10x
  when generating past their trained range. **This is exactly our failure mode.**
- **The fix is duration-MIXED training data, not uniformly long clips.** HALL-E's
  MinutesSpeech deliberately spans 3-180s "with a balanced duration distribution";
  short-only (4-10s) training fails on long utterances.
- **The field's sweet spot for long-form/audiobook AR TTS training is ~15-30s.**
  [Context-aware memory paper](https://arxiv.org/html/2508.14713v1): clips capped <30s,
  mean 16.2s. VoiceStar caps training at 30s. Nobody trains AR codec TTS on 60s+ clips —
  past the token budget, longer raw clips are the wrong lever.
- **Long clips must be natural continuous utterances.** Concatenating short clips or
  slicing mid-utterance "introduces prosodic discontinuities and unnatural acoustic
  boundaries" (NeuTTS LoRA study, arXiv 2603.10904 class). Merge ADJACENT aligned
  sentences; never splice non-adjacent audio.
- **Early EOS is also partly exposure bias** (train-with-teacher-forcing vs
  inference-on-own-tokens; [arXiv 2509.17021](https://arxiv.org/html/2509.17021v1)) —
  error accumulates with length, so mitigations help long clips disproportionately. We
  can't fix that at the dataset level, which is another reason to keep the inference-side
  guard even after retraining.
- **Diversity beats homogeneity; loss is a bad checkpoint selector.** LoRA TTS fine-tunes
  on acoustically homogeneous data show limited gains or degradation, and
  "loss improves monotonically while DNS-MOS may degrade" — **pick checkpoints by ear**
  (we already learned this independently: ep15/ep400 keepers).

## Sampling/inference interactions (for completeness)

- `repetition_penalty >= 1.1` is REQUIRED for stable Orpheus generations (official), and
  raising rep_pen/temperature "makes the model speak faster". Rep-pen penalizes the whole
  SNAC stream, which relatively boosts the (never-repeated) EOS logit as generations get
  longer — a contributing pressure toward early EOS on long chunks that we cannot remove
  (stability floor). Don't raise rep_pen above 1.1 for audiobook renders.
- **vLLM `min_tokens`** ("minimum tokens before EOS or stop_token_ids can be generated")
  is the direct inference-side EOS suppressor, BUT vLLM has a documented history of
  min_tokens-vs-stop bugs (fixed for stop strings Aug 2025 in PR #22014; the
  stop_token_ids/EOS path tracked separately in issue #21950). **Our WSL env pins vLLM
  0.7.3 — if we ever wire min_tokens, TEST it actually suppresses 128258 in 0.7.3
  first.** We chose the duration-guard + re-split approach instead partly for this reason.
- Some "skipped words" reports are inference-stack artifacts (same checkpoint drops words
  under vLLM but not transformers, [#251](https://github.com/canopyai/Orpheus-TTS/issues/251))
  — don't reflexively blame the dataset for every artifact.

## Recommendations for the deathstalker (and future voice) re-cut

1. **Raise `max_seq_length` 2048 → 4096** in `orpheus_owen.py` (TrainingConfig). Matches
   inference `max_model_len=4096`. Community-proven. VRAM cost is real but batch size 1 +
   grad-accum already accommodates it; unsloth LoRA at 4096 fits a 24GB card.
2. **Raise `MAX_CLIP_SECONDS` 20 → ~38-40** and re-derive the token-budget guard:
   `seconds × ~83 + TEXT_TOKEN_RESERVE + 7 specials ≤ 4096` → ~40s hard cap. Keep the
   existing measured-tok/s check (duplicate-frame removal makes effective rate slightly
   lower — and NOTE: frame-dedup compresses long silences in targets; it was the
   5s-pause culprit once already (`_normalize_pauses` history), re-verify pause behavior
   on long clips).
3. **Build a duration-MIXED dataset, not uniform-long**: roughly 1/3 short (3-10s,
   single sentences — protects short-line quality), 1/2 medium (10-25s), and a deliberate
   long tail (25-38s, multi-sentence) so the model sees EOS placed at every scale
   including past our 450-char (~30s) target. Merge only ADJACENT sentences from the
   `align_from_epub.py` alignment (book-as-truth), keeping internal pauses ≤2s (v2 rule);
   never splice non-adjacent audio.
4. **Hard-filter, loudly**: drop/split any example whose tokenized row exceeds 4096 and
   PRINT the count (over-length rows crash or silently truncate — and a truncated row has
   no EOS, which actively mis-trains stopping).
5. **Keep ~300+ examples per voice** (official quality bar); 1 epoch (community: eval
   loss rises and audio worsens after epoch 1 on small sets); checkpoints judged by ear.
6. **Success metric**: render a chapter at the OLD 450-char packing
   (`ORPHEUS_MAX_CHARS_PER_SEC` guard active) and count guard-trip lines in the worker
   log. Zero trips at 450 chars = the retrain worked; then e2a packing can go back up.
7. **Keep the inference guard forever** — early EOS is stochastic and partly exposure
   bias; the guard is cheap insurance even with a perfect dataset.

## Source audio note (2026-07-11)

Adobe Podcast enhancement adds audible artifacts; Owen is considering re-cleaning
manually. Original UNCUT deathstalker audiobooks (all 11 incl. preludes) are intact at
`E:\Shared\BookForge\projects\Deathstalker_*\output\*.m4b` (Book 2 was ad-trimmed; its
untrimmed original is beside it as `.m4b.prebak`). Pre-Adobe raw silence-split parts for
Books 1-3: `C:\Users\tellt\Downloads\Deathstalker Collection\Book#_recut\Book#_pt##.m4a`
(+ `recut_manifest.json` time-range maps). `enhanced/` subfolders = Adobe output (the
artifact source).
