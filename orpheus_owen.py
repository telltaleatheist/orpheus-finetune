#!/usr/bin/env python3
r"""
orpheus_owen.py — SINGLE SOURCE OF TRUTH for fine-tuning Orpheus-3B on Owen's voice.

This one file holds (a) the full spec for an Orpheus fine-tuning dataset as a block
of named constants, and (b) the pipeline that turns the existing XTTS-era `dataset_v5`
into a training-ready Orpheus dataset. Anything we *discovered* from the docs is baked
into the constants below with a citation. Anything still *uncertain* is marked `VERIFY:`
and, where possible, MEASURED at runtime instead of trusted.

Run it with subcommands:
    python orpheus_owen.py audit        # light deps (numpy, soundfile) — validate dataset_v5
    python orpheus_owen.py build        # heavy deps — resample/trim/repackage -> HF dataset
    python orpheus_owen.py train        # heavy deps — Unsloth LoRA fine-tune  (scaffold)

Everything runs in WSL (Linux) — Unsloth + bitsandbytes + SNAC are painful on native
Windows. The dataset lives at the WSL path in DATASET_DIR.

=============================================================================
 SPEC — Orpheus-3B fine-tuning dataset requirements  (what we learned)
=============================================================================

AUDIO
  * Sample rate ........ 24000 Hz, MONO.                      [Unsloth/Canopy docs]
      dataset_v5 is 22050 Hz mono -> MUST be resampled to 24000. This is the only
      hard audio transform Orpheus requires of our existing data.
  * Bit depth .......... irrelevant after we load to float32; SNAC takes a waveform.

CLIP LENGTH
  * Bounded by the training sequence length, not by a hard "max seconds" rule.
    We now train at max_seq_length = 4096 (raised from Unsloth's 2048 default,
    2026-07-11 — see TRAINING_CLIP_LENGTH_RESEARCH.md). Audio tokenizes at a
    MEASURED 82.5-84.1 SNAC tokens/sec, so with TEXT_TOKEN_RESERVE (256) for
    text + special tokens the audio budget is 3840 tokens ≈ 46.5 s; the 38 s
    MAX_CLIP_SECONDS cap sits under that with ~20% margin.
  * WHY the raise: EOS placement is learned directly from the clip-duration
    distribution ([SOH] text [EOH][SOA][SOS] codes [EOS][EOA], labels=input_ids).
    A dataset whose every example ends by 20 s teaches "utterances end by ~20 s"
    and the voice silently truncates ~30 s (450-char) chunks at a plausible
    sentence boundary (the Black Sun early-EOS bug). Training must SHOW the model
    EOS at every scale up to and past the production chunk length.
  * The dataset should be duration-MIXED (1/3 short 3-10 s, ~1/2 medium 10-25 s,
    deliberate 25-38 s long tail), NOT uniformly long — short-only fails long
    utterances, and homogeneous-long degrades short-line quality (HALL-E /
    VoiceStar length-attractor literature).
  * We still DROP clips shorter than MIN_CLIP_SECONDS: very short clips make
    autoregressive codec-LMs unstable (the same "tiny-chunk gibberish" failure we
    hit with XTTS). And we DROP/flag anything above MAX_CLIP_SECONDS as a token-budget
    guard — an over-length row is truncated by the trainer and loses its EOS
    entirely, which actively mis-trains stopping.

  * >>> PLANNED RE-CUT (decided 2026-06-27): before the real training run we will
    RE-CUT to LONGER clips (target ~TARGET_CLIP_SECONDS, cap MAX_CLIP_SECONDS) that
    KEEP natural inter-sentence pauses — Orpheus's prosody advantage comes from that
    rhythm, which the sentence-level dataset_v5 cut discards.
      DO NOT merge dataset_v5 clips: they were silence-trimmed at BOTH ends, so gluing
      them only restores the ~0.1s pads, not Owen's real pauses. Instead RE-SEGMENT
      from the ORIGINAL `E:\final` chapter audio so the genuine inter-sentence silence
      survives.
      FOUNDATION = the ALIGNMENT, not the clips: the Whisper word/sentence timestamps
      against E:\final + epub text truth ([[dataset-v5-book-aligned-pipeline]]). Pick
      longer spans (greedy-pack sentences toward TARGET) at sentence boundaries, copy
      the audio span verbatim from E:\final (pauses intact), and trim ONLY the outer
      edges of each span — never the interior.
      RESOLVED 2026-06-27: the alignment intermediates were NOT persisted
      (/home/telltale/xtts_ft/booktext_align/ is gone) — only final clips survive. So
      the re-cut must REGENERATE alignment, but the raw inputs are intact:
        - audio:  E:\final\  (18 chapter WAVs, 48kHz/24-bit stereo, ~7 GB, ~6.7h)
        - text:   C:\Users\tellt\Downloads\godspeople.epub
        - pipeline (xtts-finetune/scripts/): extract_epub_text.py -> batch_transcribe.py
          (faster-whisper -> *.whisper.json) -> batch_align.py (epub-correct ->
          *.aligned.json) -> segment_aligned.py (cuts clips from /mnt/e/final).
      For ORPHEUS, the re-cut is mostly a PARAM change to segment_aligned.py: bump the
      ~9s target to ~TARGET_CLIP_SECONDS and max 11s -> ~MAX, and stop trimming the
      outer edges so a natural pad/pause remains (it already extracts a CONTINUOUS span
      per group, so interior pauses are preserved for free). PERSIST the alignment JSONs
      this time. Confirm real SNAC tok/s before locking TARGET/MAX to max_seq_length=4096.

SILENCE / PAUSES  (docs are SILENT on this — this is our reasoned policy)
  * Boundary (leading/trailing) silence: TRIM to a small consistent pad.
    Orpheus is autoregressive and learns to tie end-of-text -> end_of_speech token.
    Excess trailing silence in training clips => it learns to emit long silent tails
    (the exact long-pause bug we fought in XTTS). dataset_v5 is already silence-snapped
    at boundaries, so this pass is light / near-idempotent.
  * Internal pauses: KEEP THEM. Unlike our aggressive XTTS silence-trimming, Orpheus
    models prosody in the LLM; natural clause pauses inside a clip are signal, not noise.
    Compressing them makes the voice read rushed/flat. => we never touch interior silence.
  * No mid-word cuts: clips must be whole utterances. dataset_v5 is breath-safe. ✓

TRANSCRIPTS
  * Normalized text (numbers/dates spelled out). dataset_v5 already is. ✓
  * Optional emotion tags Orpheus understands: <laugh> <chuckle> <sigh> <cough>
    <sniffle> <groan> <yawn> <gasp>. We have none; that's fine.
  * Prompt format at train & inference is  f"{source}: {text}"  where `source` is the
    voice name. Our speaker column is "owen" -> source = "owen" -> inference will use
    `--fine_tuned owen` in BookForge.

DATASET SCHEMA (what `build` emits — matches canopylabs/zac-sample-dataset)
  * A HuggingFace dataset saved with save_to_disk, columns:
        source : str   (voice name, "owen")
        text   : str   (normalized transcript)
        audio  : Audio  (24 kHz mono waveform + sampling_rate)
  * SNAC tokenization (waveform -> discrete codes, 7 codes/frame, wrapped with
    start_of_speech/end_of_speech special tokens) happens in `train`, matching the
    Unsloth notebook — so the on-disk dataset stays inspectable / re-tokenizable.

TRAINING (baked-in defaults; tune in TrainingConfig)
  * Method ............. LoRA via Unsloth (parameter-efficient; ~3.24% of params).
  * LoRA rank .......... r = 64   (higher than text-only; TTS benefits)   [Unsloth]
  * LoRA alpha ......... 64
  * Learning rate ...... 2e-4
  * Optimizer .......... adamw_8bit
  * Batch ............. per_device_train_batch_size = 1, grad_accum = 4
  * Epochs ............ 1–3 (start at 1, listen, extend if under-fit)
  * max_seq_length .... 4096 (raised from 2048, 2026-07-11: fits 38 s clips so the
    EOS prior covers production 450-char chunks; community-proven, fits 24 GB
    with unsloth gradient checkpointing at batch 1)
  * Base model ........ fine-tune from the multispeaker PRETRAINED base for a clean
    single-speaker imprint, NOT the already-ft model. (Test both; see BASE_MODEL.)

SNAC TOKEN RATE — the one number we refuse to trust
  * Community/docs imply ~85 tokens/sec but one Unsloth page says "125 tokens ≈ 10 s"
    (=12.5/s), which is inconsistent. So `build`/`train` MEASURE the real tokens/clip
    from SNAC on the first MEASURE_N clips and warn if our 2048 budget is at risk.
    Never silently rely on AUDIO_TOKENS_PER_SEC for correctness — it's planning only.

Sources:
  - github.com/canopyai/Orpheus-TTS (README + finetune/, zac-sample-dataset format)
  - docs.unsloth.ai/basics/text-to-speech-tts-fine-tuning  (24kHz, r=64, lr2e-4, batch)
  - canopylabs/orpheus-3b-0.1-ft  (voice-name prompt format, emotion tags)
=============================================================================
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from training_profiles import ProfileError, TrainingProfile, load_training_profile

# ----------------------------------------------------------------------------
# CONSTANTS — the single source of truth. Change behavior HERE, not in functions.
# ----------------------------------------------------------------------------

# DEFINITIVE datasets (rebuilt 2026-07-07, book-corrected transcripts + natural pauses):
#   owen        -> /home/telltale/xtts_ft/owen_v2/raw        (God's People, /mnt/e/final)
#   mistborn    -> /home/telltale/xtts_ft/mistborn_v2/raw    (A Prince's Errand, M. Kramer)
#   thirdreich  -> /home/telltale/xtts_ft/thirdreich_v2/raw  (Coming of the Third Reich, S. Pratt)
# Each holds metadata_train/eval.csv + wavs; the OLD dataset_v5/dataset_orpheus_raw/*_raw_clean
# sets were cleared (regenerable via epub -> align_from_epub.py -> recut). Override --dataset-dir.
DATASET_DIR = Path("/home/telltale/xtts_ft/owen_v2/raw")
METADATA_FILES = ("metadata_train.csv", "metadata_eval.csv")
METADATA_DELIM = "|"
METADATA_HEADER = ("audio_file", "text", "speaker_name")  # dataset_v5's columns

# Where `build` writes the Orpheus-ready HF dataset.
OUTPUT_DIR = Path("/home/telltale/xtts_ft/orpheus_owen_ds")

# --- Re-cut (longer, pause-preserving) inputs/outputs ---
# `recut` reads the regenerated book alignment and cuts NEW longer clips straight from
# the original 48kHz chapters (so real inter-sentence pauses survive). Its output dir
# then feeds `build`. See SPEC > PLANNED RE-CUT.
ALIGN_DIR = Path("/home/telltale/xtts_ft/booktext_align")        # owen *.aligned.json (batch_align)
SRC_CHAPTERS = Path("/mnt/e/final")                              # owen original 48kHz chapter WAVs
RECUT_DIR = Path("/home/telltale/xtts_ft/owen_v2/raw")           # owen definitive clips + metadata

# Silence-cut params (ported verbatim from the proven xtts segment_aligned.py logic;
# only the duration targets change for Orpheus — longer, see TARGET/MAX_CLIP_SECONDS).
RECUT_TOP_DB = 35          # librosa.effects.split threshold (dB below peak = silence)
RECUT_MIN_SIL = 0.12       # s; min gap to register as silence
RECUT_STRONG_SIL = 0.28    # s; only gaps >= this become candidate CUT points (breath-safe)
RECUT_EDGE_PAD = 0.06      # s; keep cut points away from a gap's own edges
RECUT_EDGE_CHECK_MS = 15   # require both clip edges to be quiet
RECUT_EDGE_RMS_MAX = 0.02
# NOTE: cuts land at the QUIETEST point inside a strong gap, so every clip starts/ends
# mid-pause. The OLD assumption — that this alone gives "a consistent ~0.15-0.2s pad, no
# long tails by construction" — was WRONG: the quietest point of a multi-second CHAPTER gap
# leaves ~half that gap as an edge pad, and the merge step glues such gaps into clip
# interiors. So `_normalize_pauses` (applied per clip in `_recut_chapter`) now caps interior
# silence to MAX_INTERNAL_PAUSE_SECONDS and trims edges to LEAD_PAD/TRAIL_PAD. `build` still
# runs with TRIM_BOUNDARY_SILENCE off (the recut already produced normalized edges).

# --- Audio spec (hard requirements) ---
TARGET_SAMPLE_RATE = 24000   # Orpheus/SNAC expected rate. dataset_v5 is 22050 -> resample.
TARGET_CHANNELS = 1          # mono

# --- Clip-length policy ---
MIN_CLIP_SECONDS = 1.5       # drop shorter: codec-LM instability on tiny chunks
MAX_CLIP_SECONDS = 20.0      # PROVEN CEILING (2026-07-12): training on 38s clips @
                             # max_seq_length 4096 broke EOS reliability on multi-
                             # sentence chunks (rohan-v2 capped 19% vs 0/126 for every
                             # 20s/2048 voice; recut+retrain at 20s/2048 → ~0). The
                             # model learns "audio can run 30s+ before EOS" and stalls
                             # at internal sentence boundaries. Do NOT raise again.
# (raised 20->38 with seq 2048->4096, 2026-07-11: the 20s cap put the learned EOS
#  prior at ~20s and long 450-char chunks truncated early — see CLIP LENGTH spec)
# Above MAX we DROP by default (set DROP_OVER_MAX=False to only warn).
DROP_OVER_MAX = True
# Target for the PLANNED longer re-cut (see SPEC > PLANNED RE-CUT). Greedy-pack
# consecutive same-chapter dataset_v5 spans toward this, capped at MAX_CLIP_SECONDS.
# Provisional until the real SNAC tok/s is measured (build auto-measures it).
TARGET_CLIP_SECONDS = 15.0

# --- Silence policy (see SPEC). ---
# Default OFF: the `recut` path places every clip boundary at a silence trough, giving
# a consistent natural ~0.15-0.2s pad already. Trimming again would strip the lead-in
# Orpheus benefits from. (Only turn on if building from a NON-recut, untrimmed source.)
TRIM_BOUNDARY_SILENCE = False  # trim only the leading/trailing edges when True
KEEP_INTERNAL_PAUSES = True    # keep natural clause pauses (prosody signal) — but see CAP below
SILENCE_DB_FLOOR = -40.0       # below this (dBFS) at the edges counts as silence
LEAD_PAD_SECONDS = 0.02        # keep a hair of lead-in
TRAIL_PAD_SECONDS = 0.10       # consistent short tail (matches dataset_v4 philosophy)
# Internal-pause policy (REVISED 2026-07-07). Orpheus reads BEST with natural pauses kept
# exactly as the narrator performed them — the "model drops pauses in illogical spots" bug
# once blamed on long pauses was actually the TRAINING dedup flag collapsing pause frames
# (fixed via --no-dedup), NOT the clip pauses. So we now KEEP every natural pause verbatim
# and only tame genuine STRUCTURAL gaps (chapter/scene switches), which show up as multi-
# second silences a clip can accidentally straddle (measured in the alignments: 3–25s gaps).
# Two-tier in _normalize_pauses: a gap <= NATURAL_PAUSE_MAX is kept EXACTLY; a longer one
# (structural) is trimmed to STRUCTURAL_GAP_TRIM so no long dead air is baked into a clip.
NATURAL_PAUSE_MAX_SECONDS = 2.0    # longest gap treated as a real pause (kept verbatim)
STRUCTURAL_GAP_TRIM_SECONDS = 0.4  # chapter/scene gaps (> the max) trimmed to this
MAX_INTERNAL_PAUSE_SECONDS = NATURAL_PAUSE_MAX_SECONDS  # back-compat alias (--max-pause default)

# --- SNAC token accounting ---
# MEASURED 2026-06-27 on 32 dataset_v5 clips (cuda): 82.5 tok/s (range 82.0–84.1).
# => at max_seq_length 4096: budget 4096-256 = 3840 audio tokens ≈ 46.5s at 82.5
#    (44.3s at the worst measured 84.1). MAX 38s fits with ~20% margin; a 38s clip
#    is ~3196 tokens worst-case. `measure`/`build` re-check the rate on any new cut.
AUDIO_TOKENS_PER_SEC = 82.5    # measured; `measure` re-checks on any new cut
TEXT_TOKEN_RESERVE = 256       # headroom for "{source}: {text}" + special tokens
MEASURE_N = 32                 # clips to SNAC-encode for the empirical rate check

# --- Voice identity ---
SOURCE_NAME = "owen"           # becomes prompt prefix "owen: ..." and --fine_tuned owen

# --- Model / training (baked-in defaults) ---
# Fine-tune from the multispeaker pretrained base for a clean single-speaker imprint.
BASE_MODEL = "canopylabs/orpheus-3b-0.1-pretrained"   # alt: unsloth/orpheus-3b-0.1-ft


@dataclass
class TrainingConfig:
    # The Unsloth Orpheus notebook fine-tunes from the already-FT model (validated path);
    # BASE_MODEL (…-pretrained) is the alternative clean-base option to A/B later.
    train_from: str = "unsloth/orpheus-3b-0.1-ft"
    base_model: str = BASE_MODEL
    max_seq_length: int = 2048   # REVERTED to 2048 (2026-07-12): the 4096 raise (for
                                 # 38s clips / 450-char chunks) is PROVEN to break EOS
                                 # on multi-sentence chunks — see MAX_CLIP_SECONDS note.
                                 # Inference packs 200-char/2-sentence chunks now, so
                                 # long-clip training has no remaining motivation.
    lora_r: int = 64
    lora_alpha: int = 64
    lora_dropout: float = 0.0
    learning_rate: float = 2e-4
    optim: str = "adamw_8bit"
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    num_train_epochs: int = 2          # used only when --no-eval (fixed-epoch mode)
    max_epochs: int = 10               # CAP for eval+early-stopping mode (the default)
    early_stopping_patience: int = 2   # stop if eval_loss doesn't improve for N epochs
    save_total_limit: int = 6          # keep best/last few epoch checkpoints
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    lr_scheduler_type: str = "linear"
    seed: int = 3407
    load_in_4bit: bool = False
    target_modules: tuple = ("q_proj", "k_proj", "v_proj", "o_proj",
                             "gate_proj", "up_proj", "down_proj")
    output_dir: str = "/home/telltale/xtts_ft/orpheus_owen_lora"
    merged_dir: str = "/home/telltale/xtts_ft/orpheus_owen_merged"
    emotion_tags: tuple = ("<laugh>", "<chuckle>", "<sigh>", "<cough>",
                           "<sniffle>", "<groan>", "<yawn>", "<gasp>")


def training_config_from_profile(profile: TrainingProfile) -> TrainingConfig:
    """Materialize a profile without silently filling in model/training settings."""
    return TrainingConfig(
        train_from=profile.model_id,
        max_seq_length=profile.max_seq_length,
        lora_r=profile.lora_r,
        lora_alpha=profile.lora_alpha,
        lora_dropout=profile.lora_dropout,
        learning_rate=profile.learning_rate,
        optim=profile.optimizer,
        per_device_train_batch_size=profile.per_device_train_batch_size,
        gradient_accumulation_steps=profile.gradient_accumulation_steps,
        max_epochs=profile.max_epochs,
        early_stopping_patience=profile.early_stopping_patience,
        save_total_limit=profile.save_total_limit,
        warmup_ratio=profile.warmup_ratio,
        weight_decay=profile.weight_decay,
        lr_scheduler_type=profile.lr_scheduler_type,
        seed=profile.seed,
        load_in_4bit=profile.load_in_4bit,
        target_modules=profile.target_modules,
    )


def output_base_from_args(args, profile: TrainingProfile) -> Path:
    """Resolve an explicitly supplied output root or a profile-owned default."""
    if args.out_base:
        return Path(args.out_base)
    if profile.default_out_base:
        return Path(profile.default_out_base)
    raise SystemExit(
        f"profile '{profile.name}' has no default output root; pass --out-base explicitly"
    )


# ----------------------------------------------------------------------------
# Shared: read dataset_v5 metadata into rows of (wav_path, text, speaker)
# ----------------------------------------------------------------------------

@dataclass
class Clip:
    wav: Path
    text: str
    speaker: str
    seconds: float = 0.0
    sample_rate: int = 0
    channels: int = 0


def read_metadata(dataset_dir: Path, files=METADATA_FILES) -> list[Clip]:
    clips: list[Clip] = []
    for name in files:
        meta = dataset_dir / name
        if not meta.exists():
            raise FileNotFoundError(f"metadata not found: {meta}")
        with meta.open(encoding="utf-8") as fh:
            reader = csv.reader(fh, delimiter=METADATA_DELIM)
            header = next(reader)
            if tuple(header) != METADATA_HEADER:
                raise ValueError(
                    f"{meta}: unexpected header {header}, expected {METADATA_HEADER}"
                )
            for row in reader:
                if not row:
                    continue
                rel, text, speaker = row[0], row[1], row[2]
                clips.append(Clip(wav=dataset_dir / rel, text=text, speaker=speaker))
    return clips


# ----------------------------------------------------------------------------
# audit — light deps only (numpy, soundfile). Validate dataset_v5 vs the SPEC.
# ----------------------------------------------------------------------------

def cmd_audit(args) -> int:
    import numpy as np
    import soundfile as sf

    dataset_dir = Path(args.dataset_dir)
    clips = read_metadata(dataset_dir)
    print(f"[audit] {len(clips)} clips from {dataset_dir}")

    durations = []
    bad_sr, bad_ch, missing = [], [], []
    too_short, too_long, speakers = [], [], set()

    for c in clips:
        speakers.add(c.speaker)
        if not c.wav.exists():
            missing.append(c)
            continue
        info = sf.info(str(c.wav))
        c.seconds = info.frames / info.samplerate
        c.sample_rate = info.samplerate
        c.channels = info.channels
        durations.append(c.seconds)
        if info.samplerate != TARGET_SAMPLE_RATE:
            bad_sr.append(c)
        if info.channels != TARGET_CHANNELS:
            bad_ch.append(c)
        if c.seconds < MIN_CLIP_SECONDS:
            too_short.append(c)
        if c.seconds > MAX_CLIP_SECONDS:
            too_long.append(c)

    d = np.array(durations) if durations else np.array([0.0])
    total_h = float(d.sum()) / 3600.0
    est_tokens = d * AUDIO_TOKENS_PER_SEC
    budget = TrainingConfig().max_seq_length - TEXT_TOKEN_RESERVE
    over_budget = int((est_tokens > budget).sum())

    print("\n=== DURATION ===")
    print(f"  count {len(d)}   total {total_h:.2f} h")
    print(f"  min {d.min():.2f}s  median {np.median(d):.2f}s  "
          f"mean {d.mean():.2f}s  max {d.max():.2f}s")
    print("\n=== SPEC CONFORMANCE ===")
    print(f"  speakers seen ............ {sorted(speakers)} "
          f"(source-name={args.source_name!r})")
    print(f"  missing files ............ {len(missing)}")
    print(f"  need resample (!=24k) .... {len(bad_sr)}  "
          f"(dataset_v5 is 22050 — expected; build resamples all)")
    print(f"  non-mono ................. {len(bad_ch)}")
    print(f"  shorter than {MIN_CLIP_SECONDS}s ...... {len(too_short)}  (will DROP)")
    print(f"  longer than {MAX_CLIP_SECONDS}s ...... {len(too_long)}  "
          f"({'DROP' if DROP_OVER_MAX else 'WARN'})")
    print(f"  est. over token budget ... {over_budget}  "
          f"(planning rate {AUDIO_TOKENS_PER_SEC}/s, budget {budget} tok — "
          f"build MEASURES real rate)")

    kept = len(d) - len(too_short) - (len(too_long) if DROP_OVER_MAX else 0) - len(missing)
    print(f"\n[audit] would keep ~{kept} clips (~{total_h:.1f} h gross). "
          f"Reference single-speaker finetune ≈ 3 h / 1200 clips — we're well over.")
    if too_short[:5]:
        print("  e.g. too-short:", [f"{c.wav.name}:{c.seconds:.1f}s" for c in too_short[:5]])
    return 0


# ----------------------------------------------------------------------------
# recut — heavy deps. Cut NEW longer (pause-preserving) clips straight from the 48kHz
#         E:\final chapters using the regenerated book alignment. Ported from the proven
#         xtts segment_aligned.py silence logic; only the duration targets change
#         (longer) and we cut at TARGET_SAMPLE_RATE (24k) directly. Writes RECUT_DIR.
# ----------------------------------------------------------------------------

def _silence_gaps(y, np, librosa):
    sr = TARGET_SAMPLE_RATE
    iv = librosa.effects.split(y, top_db=RECUT_TOP_DB, frame_length=2048, hop_length=512)
    gaps = []
    if len(iv) == 0:
        return [(0, len(y))]
    if iv[0][0] > 0:
        gaps.append((0, iv[0][0]))
    for k in range(len(iv) - 1):
        gaps.append((iv[k][1], iv[k + 1][0]))
    if iv[-1][1] < len(y):
        gaps.append((iv[-1][1], len(y)))
    return [(a, b) for a, b in gaps if (b - a) / sr >= RECUT_MIN_SIL]


def _quietest_point(y, a, b, np):
    sr = TARGET_SAMPLE_RATE
    a = int(a); b = int(b)
    if b - a < int(0.02 * sr):
        return int((a + b) / 2)
    win = max(1, int(0.010 * sr))
    seg = y[a:b].astype(np.float64)
    csum = np.concatenate([[0.0], np.cumsum(seg * seg)])
    n = len(seg)
    centers = np.arange(0, n)
    lo = np.clip(centers - win, 0, n)
    hi = np.clip(centers + win, 0, n)
    rms = np.sqrt((csum[hi] - csum[lo]) / np.maximum(hi - lo, 1))
    pad = int(RECUT_EDGE_PAD * sr)
    if n > 2 * pad:
        rms[:pad] = np.inf
        rms[-pad:] = np.inf
    return a + int(np.argmin(rms))


def _edge_silent(y, a, b, np):
    sr = TARGET_SAMPLE_RATE
    w = int(RECUT_EDGE_CHECK_MS / 1000 * sr)
    head = y[a:a + w]; tail = y[b - w:b]
    if len(head) < w or len(tail) < w:
        return False
    return (np.sqrt(np.mean(head ** 2)) < RECUT_EDGE_RMS_MAX and
            np.sqrt(np.mean(tail ** 2)) < RECUT_EDGE_RMS_MAX)


def _words_in(words, a, b):
    sr = TARGET_SAMPLE_RATE
    return " ".join(w for (w, ws, we) in words if a <= int(((ws + we) / 2) * sr) < b)


def _nkey(s):
    import re
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def _normalize_pauses(clip, np, librosa, max_pause=NATURAL_PAUSE_MAX_SECONDS,
                      structural_trim=STRUCTURAL_GAP_TRIM_SECONDS):
    """Keep NATURAL pauses exactly; tame only structural (chapter/scene) gaps + tighten edges.

    Reuses the recut's peak-relative silence detector (RECUT_TOP_DB) to find non-silent
    speech intervals; rebuilds the clip keeping every speech interval verbatim. Two-tier on
    each interior silence: a gap <= max_pause is a real performance pause and is kept EXACTLY
    (verbatim); a longer gap is a structural break the merge accidentally straddled and is
    trimmed to `structural_trim` so no multi-second dead air is baked into the clip.
    Leading/trailing silence is trimmed to LEAD_PAD/TRAIL_PAD. Returns the waveform.
    """
    sr = TARGET_SAMPLE_RATE
    iv = librosa.effects.split(clip, top_db=RECUT_TOP_DB, frame_length=2048, hop_length=512)
    if len(iv) == 0:
        return clip
    max_int = int(max_pause * sr)
    trim = int(structural_trim * sr)
    lead = int(LEAD_PAD_SECONDS * sr)
    trail = int(TRAIL_PAD_SECONDS * sr)
    pieces = []
    # leading silence -> at most LEAD_PAD
    if iv[0][0] > 0:
        keep = min(int(iv[0][0]), lead)
        pieces.append(clip[iv[0][0] - keep: iv[0][0]])
    for k in range(len(iv)):
        a, b = int(iv[k][0]), int(iv[k][1])
        pieces.append(clip[a:b])                       # speech verbatim
        if k < len(iv) - 1:                            # interior gap
            g0, g1 = b, int(iv[k + 1][0])
            glen = g1 - g0
            if glen <= max_int:
                pieces.append(clip[g0:g1])             # natural pause -> kept EXACTLY
            else:
                off = (glen - trim) // 2               # structural gap -> short centered remnant
                pieces.append(clip[g0 + off: g0 + off + trim])
    # trailing silence -> at most TRAIL_PAD
    if iv[-1][1] < len(clip):
        keep = min(len(clip) - int(iv[-1][1]), trail)
        pieces.append(clip[int(iv[-1][1]): int(iv[-1][1]) + keep])
    return np.concatenate(pieces) if pieces else clip


def _recut_chapter(name, wav, aligned_json, out_wavs, rows, np, librosa, sf,
                   source_name=SOURCE_NAME, cap_seconds=None,
                   max_pause=MAX_INTERNAL_PAUSE_SECONDS):
    import json, re, os
    sr = TARGET_SAMPLE_RATE
    y, _ = librosa.load(wav, sr=sr, mono=True)            # 48kHz E:\final -> 24k clean
    aligned = json.load(open(aligned_json, encoding="utf-8"))
    words = [(a["w"], a["start"], a["end"]) for a in aligned]
    if not words:
        print(f"  {name}: no words"); return 0
    gaps = _silence_gaps(y, np, librosa)
    strong = [_quietest_point(y, a, b, np) for a, b in gaps if (b - a) / sr >= RECUT_STRONG_SIL]
    bounds = sorted(set([0] + strong + [len(y)]))

    raw = []
    for i in range(len(bounds) - 1):
        a, b = bounds[i], bounds[i + 1]
        txt = _words_in(words, a, b)
        if txt.strip():
            raw.append([a, b, txt])

    # merge toward TARGET_CLIP_SECONDS, cap MAX_CLIP_SECONDS (longer than xtts)
    merged = []
    for span in raw:
        if merged and (merged[-1][1] - merged[-1][0]) / sr < TARGET_CLIP_SECONDS and \
           (span[1] - merged[-1][0]) / sr <= MAX_CLIP_SECONDS:
            merged[-1][1] = span[1]
        else:
            merged.append(span[:])

    def split_long(a, b):
        if (b - a) / sr <= MAX_CLIP_SECONDS:
            return [(a, b)]
        interior = [(g0, g1) for (g0, g1) in gaps if a < g0 and g1 < b]
        if not interior:
            return [(a, b)]
        g0, g1 = max(interior, key=lambda g: g[1] - g[0])
        mid = _quietest_point(y, g0, g1, np)
        if mid <= a or mid >= b:
            return [(a, b)]
        return split_long(a, mid) + split_long(mid, b)

    final = []
    for a, b, _ in merged:
        final.extend(split_long(a, b))

    kept = 0
    emitted = 0.0
    for a, b in final:
        dur = (b - a) / sr
        if dur < MIN_CLIP_SECONDS or dur > MAX_CLIP_SECONDS:
            continue
        if not _edge_silent(y, a, b, np):
            continue
        clean = re.sub(r"\s+", " ", _words_in(words, a, b)).strip()
        if len(clean) < 3:
            continue
        clip = _normalize_pauses(y[a:b], np, librosa, max_pause)  # cap chapter/section gaps, tighten edges
        cdur = len(clip) / sr
        if cdur < MIN_CLIP_SECONDS:                      # guard: pause-capping shouldn't, but never emit a runt
            continue
        safe = re.sub(r"\s+", "_", name)
        fn = f"{safe}_{str(kept).zfill(8)}.wav"
        sf.write(os.path.join(out_wavs, fn), clip, sr)
        rows.append((f"wavs/{fn}", clean, source_name))
        kept += 1
        emitted += cdur
        if cap_seconds is not None and emitted >= cap_seconds:
            print(f"  {name}: hit cap {cap_seconds / 60:.1f} min -> stopping", flush=True)
            break
    print(f"  {name}: {len(words)} words -> {kept} clips "
          f"({emitted / 60:.1f} min, from {len(final)} spans)", flush=True)
    return kept, emitted


def cmd_recut(args) -> int:
    import os, glob
    import numpy as np
    import librosa
    import soundfile as sf
    import pandas as pd

    align_dir = Path(args.align_dir)
    out_ds = Path(args.recut_dir)
    out_wavs = out_ds / "wavs"
    out_wavs.mkdir(parents=True, exist_ok=True)

    aligned_files = sorted(glob.glob(str(align_dir / "*.aligned.json")))
    if not aligned_files:
        raise FileNotFoundError(f"no *.aligned.json in {align_dir} — run batch_align first")
    src_chapters = Path(args.src_chapters)
    wav_candidates = glob.glob(str(src_chapters / "*.wav"))
    source_name = args.source_name
    lq_prefix = args.lq_prefix
    lq_cap_s = (args.lq_cap_minutes * 60.0) if args.lq_cap_minutes else None
    lq_emitted = 0.0
    # HQ sorts before "lq_" alphabetically, so HQ is cut in full first and the LQ cap
    # only limits the low-quality fill. Provenance lives in the clip filename prefix.
    rows, total = [], 0
    for aj in aligned_files:
        stem = os.path.basename(aj).replace(".aligned.json", "")
        match = None
        for w in wav_candidates:
            wn = os.path.splitext(os.path.basename(w))[0]
            if _nkey(wn) == _nkey(stem):
                match = (wn, w); break
        if not match:
            print(f"  no wav for {stem}"); continue
        name, wav = match
        if args.only and args.only not in name:
            continue
        is_lq = os.path.basename(name).startswith(lq_prefix)
        cap = None
        if is_lq and lq_cap_s is not None:
            remaining = lq_cap_s - lq_emitted
            if remaining <= 0:
                print(f"  {name}: LQ cap ({lq_cap_s / 60:.1f} min) reached -> skipping")
                continue
            cap = remaining
        kept, emitted = _recut_chapter(name, wav, aj, str(out_wavs), rows, np, librosa, sf,
                                       source_name=source_name, cap_seconds=cap,
                                       max_pause=args.max_pause)
        total += kept
        if is_lq:
            lq_emitted += emitted

    # Optional: trim to a TARGET duration by keeping an EVENLY-SPREAD subset of clips
    # (variety across the whole source beats a contiguous head-slice). Deletes the rest.
    if args.target_minutes and rows:
        durs = [sf.info(str(out_wavs / os.path.basename(rel))).frames
                / sf.info(str(out_wavs / os.path.basename(rel))).samplerate
                for rel, _, _ in rows]
        total_secs = sum(durs)
        target_secs = args.target_minutes * 60.0
        if total_secs > target_secs:
            mean = total_secs / len(rows)
            keep_n = max(1, int(round(target_secs / mean)))
            keep = set(int(round(x)) for x in np.linspace(0, len(rows) - 1, keep_n))
            new_rows, kept_secs = [], 0.0
            for i, r in enumerate(rows):
                if i in keep:
                    new_rows.append(r); kept_secs += durs[i]
                else:
                    try:
                        (out_wavs / os.path.basename(r[0])).unlink()
                    except FileNotFoundError:
                        pass
            print(f"[recut] target {args.target_minutes:.0f} min -> kept {len(new_rows)}/{len(rows)} "
                  f"clips ({kept_secs / 60:.1f} min, evenly spread); deleted {len(rows) - len(new_rows)}")
            rows = new_rows
            total = len(rows)
        else:
            print(f"[recut] target {args.target_minutes:.0f} min >= available "
                  f"{total_secs / 60:.1f} min — keeping all")

    if not args.only:
        df = pd.DataFrame(rows, columns=["audio_file", "text", "speaker_name"])
        df = df.sample(frac=1, random_state=13)
        nval = int(len(df) * 0.15)
        df[nval:].sort_values("audio_file").to_csv(out_ds / "metadata_train.csv", sep="|", index=False)
        df[:nval].sort_values("audio_file").to_csv(out_ds / "metadata_eval.csv", sep="|", index=False)
        (out_ds / "lang.txt").write_text("en\n")
        print(f"  wrote metadata to {out_ds}")
    print(f"\n[recut] TOTAL clips: {total}  (target {TARGET_CLIP_SECONDS}s, cap {MAX_CLIP_SECONDS}s, {TARGET_SAMPLE_RATE}Hz)")
    print(f"[recut] next: `build --dataset-dir {out_ds}`")
    return 0


# ----------------------------------------------------------------------------
# build — heavy deps. Resample->24k, light boundary-trim, repackage as HF dataset.
#         SNAC tokenization is deferred to `train` (keeps on-disk data inspectable).
# ----------------------------------------------------------------------------

def _trim_boundary_silence(y, sr):
    """Trim only leading/trailing silence; NEVER interior (KEEP_INTERNAL_PAUSES)."""
    import numpy as np
    if not TRIM_BOUNDARY_SILENCE:
        return y
    amp_floor = 10.0 ** (SILENCE_DB_FLOOR / 20.0)
    nz = np.where(np.abs(y) > amp_floor)[0]
    if nz.size == 0:
        return y  # all-silence guard: leave as-is, let it surface downstream
    start = max(0, nz[0] - int(LEAD_PAD_SECONDS * sr))
    end = min(len(y), nz[-1] + int(TRAIL_PAD_SECONDS * sr))
    return y[start:end]


def cmd_build(args) -> int:
    import soundfile as sf
    from datasets import Dataset

    dataset_dir = Path(args.dataset_dir)
    out_dir = Path(args.output_dir)
    clips = read_metadata(dataset_dir)
    print(f"[build] {len(clips)} clips -> {out_dir}")

    paths, texts, sources, durs = [], [], [], []
    kept = dropped = bad_sr = 0
    for c in clips:
        if not c.wav.exists():
            dropped += 1
            continue
        info = sf.info(str(c.wav))          # metadata only, no full decode
        secs = info.frames / info.samplerate
        if info.samplerate != TARGET_SAMPLE_RATE:
            bad_sr += 1                     # recut already writes 24k; flag if not
        if secs < MIN_CLIP_SECONDS or (secs > MAX_CLIP_SECONDS and DROP_OVER_MAX):
            dropped += 1
            continue
        paths.append(str(c.wav.resolve()))
        texts.append(c.text)
        sources.append(args.source_name)    # normalize speaker -> the voice name
        durs.append(round(secs, 3))
        kept += 1

    # Lightweight MANIFEST dataset: audio stays as 24kHz wavs on disk; `train` SNAC-encodes
    # from audio_path. This deliberately avoids HF datasets' Audio() feature, which in
    # datasets>=4 requires torchcodec (fragile ffmpeg dep) — we don't need lazy decoding
    # since we own the training loop.
    ds = Dataset.from_dict({"audio_path": paths, "text": texts,
                            "source": sources, "duration": durs})
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(out_dir))
    total_h = sum(durs) / 3600.0
    msg = f"[build] kept {kept}, dropped {dropped}"
    if bad_sr:
        msg += f", WARNING {bad_sr} clips not {TARGET_SAMPLE_RATE}Hz"
    print(msg + f". {total_h:.2f} h. Saved manifest dataset to {out_dir}")

    # Empirical SNAC-rate check on the real (longer) clips — MEASURE, don't trust constant.
    _measure_snac_paths(paths[:MEASURE_N])
    return 0


def _measure_snac_paths(paths) -> None:
    """SNAC-encode a sample of wav paths; report real tokens/sec vs the planning constant."""
    try:
        import numpy as np
        import torch
        import librosa
        from snac import SNAC
    except Exception as e:  # noqa: BLE001 — measurement is best-effort, not correctness
        print(f"[measure] skipped SNAC check ({e}). Install `snac` to validate.")
        return
    if not paths:
        return
    model = SNAC.from_pretrained(SNAC_MODEL).eval()
    if torch.cuda.is_available():
        model = model.cuda()
    rates = []
    for p in paths:
        y, _ = librosa.load(p, sr=TARGET_SAMPLE_RATE, mono=True)
        rates.append(_snac_token_count(model, y, np, torch) / (len(y) / TARGET_SAMPLE_RATE))
    _report_snac_rate(rates, np)


SNAC_MODEL = "hubertsiuzdak/snac_24khz"   # the 24 kHz SNAC codec Orpheus is trained on

# --- Orpheus token layout (VERBATIM from the Unsloth Orpheus notebook) ---
# Audio = 7 SNAC codes per frame, each offset by 128266 + codebook*4096 into the
# extended vocab. The sequence wraps text + speech with these special tokens.
TOKENISER_LENGTH = 128256
END_OF_TEXT = 128009
START_OF_SPEECH = TOKENISER_LENGTH + 1   # 128257
END_OF_SPEECH = TOKENISER_LENGTH + 2     # 128258
START_OF_HUMAN = TOKENISER_LENGTH + 3    # 128259
END_OF_HUMAN = TOKENISER_LENGTH + 4      # 128260
START_OF_AI = TOKENISER_LENGTH + 5       # 128261
END_OF_AI = TOKENISER_LENGTH + 6         # 128262
CODE_OFFSET = 128266                     # base offset for SNAC code 0 of codebook 0


def _orpheus_codes_from_wav(path, snac_model, np, torch, librosa):
    """Load a 24kHz wav and SNAC-encode into Orpheus's flat 7-per-frame token list."""
    y, _ = librosa.load(path, sr=TARGET_SAMPLE_RATE, mono=True)  # clips already 24k
    wav = torch.from_numpy(y).unsqueeze(0).unsqueeze(0).to(dtype=torch.float32)
    if next(snac_model.parameters()).is_cuda:
        wav = wav.cuda()
    with torch.inference_mode():
        codes = snac_model.encode(wav)   # 3 hierarchical codebooks
    out = []
    for i in range(codes[0].shape[1]):
        out.append(codes[0][0][i].item() + CODE_OFFSET)
        out.append(codes[1][0][2 * i].item() + CODE_OFFSET + 4096)
        out.append(codes[2][0][4 * i].item() + CODE_OFFSET + 2 * 4096)
        out.append(codes[2][0][4 * i + 1].item() + CODE_OFFSET + 3 * 4096)
        out.append(codes[1][0][2 * i + 1].item() + CODE_OFFSET + 4 * 4096)
        out.append(codes[2][0][4 * i + 2].item() + CODE_OFFSET + 5 * 4096)
        out.append(codes[2][0][4 * i + 3].item() + CODE_OFFSET + 6 * 4096)
    return out


def _dedup_frames(vals):
    """Drop consecutive frames whose first code repeats (Unsloth's remove_duplicate_frames)."""
    if len(vals) % 7 != 0:
        raise ValueError("code list not divisible by 7")
    result = vals[:7]
    for i in range(7, len(vals), 7):
        if vals[i] != result[-7]:
            result.extend(vals[i:i + 7])
    return result


def _snac_token_count(model, y, np, torch) -> int:
    """SNAC-encode a float32 24kHz mono waveform; return total flattened code count."""
    wav = torch.from_numpy(np.asarray(y, dtype=np.float32)).unsqueeze(0).unsqueeze(0)
    if next(model.parameters()).is_cuda:
        wav = wav.cuda()
    with torch.inference_mode():
        codes = model.encode(wav)  # list of code tensors across SNAC's 3 levels (7/frame)
    return sum(int(c.numel()) for c in codes)


def _report_snac_rate(rates, np) -> None:
    r = float(np.mean(rates))
    budget = TrainingConfig().max_seq_length - TEXT_TOKEN_RESERVE
    max_secs = budget / r
    print(f"\n[measure] real SNAC rate ≈ {r:.1f} tok/s  (min {min(rates):.1f}, "
          f"max {max(rates):.1f}; planning constant was {AUDIO_TOKENS_PER_SEC}).")
    print(f"[measure] audio token budget = {budget} (max_seq_length "
          f"{TrainingConfig().max_seq_length} − {TEXT_TOKEN_RESERVE} text reserve)")
    print(f"[measure] => longest safe clip ≈ {max_secs:.1f}s.")
    for label, val in (("TARGET_CLIP_SECONDS", TARGET_CLIP_SECONDS),
                       ("MAX_CLIP_SECONDS", MAX_CLIP_SECONDS)):
        verdict = "OK" if val <= max_secs else "TOO HIGH — lower it"
        print(f"           {label} = {val}s -> {verdict}")


def cmd_measure(args) -> int:
    """Measure real SNAC tokens/sec on a sample of dataset_v5 clips (settles the budget)."""
    import numpy as np
    import torch
    import librosa
    from snac import SNAC

    dataset_dir = Path(args.dataset_dir)
    clips = read_metadata(dataset_dir)
    n = min(MEASURE_N, len(clips))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[measure] SNAC {SNAC_MODEL} on {dev}; sampling {n}/{len(clips)} clips")
    model = SNAC.from_pretrained(SNAC_MODEL).eval()
    if dev == "cuda":
        model = model.cuda()

    rates = []
    for c in clips[:n]:
        y, _ = librosa.load(str(c.wav), sr=TARGET_SAMPLE_RATE, mono=True)  # 22050->24000
        secs = len(y) / TARGET_SAMPLE_RATE
        toks = _snac_token_count(model, y, np, torch)
        rates.append(toks / secs)
    _report_snac_rate(rates, np)
    return 0


def _measure_snac_rate(ds, n: int) -> None:
    """build-time variant: same check over an in-memory HF dataset (best-effort)."""
    try:
        import numpy as np
        import torch
        from snac import SNAC
    except Exception as e:  # noqa: BLE001 — measurement is best-effort, not correctness
        print(f"[measure] skipped SNAC rate check ({e}). Install `snac` to validate.")
        return
    model = SNAC.from_pretrained(SNAC_MODEL).eval()
    if torch.cuda.is_available():
        model = model.cuda()
    rates = []
    for i in range(n):
        y = np.asarray(ds[i]["audio"]["array"], dtype=np.float32)
        rates.append(_snac_token_count(model, y, np, torch) / (len(y) / TARGET_SAMPLE_RATE))
    _report_snac_rate(rates, np)


# ----------------------------------------------------------------------------
# train — Unsloth LoRA fine-tune with held-out eval + early-stopping + best-checkpoint
#         selection (no guessing the epoch count). SNAC token layout is verbatim from
#         the Unsloth Orpheus notebook.
# ----------------------------------------------------------------------------

def _encode_clips(clips, tokenizer, snac_model, cfg, np, torch, librosa, label, source,
                  mask_prompt=False, dedup=True):
    """SNAC-encode clips into Orpheus input_ids; return an HF Dataset.

    `source` is the voice/prompt token and is REQUIRED — training prompts are built
    from it, NOT from the CSV speaker column. (Bug found 2026-07-07: this used
    c.speaker, so `--source-name` renamed only the output dirs and the "rohan"
    retrain silently trained on the CSV's "deathstalker:" prompts.)

    mask_prompt: label the text-prompt span -100 so loss (train AND eval) applies only
    to the audio tokens — all gradient goes to p(audio|text) instead of re-learning to
    predict the transcript, and eval_loss becomes a pure-audio model-selection metric.
    NOTE: eval_loss under masking is NOT comparable to unmasked runs.

    dedup=False keeps duplicate SNAC frames. _dedup_frames collapses repeated coarse
    codes — which is mostly SILENCE — so dedup time-compresses natural pauses in the
    training targets (~0.086s/frame; a 0.5s pause -> ~0.1-0.2s) and the model learns
    rushed pacing. Keeping duplicates preserves real pause durations.
    """
    from datasets import Dataset
    rows = {"input_ids": [], "labels": [], "attention_mask": []}
    err = over = 0
    for i, c in enumerate(clips):
        try:
            codes = _orpheus_codes_from_wav(str(c.wav), snac_model, np, torch, librosa)
            if dedup:
                codes = _dedup_frames(codes)
        except Exception:  # noqa: BLE001 — one bad clip shouldn't kill the run
            err += 1
            continue
        text_ids = tokenizer.encode(f"{source}: {c.text}", add_special_tokens=True) + [END_OF_TEXT]
        prompt_ids = [START_OF_HUMAN] + text_ids + [END_OF_HUMAN] + [START_OF_AI] + [START_OF_SPEECH]
        ids = prompt_ids + codes + [END_OF_SPEECH] + [END_OF_AI]
        if len(ids) > cfg.max_seq_length:
            over += 1
            continue
        rows["input_ids"].append(ids)
        rows["labels"].append([-100] * len(prompt_ids) + ids[len(prompt_ids):]
                              if mask_prompt else list(ids))
        rows["attention_mask"].append([1] * len(ids))
        if (i + 1) % 300 == 0:
            print(f"  [{label}] encoded {i + 1}/{len(clips)}", flush=True)
    print(f"[train] {label}: encoded {len(rows['input_ids'])} (skipped {err} err, {over} >maxlen)")
    return Dataset.from_dict(rows)


def _cmd_train_orpheus(args, profile: TrainingProfile) -> int:
    if profile.kind != "orpheus_tts":
        raise SystemExit(f"profile '{profile.name}' is not an Orpheus TTS profile")
    import time
    import numpy as np
    import torch
    import librosa
    from snac import SNAC
    from unsloth import FastLanguageModel
    from transformers import TrainingArguments, Trainer, EarlyStoppingCallback, TrainerCallback
    try:
        import run_metrics
    except Exception:
        run_metrics = None

    cfg = training_config_from_profile(profile)
    if args.max_seq_length:
        cfg.max_seq_length = args.max_seq_length
    # Per-voice output dirs (owen defaults reproduce the original paths exactly).
    source = args.source_name
    cfg.output_dir, cfg.merged_dir = (str(path) for path in profile.output_dirs(
        output_base_from_args(args, profile), source_name=source))
    base = args.train_from or cfg.train_from
    smoke = bool(args.max_steps)            # max_steps => fixed-step smoke run (no eval)
    epochs_cap = args.epochs or cfg.max_epochs

    # 1. model + LoRA (16-bit; fits the 3090 Ti for a 3B)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base, max_seq_length=cfg.max_seq_length, dtype=None,
        load_in_4bit=cfg.load_in_4bit)
    model = FastLanguageModel.get_peft_model(
        model, r=cfg.lora_r, target_modules=list(cfg.target_modules),
        lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout, bias="none",
        use_gradient_checkpointing="unsloth", random_state=cfg.seed)

    # 2. SNAC-encode TRAIN and EVAL splits separately (eval is the held-out overtraining
    #    detector; recut wrote metadata_train.csv = 85% and metadata_eval.csv = 15%).
    snac_model = SNAC.from_pretrained(SNAC_MODEL).eval().cuda()
    train_clips = read_metadata(Path(args.recut_dir), ("metadata_train.csv",))
    eval_clips = read_metadata(Path(args.recut_dir), ("metadata_eval.csv",))
    if args.limit:
        train_clips = train_clips[:args.limit]
        eval_clips = eval_clips[:max(2, args.limit // 8)]
    if args.lr_schedule:
        cfg.lr_scheduler_type = args.lr_schedule
    mask_prompt = bool(args.mask_prompt_loss)
    dedup = not args.no_dedup
    mode = (f"steps={args.max_steps}" if smoke
            else (f"epochs={epochs_cap} OVERTRAIN (no early-stop, keep last)"
                  if args.no_early_stop else f"epochs<={epochs_cap} + early-stop(eval_loss)"))
    print(f"[train] base={base} train={len(train_clips)} eval={len(eval_clips)} "
          f"r={cfg.lora_r} lr={cfg.learning_rate} sched={cfg.lr_scheduler_type} "
          f"mask_prompt={mask_prompt} dedup={dedup} {mode}")
    if mask_prompt:
        print("[train] NOTE: eval_loss is audio-only under --mask-prompt-loss — "
              "NOT comparable to unmasked runs.")
    csv_speakers = {c.speaker for c in train_clips} | {c.speaker for c in eval_clips}
    if csv_speakers != {source}:
        print(f"[train] NOTE: CSV speaker_name = {sorted(csv_speakers)} but prompt token "
              f"(--source-name) = '{source}' — prompts are built from '{source}:'.")
    train_ds = _encode_clips(train_clips, tokenizer, snac_model, cfg, np, torch, librosa, "train",
                             source, mask_prompt=mask_prompt, dedup=dedup)
    eval_ds = _encode_clips(eval_clips, tokenizer, snac_model, cfg, np, torch, librosa, "eval",
                            source, mask_prompt=mask_prompt, dedup=dedup)
    del snac_model
    torch.cuda.empty_cache()

    # 3. train. Default = eval each epoch, keep every epoch's checkpoint, early-stop when
    #    eval_loss stops improving, and load_best_model_at_end => final model is the BEST
    #    epoch (not the last). Smoke/max_steps skips eval. Batch 1 => no padding collator.
    use_eval = not smoke and len(eval_ds) > 0
    # overtraining mode: still EVAL each epoch (so we see the loss curve) but never
    # early-stop and never load-best — the final model is the LAST epoch, and every
    # epoch checkpoint is kept for auditioning. See --no-early-stop help.
    no_early_stop = bool(args.no_early_stop)
    stop_overtrain = bool(getattr(args, "stop_overtrain", False))
    if no_early_stop and stop_overtrain:
        raise SystemExit("--no-early-stop and --stop-overtrain are mutually exclusive")
    if stop_overtrain and not use_eval:
        raise SystemExit("--stop-overtrain needs an eval set (it stops on eval_loss)")
    # THREE modes:
    #   default          -> early-stop(patience) + load BEST epoch  (final = best)
    #   --no-early-stop  -> run ALL epochs, keep every ckpt         (final = last)
    #   --stop-overtrain -> run until eval_loss rises past best for --overtrain-patience
    #                       epoch(s), then STOP; keep every ckpt    (final = settle+patience)
    #                       Captures the LATE pause-consolidation epoch without wasting
    #                       epochs beyond overtrain+patience. The settle epoch stays on
    #                       disk (save_total_limit=None) for the keeper ear-check.
    load_best = use_eval and not no_early_stop and not stop_overtrain
    keep_all_ckpts = cfg.save_total_limit if load_best else None   # None => keep ALL
    track_metric = use_eval and (load_best or stop_overtrain)
    targs = TrainingArguments(
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=1,       # variable-length seqs => batch 1 (no padding)
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        warmup_ratio=cfg.warmup_ratio,
        num_train_epochs=(1 if smoke else epochs_cap),
        max_steps=(args.max_steps if smoke else -1),
        learning_rate=cfg.learning_rate, logging_steps=10, optim=cfg.optim,
        weight_decay=cfg.weight_decay, lr_scheduler_type=cfg.lr_scheduler_type,
        seed=cfg.seed, output_dir=cfg.output_dir, report_to="none",
        save_strategy=("no" if smoke else "epoch"),
        eval_strategy=("epoch" if use_eval else "no"),
        save_total_limit=keep_all_ckpts,
        load_best_model_at_end=load_best,
        metric_for_best_model=("eval_loss" if track_metric else None),
        greater_is_better=False)

    class OvertrainStopCallback(TrainerCallback):
        """Stop `patience` epoch(s) after eval_loss stops improving; keep the LAST
        model (= settle + patience), NOT the best. All epoch checkpoints are retained
        so the settle epoch is still on disk for the keeper ear-check."""
        def __init__(self, patience):
            self.patience = patience; self.best = None; self.bad = 0
        def on_evaluate(self, a, state, control, metrics=None, **kw):
            v = (metrics or {}).get("eval_loss")
            if v is None:
                return control
            if self.best is None or v < self.best:
                self.best = v; self.bad = 0
            else:
                self.bad += 1
                if self.bad >= self.patience:
                    print(f"[train] eval_loss rose {self.bad} epoch(s) past best "
                          f"{self.best:.4f} -> stopping (overtrain+{self.patience}).",
                          flush=True)
                    control.should_training_stop = True
            return control

    class TrainMetricsCallback(TrainerCallback):
        """Per-epoch wall-time + GPU health -> run_metrics.jsonl + a per-run JSON.
        Rising temp + sagging SM clock + growing epoch_s = throttling/dying GPU."""
        def __init__(self, meta):
            self.meta = meta; self.t0 = time.time(); self.epoch_t = self.t0; self.epochs = []
        def on_evaluate(self, a, state, control, metrics=None, **kw):
            now = time.time()
            gpu = run_metrics.gpu_snapshot() if run_metrics else {}
            self.epochs.append({"epoch": round(float(state.epoch or 0), 2),
                                "epoch_s": round(now - self.epoch_t, 1),
                                "eval_loss": (metrics or {}).get("eval_loss"),
                                **{f"gpu_{k}": v for k, v in gpu.items()}})
            self.epoch_t = now
            return control
        def on_train_end(self, a, state, control, **kw):
            if not run_metrics:
                return control
            total = round(time.time() - self.t0, 1)
            valid = [e for e in self.epochs if e.get("eval_loss") is not None]
            best = min(valid, key=lambda e: e["eval_loss"]) if valid else None
            rec = {**self.meta, "total_s": total, "n_epochs": len(self.epochs),
                   "best_epoch": best["epoch"] if best else None,
                   "best_eval_loss": best["eval_loss"] if best else None, "epochs": self.epochs}
            run_metrics.record("train", rec)
            p = run_metrics.save_run(f"train_{self.meta.get('voice','voice')}", rec)
            print(f"[metrics] run saved -> {p}  (total {total}s, {len(self.epochs)} epochs)", flush=True)
            return control

    run_meta = {"voice": source, "n_clips": len(train_clips) + len(eval_clips),
                "n_train": len(train_clips), "n_eval": len(eval_clips), "base": base,
                "recut_dir": args.recut_dir, "max_seq": cfg.max_seq_length,
                "lora_r": cfg.lora_r, "lr": cfg.learning_rate,
                "batch": cfg.per_device_train_batch_size,
                "grad_accum": cfg.gradient_accumulation_steps, "epochs_cap": epochs_cap,
                "mode": ("no-early-stop" if no_early_stop else
                         "stop-overtrain" if stop_overtrain else "early-stop"),
                "overtrain_patience": (int(args.overtrain_patience) if stop_overtrain else None),
                "lr_schedule": cfg.lr_scheduler_type,
                "mask_prompt": bool(args.mask_prompt_loss), "no_dedup": bool(args.no_dedup)}
    if load_best:
        callbacks = [EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience)]
    elif stop_overtrain:
        callbacks = [OvertrainStopCallback(int(args.overtrain_patience))]
    else:
        callbacks = []
    callbacks.append(TrainMetricsCallback(run_meta))
    trainer = Trainer(model=model, train_dataset=train_ds,
                      eval_dataset=(eval_ds if use_eval else None),
                      args=targs, callbacks=callbacks)
    # Resume restores optimizer/LR/step/RNG from the checkpoint's trainer_state and
    # fast-forwards the dataloader. Fail loud if a resume was requested but can't be
    # honored — never silently start from scratch (NO FALLBACKS).
    resume = args.resume_from_checkpoint
    if resume:
        if resume == "latest":
            ckpts = sorted(Path(cfg.output_dir).glob("checkpoint-*"),
                           key=lambda p: int(p.name.split("-")[1]))
            if not ckpts:
                raise SystemExit(f"[train] --resume-from-checkpoint latest: no checkpoint-* in {cfg.output_dir}")
            resume = str(ckpts[-1])
        elif not Path(resume).is_dir():
            raise SystemExit(f"[train] --resume-from-checkpoint: not a directory: {resume}")
        print(f"[train] RESUMING from {resume}", flush=True)
    trainer.train(resume_from_checkpoint=resume)
    if use_eval and trainer.state.best_model_checkpoint:
        print(f"[train] BEST epoch checkpoint = {trainer.state.best_model_checkpoint} "
              f"(eval_loss {trainer.state.best_metric:.4f}) — loaded as the final model.")

    # 4. save the best LoRA (+ optional merge to 16-bit for vLLM/BookForge)
    model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    print(f"[train] saved best LoRA adapters -> {cfg.output_dir}")
    if args.merge:
        model.save_pretrained_merged(cfg.merged_dir, tokenizer, save_method="merged_16bit")
        print(f"[train] merged 16-bit model -> {cfg.merged_dir}  (load in vLLM; "
              f"voice = '{source}', prompt '{source}: <text>')")
    else:
        print("[train] to merge for vLLM later: re-run with --merge")
    return 0


def cmd_train(args) -> int:
    """Dispatch to a strict, named training profile."""
    try:
        profile = load_training_profile(Path(args.training_config), args.profile)
    except ProfileError as error:
        raise SystemExit(f"[train] invalid training profile: {error}") from error
    if profile.kind == "orpheus_tts":
        return _cmd_train_orpheus(args, profile)
    if profile.kind == "text_sft":
        from text_sft import train_text_sft
        return train_text_sft(args, profile)
    raise SystemExit(f"[train] unsupported profile kind '{profile.kind}' in '{profile.name}'")


# ----------------------------------------------------------------------------
# speak — synthesize speech from the fine-tuned model. Inverse of the training
#         tokenization: prompt -> generate audio tokens -> reverse the 7-per-frame
#         SNAC layout -> snac.decode -> waveform. Used to audition the trained voice.
# ----------------------------------------------------------------------------

DEFAULT_SAMPLE_TEXT = (
    "The question of how ordinary people come to embrace extraordinary cruelty has "
    "haunted historians for generations. It is tempting to imagine that the architects "
    "of atrocity were monsters, set apart from the rest of us by some defect of "
    "character. The truth is far more unsettling.\n"
    "What the record shows, again and again, is that the machinery of persecution was "
    "built and operated by people who thought of themselves as decent, even devout. "
    "They went to church on Sunday, loved their children, and believed they were "
    "defending something sacred. That is the warning history leaves us: not that evil "
    "is rare, but that it so often wears a familiar and respectable face."
)


def _split_for_tts(text, max_chars=280, min_chars=25, pack=False):
    """Split text into TTS chunks.

    Default (pack=False): one sentence per chunk so every sentence boundary gets a
    gap. Very short fragments (interjections like "So.") merge into the previous
    chunk to avoid tiny unstable generations; a chunk that would exceed max_chars
    (~17s) is kept whole anyway (never split mid-sentence).

    pack=True: greedily pack ADJACENT sentences up to max_chars so each generation
    spans 2-3 sentences — mirrors e2a core.py's Orpheus multi-sentence packing so an
    A/B here reflects the pipeline. Within-chunk sentence boundaries get the model's
    own prosody (no inserted gap) instead of a hard fixed gap."""
    import re
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", " ".join(text.split())) if s.strip()]
    chunks = []
    if pack:
        for s in sents:
            if chunks and len(chunks[-1]) + 1 + len(s) <= max_chars:
                chunks[-1] = f"{chunks[-1]} {s}".strip()
            else:
                chunks.append(s)
        return chunks
    for s in sents:
        if chunks and (len(s) < min_chars or len(chunks[-1]) < min_chars):
            chunks[-1] = f"{chunks[-1]} {s}".strip()
        else:
            chunks.append(s)
    return chunks


def _redistribute_and_decode(speech_tokens, snac_model, torch):
    """Reverse the 7-per-frame Orpheus layout -> 3 SNAC codebooks -> waveform."""
    code_list = [t - CODE_OFFSET for t in speech_tokens]
    n = len(code_list) // 7
    l1, l2, l3 = [], [], []
    for i in range(n):
        b = code_list[7 * i:7 * i + 7]
        l1.append(b[0])
        l2.append(b[1] - 4096)
        l3.append(b[2] - 2 * 4096)
        l3.append(b[3] - 3 * 4096)
        l2.append(b[4] - 4 * 4096)
        l3.append(b[5] - 5 * 4096)
        l3.append(b[6] - 6 * 4096)

    def t(layer):
        x = torch.tensor(layer, dtype=torch.int32).clamp_(0, 4095)
        return x.unsqueeze(0).cuda()
    with torch.inference_mode():
        audio = snac_model.decode([t(l1), t(l2), t(l3)])
    return audio.squeeze().detach().cpu().float().numpy()


def cmd_speak(args) -> int:
    # ============================================================================
    # DEBUG-ONLY (2026-07-11). This standalone HF-transformers `.generate` path is
    # NO LONGER the way to render/audition a trained voice. It has NO VRAM cap (it
    # OOM'd the whole desktop at batch 8 / max-chars 450) and none of BookForge's
    # WSL guards. Render through the BookForge TTS CLI instead — it drives BookForge's
    # compiled worker pool, inheriting the WSL kill-ladder, vLLM gpu_memory_utilization
    # memory tiers + safe GPU sizing, and model resolution:
    #     python cli/bookforge-tts.py --tts --engine=orpheus --voice=<v> \
    #         --input passage.txt --out sample.wav [--tier fast]
    # (BookForge repo; must be built, need not be running.) Keep `speak` ONLY for
    # debugging this finetune pipeline — it was invaluable for that. See memory
    # bookforge-tts-cli.md. The ORPHEUS_GPU_MEM_UTIL cap below is a partial backstop.
    # ============================================================================
    import os
    import numpy as np
    import torch
    import soundfile as sf
    from snac import SNAC
    from unsloth import FastLanguageModel

    # Optional GPU memory bound (2026-07-11): ORPHEUS_GPU_MEM_UTIL=0.54 hard-caps
    # this process's torch allocations at that fraction of VRAM (mirrors
    # BookForge's "Fast" tier ~13 GiB on the 24 GB card) so audition runs can't
    # starve a desktop in active use. Applies to the caching allocator, so it
    # also bounds any engine built on torch tensors; an over-ask OOMs THIS
    # process fast instead of squeezing the host. Unset = old behavior.
    _util = os.environ.get("ORPHEUS_GPU_MEM_UTIL")
    if _util:
        frac = float(_util)
        torch.cuda.set_per_process_memory_fraction(frac)
        total = torch.cuda.get_device_properties(0).total_memory
        print(f"[speak] GPU hard cap: {frac:.2f} of VRAM "
              f"({frac * total / 2**30:.1f} GiB) via ORPHEUS_GPU_MEM_UTIL", flush=True)

    try:
        profile = load_training_profile(Path(args.training_config), args.profile)
    except ProfileError as error:
        raise SystemExit(f"[speak] invalid training profile: {error}") from error
    if profile.kind != "orpheus_tts":
        raise SystemExit("[speak] requires an orpheus_tts profile")
    cfg = training_config_from_profile(profile)
    source = args.source_name
    _, merged_dir = profile.output_dirs(output_base_from_args(args, profile), source_name=source)
    cfg.merged_dir = str(merged_dir)
    model_dir = args.model or cfg.merged_dir
    if args.text_file:
        text = Path(args.text_file).read_text(encoding="utf-8")
    else:
        text = args.text or DEFAULT_SAMPLE_TEXT
    batch = args.batch or 8
    print(f"[speak] model={model_dir}  voice={source}  batch={batch}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_dir, max_seq_length=cfg.max_seq_length, dtype=None,
        load_in_4bit=cfg.load_in_4bit)
    FastLanguageModel.for_inference(model)
    tokenizer.padding_side = "left"          # left-pad so generated tokens align across the batch
    snac_model = SNAC.from_pretrained(SNAC_MODEL).eval().cuda()

    chunks = _split_for_tts(text, max_chars=args.max_chars, pack=args.pack)
    gap = np.zeros(int(args.gap * TARGET_SAMPLE_RATE), dtype=np.float32)
    prompts = [([START_OF_HUMAN]
                + tokenizer.encode(f"{source}: {ch}", add_special_tokens=True)
                + [END_OF_TEXT, END_OF_HUMAN, START_OF_AI, START_OF_SPEECH]) for ch in chunks]
    print(f"[speak] {len(prompts)} chunks, {len(prompts)//batch + 1} batches")
    PAD = END_OF_TEXT                        # masked out; < CODE_OFFSET so it's filtered later
    segments = []
    for bstart in range(0, len(prompts), batch):
        group = prompts[bstart:bstart + batch]
        mlen = max(len(p) for p in group)
        ids = torch.tensor([[PAD] * (mlen - len(p)) + p for p in group]).cuda()
        attn = torch.tensor([[0] * (mlen - len(p)) + [1] * len(p) for p in group]).cuda()
        # max_new_tokens must cover the LONGEST possible render for this chunk
        # size or the harness itself truncates (~21.8 s at the old hardcoded
        # 1800 = the audio budget bug found 2026-07-11 during the v2 acid test:
        # a 450-char chunk needs ~30 s ≈ 2500 tokens). Auto-fit to the model's
        # context: whole window minus this batch's prompt length.
        max_new = cfg.max_seq_length - mlen - 8
        with torch.inference_mode():
            out = model.generate(ids, attention_mask=attn, max_new_tokens=max_new,
                                 do_sample=True, temperature=args.temperature, top_p=args.top_p,
                                 repetition_penalty=args.repetition_penalty, eos_token_id=END_OF_SPEECH,
                                 pad_token_id=PAD)
        for j in range(len(group)):
            gen = out[j].tolist()[mlen:]                       # generated part (left-padded prompt)
            if END_OF_SPEECH in gen:
                gen = gen[:gen.index(END_OF_SPEECH)]
            gen = [tk for tk in gen if tk >= CODE_OFFSET]      # keep only speech codes
            if len(gen) >= 7:
                segments.append(_redistribute_and_decode(gen, snac_model, torch))
                segments.append(gap)
        print(f"  batch {bstart // batch + 1} ({len(group)} chunks) done", flush=True)
    if not segments:
        print("[speak] nothing generated"); return 1
    audio = np.concatenate(segments)
    sf.write(args.out, audio, TARGET_SAMPLE_RATE)
    print(f"[speak] wrote {len(audio) / TARGET_SAMPLE_RATE:.1f}s -> {args.out}")
    return 0


# ----------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description="Profile-driven Orpheus or text-model LoRA training")
    p.add_argument("--training-config", default=str(Path(__file__).with_name("training_profiles.json")),
                   help="JSON file holding named model/training profiles")
    p.add_argument("--profile", default="orpheus",
                   help="named profile in --training-config (default: orpheus)")
    p.add_argument("--dataset-dir", default=str(DATASET_DIR))
    p.add_argument("--output-dir", default=str(OUTPUT_DIR))
    p.add_argument("--align-dir", default=str(ALIGN_DIR))
    p.add_argument("--recut-dir", default=str(RECUT_DIR))
    p.add_argument("--only", default=None, help="recut: limit to one chapter substring")
    # --- voice parameterization (defaults reproduce the Owen run exactly) ---
    p.add_argument("--source-name", default=SOURCE_NAME,
                   help="voice/prompt name; metadata speaker col + 'name: text' prefix")
    p.add_argument("--src-chapters", default=str(SRC_CHAPTERS),
                   help="recut: dir of source narration wavs to cut from")
    p.add_argument("--lq-prefix", default="lq_",
                   help="recut: source-wav name prefix marking low-quality fill")
    p.add_argument("--max-pause", type=float, default=NATURAL_PAUSE_MAX_SECONDS,
                   help="recut: longest gap treated as a NATURAL pause, kept verbatim (default "
                        "2.0s); longer gaps are structural chapter/scene breaks trimmed to "
                        f"{STRUCTURAL_GAP_TRIM_SECONDS}s so no long dead air is baked in")
    p.add_argument("--lq-cap-minutes", type=float, default=0.0,
                   help="recut: cap total minutes taken from lq-prefixed sources (0=no cap)")
    p.add_argument("--target-minutes", type=float, default=0.0,
                   help="recut: trim final set to ~N min by keeping an EVENLY-SPREAD clip "
                        "subset (variety beats a head-slice); deletes the rest (0=keep all)")
    p.add_argument("--out-base", default=None,
                   help="train/speak: output root; required for text profiles, Orpheus profile supplies its configured default")
    p.add_argument("--max-steps", type=int, default=0, help="train: cap steps (smoke test)")
    p.add_argument("--limit", type=int, default=0, help="train: only first N clips (smoke test)")
    p.add_argument("--epochs", type=int, default=0, help="train: max epochs cap (default cfg.max_epochs)")
    p.add_argument("--max-seq-length", type=int, default=0,
                   help="train: override max_seq_length (e.g. 2048 = the recipe every "
                        "pre-rohan voice was trained with; default cfg.max_seq_length)")
    p.add_argument("--resume-from-checkpoint", default=None,
                   help="train: resume from a checkpoint dir, or 'latest' for the highest "
                        "checkpoint-N in the output dir. Restores optimizer/LR/step/RNG state "
                        "and fast-forwards the dataloader — the SAME train flags must be passed "
                        "so the re-encoded dataset is identical.")
    p.add_argument("--no-early-stop", action="store_true",
                   help="train: disable early-stopping AND best-model loading — run the full "
                        "--epochs and keep the LAST epoch. Use for OVERTRAINING experiments: "
                        "Orpheus consolidates inter-sentence PAUSE reproduction ~1 epoch AFTER "
                        "eval_loss bottoms, so the best-eval checkpoint can squish pauses "
                        "(proven 2026-07-13). Keeps every epoch checkpoint to audition.")
    p.add_argument("--stop-overtrain", action="store_true",
                   help="train: like --no-early-stop but STOP once eval_loss rises past the best "
                        "for --overtrain-patience epoch(s) — don't waste epochs beyond "
                        "overtrain+patience. Keeps every checkpoint; final model = settle+patience "
                        "(the pause-consolidation epoch). Pick the keeper (settle vs settle+1) by ear.")
    p.add_argument("--overtrain-patience", type=int, default=1,
                   help="train: with --stop-overtrain, how many epochs eval_loss may rise past best "
                        "before stopping (default 1 = stop at settle+1; use 2 to tolerate a "
                        "1-epoch eval-noise wobble before the true settle).")
    p.add_argument("--merge", action="store_true", help="train: merge LoRA to 16-bit for vLLM")
    p.add_argument("--train-from", default=None, help="train: override base model")
    p.add_argument("--train-data", default=None,
                   help="text_sft train: chronological chat JSONL training split")
    p.add_argument("--eval-data", default=None,
                   help="text_sft train: chronological held-out chat JSONL evaluation split")
    p.add_argument("--run-name", default=None,
                   help="text_sft train: required output name used by the profile's output templates")
    p.add_argument("--mask-prompt-loss", action="store_true",
                   help="train: loss on audio tokens only (mask the text prompt with -100); "
                        "eval_loss becomes audio-only — not comparable to unmasked runs")
    p.add_argument("--no-dedup", action="store_true",
                   help="train: keep duplicate SNAC frames so real pause durations survive "
                        "(default dedup compresses silence in the training targets)")
    p.add_argument("--lr-schedule", default=None,
                   help="train: override lr_scheduler_type (e.g. constant_with_warmup so "
                        "early-stopping doesn't interact with a decaying schedule)")
    p.add_argument("--text", default=None, help="speak: text to synthesize (default sample)")
    p.add_argument("--text-file", default=None, help="speak: read text from a file")
    p.add_argument("--batch", type=int, default=8, help="speak: chunks generated in parallel")
    p.add_argument("--max-chars", type=int, default=280, help="speak: max chars per chunk")
    p.add_argument("--pack", action="store_true",
                   help="speak: greedily pack 2-3 sentences per chunk up to --max-chars "
                        "(mirrors e2a Orpheus multi-sentence chunking)")
    p.add_argument("--model", default=None, help="speak: model dir (default merged_dir)")
    p.add_argument("--out", default="/mnt/c/tmp/owen_sample.wav", help="speak: output wav")
    p.add_argument("--temperature", type=float, default=0.6, help="speak: sampling temperature (lower = steadier pitch)")
    p.add_argument("--top-p", type=float, default=0.8, help="speak: nucleus sampling top_p (0.8 = e2a/BookForge production default)")
    p.add_argument("--repetition-penalty", type=float, default=1.1, help="speak: repetition penalty")
    p.add_argument("--gap", type=float, default=0.5, help="speak: seconds of silence inserted between sentences")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("audit", help="validate a dataset against the spec (light deps)")
    sub.add_parser("measure", help="measure real SNAC tokens/sec to settle clip budget")
    sub.add_parser("recut", help="cut longer pause-preserving clips from E:\\final (heavy)")
    sub.add_parser("build", help="resample/trim/repackage -> HF dataset (heavy deps)")
    sub.add_parser("train", help="Unsloth LoRA fine-tune (eval + early-stopping)")
    sub.add_parser("speak", help="synthesize speech from the trained model (audition)")
    args = p.parse_args()

    return {"audit": cmd_audit, "measure": cmd_measure, "recut": cmd_recut,
            "build": cmd_build, "train": cmd_train, "speak": cmd_speak}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
