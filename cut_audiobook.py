#!/usr/bin/env python3
r"""cut_audiobook.py — cut a CONTIGUOUS full-book audiobook into an Orpheus-ready,
duration-MIXED training dataset, using a BookForge epub-aligned VTT for the text
and the WAVEFORM for every actual cut point.

WHY THIS EXISTS (vs cut_excerpts.py, which is for scattered GraphicAudio excerpts):
the source here is ONE narrator reading a whole book start-to-finish, and we
already have BookForge's epub-corrected sentence VTT (proper nouns exact). So we
skip Whisper+align entirely and drive cutting off the VTT — BUT the VTT cue
timestamps are Whisper-grade approximate (MEASURED on this book: ~50% of sentence
boundaries land mid-word, nearest real silence median 25ms / max 80ms away). So
the VTT is used ONLY for (a) exact text, (b) approximate sentence LOCATIONS, and
(c) chapter/subheading detection. Every real cut is SILENCE-SNAPPED to the
waveform — never placed at a raw cue timestamp.

THE CUTTING PHILOSOPHY (Owen's spec, 2026-07-11):
  * Cut at SENTENCE boundaries only (a clip always ends on a sentence end).
  * Keep EVERY natural pause verbatim — no capping, no compression.
  * PRESERVE the trailing pause: cut at the FAR edge of the sentence-end silence
    (next word's onset), so each clip carries its own end-of-sentence pause. This
    teaches the model to end utterances with a pause, so e2a needn't insert gaps.
  * The ONLY pauses removed are STRUCTURAL ones (chapter / subheading / part
    transitions) — detected semantically from the epub headings AND as long
    waveform silences. A clip ending at a structural boundary has its trailing
    pause capped (STRUCT_TRAIL_CAP); the long dead air is dropped.
  * Duration-MIXED buckets so EOS is learned at every scale (see
    TRAINING_CLIP_LENGTH_RESEARCH.md): default 2:3-10, 3:10-25, 1:25-38 seconds.
    Hard cap MAX_CLIP_SECONDS (38s = the max_seq_length=4096 token budget).
  * ~target-minutes of clips, EVENLY SPREAD across the whole book (variety).

Reuses orpheus_owen.py's validated, breath-safe silence machinery (do NOT
reimplement): _silence_gaps / RECUT_STRONG_SIL / _quietest_point.

Outputs (a NEW dataset dir; never touches the source audio):
  <out>/wavs/<stem>_########.wav      24000 Hz mono (Orpheus/SNAC rate)
  <out>/metadata_train.csv, metadata_eval.csv   audio_file|text|speaker_name
  <out>/clips.json                    provenance + trailing_pause + n_sentences
  <out>/CUT_REPORT.md                 totals, histograms, structural boundaries

Usage (env needs numpy/librosa/soundfile — e2a python_env works on Windows):
  python cut_audiobook.py --vtt rohan.vtt --audio book.flac --epub book.epub \
      --out-dir <.../rohan> --source-name rohan --target-minutes 180
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import orpheus_owen as oo                                   # noqa: E402
from cut_excerpts import expand_digits                      # noqa: E402
from align_excerpts import extract_book, _SMART, num_to_words  # noqa: E402

SR = oo.TARGET_SAMPLE_RATE           # 24000
MIN_CLIP = oo.MIN_CLIP_SECONDS       # 1.5
MAX_CLIP = oo.MAX_CLIP_SECONDS       # 38.0
STRONG_SIL = oo.RECUT_STRONG_SIL     # 0.28 s — a gap must be this long to be a cut
LEAD_PAD = oo.LEAD_PAD_SECONDS       # 0.02 s — hair of lead-in before the first word

# A waveform silence at least this long is treated as STRUCTURAL (chapter/scene) —
# in this narration natural performance pauses run <~1.5s, so anything past this is
# a section break. Reported in CUT_REPORT so the threshold is auditable/tunable.
STRUCT_GAP_SECONDS = 2.5
STRUCT_TRAIL_CAP = 0.5               # trailing pause kept at a structural boundary
SNAP_WINDOW = 0.15                   # s; how far a cue boundary may snap to real silence
SENT_FINAL = ('.', '?', '!', '"', '”', ')')
DQUOTES = '"“”'                       # any double-quote mark toggles in/out of a quotation


def dquote_open_after(cues):
    """For each cue index, True if we are INSIDE an open double-quote AFTER that cue
    (odd running count of double-quote marks across the book so far). Drives
    dialogue-aware clip planning: a clip should not END while a quotation is still
    open, so a quote (and, when it's a single sentence, its same-sentence 'he said'
    attribution) is never split across two clips. Counts straight and smart double
    quotes; single quotes/apostrophes are ignored (ambiguous with contractions)."""
    out, cnt = [], 0
    for (_, _, txt) in cues:
        cnt += sum(txt.count(q) for q in DQUOTES)
        out.append(cnt % 2 == 1)
    return out


# --------------------------------------------------------------------------- VTT
def parse_vtt(path):
    """Return list of cues [(start_s, end_s, text)].

    Cues tagged `NOTE asr-fallback` by align_audiobook.py are DROPPED here:
    they are whisper ASR text filling audioNotInEpub holes (intros/ads/outros),
    not book truth — training on them would inject transcription errors. This
    is automatic; --exclude-ranges remains for manual surgical exclusions only.
    """
    def ts(t):
        p = t.strip().split(":")
        return (int(p[0]) * 3600 + int(p[1]) * 60 + float(p[2])) if len(p) == 3 \
            else int(p[0]) * 60 + float(p[1])
    cues = []
    dropped_asr = 0
    pending_asr = False
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        if lines[i].strip().startswith("NOTE asr-fallback"):
            pending_asr = True          # tags the NEXT cue
            i += 1
            continue
        m = re.match(r"(\d[\d:.]+)\s*-->\s*(\d[\d:.]+)", lines[i])
        if m:
            st, en = ts(m.group(1)), ts(m.group(2))
            buf, j = [], i + 1
            while j < len(lines) and lines[j].strip():
                buf.append(lines[j].strip())
                j += 1
            if pending_asr:
                dropped_asr += 1
                pending_asr = False
            else:
                # normalize smart quotes/dashes -> ASCII (consistency with other voices)
                cues.append((st, en, " ".join(buf).translate(_SMART)))
            i = j
        else:
            i += 1
    if dropped_asr:
        print(f"[vtt] dropped {dropped_asr} asr-fallback cue(s) (whisper hole-fill, not book text)")
    return cues


def _norm_head(s):
    # drop a leading "N: " / "N. " chapter number, normalize to alnum words
    s = re.sub(r"^\s*[\dIVXLC]+[:.]\s*", "", s)
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).split()


def heading_cue_idxs(cues, epub):
    """Indices of cues that ARE a chapter/subheading (match an epub heading).
    Matches on the distinctive tail of the heading (e.g. 'discipline of
    meditation') so 'Chapter Two, The Discipline of Meditation' still hits."""
    if not epub:
        return set()
    _, meta = extract_book(epub)
    heads = []
    for m in meta:
        h = m.get("heading")
        if not h:
            continue
        w = _norm_head(h)
        if w:
            heads.append(w[-4:] if len(w) > 4 else w)   # distinctive tail
    hit = set()
    for ci, (_, _, txt) in enumerate(cues):
        cw = _norm_head(txt)
        if not cw or len(cw) > 12:      # headings are short cues
            continue
        cset = set(cw)
        for hw in heads:
            if hw and all(x in cset for x in hw):
                hit.add(ci)
                break
    return hit


# ----------------------------------------------------------------- cut planning
def build_boundaries(cues, gaps_strong, heading_idxs, np, snap_window=SNAP_WINDOW,
                     parity_after=None):
    """Snap each sentence-final cue end to the nearest strong silence gap, and
    tag structural boundaries. Returns sorted boundaries:
      [{"pos": g1_sample, "gstart": g0_sample, "struct": bool, "quote_safe": bool}]
      (word onsets).
    snap_window: how far (s) a cue end may be from a silence and still snap. Epub-
    align cue ends are tight (0.15 default); Whisper segment ends are much rougher
    (±0.5-1s), so whisper-transcribed sources need a wider window or nearly every
    sentence-end fails to snap and clips merge into >max_clip spans that get dropped.
    parity_after: optional per-cue 'inside an open quote?' flags (dquote_open_after).
    A boundary is quote_safe when closing a clip there won't split a quotation
    (structural walls are always safe). None => every boundary quote_safe (feature off)."""
    # gap intervals in seconds. BookForge sets a cue's end ~= the next word's
    # onset (the gap's FAR edge), so we snap when the cue end falls within the
    # silence interval [g0, g1] (+/- window), not merely near its start g0.
    if not gaps_strong:
        return []
    g0s = np.array([g0 / SR for (g0, g1) in gaps_strong])
    g1s = np.array([g1 / SR for (g0, g1) in gaps_strong])
    boundaries = {}
    for ci, (st, en, txt) in enumerate(cues):
        is_head_next = (ci + 1) in heading_idxs      # a heading FOLLOWS this cue
        if not (txt.rstrip().endswith(SENT_FINAL) or is_head_next):
            continue
        is_head = ci in heading_idxs or is_head_next   # gap borders a heading cue
        mask = (g0s - snap_window <= en) & (en <= g1s + snap_window)
        if not mask.any():
            continue                                 # no real pause here -> merge through
        idxs = np.where(mask)[0]
        dist = np.maximum(0.0, np.maximum(g0s[idxs] - en, en - g1s[idxs]))
        k = int(idxs[np.argmin(dist)])
        g0, g1 = gaps_strong[k]
        glen = (g1 - g0) / SR
        struct = bool(glen >= STRUCT_GAP_SECONDS or is_head)
        b = boundaries.get(g1)
        struct = struct or (b["struct"] if b else False)
        # A structural wall is never mid-quote; otherwise the closing sentence is cue
        # ci, so it's quote_safe iff we are NOT inside an open quote after ci.
        quote_safe = struct or parity_after is None or (not parity_after[ci])
        boundaries[g1] = {"pos": g1, "gstart": g0, "struct": struct,
                          "quote_safe": bool(quote_safe)}
    return sorted(boundaries.values(), key=lambda d: d["pos"])


def plan_clips(boundaries, first_speech, sampler, max_clip, dialogue_aware=True):
    """Merge sentence units between boundaries into duration-mixed clips, never
    crossing a structural boundary, each ending on a sentence. Boundaries are
    word-onset samples (the far edge of a sentence-end silence), so a clip is
    [prev_onset, this_onset] and always carries its own trailing pause.

    DIALOGUE-AWARE (default): once the sampled target is reached, only close at a
    quote_safe boundary — if we're mid-quotation, keep extending to the next safe
    boundary so a quote (and its same-sentence attribution) stays whole. Extension
    is still bounded by max_clip: a quotation longer than the remaining budget is
    force-split at max_clip (counted as a forced split; the ≤~20s/≤38s EOS-safety
    cap always wins over keeping a quote intact). A book with no double quotes is
    unaffected (every boundary is quote_safe).

    Returns (spans, stats): spans = (a_sample, b_onset, struct_end, gap_start);
    stats = {"deferred": boundaries kept whole to avoid a split,
             "forced_splits": quotes we had to cut for max_clip}."""
    clips = []
    n = len(boundaries)
    clip_a = first_speech
    target = sampler()
    deferred = forced = 0
    for i, b in enumerate(boundaries):
        cur_dur = (b["pos"] - clip_a) / SR
        next_dur = (boundaries[i + 1]["pos"] - clip_a) / SR if i + 1 < n else 1e9
        qsafe = b.get("quote_safe", True)
        reached = cur_dur >= target
        forced_close = b["struct"] or next_dur > max_clip
        if forced_close:
            close = True
            if dialogue_aware and not b["struct"] and not qsafe:
                forced += 1          # max_clip forced a cut inside a quotation
        elif reached:
            if dialogue_aware and not qsafe:
                close = False        # past target but mid-quote — defer to a safe end
                deferred += 1
            else:
                close = True
        else:
            close = False
        if close:
            clips.append((clip_a, b["pos"], b["struct"], b["gstart"]))
            clip_a = b["pos"]            # next clip starts at this word onset
            target = sampler()
    return clips, {"deferred": deferred, "forced_splits": forced}


# --------------------------------------------------------------------------- io
def cue_text_in(cues, a_s, b_s):
    """Join epub-exact cue text whose midpoint falls in [a_s, b_s] seconds."""
    parts = [t for (s, e, t) in cues if a_s <= (s + e) / 2 < b_s]
    return re.sub(r"\s+", " ", " ".join(parts)).strip()


def _num(n):
    x = num_to_words(int(n))
    return " ".join(x) if x else str(n)


def normalize_scripture(text):
    """This devotional book is dense with Bible references the narrator SPEAKS
    (confirmed by chars/sec: ref clips read at a normal 14.7 c/s after this norm).
    The epub prints them as digits — expand to the spoken form so training text
    matches the audio: '1 John 1:9' -> 'First John one nine', 'Matthew 5:16-18'
    -> 'Matthew five sixteen through eighteen'. Run BEFORE expand_digits so the
    ordinal book prefix isn't turned into a cardinal ('one John')."""
    ORD = {"1": "First", "2": "Second", "3": "Third"}
    text = re.sub(r"\b([123]) ([A-Z][a-z])",
                  lambda m: ORD[m.group(1)] + " " + m.group(2), text)

    def repl(m):
        s = f"{_num(m.group('c'))} {_num(m.group('v'))}"
        if m.group('v2'):
            s += f" through {_num(m.group('v2'))}"
        return s
    return re.sub(r"(?P<c>\d+):(?P<v>\d+)(?:[–-](?P<v2>\d+))?(?:ff\.)?",
                  repl, text)


def silence_analysis(y, np, librosa, chunk_sec=600, ov_sec=3.0):
    """Memory-safe silence detection over a multi-hour signal. librosa.effects.split
    on the whole 7h builds a 9GB framed array; instead run it on overlapping windows,
    merge the speech intervals (the overlap bridges seams so no word is split at a
    chunk edge), then derive gaps. Returns (gaps, speech_intervals) in global samples;
    mirrors orpheus_owen._silence_gaps (top_db=RECUT_TOP_DB, min gap RECUT_MIN_SIL)."""
    n = len(y)
    step, ov = int(chunk_sec * SR), int(ov_sec * SR)
    speech = []
    start = 0
    while start < n:
        seg = y[start:min(n, start + step + ov)]
        for a, b in librosa.effects.split(seg, top_db=oo.RECUT_TOP_DB,
                                          frame_length=2048, hop_length=512):
            speech.append((start + int(a), start + int(b)))
        start += step
    speech.sort()
    merged = []
    for a, b in speech:
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    gaps = []
    if merged and merged[0][0] > 0:
        gaps.append((0, merged[0][0]))
    for k in range(len(merged) - 1):
        gaps.append((merged[k][1], merged[k + 1][0]))
    if merged and merged[-1][1] < n:
        gaps.append((merged[-1][1], n))
    gaps = [(a, b) for a, b in gaps if (b - a) / SR >= oo.RECUT_MIN_SIL]
    return gaps, merged


def load_silence_map(path, n, np):
    """Load an auto-editor silence map (JSON {'silences':[[s0,s1],...]} in seconds) and
    return (gaps, speech) in global samples at SR — SAME contract as silence_analysis,
    so all the validated snapping/structural logic downstream is unchanged; only the
    silence DETECTOR swaps from librosa to auto-editor. Owen's auto-editor-driven cut."""
    import json
    d = json.load(open(path, encoding="utf-8"))
    sil = d["silences"] if isinstance(d, dict) else d
    raw = []
    for s0, s1 in sil:
        a, b = int(round(float(s0) * SR)), int(round(float(s1) * SR))
        a = max(0, min(a, n)); b = max(0, min(b, n))
        if b > a:
            raw.append((a, b))
    raw.sort()
    speech = []; prev = 0                       # speech = complement of ALL silences
    for a, b in raw:
        if a > prev: speech.append((prev, a))
        prev = max(prev, b)
    if prev < n: speech.append((prev, n))
    gaps = [(a, b) for a, b in raw if (b - a) / SR >= oo.RECUT_MIN_SIL]  # same floor
    return gaps, speech


SENT_TOL = 0.3          # s: slack around a silence interval when testing whether a
                        # sentence-final cue end falls inside it (cue timestamps are
                        # approximate). A gap containing such a cue end = between-
                        # sentence; anything else = mid-sentence.


def cap_internal_silences(clip, cue_ends_rel, max_between, max_mid, np, librosa):
    """Return (clip, mid_capped, max_mid_gap_s) with internal silences compressed
    SENTENCE-AWARELY:
      * a gap whose center is within SENT_TOL of a sentence-final cue end is a
        BETWEEN-sentence pause -> capped at max_between (e.g. 1.0s: tame long
        paragraph/context breaks, keep natural rhythm).
      * ANY OTHER internal gap is MID-sentence (e.g. a removed sound-effect gap,
        'a tall glass [gap] and gulped it down') -> capped tight at max_mid (e.g.
        0.4s) so it doesn't break prosody or teach a pause-before-dialogue.
    Lead-in and trailing tail are untouched (trail_cap handles the tail). Also
    reports how many mid-sentence gaps exceeded max_mid (i.e. lingering SFX-style
    gaps) and the longest one found, for the CUT_REPORT.
    cue_ends_rel: sentence-final cue END times in CLIP-relative seconds."""
    iv = librosa.effects.split(clip, top_db=oo.RECUT_TOP_DB,
                               frame_length=2048, hop_length=512)
    if len(iv) <= 1:
        return clip, 0, 0.0
    cb = int(round(max_between * SR))
    cm = int(round(max_mid * SR))
    parts = [clip[:int(iv[0][1])]]                      # lead-in + first speech run
    mid_capped = 0
    max_mid_gap = 0.0
    for k in range(1, len(iv)):
        s0, s1 = int(iv[k - 1][1]), int(iv[k][0])
        gap = s1 - s0
        # A sentence-final cue END lands INSIDE a between-sentence silence (it marks
        # the next word's onset, per build_boundaries). So the gap is between-sentence
        # iff some cue end falls within [s0, s1] (+/- tol). A mid-sentence SFX gap has
        # the cue spanning across it -> no cue end inside -> tight cap.
        a0, a1 = s0 / SR, s1 / SR
        between = any(a0 - SENT_TOL <= ce <= a1 + SENT_TOL for ce in cue_ends_rel)
        keep = min(gap, cb if between else cm)
        if not between and gap > cm:                    # a compressed mid-sentence gap
            mid_capped += 1
            max_mid_gap = max(max_mid_gap, gap / SR)
        parts.append(clip[s0:s0 + keep])
        parts.append(clip[int(iv[k][0]):int(iv[k][1])])  # this speech run
    parts.append(clip[int(iv[-1][1]):])                 # trailing tail (trail_cap trims)
    return np.concatenate(parts), mid_capped, max_mid_gap


def main():
    import time as _time
    _t0 = _time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--vtt", required=True)
    ap.add_argument("--audio", required=True, help="contiguous full-book audio (flac/wav)")
    ap.add_argument("--epub", default=None, help="for chapter/subheading detection")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--source-name", required=True)
    ap.add_argument("--mix", default="2:3-10,3:10-25,1:25-38")
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--max-clip", type=float, default=MAX_CLIP)
    ap.add_argument("--trail-cap", type=float, default=None,
                    help="trim each clip's trailing silence to at most this many "
                         "seconds (e.g. 0.1). Omit to keep full natural tails — but "
                         "uncapped tails train runaway silence into Orpheus.")
    ap.add_argument("--max-internal-sil", type=float, default=None,
                    help="compress BETWEEN-SENTENCE silence inside a clip longer than "
                         "this many seconds down to this length (e.g. 1.0). The waveform "
                         "cut only makes structural gaps (>=2.5s) into clip boundaries, "
                         "so long paragraph/context-break pauses up to ~2.5s otherwise "
                         "survive inside a clip and get LEARNED as spurious output "
                         "pauses. Natural pauses shorter than this are untouched. Omit "
                         "to keep all internal pauses verbatim.")
    ap.add_argument("--max-midsentence-sil", type=float, default=0.4,
                    help="tighter cap (default 0.4s) for silences NOT at a sentence "
                         "boundary — i.e. MID-sentence gaps, e.g. a removed sound-effect "
                         "gap ('a tall glass [gap] and gulped it down'). Only active when "
                         "--max-internal-sil is set. A gap containing a sentence-final "
                         "cue end (+/- 0.3s) is treated as between-sentence.")
    ap.add_argument("--snap-window", type=float, default=SNAP_WINDOW,
                    help=f"how far (s, default {SNAP_WINDOW}) a sentence-end cue may be "
                         "from a real silence and still snap to it. Epub-align VTTs are "
                         "tight; WHISPER-transcribed sources (no epub) have rough segment "
                         "timestamps — use ~0.6-0.8 for them or most sentence-ends fail "
                         "to snap and clips merge past --max-clip and get dropped.")
    ap.add_argument("--dialogue-aware", dest="dialogue_aware", action="store_true",
                    default=True,
                    help="(default ON) keep quotations whole: never end a clip inside "
                         "an open double-quote — defer the cut to the next sentence "
                         "boundary that closes the quote, so a quote and its 'he said' "
                         "attribution train together. Still bounded by --max-clip.")
    ap.add_argument("--no-dialogue-aware", dest="dialogue_aware", action="store_false",
                    help="disable dialogue-aware planning (cut purely on duration).")
    ap.add_argument("--target-minutes", type=float, default=180.0,
                    help="even-spread subset to ~this many minutes (0 = keep all)")
    ap.add_argument("--eval-frac", type=float, default=0.15)
    ap.add_argument("--gain-db", type=float, default=0.0,
                    help="STATIC gain to -20 LUFS (never per-clip loudnorm)")
    ap.add_argument("--dry-run", type=int, default=0,
                    help="write only the first N clips (spot-check), skip subset")
    ap.add_argument("--exclude-ranges", default=None,
                    help="comma-separated start-end second ranges whose cues are "
                         "dropped before cutting (e.g. '0-239.38'). Use for "
                         "audioNotInEpub holes: their cues are whisper-fallback ASR "
                         "text, not book truth — never train on them.")
    ap.add_argument("--silence-map", default=None,
                    help="JSON silence map from autoeditor_silences.py (auto-editor "
                         "loudness -> [[start_s,end_s],...]). When set, cut points snap "
                         "to THESE real silences instead of librosa's detection. "
                         "Owen's auto-editor-driven cutting (no mid-word slices).")
    args = ap.parse_args()

    excludes = []
    if args.exclude_ranges:
        for part in args.exclude_ranges.split(","):
            m = re.fullmatch(r"\s*([\d.]+)\s*-\s*([\d.]+)\s*", part)
            if not m:
                raise SystemExit(f"--exclude-ranges: malformed range {part!r} "
                                 f"(expected 'start-end' in seconds)")
            a, b = float(m.group(1)), float(m.group(2))
            if b <= a:
                raise SystemExit(f"--exclude-ranges: end <= start in {part!r}")
            excludes.append((a, b))

    import random
    import numpy as np
    import librosa
    import soundfile as sf
    from cut_excerpts import make_mix_sampler

    max_clip = args.max_clip
    trail_cap = args.trail_cap
    max_internal_sil = args.max_internal_sil
    max_mid_sil = args.max_midsentence_sil
    sampler = make_mix_sampler(args.mix, random.Random(args.seed))

    print(f"[load] {args.audio}", flush=True)
    # librosa.load reads the ENTIRE file at its NATIVE rate/channels before
    # resampling — a 24h/44.1k/stereo book is ~31 GB and OOMs a 32 GB box (and
    # blows the WSL RAM cap). Decode to 24k mono FIRST with ffmpeg (streamed, low
    # RAM) to a temp FLAC, then load that (~8 GB for 24h). Same samples, safe for
    # any length. FLAC temp (not WAV) because a >6.8h 24k-mono WAV exceeds 4 GB.
    import subprocess as _sp
    import tempfile as _tf
    _fd, _tmp = _tf.mkstemp(suffix=".flac")
    import os as _os
    _os.close(_fd)
    try:
        # soxr (precision 28) for the downsample to 24k — cleaner/alias-free vs
        # ffmpeg's default swr resampler. -nostdin so this ffmpeg never eats the
        # parent's stdin (which breaks tr|bash / heredoc script runs).
        # Windows conda/choco ffmpeg builds often LACK libsoxr ("resampling engine
        # unavailable"); fall back to the default swr resampler, which is transparent
        # for a <10 kHz voice at 44.1->24k. Linux/WSL ffmpeg keeps soxr.
        _base = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", args.audio]
        _tail = ["-ac", "1", "-c:a", "flac", _tmp]
        _r = _sp.run(_base + ["-af", f"aresample={SR}:resampler=soxr:precision=28"] + _tail)
        if _r.returncode != 0:
            print("[resample] libsoxr unavailable in this ffmpeg -> default swr "
                  "(transparent for this voice at 44.1->24k)", flush=True)
            _sp.run(_base + ["-af", f"aresample={SR}"] + _tail, check=True)
        y, _ = librosa.load(_tmp, sr=SR, mono=True)
    finally:
        try:
            _os.remove(_tmp)
        except OSError:
            pass
    if args.gain_db:
        y = y * (10.0 ** (args.gain_db / 20.0))
        if float(np.max(np.abs(y))) >= 1.0:
            raise SystemExit(f"gain {args.gain_db:+.1f} dB clips — lower it")
    dur_h = len(y) / SR / 3600
    print(f"[load] {dur_h:.2f}h @ {SR}Hz mono", flush=True)

    cues = parse_vtt(args.vtt)
    if excludes:
        before = len(cues)
        cues = [(st, en, txt) for (st, en, txt) in cues
                if not any(st < b and en > a for (a, b) in excludes)]
        dropped = before - len(cues)
        print(f"[vtt] excluded {dropped} cue(s) in "
              f"{','.join(f'{a:.2f}-{b:.2f}s' for a, b in excludes)} "
              f"(non-book/ASR regions)", flush=True)
        if dropped == 0:
            raise SystemExit("--exclude-ranges matched no cues — the ranges are "
                             "probably wrong for this VTT; refusing to continue")
    heads = heading_cue_idxs(cues, args.epub)
    print(f"[vtt] {len(cues)} cues, {len(heads)} heading cues", flush=True)

    if args.silence_map:
        gaps, speech = load_silence_map(args.silence_map, len(y), np)
        print(f"[gaps] auto-editor silence map: {len(gaps)} silences from {args.silence_map}", flush=True)
    else:
        gaps, speech = silence_analysis(y, np, librosa)
    strong = [(g0, g1) for (g0, g1) in gaps if (g1 - g0) / SR >= STRONG_SIL]
    print(f"[gaps] {len(gaps)} silences, {len(strong)} strong (>= {STRONG_SIL}s)",
          flush=True)

    parity_after = dquote_open_after(cues) if args.dialogue_aware else None
    boundaries = build_boundaries(cues, strong, heads, np, snap_window=args.snap_window,
                                  parity_after=parity_after)
    n_struct = sum(1 for b in boundaries if b["struct"])
    n_unsafe = sum(1 for b in boundaries if not b.get("quote_safe", True))
    print(f"[plan] {len(boundaries)} sentence cut points, {n_struct} structural"
          + (f", {n_unsafe} mid-quote (won't be cut on)" if args.dialogue_aware else ""),
          flush=True)

    first_speech = int(speech[0][0]) if speech else 0
    spans, plan_stats = plan_clips(boundaries, first_speech, sampler, max_clip,
                                   dialogue_aware=args.dialogue_aware)
    print(f"[plan] {len(spans)} raw clips"
          + (f" (dialogue-aware: {plan_stats['deferred']} boundaries kept whole, "
             f"{plan_stats['forced_splits']} quotes force-split at max_clip)"
             if args.dialogue_aware else ""), flush=True)

    # even-spread subset to target minutes
    keep_idx = list(range(len(spans)))
    if args.target_minutes and not args.dry_run:
        total = sum((b - a) / SR for (a, b, *_ ) in spans)
        tgt = args.target_minutes * 60
        if total > tgt:
            n_keep = max(1, int(round(len(spans) * tgt / total)))
            keep_idx = sorted(set(int(round(x)) for x in
                                  np.linspace(0, len(spans) - 1, n_keep)))
            print(f"[subset] {total/60:.0f}min -> keep {len(keep_idx)}/{len(spans)} "
                  f"(~{args.target_minutes:.0f}min even-spread)", flush=True)

    out = Path(args.out_dir)
    out_wavs = out / "wavs"
    out_wavs.mkdir(parents=True, exist_ok=True)

    rows, meta, leftovers = [], [], []
    kept = 0
    excluded_clips = 0
    mid_capped_total = 0                 # lingering mid-sentence (SFX-style) gaps compressed
    mid_gap_max = 0.0                    # longest such gap found (s)
    for oi in keep_idx:
        a, b, struct_end, gstart = spans[oi]
        a = max(0, int(a) - int(LEAD_PAD * SR))     # hair of lead-in
        if struct_end:                               # cap trailing pause at a wall
            b = min(int(b), int(gstart) + int(STRUCT_TRAIL_CAP * SR))
        a, b = int(a), int(b)
        dur = (b - a) / SR
        if dur < MIN_CLIP or dur > max_clip + 0.5:
            continue
        # A clip may not OVERLAP an excluded range at all — cue filtering alone
        # is not enough: a span can swallow an excluded sentence's AUDIO while
        # cue_text_in only sees the neighbors' text (audio/text mismatch).
        if any(a / SR < xb and b / SR > xa for (xa, xb) in excludes):
            excluded_clips += 1
            continue
        txt = cue_text_in(cues, a / SR, b / SR)
        if len(txt) < 3:
            continue
        txt = normalize_scripture(txt)          # Bible refs -> spoken words FIRST
        txt, lo = expand_digits(txt)            # then any remaining bare numbers
        leftovers.extend(lo)
        clip = y[a:b]
        if max_internal_sil is not None:
            # Compress overlong pauses INSIDE the clip, sentence-awarely: long
            # paragraph/context breaks between sentences -> max_internal_sil; any
            # mid-sentence gap (e.g. a removed sound-effect gap) -> the tighter
            # max_mid_sil. Done BEFORE trail_cap (which handles only the tail). The
            # sentence boundaries are the sentence-final cue ends inside this clip.
            cue_ends_rel = [en - a / SR for (st, en, tx) in cues
                            if a / SR <= (st + en) / 2 < b / SR
                            and tx.rstrip().endswith(SENT_FINAL)]
            clip, _nmid, _mxmid = cap_internal_silences(
                clip, cue_ends_rel, max_internal_sil, max_mid_sil, np, librosa)
            mid_capped_total += _nmid
            if _mxmid > mid_gap_max:
                mid_gap_max = _mxmid
        if trail_cap is not None:
            # Trim trailing silence to at most trail_cap. Uncapped training tails
            # taught Orpheus runaway silence (proven: owen trimmed=0 runaway vs
            # rohan uncapped=10%); the end-of-sentence pause is added DETERMINISTICALLY
            # at generation instead, so it never has to be learned from open-ended
            # silence in the data.
            if args.silence_map:
                # auto-editor map (absolute threshold) reliably finds the trailing
                # silence: last-speech-end = start of the last silence that begins
                # inside this clip. librosa's RELATIVE top_db can read a quiet clip's
                # trailing breath as speech and leave a long tail (proven on clip #8).
                last_sp = b
                for (g0, g1) in gaps:
                    if g0 >= b:
                        break
                    if g0 > a and g1 > a:
                        last_sp = g0
                clip = clip[:min(len(clip), (int(last_sp) - a) + int(trail_cap * SR))]
            else:
                iv = librosa.effects.split(clip, top_db=oo.RECUT_TOP_DB,
                                           frame_length=2048, hop_length=512)
                if len(iv):
                    clip = clip[:min(len(clip), int(iv[-1][1]) + int(trail_cap * SR))]
            trail = trail_cap
        else:
            trail = (b - gstart) / SR if not struct_end else STRUCT_TRAIL_CAP
        dur = len(clip) / SR
        if dur < MIN_CLIP:
            continue
        fn = f"{args.source_name}_{str(kept).zfill(8)}.wav"
        sf.write(str(out_wavs / fn), clip, SR)
        rows.append((f"wavs/{fn}", txt))
        meta.append({"file": f"wavs/{fn}", "src_start": round(a / SR, 3),
                     "src_end": round(b / SR, 3), "duration": round(dur, 2),
                     "trailing_pause": round(max(0.0, trail), 2),
                     "struct_end": bool(struct_end), "text": txt})
        kept += 1
        if args.dry_run and kept >= args.dry_run:
            break

    if excluded_clips:
        print(f"[exclude] dropped {excluded_clips} clip span(s) overlapping excluded ranges",
              flush=True)
    if not rows:
        raise SystemExit("no clips emitted — check inputs")

    # evenly-spread eval split
    n_eval = max(1, int(round(len(rows) * args.eval_frac)))
    eval_idx = set(int(round(x)) for x in np.linspace(0, len(rows) - 1, n_eval))

    def write_csv(path, idxs):
        # Proper CSV (csv.writer) — NOT a raw f-string. Prose text contains "
        # (dialogue); an unquoted "..." that spans a clip boundary makes the
        # csv.reader in read_metadata() swallow following rows until the quote
        # balances, corrupting text<->audio pairing. QUOTE_MINIMAL quotes only
        # the fields that need it so the reader round-trips exactly.
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="|", quoting=csv.QUOTE_MINIMAL,
                           lineterminator="\n")
            w.writerow(["audio_file", "text", "speaker_name"])
            for i in idxs:
                rel, txt = rows[i]
                w.writerow([rel, txt.replace("|", "/"), args.source_name])
    write_csv(out / "metadata_train.csv",
              [i for i in range(len(rows)) if i not in eval_idx])
    write_csv(out / "metadata_eval.csv", sorted(eval_idx))
    json.dump(meta, open(out / "clips.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    durs = [c["duration"] for c in meta]
    trails = [c["trailing_pause"] for c in meta]
    hist = {}
    for d in durs:
        k = f"{int(d // 5) * 5}-{int(d // 5) * 5 + 5}s"
        hist[k] = hist.get(k, 0) + 1
    lines = [
        "# Cut report (cut_audiobook.py — VTT + waveform-snap)", "",
        f"- Clips: **{len(durs)}**, total **{sum(durs)/60:.1f} min**",
        f"- Duration: min {min(durs):.1f}s / median {sorted(durs)[len(durs)//2]:.1f}s "
        f"/ max {max(durs):.1f}s  (mix {args.mix}, cap {max_clip}s, seed {args.seed})",
        f"- Duration histogram: " + ", ".join(f"{k}: {v}" for k, v in sorted(hist.items())),
        f"- Trailing pause kept: median {sorted(trails)[len(trails)//2]:.2f}s "
        f"/ max {max(trails):.2f}s (structural ends capped at {STRUCT_TRAIL_CAP}s)",
        f"- Structural boundaries: {sum(1 for c in meta if c['struct_end'])} clips end at a chapter/section wall",
        (f"- Dialogue-aware: ON — {plan_stats['deferred']} clip boundaries deferred to keep "
         f"a quotation whole; {plan_stats['forced_splits']} quote(s) force-split at max_clip"
         if args.dialogue_aware else "- Dialogue-aware: OFF (cut purely on duration)"),
        (f"- Internal-silence cap: between-sentence <= {max_internal_sil}s, "
         f"mid-sentence <= {max_mid_sil}s. **Mid-sentence gaps compressed "
         f"(lingering SFX-style): {mid_capped_total}** (longest found "
         f"{mid_gap_max:.2f}s)." if max_internal_sil is not None
         else "- Internal-silence cap: OFF (pauses kept verbatim)"),
        f"- Sample rate: {SR} Hz mono (Orpheus/SNAC)",
        f"- Train/eval: {len(rows)-n_eval}/{n_eval}",
        f"- Unexpanded digit tokens: {sorted(set(leftovers)) if leftovers else 'none'}",
        "", f"Structural gap threshold {STRUCT_GAP_SECONDS}s; snap window {SNAP_WINDOW}s "
        f"(cue timestamps are approximate — every cut is silence-snapped).",
    ]
    (out / "CUT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[done] {out}: {len(durs)} clips, {sum(durs)/60:.1f} min"
          + (f", mid-sentence gaps compressed: {mid_capped_total} (longest {mid_gap_max:.2f}s)"
             if max_internal_sil is not None else ""), flush=True)
    try:
        import run_metrics
        run_metrics.record("cut", {"source": args.source_name, "vtt": Path(args.vtt).name,
                                   "clips": len(durs), "minutes": round(sum(durs)/60, 1),
                                   "gain_db": args.gain_db, "elapsed_s": round(_time.time() - _t0, 1)})
    except Exception:
        pass


if __name__ == "__main__":
    main()
