#!/usr/bin/env python
"""Rebuild ALL Orpheus narrator corpora with the 2026-07-19 principles baked in.

Findings this build encodes (measured, not theorized):
  1. EOS margin needs a genuine room-hiss bed at -65 dBFS with random offsets
     (v5 result: full book 0 runaways).
  2. THE RINGING (KA/Ghostworld 8.4 kHz comb, 46% of speech windows): SNAC
     round-trip ADDS narrowband tones (8423 Hz +10-11 dB, 7705, 7210, 10365)
     when rendering low-level HF hiss. The bed taught the model to emit hiss
     codes under speech -> comb everywhere. Old no-bed model = clean.
     FIX: deny the codec HF hiss to render. LP the bed at 6.5 kHz; LP speech
     at 6.5 kHz for band-limited sources (MM rolls off ~4.5k, thirdreich ~4.9k)
     which also kills their real HF birdies (MM 8.25-9.5k x10-15% of clips,
     thirdreich 10-10.75k x20%). Mistborn AoL speech is genuinely wideband
     (8.7k) and scanned clean -> left full-band.
  3. The ~99 Hz hum users had to remove in reassembly came in with the bed:
     HP the bed at 120 Hz. (Beds only — never HP speech; the narrator f0
     lives down there.)
  4. Pause tails (punctuation-scaled 0.40s / 0.20s of bed) so the model emits
     its own inter-chunk pause. Validated at 120-clip scale: 480ms median
     self-generated tail, gates clean.
  5. No pause-capping: v5 shipped bed-only on 31%-mass clips and ran a full
     book clean, so the bed is the load-bearing fix; caps stay out.

Post-build self-verification per corpus: duration ceiling, HF tone scan,
bed-liveness. CPU only.
"""
import csv, os, sys
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt, resample_poly, iirnotch, filtfilt
from math import gcd

FT = "/home/telltale/xtts_ft"
BEDS = "/home/telltale/beds"
LEVEL = 10 ** (-65 / 20)
TAIL_FULL, TAIL_HALF = 0.40, 0.20
CEIL_S = 19.9          # hard clip+tail ceiling (training drops rows > 20s)
SR = 24000

LP_SOS = butter(8, 6500, "lowpass", fs=SR, output="sos")
HP_SOS = butter(4, 120, "highpass", fs=SR, output="sos")

os.makedirs(BEDS, exist_ok=True)


def filter_bed(y):
    """THE E RECIPE bed (locked by Owen's ear, 2026-07-20, five-arm sprint):
    HP 120 Hz (kills the rumble users had to de-hum) + notch any narrowband
    whine (raw tr bed carried +11 dB lines at 4430/4786) — and then scale to
    RMS -6 dBFS, NOT unit. Renormalizing to unit packs the removed rumble's
    energy into the audible band (arm B: +6 dB hiss, "too loud"; arm D same
    mechanism milder). -6 dBFS file RMS × the -65 dB application = the exact
    in-band hiss level of arm A/E, which Owen signed off as "no noise problem".
    NO low-pass: banned (arm B/v7) — and the SNAC comb it targeted is handled
    by per-model output notch maps instead.
    """
    y = sosfiltfilt(HP_SOS, y)
    for _ in range(8):
        peaks = tone_check(y, SR, fmin=300, fmax=11500, prom=6)
        if not peaks:
            break
        f0 = peaks[0][0]
        b, a = iirnotch(f0, 30, fs=SR)
        y = filtfilt(b, a, y)
    y = y * (10 ** (-6.0 / 20) / (np.sqrt(np.mean(y ** 2)) + 1e-12))
    return y.astype(np.float32)


def tone_check(y, sr, fmin=800, fmax=11500, prom=5):
    NF = 16384
    w = np.hanning(NF)
    m = (len(y) - NF) // (NF // 2) + 1
    ps = np.zeros(NF // 2 + 1)
    for j in range(min(m, 4000)):
        s = y[j * NF // 2: j * NF // 2 + NF] * w
        ps += np.abs(np.fft.rfft(s)) ** 2
    db = 10 * np.log10(ps + 1e-20)
    f = np.fft.rfftfreq(NF, 1 / sr)
    lo, hi = np.searchsorted(f, fmin), np.searchsorted(f, min(fmax, sr / 2 - 200))
    band = db[lo:hi]
    bw = int(400 / (sr / NF)); ex = int(60 / (sr / NF))
    out = []
    for i in range(bw, len(band) - bw):
        v = band[i]
        neigh = np.concatenate([band[i - bw:i - ex], band[i + ex:i + bw]])
        if v - np.median(neigh) >= prom and v == band[max(0, i - ex):i + ex + 1].max():
            out.append((round(f[lo + i]), round(v - np.median(neigh), 1)))
    return sorted(out, key=lambda t: -t[1])[:8]


def harvest_bed(src_path, out_path, max_read_s=2400, lo_db=-58, hi_db=-45,
                target_s=90.0):
    """Genuine room tone from a source's quiet regions, 10ms crossfaded,
    HP120+LP6500, unit RMS."""
    info = sf.info(src_path)
    frames = min(info.frames, int(max_read_s * info.samplerate))
    y, sr = sf.read(src_path, frames=frames)
    if y.ndim > 1:
        y = y.mean(1)
    if sr != SR:
        g = gcd(SR, sr)
        y = resample_poly(y, SR // g, sr // g)
        sr = SR
    y = y.astype(np.float32)
    w = int(0.05 * sr)
    m = len(y) // w
    rms = np.sqrt(np.mean(y[:m * w].reshape(m, w) ** 2, axis=1))
    db = 20 * np.log10(rms + 1e-12)
    for widen in (0, 3, 6):
        sel = (db >= lo_db - widen) & (db <= hi_db + widen)
        # merge adjacent windows into runs of >= 4 windows (200ms) to avoid
        # speech-skirt slivers
        segs, i = [], 0
        while i < m:
            if sel[i]:
                j = i
                while j < m and sel[j]:
                    j += 1
                if j - i >= 4:
                    segs.append((i * w, j * w))
                i = j
            else:
                i += 1
        total = sum(b - a for a, b in segs) / sr
        if total >= target_s:
            break
    if total < 30:
        raise SystemExit(f"FATAL bed harvest from {src_path}: only {total:.0f}s "
                         f"of quiet regions found — refusing to build a thin bed")
    # Equal-power OVERLAP crossfade. The first cut concatenated ramped pieces,
    # leaving a ~20ms V-dip to zero at every boundary — 245 dips read as a
    # -82 dBFS quiet floor, i.e. periodic dead-ish air inside the very bed
    # whose job is to keep silence spectrally alive.
    xf = int(0.010 * sr)
    t = np.linspace(0, np.pi / 2, xf, dtype=np.float32)
    fin, fout = np.sin(t), np.cos(t)
    bed = None
    for a, b in segs:
        s = y[a:b].copy()
        if bed is None:
            bed = s
        else:
            bed[-xf:] = bed[-xf:] * fout + s[:xf] * fin
            bed = np.concatenate([bed, s[xf:]])
    bed = filter_bed(bed)
    sf.write(out_path, bed, SR, subtype="FLOAT")
    print(f"[bed] {out_path}: {len(bed)/SR:.0f}s from {len(segs)} regions "
          f"({total:.0f}s raw) | residual tones: {tone_check(bed, SR) or 'NONE'}",
          flush=True)
    return bed


def notch_birdies(y, sr):
    """Surgically notch MEASURED narrowband tones 6-12k in a speech clip.

    Replaces the 6.5k speech low-pass (2026-07-19 night): rolloff-99 said MM had
    no content above ~4.5k, but the human source measures -7dB at 6.5-8k — the
    LP amputated real sibilance/air and Owen heard it instantly ("SUPER
    muffled"), while the codec comb just moved down and survived. Energy
    percentile != perceptual bandwidth. Narrow notches at detected birdie
    frequencies remove what a model could learn without touching brightness.
    """
    peaks = tone_check(y, sr, fmin=6000, fmax=11800, prom=8)
    for f0, _ in peaks[:3]:
        b, a = iirnotch(f0, 40, fs=sr)
        y = filtfilt(b, a, y).astype(np.float32)
    return y, [round(f) for f, _ in peaks[:3]]


def tail_len(text):
    t = text.rstrip()
    if t.endswith(('"', "'", "”", "’")):
        t = t[:-1].rstrip()
    return TAIL_HALF if t.endswith((",", ";", ":", "—", "-")) else TAIL_FULL


def build(voice, src, dst, bed, lp_speech, seed):
    # lp_speech is retired — kept in the signature so old call sites fail loud
    # if someone passes True again. Speech cleaning = notch_birdies only.
    assert lp_speech is False, "speech low-pass is BANNED (muffles the voice; 2026-07-19)"
    rng = np.random.default_rng(seed)

    def bed_slice(n):
        if n >= len(bed):
            parts, need = [], n
            while need > 0:
                off = rng.integers(0, len(bed) // 2)
                take = min(need, len(bed) - off)
                parts.append(bed[off:off + take])
                need -= take
            return np.concatenate(parts)[:n] * LEVEL
        off = rng.integers(0, len(bed) - n)
        return bed[off:off + n] * LEVEL

    os.makedirs(f"{dst}/wavs", exist_ok=True)
    stats = {"clips": 0, "tail_added_s": 0.0, "no_tail": 0, "trimmed_tail": 0,
             "half": 0, "durs": []}
    for split in ("train", "eval"):
        rows = list(csv.reader(open(f"{src}/metadata_{split}.csv"), delimiter="|"))
        out = [rows[0]]
        for k, r in enumerate(rows[1:]):
            y, sr = sf.read(os.path.join(src, r[0]))
            if y.ndim > 1:
                y = y.mean(1)
            assert sr == SR, f"{r[0]} is {sr}"
            y = y.astype(np.float32)
            if lp_speech:
                y = sosfiltfilt(LP_SOS, y).astype(np.float32)
            # bed under the whole clip (the EOS fix)
            y = y + bed_slice(len(y))
            # punctuation-scaled tail of the same bed, capped at the ceiling
            tl = tail_len(r[1])
            if tl == TAIL_HALF:
                stats["half"] += 1
            room = CEIL_S - len(y) / sr
            if room <= 0.05:
                tl = 0.0
                stats["no_tail"] += 1
            elif tl > room:
                tl = room
                stats["trimmed_tail"] += 1
            if tl > 0:
                y = np.concatenate([y, bed_slice(int(tl * sr))])
                stats["tail_added_s"] += tl
            np.clip(y, -1.0, 1.0, out=y)
            fn = ("e_" if split == "eval" else "") + os.path.basename(r[0])
            sf.write(os.path.join(dst, "wavs", fn), y, sr)
            out.append([f"wavs/{fn}", r[1], r[2]])
            stats["clips"] += 1
            stats["durs"].append(len(y) / sr)
            if (k + 1) % 400 == 0:
                print(f"[{voice}/{split}] {k+1}/{len(rows)-1}", flush=True)
        with open(f"{dst}/metadata_{split}.csv", "w", newline="") as f:
            csv.writer(f, delimiter="|").writerows(out)
        print(f"[{voice}/{split}] {len(out)-1} clips written", flush=True)

    d = np.array(stats["durs"])
    print(f"[{voice}] DONE: {stats['clips']} clips | +{stats['tail_added_s']/60:.1f} min tails "
          f"({stats['half']} half, {stats['no_tail']} skipped, {stats['trimmed_tail']} trimmed) | "
          f"dur median {np.median(d):.1f}s p99 {np.percentile(d,99):.1f}s max {d.max():.1f}s | "
          f">20s: {(d>20).sum()}", flush=True)

    # verification: HF tones must be gone from a sample; bed must be alive
    flagged = 0
    freqs = []
    files = sorted(os.listdir(f"{dst}/wavs"))[::max(1, stats["clips"] // 40)][:40]
    floor = []
    for fn in files:
        y, sr = sf.read(os.path.join(dst, "wavs", fn))
        t = [x for x in tone_check(y, sr, fmin=6000, prom=8)]
        if t:
            flagged += 1
            freqs += [x[0] for x in t]
        w = int(0.05 * sr)
        mm = len(y) // w
        rms = np.sqrt(np.mean(y[:mm * w].reshape(mm, w) ** 2, axis=1))
        floor.append(20 * np.log10(rms.min() + 1e-12))
    print(f"[{voice}] VERIFY: HF tones 6-12k flagged {flagged}/{len(files)} "
          f"{sorted(set(freqs))[:8] if freqs else ''} | quiet-floor median "
          f"{np.median(floor):.0f} dBFS (want ~-65, NOT < -80)", flush=True)


which = sys.argv[1] if len(sys.argv) > 1 else "all"

# ---- beds -------------------------------------------------------------
if which in ("all", "beds"):
    y, bsr = sf.read("/home/telltale/mm_hiss_bed.wav")
    assert bsr == SR
    if y.ndim > 1:
        y = y.mean(1)
    mm_bed = filter_bed(y)
    sf.write(f"{BEDS}/mm_bed_v2.wav", mm_bed, SR, subtype="FLOAT")
    print(f"[bed] mm_bed_v2: filtered existing 60s bed | residual tones: "
          f"{tone_check(mm_bed, SR) or 'NONE'}", flush=True)
    aol_bed = harvest_bed("/mnt/e/training/mistborn/source/alloy of law/chapter 1.wav",
                          f"{BEDS}/aol_bed_v2.wav")
    tr_bed = harvest_bed("/mnt/e/training/thirdreich/source/third reich FULL merged.flac",
                         f"{BEDS}/tr_bed_v2.wav")
else:
    mm_bed = sf.read(f"{BEDS}/mm_bed_v2.wav")[0].astype(np.float32)
    aol_bed = sf.read(f"{BEDS}/aol_bed_v2.wav")[0].astype(np.float32)
    tr_bed = sf.read(f"{BEDS}/tr_bed_v2.wav")[0].astype(np.float32)

# ---- corpora (E-recipe names; v7/v2_tails corpora are CONDEMNED — muffled) --
if which in ("all", "deathstalker"):
    build("deathstalker", f"{FT}/deathstalker_mm8_v2", f"{FT}/deathstalker_mm8_v8_tails",
          mm_bed, lp_speech=False, seed=11)
if which in ("all", "mistborn"):
    build("mistborn", f"{FT}/mistborn_aol8", f"{FT}/mistborn_aol8_v3_tails",
          aol_bed, lp_speech=False, seed=23)
if which in ("all", "thirdreich"):
    build("thirdreich", f"{FT}/thirdreich_np_8h", f"{FT}/thirdreich_np8h_v3_tails",
          tr_bed, lp_speech=False, seed=37)

print("REBUILD_ALL_DONE", flush=True)
