#!/usr/bin/env python
"""Dense independent VTT verification for training-data cutting.

Extends drift_check.py (Downloads/Deathstalker Collection): instead of 9 spots,
probes a window every STEP_S seconds across the whole file PLUS explicit extra
targets (chunk boundaries, truncated-chunk region). For each probed cue:
  - faster-whisper (small, CPU, int8) transcribes a 16s window around cue start
  - the cue's first ~6 normalized words are searched in the whisper word stream
  - offset = |whisper word time - cue start|
Prints per-probe lines for offenders (> WARN_S) and a summary:
  USABLE=<n> SKIPPED=<n> MEDOFF= P95OFF= MAXOFF=
Exit code 1 if MAXOFF > FAIL_S (so it can gate the pipeline).

Usage: python verify_vtt.py <vtt> <audio> [extra_target_sec ...]
"""
import sys, os, subprocess, re, statistics, tempfile

STEP_S = 120.0     # one probe every 2 minutes
WARN_S = 0.75      # report cues off by more than this
FAIL_S = 1.5       # gate: any usable cue off by more than this fails the run

vtt, audio = sys.argv[1], sys.argv[2]
extra = []
excludes = []
for arg in sys.argv[3:]:
    if arg.startswith("--exclude="):
        for part in arg[len("--exclude="):].split(","):
            a, b = part.split("-")
            excludes.append((float(a), float(b)))
    else:
        extra.append(float(arg))

def normw(s): return re.sub(r'[^a-z0-9]', '', s.lower())
def toks(s):
    # split hyphens/dashes/slashes BEFORE normalizing: the book writes
    # "fourteenth-century" as one token, ASR writes two — unmatchable otherwise
    s = re.sub(r"[-–—/]", " ", s)
    return [t for t in (normw(w) for w in s.split()) if t]
def ts(t):
    p = t.split(':'); h, m, s = (p if len(p) == 3 else ['0'] + p)
    return int(h)*3600 + int(m)*60 + float(s)
def fmt(sec):
    h = int(sec//3600); m = int(sec%3600//60); s = sec%60
    return f"{h}:{m:02d}:{s:05.2f}"

cues = []; L = open(vtt, encoding='utf-8').read().splitlines(); i = 0
while i < len(L):
    if '-->' in L[i]:
        a, _ = L[i].split('-->'); st = ts(a.strip()); txt = []; j = i+1
        while j < len(L) and L[j].strip() != '': txt.append(L[j]); j += 1
        cues.append((st, ' '.join(txt))); i = j
    else: i += 1
if not cues:
    print("USABLE=0 SKIPPED=0 MEDOFF=-1 P95OFF=-1 MAXOFF=-1"); sys.exit(1)

dur = cues[-1][0]
targets = sorted(set([round(t, 1) for t in
                      [x * STEP_S for x in range(1, int(dur // STEP_S) + 1)] + extra]))
print(f"[verify] {len(cues)} cues, span {fmt(dur)}, {len(targets)} probe targets", flush=True)

from faster_whisper import WhisperModel
m = WhisperModel("small", device="cpu", compute_type="int8")

wav = os.path.join(tempfile.gettempdir(), "verify_vtt_slice.wav")

def window_words(a, t):
    """Transcribe [a, a+t] and return [(abs_time, norm_token), ...]."""
    subprocess.run(["ffmpeg","-nostdin","-hide_banner","-v","error","-y","-ss",f"{a:.3f}",
                    "-t",f"{t:.3f}","-i",audio,"-ac","1","-ar","16000",wav], check=True)
    segs, _ = m.transcribe(wav, language="en", vad_filter=False, word_timestamps=True)
    words = []
    for s in segs:
        for w in (s.words or []):
            for n in toks(w.word):
                words.append((a + w.start, n))
    return words

def best_offset(words, tk, ct):
    """All positions where the opening tokens match; closest-to-claimed wins.
    (First-match anchoring gets stolen by e.g. '1 Peter 3:3' -> ASR 'First Peter'
    right before a sentence starting 'First,' — found the hard way.)"""
    matches = []
    for wi in range(len(words)):
        if words[wi][1] == tk[0]:
            mm = 1; k = wi+1
            while k < min(len(words), wi+12) and mm < len(tk):
                if words[k][1] == tk[mm]: mm += 1
                k += 1
            if mm >= max(2, len(tk)-1):
                matches.append(words[wi][0] - ct)
    return min(matches, key=abs) if matches else None

def onset_near(ct):
    """Nearest silence->speech onset to ct within +/-0.35s, from the WAVEFORM.
    Whisper stamps pause-preceded sentence starts up to ~2s early (ground-truthed
    on this book); the waveform edge is exact. Returns onset time or None."""
    import numpy as np
    a = max(0.0, ct - 1.5)
    pcm = subprocess.run(
        ["ffmpeg","-nostdin","-hide_banner","-v","error","-ss",f"{a:.3f}","-t","3.0",
         "-i",audio,"-ac","1","-ar","16000","-f","f32le","-"],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout
    y = np.frombuffer(pcm, dtype=np.float32)
    sr = 16000; win = int(0.020 * sr)
    n = len(y) // win
    if n < 20: return None
    rms = np.sqrt(np.mean(y[:n*win].reshape(n, win)**2, axis=1))
    thr = max(0.012, float(np.percentile(rms, 20)) * 1.8)
    k = 7   # 140ms of silence required before the onset
    onsets = [a + i*win/sr for i in range(k, n)
              if np.all(rms[i-k:i] < thr) and rms[i] >= thr*2]
    if not onsets: return None
    best = min(onsets, key=lambda t: abs(t-ct))
    return best if abs(best-ct) <= 0.35 else None

offs = []; skipped = 0; refined = 0; probed_cues = set()
def probe_ok(cue):
    """A cue is probe-friendly if its opening tokens are digit-free (whisper writes
    '35'/'5:17' where the book has 'thirty-five' — unmatchable without a full
    number normalizer), long enough to anchor, and not inside an excluded range
    (excluded cues are never cut into training clips — don't verify what won't train)."""
    st, text = cue
    if any(a <= st <= b for (a, b) in excludes):
        return False
    tk = toks(text)
    return len(tk) >= 10 and not any(re.search(r"\d", t) for t in tk[:6])

for T in targets:
    cand = [c for c in cues if probe_ok(c) and abs(c[0]-T) < 240 and c[0] not in probed_cues]
    if not cand:
        skipped += 1; continue
    ct, ctext = min(cand, key=lambda c: abs(c[0]-T))
    probed_cues.add(ct)
    tk = toks(ctext)[:6]
    off = best_offset(window_words(max(0.0, ct-12), 24), tk, ct)
    if off is not None and abs(off) > 1.0:
        # whisper's word timestamps skew when a long pause precedes the word in a
        # thin window (measured: 'First,' stamped 2s early into its own dramatic
        # pause). Confirm with a fresh window CENTERED on the claimed match —
        # generous context on both sides fixes the stamp. Keep the better reading.
        off2 = best_offset(window_words(max(0.0, ct + off - 15), 30), tk, ct)
        if off2 is not None and abs(off2) < abs(off):
            off = off2
    if off is not None and 1.0 < abs(off) <= 2.5:
        # word identity is confirmed nearby but the stamp is suspect (pause-preceded
        # start). If the cue sits on a real silence->speech onset, THAT is the
        # sentence start — measure at waveform precision instead of the ASR stamp.
        o = onset_near(ct)
        if o is not None:
            off = o - ct
            refined += 1
    if off is not None:
        offs.append(abs(off))
        if abs(off) > WARN_S:
            print(f"[OFF {off:+.2f}s] cue@{fmt(ct)}: {ctext[:80]}", flush=True)
    else:
        skipped += 1
        print(f"[no-match] cue@{fmt(ct)}: {ctext[:80]}", flush=True)

if offs:
    offs_sorted = sorted(offs)
    p95 = offs_sorted[min(len(offs)-1, int(round(0.95*(len(offs)-1))))]
    print(f"USABLE={len(offs)} SKIPPED={skipped} MEDOFF={statistics.median(offs):.2f} "
          f"P95OFF={p95:.2f} MAXOFF={max(offs):.2f} REFINED={refined}")
    sys.exit(0 if max(offs) <= FAIL_S else 1)
print(f"USABLE=0 SKIPPED={skipped} MEDOFF=-1 P95OFF=-1 MAXOFF=-1"); sys.exit(1)
