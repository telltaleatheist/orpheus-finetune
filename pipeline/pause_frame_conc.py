#!/usr/bin/env python
"""Pause-frame concentration from FULL-CLIP SNAC encodes — the proven EOS
instrument (2026-07-19). Encoding isolated segments gives the WRONG answer;
always encode the whole clip and mask pause frames afterwards.

Benchmarks: dead-silence pauses 27-44% top-3 share (attractor, breaks EOS),
v5 genuine hiss bed 1.4%, healthy novels corpus ~5%.

Frame = the full 7-code tuple (l1[i], l2[2i:2i+2], l3[4i:4i+4]), one per
2048 samples (85.3ms) at 24 kHz. Pause frame = audio RMS < -45 dBFS.
Usage: pause_frame_conc.py <corpus_wavs_dir> [n_clips]
"""
import glob, os, sys
from collections import Counter
import numpy as np, soundfile as sf
import torch
from snac import SNAC

d = sys.argv[1]
n_clips = int(sys.argv[2]) if len(sys.argv) > 2 else 20
model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()

files = sorted(glob.glob(os.path.join(d, "*.wav")))
files = files[::max(1, len(files) // n_clips)][:n_clips]
FR = 2048
cnt = Counter()
npause = 0
for p in files:
    y, sr = sf.read(p)
    if y.ndim > 1:
        y = y.mean(1)
    assert sr == 24000
    y = y.astype(np.float32)
    with torch.inference_mode():
        codes = model.encode(torch.from_numpy(y)[None, None])
    l1, l2, l3 = [c[0].tolist() for c in codes]
    nfr = min(len(l1), len(l2) // 2, len(l3) // 4, len(y) // FR)
    for i in range(nfr):
        seg = y[i * FR:(i + 1) * FR]
        if np.sqrt(np.mean(seg ** 2)) < 10 ** (-45 / 20):
            key = (l1[i], tuple(l2[2 * i:2 * i + 2]), tuple(l3[4 * i:4 * i + 4]))
            cnt[key] += 1
            npause += 1
top3 = sum(c for _, c in cnt.most_common(3))
name = os.path.basename(os.path.dirname(d.rstrip("/")) if d.rstrip("/").endswith("wavs") else d.rstrip("/"))
print(f"{name}: {len(files)} clips, {npause} pause frames, distinct {len(cnt)}, "
      f"top-3 share {100 * top3 / max(npause, 1):.1f}%  "
      f"(dead=27-44% | v5 bed=1.4% | novels=5%)", flush=True)
