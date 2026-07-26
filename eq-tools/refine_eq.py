"""Iteration 2: measure residual (ref - matched) in the same 27 bands, add it to the
original gain plan, re-render from ORIGINALS."""
import subprocess, sys, os, json, re
import numpy as np
sys.path.insert(0, os.path.dirname(sys.argv[0]))
from match_eq import decode, speech_psd, band_levels, smooth, edges  # reuse

D = sys.argv[1]; MATCHED = os.path.join(D, 'matched')
TARGET_LUFS = -19.7; CLAMP = 6.0
centers = np.sqrt(edges[:-1]*edges[1:])
plan = json.load(open(os.path.join(os.path.dirname(sys.argv[0]),'eq_plan.json')))

rp, rf = speech_psd(decode(os.path.join(D,'deathstalker legacy.wav')))
ref = band_levels(rp, rf)

for orig_path, entries in plan.items():
    name = os.path.basename(orig_path)
    mp, mf = speech_psd(decode(os.path.join(MATCHED, name)))
    mb = band_levels(mp, mf)
    resid = ref - mb
    resid -= resid[(centers>=630)&(centers<1250)].mean()
    old = np.array([float(x) for x in re.findall(r'entry\([0-9.]+,(-?[0-9.]+)\)', entries)])
    total = np.clip(old + smooth(resid, k=2), -CLAMP, CLAMP)
    ent = ';'.join(f"entry({c:.0f},{g:.2f})" for c,g in zip(centers, total))
    plan[orig_path] = ent
    eq = f"firequalizer=gain_entry='{ent}'"
    r = subprocess.run(['ffmpeg','-v','info','-i',orig_path,'-af',f"{eq},ebur128=framelog=quiet",'-f','null','-'],
                       capture_output=True, text=True)
    lufs = float(re.search(r'Integrated loudness:\s*\n\s*I:\s*(-?[0-9.]+) LUFS', r.stderr).group(1))
    subprocess.run(['ffmpeg','-v','error','-y','-i',orig_path,'-af',f"{eq},volume={TARGET_LUFS-lufs:.2f}dB",
                    '-c:a','pcm_s24le', os.path.join(MATCHED, name)], check=True)
    print(f"[v2] {name}: resid max {np.abs(resid).max():.1f} dB, trim {TARGET_LUFS-lufs:+.2f} dB")

json.dump(plan, open(os.path.join(os.path.dirname(sys.argv[0]),'eq_plan.json'),'w'), indent=1)
