"""Match each input file's speech-band spectrum to a reference (LTAS matching).
Emits per-file ffmpeg firequalizer gain_entry strings + runs the EQ pass."""
import subprocess, sys, os, json
import numpy as np

SR = 24000          # analysis rate (12 kHz Nyquist covers narration)
NFFT, HOP = 4096, 2048
MAX_FRAMES = 40000
CLAMP_DB = 6.0

def decode(path, sr=SR):
    p = subprocess.run(['ffmpeg','-v','error','-i',path,'-ac','1','-ar',str(sr),'-f','f32le','-'],
                       capture_output=True)
    return np.frombuffer(p.stdout, dtype=np.float32)

def speech_psd(x):
    nfrm = (len(x)-NFFT)//HOP
    frames = np.lib.stride_tricks.as_strided(x, (nfrm,NFFT), (x.strides[0]*HOP, x.strides[0]))
    rms = np.sqrt((frames**2).mean(axis=1))
    gate = np.percentile(rms[rms>0], 60)
    keep = np.where(rms >= gate)[0]
    if len(keep) > MAX_FRAMES:
        keep = keep[np.linspace(0, len(keep)-1, MAX_FRAMES).astype(int)]
    win = np.hanning(NFFT)
    psd = np.zeros(NFFT//2+1)
    for i in keep:
        s = np.fft.rfft(frames[i]*win)
        psd += (s.real**2 + s.imag**2)
    return psd/len(keep), np.fft.rfftfreq(NFFT, 1/SR)

# ~1/6-octave log bands, 60 Hz - 11.5 kHz
edges = np.geomspace(60, 11500, 28)
def band_levels(psd, freqs):
    out = []
    for lo,hi in zip(edges[:-1], edges[1:]):
        m = (freqs>=lo)&(freqs<hi)
        out.append(10*np.log10(psd[m].mean()+1e-20))
    return np.array(out)

def smooth(g, k=3):
    pad = np.pad(g, k, mode='edge')
    return np.convolve(pad, np.ones(2*k+1)/(2*k+1), mode='valid')

if __name__ == '__main__':
    ref_path = sys.argv[1]
    targets = sys.argv[2:]
    print(f"[ref] {os.path.basename(ref_path)}")
    rp, rf = speech_psd(decode(ref_path))
    ref_bands = band_levels(rp, rf)
    centers = np.sqrt(edges[:-1]*edges[1:])
    
    plan = {}
    for path in targets:
        p, f = speech_psd(decode(path))
        b = band_levels(p, f)
        gain = ref_bands - b
        gain -= gain[(centers>=630)&(centers<1250)].mean()  # anchor mids: EQ shapes, volume pass levels
        gain = np.clip(smooth(gain), -CLAMP_DB, CLAMP_DB)
        entries = ';'.join(f"entry({c:.0f},{g:.2f})" for c,g in zip(centers, gain))
        plan[path] = entries
        print(f"[gain] {os.path.basename(path)}: " + " ".join(f"{g:+.1f}" for g in gain))
    with open(sys.argv[0].replace('match_eq.py','eq_plan.json'),'w') as fh:
        json.dump(plan, fh, indent=1)
    print("[ok] plan written")
