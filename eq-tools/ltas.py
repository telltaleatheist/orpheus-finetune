import subprocess, sys, numpy as np

SR = 24000
def ltas(path):
    # decode to mono float32 @24k via ffmpeg pipe
    p = subprocess.run(['ffmpeg','-v','error','-i',path,'-ac','1','-ar',str(SR),'-f','f32le','-'],
                       capture_output=True)
    x = np.frombuffer(p.stdout, dtype=np.float32)
    n = 4096; hop = 2048
    nfrm = (len(x)-n)//hop
    win = np.hanning(n)
    # frame energies for speech gate
    frames = np.lib.stride_tricks.as_strided(x, (nfrm,n), (x.strides[0]*hop, x.strides[0]))
    rms = np.sqrt((frames**2).mean(axis=1))
    gate = np.percentile(rms[rms>0], 60)   # keep top-40% energy frames = confident speech
    keep = np.where(rms >= gate)[0][:40000]
    psd = np.zeros(n//2+1)
    for i in keep:
        s = np.fft.rfft(frames[i]*win)
        psd += (s.real**2 + s.imag**2)
    psd /= len(keep)
    freqs = np.fft.rfftfreq(n, 1/SR)
    return freqs, psd

BANDS = [(80,160),(160,315),(315,630),(630,1250),(1250,2500),(2500,5000),(5000,8000),(8000,11500)]
names = sys.argv[1:]
rows = {}
for path in names:
    f, p = ltas(path)
    row = []
    for lo,hi in BANDS:
        m = (f>=lo)&(f<hi)
        row.append(10*np.log10(p[m].mean()+1e-20))
    rows[path] = np.array(row)
# print relative to per-file 630-1250 band (mid anchor) so shape differences pop
hdr = "file".ljust(20) + "".join(f"{lo}-{hi}".rjust(11) for lo,hi in BANDS)
print(hdr)
for path,row in rows.items():
    anchored = row - row[3]
    name = path.split('/')[-1].replace('.wav','')[:19]
    print(name.ljust(20) + "".join(f"{v:11.1f}" for v in anchored))
