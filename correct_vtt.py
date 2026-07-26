"""correct_vtt.py — VTT fidelity corrector for paraphrased narration sources.

Problem: GraphicAudio Deathstalker fragments PARAPHRASE the novel. epub-align
emits the BOOK's exact prose, which is right where Rohan reads verbatim (majority,
correct proper nouns) but WRONG where the script reworded/embellished — training
poison. Also the highlight-reel montage overruns cue windows at scene cuts, and a
few cues are OTHER actors' dialogue.

This pass makes every cue's TEXT faithfully match its AUDIO, so the existing cut
pipeline (session.py) can consume the corrected VTT unchanged.

Per cue (whisper-large-v3 transcript of the cue's audio span, one pass):
  0. DROP cues containing quote marks -> other-actor dialogue guard.
  1. classify:  cov = LCS(epub, spoken)/len(epub);  ratio = len(spoken)/len(epub)
       verbatim  : cov>=0.85, ratio<=1.4  -> keep EPUB text (exact, names right)
       overrun   : cov>=0.85, ratio >1.4  -> keep EPUB text, TIGHTEN end to where
                                             the epub prose ends in the audio
       paraphrase: cov<0.85               -> use WHISPER text + epub name-glossary fix
  2. cues inside coverage 'audioNotInEpub' holes are whisper-fallback already ->
     force the paraphrase route (upgrades their text to large-v3 quality).

Outputs a corrected .vtt + a .correction.json report (every route/drop/name-fix
logged for review).

Usage:
  python correct_vtt.py --vtt in.vtt --audio in.wav --epub book.epub \
      --out out.vtt [--coverage in.coverage.json] [--report out.correction.json] \
      [--model large-v3] [--device cuda]
"""
import argparse, html.parser, json, os, re, sys, zipfile

# ------------------------- epub text + name glossary -------------------------

class _Strip(html.parser.HTMLParser):
    def __init__(self): super().__init__(); self.buf=[]; self.skip=0
    def handle_starttag(self,t,a):
        if t in ("script","style"): self.skip+=1
    def handle_endtag(self,t):
        if t in ("script","style") and self.skip: self.skip-=1
    def handle_data(self,d):
        if not self.skip: self.buf.append(d)

def epub_text(path):
    parts=[]
    with zipfile.ZipFile(path) as z:
        names=[n for n in z.namelist() if n.lower().endswith((".xhtml",".html",".htm"))]
        for n in sorted(names):
            try: raw=z.read(n).decode("utf-8","ignore")
            except Exception: continue
            p=_Strip(); p.feed(raw); parts.append(" ".join(p.buf))
    return re.sub(r"\s+"," "," ".join(parts))

_COMMON = set("""The A An And But Or Nor For Yet So He She It They We You I His Her Its
Their Our Your My Him Them Us Me When While Then There Here Now If As At In On Of To
By With From That This These Those What Who Whom Which Why How Not No Yes Well Oh Ah
Come Go Get Let Do Did Was Were Is Are Be Been Had Has Have Will Would Could Should
Can May Might Must Said One Two Three First Last Some All Any Each Every Only Even
Still Just Because After Before Once Again Very Most More Down Up Out Over Under""".split())

def _clean_name(w):
    """Strip possessive 's and stray trailing apostrophe; drop contractions."""
    w=re.sub(r"[’']s$","",w); w=w.rstrip("’'")
    return w

def build_glossary(text, min_count=2):
    """Proper nouns = Capitalized tokens NOT at sentence start, freq>=min_count,
    not common words. Also capture 2-word capitalized names (Mater Mundi)."""
    from collections import Counter
    initial=set()
    for sent in re.split(r"[.!?]+", text):
        m=re.search(r"[A-Za-z][A-Za-z'\-]+", sent)
        if m: initial.add(m.group(0))
    single=Counter()
    for w in re.findall(r"\S+", text):
        core=re.sub(r"^[^A-Za-z]+|[^A-Za-z'\-]+$","",w)
        if not core or "n't" in core.lower() or core.lower() in ("i'm","i'll","i'd","i've"):
            continue
        core=_clean_name(core)
        if len(core)>=3 and core[0].isupper() and core.lower()!=core and core not in _COMMON:
            single[core]+=1
    # words that appear LOWERCASE often are ordinary vocabulary, not names —
    # derived from the book itself so it needs no external dictionary
    lower=Counter(w.lower() for w in re.findall(r"[A-Za-z][A-Za-z']+", text)
                  if w[0].islower())
    common_lower={w for w,c in lower.items() if c>=5}
    names={w for w,c in single.items() if (c>=min_count and w not in initial) or c>=4}
    names |= {w for w,c in single.items()
              if ("'" in w or "’" in w or "-" in w) and c>=2}
    # a real name never shows up as a frequent lowercase word (drops Want/Palace/Block)
    names={w for w in names if w.lower() not in common_lower}
    # bigrams of adjacent capitalized non-common tokens (BB Chojiro, Mater Mundi)
    toks=re.findall(r"[A-Za-z][A-Za-z'\-]+", text); bigrams=Counter()
    for i in range(len(toks)-1):
        a,b=_clean_name(toks[i]),_clean_name(toks[i+1])
        if a and b and a[0].isupper() and b[0].isupper() and a not in _COMMON and b not in _COMMON \
           and a.lower() not in common_lower and b.lower() not in common_lower:
            bigrams[a+" "+b]+=1
    big={g for g,c in bigrams.items() if c>=min_count}
    return names, big, common_lower

# ------------------------------- vtt io -------------------------------

def ts_to_s(t):
    h,m,rest=t.split(":"); s,ms=rest.split("."); return int(h)*3600+int(m)*60+int(s)+int(ms)/1000
def s_to_ts(x):
    h=int(x//3600); x-=h*3600; m=int(x//60); x-=m*60; s=int(x); ms=int(round((x-s)*1000))
    if ms==1000: s+=1; ms=0
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

def parse_vtt(path):
    cues=[]; block=[]
    for line in open(path,encoding="utf-8"):
        line=line.rstrip("\n")
        if line=="":
            if block: cues.append(block); block=[]
        else: block.append(line)
    if block: cues.append(block)
    out=[]
    for b in cues:
        tl=next((l for l in b if "-->" in l),None)
        if not tl: continue
        a,bb=tl.split("-->"); a=a.strip(); bb=bb.strip().split()[0]
        txt=" ".join(b[b.index(tl)+1:]).strip()
        if txt: out.append({"start":ts_to_s(a),"end":ts_to_s(bb),"text":txt})
    return out

def write_vtt(path,cues):
    with open(path,"w",encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for i,c in enumerate(cues,1):
            f.write(f"{i}\n{s_to_ts(c['start'])} --> {s_to_ts(c['end'])}\n{c['text']}\n\n")

# ------------------------------- matching -------------------------------

def norm(t):
    t=t.lower().replace("-"," ")
    t=re.sub(r"[^a-z0-9\s']"," ",t)
    return re.sub(r"\s+"," ",t).strip().split()

def lcs_len(a,b):
    n,m=len(a),len(b); dp=[0]*(m+1)
    for i in range(1,n+1):
        prev=0
        for j in range(1,m+1):
            cur=dp[j]; dp[j]=prev+1 if a[i-1]==b[j-1] else max(dp[j],dp[j-1]); prev=cur
    return dp[m]

def classify(epub,spoken):
    e,s=norm(epub),norm(spoken)
    if not e or not s: return ("paraphrase",0.0,0.0)
    cov=lcs_len(e,s)/len(e); ratio=len(s)/len(e)
    if cov>=0.85 and ratio<=1.4: return ("verbatim",cov,ratio)
    if cov>=0.85:                return ("overrun",cov,ratio)
    return ("paraphrase",cov,ratio)

def tighten_end(epub, words):
    """Greedy-consume epub words through the timestamped hyp words; return the end
    time (relative to slice) of the last epub word, or None if not resolvable."""
    e=norm(epub); ei=0; last=None
    for w in words:
        wn=norm(w["word"])
        if wn and ei<len(e) and wn[0]==e[ei]:
            last=w["end"]; ei+=1
            if ei>=len(e): break
    return last if ei>=int(len(e)*0.85) else None

# ------------------------------- name fix -------------------------------

def edit_ratio(a,b):
    a,b=a.lower(),b.lower(); n,m=len(a),len(b)
    if not n or not m: return 1.0
    d=list(range(m+1))
    for i in range(1,n+1):
        prev=d[0]; d[0]=i
        for j in range(1,m+1):
            cur=d[j]; d[j]=min(d[j]+1,d[j-1]+1,prev+(a[i-1]!=b[j-1])); prev=cur
    return d[m]/max(n,m)

def fix_names(text, names, common_lower, log):
    """Conservative: replace a spoken token with a glossary name only on a close
    fuzzy match to an invented name. NEVER touches ordinary vocabulary (common_lower)
    so 'went','pale','they' are safe. Logs each change."""
    out=[]
    for tok in text.split():
        core=re.sub(r"^[^A-Za-z]+|[^A-Za-z'\-]+$","",tok)
        base=re.sub(r"[’']s$","",core).rstrip("’'")   # possessive/plural-strip for common check
        if len(core)<4 or core in names or core.lower() in common_lower or base.lower() in common_lower:
            out.append(tok); continue
        best=None;bestr=1.0
        for nm in names:
            if abs(len(nm)-len(core))>3: continue
            r=edit_ratio(core,nm)
            if r<bestr: bestr=r;best=nm
        # skip when the only difference is punctuation (Campbell's vs Campbells)
        strip=lambda w: re.sub(r"[^a-z0-9]","",w.lower())
        if best and strip(best)==strip(core):
            out.append(tok); continue
        # tight threshold; scale allowance with length (short words need near-exact)
        thr = 0.20 if len(core)<=5 else 0.30
        if best and 0<bestr<=thr and best.lower()!=core.lower():
            out.append(tok.replace(core,best)); log.append(f"{core} -> {best} ({bestr:.2f})")
        else:
            out.append(tok)
    return " ".join(out)

# ------------------------------- main -------------------------------

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--vtt",required=True); ap.add_argument("--audio",required=True)
    ap.add_argument("--epub",required=True); ap.add_argument("--out",required=True)
    ap.add_argument("--coverage"); ap.add_argument("--report")
    ap.add_argument("--model",default="large-v3"); ap.add_argument("--device",default="cuda")
    ap.add_argument("--min-clip",type=float,default=0.0,help="skip cues shorter than this (s)")
    ap.add_argument("--review",help="write dropped-but-maybe-usable cues here (whisper text + reason)")
    ap.add_argument("--min-logprob",type=float,default=-1.0,help="drop whisper-routed cue below this avg_logprob (stricter=-0.7)")
    ap.add_argument("--max-no-speech",type=float,default=0.6,help="drop whisper-routed cue above this no_speech_prob")
    ap.add_argument("--shrink-ratio",type=float,default=0.55,help="drop if whisper words < this * epub words (truncation/halluc)")
    ap.add_argument("--divergent",choices=["keep","drop"],default="drop",
                    help="same-length epub/whisper mismatch: 'drop' (safe, -> review) or 'keep' (whisper label)")
    a=ap.parse_args()

    print(f"[correct] epub glossary from {os.path.basename(a.epub)}")
    text=epub_text(a.epub); names,bigs,common_lower=build_glossary(text)
    print(f"[correct] glossary: {len(names)} names, {len(bigs)} 2-word names, {len(common_lower)} common words")
    print("  sample:", ", ".join(sorted(names)[:20]))

    cues=parse_vtt(a.vtt); print(f"[correct] {len(cues)} cues in VTT")

    holes=[]
    if a.coverage and os.path.isfile(a.coverage):
        cov=json.load(open(a.coverage,encoding="utf-8"))
        holes=[(h["audioStart"],h["audioEnd"]) for h in cov.get("audioNotInEpub",[])]
    def in_hole(c): return any(c["start"]<he and c["end"]>hs for hs,he in holes)

    SR=16000
    from faster_whisper.audio import decode_audio
    print(f"[correct] decoding audio -> 16k mono (PyAV)...")
    wav=decode_audio(a.audio,sampling_rate=SR)   # float32 mono
    print(f"[correct] audio: {len(wav)/SR:.0f}s")

    from faster_whisper import WhisperModel
    print(f"[correct] loading whisper {a.model} on {a.device}...")
    ct = "float16" if a.device=="cuda" else "int8"
    model=WhisperModel(a.model,device=a.device,compute_type=ct)

    out=[]; review=[]; rep=[]; drops=[]; namelog_all=[]
    counts={"verbatim":0,"overrun":0,"expansion":0,"hole":0,"divergent":0,
            "drop_dialogue":0,"drop_shrink":0,"drop_lowconf":0,"drop_divergent":0}
    for i,c in enumerate(cues):
        dur=c["end"]-c["start"]
        if '"' in c["text"] or "“" in c["text"]:
            counts["drop_dialogue"]+=1; drops.append({"t":s_to_ts(c["start"]),"why":"dialogue","text":c["text"][:80]}); continue
        if dur< a.min_clip: continue
        s0=int(c["start"]*SR); s1=min(len(wav),int(c["end"]*SR))
        slice_=wav[s0:s1]
        segs,_=model.transcribe(slice_,language="en",beam_size=5,word_timestamps=True)
        segs=list(segs)
        spoken=" ".join(s.text for s in segs).strip()
        words=[{"word":w.word,"start":w.start,"end":w.end} for s in segs for w in (s.words or [])]
        if segs:
            alp=sum(s.avg_logprob for s in segs)/len(segs); nsp=max(s.no_speech_prob for s in segs)
        else:
            alp=-9.0; nsp=1.0

        forced=in_hole(c)
        route,covv,ratio=classify(c["text"],spoken)   # ratio is word-count ratio (spoken/epub)

        newc=dict(c); nlog=[]; keep=True; reason=""; final=route
        if route in ("verbatim","overrun") and not forced:
            newc["text"]=c["text"]
            if route=="overrun":
                te=tighten_end(c["text"],words)
                if te is not None: newc["end"]=round(c["start"]+te+0.05,3)
        else:
            # epub text is NOT trustworthy here (paraphrase / hole / mis-align) -> use whisper,
            # but only when whisper is itself reliable. Drop garbage rather than mislabel.
            if ratio < a.shrink_ratio:
                keep=False; final="drop_shrink"; reason=f"whisper truncated/halluc (r{ratio:.2f})"
            elif alp < a.min_logprob or nsp > a.max_no_speech:
                keep=False; final="drop_lowconf"; reason=f"low conf (alp{alp:.2f} nsp{nsp:.2f})"
            else:
                final = "hole" if forced else ("expansion" if (ratio>=1.5 and covv>=0.5) else "divergent")
                if final=="divergent" and a.divergent=="drop":
                    keep=False; final="drop_divergent"; reason=f"ambiguous same-len (cov{covv:.2f})"
                else:
                    newc["text"]=fix_names(spoken,names,common_lower,nlog)

        counts[final]+=1
        if nlog: namelog_all.extend(nlog)
        entry={"t":s_to_ts(c["start"]),"route":final,"cov":round(covv,2),"ratio":round(ratio,2),
               "alp":round(alp,2),"nsp":round(nsp,2),"forced_hole":forced,
               "epub":c["text"][:70],"spoken":spoken[:70],"names":nlog,
               "dur_before":round(dur,2),"dur_after":round(newc["end"]-newc["start"],2),"reason":reason}
        rep.append(entry)
        if keep:
            out.append(newc)
        else:
            review.append({**newc,"text":f"[{final}:{reason}] EPUB={c['text']} || WHISPER={spoken}"})
        if (i+1)%25==0:
            print(f"  {i+1}/{len(cues)}  keep{len(out)} v{counts['verbatim']} o{counts['overrun']} "
                  f"h{counts['hole']} d{counts['divergent']} x{counts['expansion']} | "
                  f"drop shrink{counts['drop_shrink']} lowconf{counts['drop_lowconf']}")

    write_vtt(a.out,out)
    kept=len(out); dropped=len(cues)-kept
    print(f"\n[correct] KEPT {kept}/{len(cues)} cues -> {a.out}")
    print(f"[correct] routes: {counts}")
    print(f"[correct] name fixes: {len(namelog_all)}  e.g. {namelog_all[:8]}")
    if a.review and review:
        write_vtt(a.review,review); print(f"[correct] {len(review)} dropped cues for review -> {a.review}")
    if a.report:
        json.dump({"counts":counts,"kept":kept,"dropped":dropped,"drops":drops,
                   "nameFixes":namelog_all,"cues":rep},
                  open(a.report,"w",encoding="utf-8"),indent=1)
        print(f"[correct] report -> {a.report}")

if __name__=="__main__":
    main()
