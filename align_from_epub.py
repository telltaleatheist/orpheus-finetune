"""Extract epub text (spine order) -> window it to the audio extent -> align
book-as-truth against Whisper timing (difflib) -> <out>.aligned.json.

Book is truth: Whisper hallucinations/insertions are DROPPED, so looping garbage
in the raw transcript disappears. Usage:
  align_from_epub.py <epub> <whisper.json> <out.aligned.json> <book_start_frac> <window_words>
book_start_frac: where in the book to start the alignment window (0.0 = start).
window_words: how many book words to include in the window (~1.6x whisper count).
"""
import sys, os, re, json, zipfile
from difflib import SequenceMatcher
from xml.etree import ElementTree as ET
from html.parser import HTMLParser

EPUB, WHISPER, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
START_FRAC = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
WINDOW = int(sys.argv[5]) if len(sys.argv) > 5 else 0

class Strip(HTMLParser):
    def __init__(self): super().__init__(); self.parts=[]; self.skip=0
    def handle_starttag(self, t, a):
        if t in ("script","style","sup"): self.skip+=1
    def handle_endtag(self, t):
        if t in ("script","style","sup") and self.skip: self.skip-=1
        if t in ("p","div","h1","h2","h3","br","li"): self.parts.append("\n")
    def handle_data(self, d):
        if not self.skip: self.parts.append(d)

def spine_files(z):
    # container.xml -> opf path -> spine itemrefs -> manifest hrefs (reading order)
    cont = z.read("META-INF/container.xml").decode("utf-8", "replace")
    opf_path = re.search(r'full-path="([^"]+)"', cont).group(1)
    opf = z.read(opf_path).decode("utf-8", "replace")
    ns = {"o":"http://www.idpf.org/2007/opf"}
    root = ET.fromstring(opf)
    base = os.path.dirname(opf_path)
    manifest = {}
    for it in root.iter():
        if it.tag.endswith("}item") or it.tag=="item":
            manifest[it.get("id")] = it.get("href")
    order = []
    for ir in root.iter():
        if ir.tag.endswith("}itemref") or ir.tag=="itemref":
            href = manifest.get(ir.get("idref"))
            if href:
                order.append(os.path.join(base, href).replace("\\","/") if base else href)
    return order

def extract(z, files):
    words = []
    for f in files:
        try: html = z.read(f).decode("utf-8", "replace")
        except KeyError: continue
        p = Strip(); p.feed(html)
        txt = "".join(p.parts)
        txt = (txt.replace("⁠","").replace("​","")
                  .replace("“",'"').replace("”",'"')
                  .replace("‘","'").replace("’","'"))
        for raw in txt.split():
            words.append(raw)
    return words

def norm(w): return re.sub(r"[^a-z0-9]","",w.lower())

with zipfile.ZipFile(EPUB) as z:
    book_words = extract(z, spine_files(z))
print(f"[epub] {len(book_words)} words total")

whisper = json.load(open(WHISPER, encoding="utf-8"))
wn = [norm(w["word"]) for w in whisper]
wn_ct = len(whisper)

book_norm_all = [norm(w) for w in book_words]
if START_FRAC < 0:
    # AUTO-ANCHOR: find where the whisper stream starts in the book via the longest
    # matching run between the full book and the whisper head (skips front matter and
    # locates a mid-book start like TR p2). Window from just before that anchor.
    head = wn[:120]
    m = SequenceMatcher(None, book_norm_all, head, autojunk=False).find_longest_match(0, len(book_norm_all), 0, len(head))
    start = max(0, m.a - m.b - 15)
    print(f"[anchor] longest head match: book@{m.a} whisper@{m.b} size={m.size} -> window start {start}")
else:
    start = int(len(book_words) * START_FRAC)
win = WINDOW if WINDOW else int(wn_ct * 1.35)
btok_raw = book_words[start:start+win]
btok = [(r, norm(r)) for r in btok_raw if norm(r)]
print(f"[window] book[{start}:{start+win}] -> {len(btok)} usable tokens vs {wn_ct} whisper words")

sm = SequenceMatcher(a=[n for _,n in btok], b=wn, autojunk=False)
aligned=[]; last_end=0.0; pending=[]; corr_blocks=[]
def flush(nxt):
    if not pending: return
    n=len(pending); span=max(0.0,nxt-last_end)
    for k,bi in enumerate(pending):
        s=last_end+span*k/n; e=last_end+span*(k+1)/n
        aligned.append({"w":btok[bi][0],"start":round(s,3),"end":round(e,3),"corr":True})
    pending.clear()
for tag,i1,i2,j1,j2 in sm.get_opcodes():
    if tag=="equal":
        flush(whisper[j1]["start"])
        for k in range(i2-i1):
            w=whisper[j1+k]; aligned.append({"w":btok[i1+k][0],"start":w["start"],"end":w["end"],"corr":False}); last_end=w["end"]
    elif tag=="replace":
        ws=whisper[j1]["start"]; we=whisper[j2-1]["end"]; flush(ws); nb=i2-i1
        for k in range(nb):
            s=ws+(we-ws)*k/nb; e=ws+(we-ws)*(k+1)/nb
            aligned.append({"w":btok[i1+k][0],"start":round(s,3),"end":round(e,3),"corr":True})
        last_end=we; corr_blocks.append((" ".join(btok[x][0] for x in range(i1,i2)), " ".join(whisper[x]["word"] for x in range(j1,j2))))
    elif tag=="delete": pending.extend(range(i1,i2))
flush(last_end)
# Trim to the actually-narrated span: drop leading front-matter and trailing
# beyond-audio words (both pile up as interpolated 'corr' words outside the real
# whisper timeline). Keep from first to last EXACT match.
exacts=[i for i,a in enumerate(aligned) if not a["corr"]]
if exacts:
    aligned = aligned[exacts[0]:exacts[-1]+1]
json.dump(aligned, open(OUT,"w",encoding="utf-8"), ensure_ascii=False)

matched=sum(1 for a in aligned if not a["corr"])
print(f"[align] aligned {len(aligned)} words: exact {matched} ({100*matched/max(1,len(aligned)):.1f}%), corrected {len(aligned)-matched}")
# find last matched book index to know where audio ended (for next segment)
last_exact = max((i for i,a in enumerate(aligned) if not a["corr"]), default=0)
print(f"[align] last exact-match at aligned index {last_exact}/{len(aligned)} (book token ~{start+last_exact})")
# gap recheck
gaps=[(aligned[i+1]["start"]-aligned[i]["end"]) for i in range(len(aligned)-1)]
big=sorted([g for g in gaps if g>=3.0], reverse=True)[:8]
print(f"[gaps] >=3s: {sum(1 for g in gaps if g>=3)}  >=5s: {sum(1 for g in gaps if g>=5)}  top: {[round(g,1) for g in big]}")
# text naturalness sample
print("[text] sample @40%:", " ".join(a["w"] for a in aligned[int(len(aligned)*0.4):int(len(aligned)*0.4)+45]))
print("[corr] sample fixes (book <= whisper):")
for b,w in corr_blocks[:12]:
    if b.lower()!=w.lower() and len(b)<50: print(f"   '{b}'  <=  '{w}'")
