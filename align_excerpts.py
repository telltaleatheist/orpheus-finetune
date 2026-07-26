#!/usr/bin/env python3
r"""align_excerpts.py — book-anchored, word-timestamped transcripts for scattered
GraphicAudio training excerpts (Deathstalker Honor / Legacy).

Unlike align_from_epub.py (which assumes ONE contiguous book window == the audio),
these excerpts are MANY short spans scattered across a ~30-hour novel, ~80% verbatim
but condensed/adapted in places, with stretches that match NO book text at all
(dramatized additions). So this tool:

  * extracts full book text from the epub (same spine-order HTML strip as
    align_from_epub.py) into a normalized word stream, keeping a map back to the
    ORIGINAL word (punctuation/casing) + spine file + char offset;
  * transcribes-in (reads the whisper.json produced by transcribe_excerpts.py);
  * GLOBALLY anchors the whisper stream: seed on exact 6-gram matches, then
    difflib-extend each seed into a maximal near-diagonal run (tolerating small
    substitutions / whisper errors / condensation deletes);
  * emits BOOK text (original punctuation) for anchored spans with whisper timing
    carried over 1:1 (interpolated across small mismatch runs); condensed-out book
    text (large deletes) is DROPPED, not invented;
  * keeps whisper text verbatim for unanchored stretches, flagged source="whisper";
  * marks every word overlapping an SFX keep-out window (in_sfx_keepout).

NUMBERS: the narrator speaks numbers as words but the epub writes digits — a real
v2-dataset mismatch (see fix_regnal_names.py). Both streams are number-normalized
(digit tokens 0-9999 expanded to words) so they align.

This tool does NOT modify the audio, epub, or sfx files. It writes:
  <out>/<stem>.words.json   ordered word list + span metadata
  <out>/<stem>.transcript.txt  readable, [WHISPER-ONLY]...[/WHISPER-ONLY] wrapped
(REPORT.md and the ffmpeg spot-checks are produced by the driver, not this file.)

Usage:
  python align_excerpts.py --epub <epub> --whisper <stem.whisper.json>
      --sfx <sfx_regions.json> --sfx-key "deathstalker honor"
      --stem "deathstalker honor" --out-dir <dir>
"""
from __future__ import annotations

import argparse
import json
import os
import re
import zipfile
from difflib import SequenceMatcher
from html.parser import HTMLParser
from xml.etree import ElementTree as ET

# ---------------------------------------------------------------------------
# epub extraction (spine-order HTML strip) — same approach as align_from_epub.py,
# extended to record per-word original text, spine index, char offset, and to
# capture chapter headings.
# ---------------------------------------------------------------------------

_SMART = str.maketrans({"⁠": "", "​": "", "“": '"', "”": '"',
                        "‘": "'", "’": "'", "—": " - ",
                        "–": "-", " ": " "})


class Strip(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.skip = 0
        self.headings = []          # (approx char offset in this file, text)
        self._in_h = None
        self._h_buf = []
        self._chars = 0

    def handle_starttag(self, t, a):
        if t in ("script", "style", "sup"):
            self.skip += 1
        if t in ("h1", "h2", "h3"):
            self._in_h = t
            self._h_buf = []

    def handle_endtag(self, t):
        if t in ("script", "style", "sup") and self.skip:
            self.skip -= 1
        if t in ("p", "div", "h1", "h2", "h3", "br", "li"):
            self.parts.append("\n")
            self._chars += 1
        if t in ("h1", "h2", "h3") and self._in_h == t:
            htext = "".join(self._h_buf).strip()
            if htext:
                self.headings.append(htext)
            self._in_h = None

    def handle_data(self, d):
        if not self.skip:
            self.parts.append(d)
            self._chars += len(d)
            if self._in_h:
                self._h_buf.append(d)


def spine_files(z):
    cont = z.read("META-INF/container.xml").decode("utf-8", "replace")
    opf_path = re.search(r'full-path="([^"]+)"', cont).group(1)
    opf = z.read(opf_path).decode("utf-8", "replace")
    root = ET.fromstring(opf)
    base = os.path.dirname(opf_path)
    manifest = {}
    for it in root.iter():
        if it.tag.endswith("}item") or it.tag == "item":
            manifest[it.get("id")] = it.get("href")
    order = []
    for ir in root.iter():
        if ir.tag.endswith("}itemref") or ir.tag == "itemref":
            href = manifest.get(ir.get("idref"))
            if href:
                order.append(os.path.join(base, href).replace("\\", "/")
                             if base else href)
    return order


class BookWord:
    __slots__ = ("orig", "spine_idx", "char_off")

    def __init__(self, orig, spine_idx, char_off):
        self.orig = orig
        self.spine_idx = spine_idx
        self.char_off = char_off


def extract_book(epub_path):
    """Return (book_words: list[BookWord], spine_meta: list[dict]).
    char_off is a running character offset across the whole reconstructed book."""
    book = []
    spine_meta = []
    with zipfile.ZipFile(epub_path) as z:
        files = spine_files(z)
        running_chars = 0
        for si, f in enumerate(files):
            try:
                html = z.read(f).decode("utf-8", "replace")
            except KeyError:
                spine_meta.append({"idx": si, "file": os.path.basename(f),
                                   "heading": None, "word_start": len(book),
                                   "word_end": len(book)})
                continue
            p = Strip()
            p.feed(html)
            txt = "".join(p.parts).translate(_SMART)
            ws = len(book)
            # walk txt tracking char offset for each word
            off = running_chars
            for m in re.finditer(r"\S+", txt):
                book.append(BookWord(m.group(0), si, off + m.start()))
            spine_meta.append({
                "idx": si, "file": os.path.basename(f),
                "heading": p.headings[0] if p.headings else None,
                "word_start": ws, "word_end": len(book),
            })
            running_chars += len(txt)
    return book, spine_meta


# ---------------------------------------------------------------------------
# number-aware normalization -> token stream + back-references
# ---------------------------------------------------------------------------

_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
         "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
         "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
         "eighty", "ninety"]


def num_to_words(n):
    if n < 0 or n > 9999:
        return None
    if n < 20:
        return [_ONES[n]]
    if n < 100:
        w = [_TENS[n // 10]]
        if n % 10:
            w.append(_ONES[n % 10])
        return w
    if n < 1000:
        w = [_ONES[n // 100], "hundred"]
        if n % 100:
            w += num_to_words(n % 100)
        return w
    w = [_ONES[n // 1000], "thousand"]
    if n % 1000:
        w += num_to_words(n % 1000)
    return w


_ALNUM = re.compile(r"[^a-z0-9]")

# ASR proper-noun fixes: whisper mishears a series-specific name as one fused
# token; the epub spelling is the truth. Applied to the whisper word list BEFORE
# alignment (fixes whisper-only output text AND helps seeds anchor). Key is the
# normalized whisper token; value is the replacement word list (timing is split
# evenly across it, punctuation from the original token is kept on the last word).
# "Mater Mundi" (Deathstalker uber-esper, "Our Mother of All Souls") confirmed
# 56x in the Honor epub; whisper writes "Matamundi".
ASR_PROPER_NOUN_FIXES = {
    "matamundi": ["Mater", "Mundi"],
}


def apply_asr_fixes(whisper):
    """Return a new whisper word list with ASR_PROPER_NOUN_FIXES applied.
    A fused token is split into N words sharing its time span evenly; leading
    punctuation stays on the first word, trailing punctuation on the last."""
    out = []
    for w in whisper:
        core = _ALNUM.sub("", w["word"].lower())
        fix = ASR_PROPER_NOUN_FIXES.get(core)
        if not fix:
            out.append(w)
            continue
        m = re.match(r"^(\W*)(.*?)(\W*)$", w["word"], re.S)
        lead, _, trail = m.group(1), m.group(2), m.group(3)
        n = len(fix)
        span = w["end"] - w["start"]
        for k, piece in enumerate(fix):
            word = piece
            if k == 0:
                word = lead + word
            if k == n - 1:
                word = word + trail
            out.append({"word": word,
                        "start": round(w["start"] + span * k / n, 3),
                        "end": round(w["start"] + span * (k + 1) / n, 3),
                        "prob": w.get("prob")})
    return out


def normalize_stream(items, get_text):
    """items: list of arbitrary objects; get_text(item)->original word string.
    Returns (norm_tokens: list[str], refs: list[int]) where refs[k] is the index
    into `items` that normalized token k came from. One original word may expand
    to several normalized tokens (hyphenated compounds, spelled-out numbers)."""
    norm = []
    refs = []
    for i, it in enumerate(items):
        raw = get_text(it).lower().translate(_SMART)
        for sub in re.split(r"[\s\-/]+", raw):
            sub = _ALNUM.sub("", sub)
            if not sub:
                continue
            if sub.isdigit() and len(sub) <= 4:
                expanded = num_to_words(int(sub))
                if expanded:
                    for w in expanded:
                        norm.append(w)
                        refs.append(i)
                    continue
            norm.append(sub)
            refs.append(i)
    return norm, refs


# ---------------------------------------------------------------------------
# global anchoring: seed on 6-grams, difflib-extend each seed
# ---------------------------------------------------------------------------

K = 6
MAXOCC = 6          # ignore book 6-grams that occur more than this (ambiguous)
BOOK_WIN = 1200     # book words to search downstream of a seed
WHIS_WIN = 800      # whisper words per seed pass (long runs re-seed & continue)
INS_BREAK = 12      # >= this many consecutive whisper-only words ends a segment
DEL_DROP = 8        # book-only run longer than this = condensed-out -> DROP
REPLACE_MAX = 6     # replace block larger than this (either side) = the diagonal
                    # broke (condensation/adaptation), not a word-substitution ->
                    # end the segment; the whisper words become whisper-only.
MIN_SEG_WORDS = 4   # discard trivially short "segments" (noise)


def build_kgram_index(bn):
    idx = {}
    for i in range(len(bn) - K + 1):
        key = tuple(bn[i:i + K])
        lst = idx.get(key)
        if lst is None:
            idx[key] = [i]
        elif len(lst) <= MAXOCC:
            lst.append(i)
    return {k: v for k, v in idx.items() if len(v) <= MAXOCC}


def _exact_run(bn, wn, bi, wi):
    """Length of the maximal exact-match run of normalized tokens starting at
    book index bi / whisper index wi (extends forward only)."""
    n = 0
    while (bi + n < len(bn) and wi + n < len(wn)
           and bn[bi + n] == wn[wi + n]):
        n += 1
    return n


def find_seed(bn, wn, wi, kidx, book_hint):
    """Find a book index for the whisper 6-gram at wi. Among candidate book
    occurrences pick the one with the LONGEST exact forward run (the true
    diagonal), tie-broken by proximity to book_hint. Robust to repeated phrases."""
    if wi + K > len(wn):
        return None
    occ = kidx.get(tuple(wn[wi:wi + K]))
    if not occ:
        return None
    if len(occ) == 1:
        return occ[0]

    def score(b):
        return (_exact_run(bn, wn, b, wi),
                -abs(b - book_hint) if book_hint is not None else 0)
    return max(occ, key=score)


def align(book, book_meta, whisper, sfx_regions, stem):
    """Core. Returns (words: list[dict], spans: list[dict], stats: dict)."""
    bn, bref = normalize_stream(book, lambda b: b.orig)
    wn, wref = normalize_stream(whisper, lambda w: w["word"])
    kidx = build_kgram_index(bn)

    # spine index -> chapter label
    def chapter_of(book_idx):
        for sm in book_meta:
            if sm["word_start"] <= book_idx < sm["word_end"]:
                return sm
        return book_meta[-1]

    out_words = []          # final flat word list (unsorted; sorted at end)
    spans = []              # book-span metadata
    consumed_w = [False] * len(wn)   # whisper norm tokens claimed by a book span

    def interp(a_start, a_end, n):
        span = max(0.0, a_end - a_start)
        return [(round(a_start + span * k / n, 3),
                 round(a_start + span * (k + 1) / n, 3)) for k in range(n)]

    wi = 0
    book_hint = None
    while wi < len(wn):
        if consumed_w[wi]:
            wi += 1
            continue
        bi = find_seed(bn, wn, wi, kidx, book_hint)
        if bi is None:
            wi += 1
            continue
        # windows
        a = bn[bi: bi + BOOK_WIN]
        aref = bref[bi: bi + BOOK_WIN]
        b = wn[wi: wi + WHIS_WIN]
        sm = SequenceMatcher(None, a, b, autojunk=False)
        opcodes = sm.get_opcodes()

        seg_words = []          # emitted output words for this segment
        seg_book_idxs = []      # original book-word indices emitted
        last_end = None
        first_wtime = None
        stop = False
        w_consumed_upto = 0     # count of whisper norm tokens consumed (rel to wi)

        def wtime(local_j):
            return whisper[wref[wi + local_j]]

        for tag, i1, i2, j1, j2 in opcodes:
            if stop:
                break
            if tag == "equal":
                for k in range(i2 - i1):
                    bw = book[aref[i1 + k]]
                    wt = wtime(j1 + k)
                    seg_words.append({"word": bw.orig, "start": wt["start"],
                                      "end": wt["end"], "source": "book",
                                      "book_idx": aref[i1 + k]})
                    seg_book_idxs.append(aref[i1 + k])
                    last_end = wt["end"]
                    if first_wtime is None:
                        first_wtime = wt["start"]
                    for jj in range(wi + j1 + k, wi + j1 + k + 1):
                        pass
                # mark consumed
                for jj in range(j1, j2):
                    consumed_w[wi + jj] = True
                w_consumed_upto = max(w_consumed_upto, j2)
            elif tag == "replace":
                nb = i2 - i1
                nw = j2 - j1
                if nb > REPLACE_MAX or nw > REPLACE_MAX:
                    # diagonal broke: this is condensation/adaptation, not a
                    # small word-substitution. End the segment; leave the whisper
                    # words (j1..) unconsumed -> collected as whisper-only.
                    stop = True
                    w_consumed_upto = max(w_consumed_upto, j1)
                    break
                ws = wtime(j1)["start"]
                we = wtime(j2 - 1)["end"]
                if first_wtime is None:
                    first_wtime = ws
                times = interp(ws, we, nb)
                for k in range(nb):
                    bw = book[aref[i1 + k]]
                    seg_words.append({"word": bw.orig, "start": times[k][0],
                                      "end": times[k][1], "source": "book",
                                      "book_idx": aref[i1 + k]})
                    seg_book_idxs.append(aref[i1 + k])
                last_end = we
                for jj in range(j1, j2):
                    consumed_w[wi + jj] = True
                w_consumed_upto = max(w_consumed_upto, j2)
            elif tag == "delete":       # book-only run (not spoken)
                run = i2 - i1
                if run <= DEL_DROP and last_end is not None:
                    # small: whisper likely missed a word or two -> keep book text,
                    # interpolate timing in the local gap
                    nxt = None
                    # find next whisper start after this delete
                    # (use last_end..last_end tiny span; refine with following op)
                    nxt = last_end
                    times = interp(last_end, last_end, run)  # zero-width; refined below
                    for k in range(run):
                        bw = book[aref[i1 + k]]
                        seg_words.append({"word": bw.orig, "start": last_end,
                                          "end": last_end, "source": "book",
                                          "book_idx": aref[i1 + k],
                                          "_interp": True})
                        seg_book_idxs.append(aref[i1 + k])
                # else: condensed-out book text -> DROP (do not invent)
            elif tag == "insert":       # whisper-only run
                run = j2 - j1
                if run >= INS_BREAK:
                    # end of this excerpt / dramatized addition -> stop segment.
                    # whisper words j1.. remain UNCONSUMED (outer loop handles them)
                    stop = True
                    w_consumed_upto = max(w_consumed_upto, j1)
                    break
                # small inline whisper insertion (spoken adaptation / whisper noise):
                # keep verbatim, source=whisper
                for k in range(run):
                    wt = wtime(j1 + k)
                    seg_words.append({"word": whisper[wref[wi + j1 + k]]["word"],
                                      "start": wt["start"], "end": wt["end"],
                                      "source": "whisper"})
                    consumed_w[wi + j1 + k] = True
                if seg_words:
                    last_end = seg_words[-1]["end"]
                w_consumed_upto = max(w_consumed_upto, j2)

        # refine zero-width interpolated deletes: spread them between neighbours
        for idx in range(len(seg_words)):
            if seg_words[idx].get("_interp"):
                # find previous real end and next real start
                prev_e = seg_words[idx - 1]["end"] if idx > 0 else seg_words[idx]["start"]
                nxt_s = None
                for j2i in range(idx + 1, len(seg_words)):
                    if not seg_words[j2i].get("_interp"):
                        nxt_s = seg_words[j2i]["start"]
                        break
                if nxt_s is None:
                    nxt_s = prev_e
                seg_words[idx]["start"] = prev_e
                seg_words[idx]["end"] = max(prev_e, nxt_s)
        for w in seg_words:
            w.pop("_interp", None)

        real_book = [w for w in seg_words if w["source"] == "book"]
        if len(real_book) >= MIN_SEG_WORDS:
            out_words.extend(seg_words)
            if seg_book_idxs:
                b0, b1 = min(seg_book_idxs), max(seg_book_idxs)
                ch = chapter_of(b0)
                spans.append({
                    "source": "book",
                    "audio_start": round(first_wtime, 3) if first_wtime is not None else None,
                    "audio_end": round(last_end, 3) if last_end is not None else None,
                    "book_word_start": b0, "book_word_end": b1,
                    "book_char_start": book[b0].char_off,
                    "book_char_end": book[b1].char_off,
                    "book_frac": round(b0 / max(1, len(book)), 4),
                    "chapter_file": ch["file"], "chapter_heading": ch["heading"],
                    "n_words": len(real_book),
                })
            book_hint = (max(seg_book_idxs) if seg_book_idxs else bi) + 1
            advance = max(w_consumed_upto, 1)
            wi = wi + advance
        else:
            # seed didn't pan out; skip this whisper word
            wi += 1

    # ---- collect unconsumed whisper words into whisper-only stretches ----
    whisper_words = []
    i = 0
    while i < len(wn):
        if consumed_w[i]:
            i += 1
            continue
        j = i
        while j < len(wn) and not consumed_w[j]:
            j += 1
        # emit whisper words i..j (dedup original word indices)
        seen_ref = None
        for k in range(i, j):
            r = wref[k]
            if r == seen_ref:
                continue
            seen_ref = r
            wt = whisper[r]
            whisper_words.append({"word": wt["word"], "start": wt["start"],
                                  "end": wt["end"], "source": "whisper"})
        i = j

    out_words.extend(whisper_words)

    # dedup book words that share an original index but got emitted per-norm-token
    # (a hyphenated/number word expands to multiple norm tokens -> one output word)
    out_words.sort(key=lambda w: (w["start"], w["end"]))
    deduped = []
    for w in out_words:
        if (w["source"] == "book" and deduped and deduped[-1]["source"] == "book"
                and deduped[-1].get("book_idx") == w.get("book_idx")):
            deduped[-1]["end"] = max(deduped[-1]["end"], w["end"])
            continue
        deduped.append(w)
    out_words = deduped

    # ---- SFX keep-out marking ----
    for w in out_words:
        w["in_sfx_keepout"] = any(
            w["start"] < r[1] and w["end"] > r[0] for r in sfx_regions)

    # strip internal book_idx from final words (kept in span metadata instead)
    for w in out_words:
        w.pop("book_idx", None)

    # ---- whisper-only stretch list (contiguous whisper runs) ----
    stretches = []
    i = 0
    while i < len(out_words):
        if out_words[i]["source"] != "whisper":
            i += 1
            continue
        j = i
        while j < len(out_words) and out_words[j]["source"] == "whisper":
            j += 1
        run = out_words[i:j]
        dur = run[-1]["end"] - run[0]["start"]
        stretches.append({
            "start": round(run[0]["start"], 3), "end": round(run[-1]["end"], 3),
            "duration": round(dur, 2), "n_words": len(run),
            "text": " ".join(w["word"] for w in run),
        })
        i = j

    n_book = sum(1 for w in out_words if w["source"] == "book")
    n_whis = sum(1 for w in out_words if w["source"] == "whisper")
    stats = {
        "total_words": len(out_words),
        "book_words": n_book,
        "whisper_words": n_whis,
        "pct_anchored": round(100 * n_book / max(1, len(out_words)), 1),
        "n_book_spans": len(spans),
        "n_whisper_stretches_all": len(stretches),
        "whisper_total_sec": round(sum(s["duration"] for s in stretches), 1),
    }
    return out_words, spans, stretches, stats


# ---------------------------------------------------------------------------
# writers
# ---------------------------------------------------------------------------

STRETCH_MIN_WORDS = 5
STRETCH_MIN_SEC = 2.0


def write_transcript_txt(path, words, mark_whisper=True):
    """Readable transcript; wrap notable whisper-only runs in markers.
    mark_whisper=False (whisper-only mode, no book exists): plain paragraphs,
    broken on pauses >= 1.5s — markers would be noise when EVERYTHING is
    whisper."""
    lines = []
    buf = []

    def flush_para():
        if buf:
            lines.append(" ".join(buf))
            buf.clear()

    if not mark_whisper:
        prev_end = None
        for w in words:
            if prev_end is not None and w["start"] - prev_end >= 1.5:
                flush_para()
            buf.append(w["word"].strip())
            prev_end = w["end"]
        flush_para()
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(lines) + "\n")
        return

    i = 0
    while i < len(words):
        w = words[i]
        if w["source"] == "whisper":
            j = i
            while j < len(words) and words[j]["source"] == "whisper":
                j += 1
            run = words[i:j]
            dur = run[-1]["end"] - run[0]["start"]
            if len(run) >= STRETCH_MIN_WORDS or dur >= STRETCH_MIN_SEC:
                flush_para()
                lines.append(f"[WHISPER-ONLY {run[0]['start']:.1f}-{run[-1]['end']:.1f}s]")
                lines.append(" ".join(x["word"].strip() for x in run))
                lines.append("[/WHISPER-ONLY]")
            else:
                buf.extend(x["word"].strip() for x in run)
            i = j
        else:
            buf.append(w["word"].strip())
            i += 1
    flush_para()
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def whisper_only(whisper, sfx_regions):
    """No source book available: whisper IS the transcript. Same output shape
    as align() — every word source='whisper', one stretch covering everything
    is NOT emitted (stretch list stays per-contiguous-run = one run)."""
    out_words = [{"word": w["word"], "start": w["start"], "end": w["end"],
                  "source": "whisper",
                  "in_sfx_keepout": any(w["start"] < e and w["end"] > s
                                        for s, e in sfx_regions)}
                 for w in whisper]
    stretches = []
    if out_words:
        stretches.append({
            "start": out_words[0]["start"], "end": out_words[-1]["end"],
            "duration": round(out_words[-1]["end"] - out_words[0]["start"], 2),
            "n_words": len(out_words),
            "text": " ".join(w["word"] for w in out_words[:20]) + " ...",
        })
    stats = {
        "total_words": len(out_words), "book_words": 0,
        "whisper_words": len(out_words), "pct_anchored": 0.0,
        "n_book_spans": 0, "n_whisper_stretches_all": len(stretches),
        "whisper_total_sec": stretches[0]["duration"] if stretches else 0.0,
    }
    return out_words, [], stretches, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epub", default=None,
                    help="source epub; omit for whisper-only mode (no book "
                         "exists — whisper is the transcript, all words "
                         "source='whisper')")
    ap.add_argument("--whisper", required=True)
    ap.add_argument("--sfx", required=True)
    ap.add_argument("--sfx-key", default=None,
                    help="key into sfx json 'files'; omit ONLY for files that "
                         "were never SFX-reviewed (no keep-out flags emitted)")
    ap.add_argument("--stem", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    whisper = apply_asr_fixes(json.load(open(args.whisper, encoding="utf-8")))

    regions = []
    if args.sfx_key is not None:
        sfx_all = json.load(open(args.sfx, encoding="utf-8"))
        if args.sfx_key not in sfx_all["files"]:
            raise SystemExit(f"--sfx-key {args.sfx_key!r} not in {args.sfx} "
                             f"(keys: {list(sfx_all['files'])}); omit --sfx-key "
                             f"only for files that were never SFX-reviewed")
        for r in sfx_all["files"][args.sfx_key].get("regions", []):
            if r.get("curation_decision") == "exclude":
                regions.append((r["snapped_start_sec"], r["snapped_end_sec"]))

    if args.epub:
        book, book_meta = extract_book(args.epub)
        out_words, spans, stretches, stats = align(book, book_meta, whisper,
                                                   regions, args.stem)
    else:
        out_words, spans, stretches, stats = whisper_only(whisper, regions)
        book = []
        book_meta = []

    os.makedirs(args.out_dir, exist_ok=True)
    words_out = {
        "stem": args.stem,
        "epub": os.path.basename(args.epub) if args.epub else None,
        "sfx_reviewed": args.sfx_key is not None,
        "stats": stats,
        "sfx_keepout_regions": ([{"start": s, "end": e} for s, e in regions]
                                if args.sfx_key is not None else None),
        "spans": spans,
        "whisper_only_stretches": stretches,
        "words": out_words,
    }
    wp = os.path.join(args.out_dir, f"{args.stem}.words.json")
    json.dump(words_out, open(wp, "w", encoding="utf-8"), ensure_ascii=False,
              indent=1)
    write_transcript_txt(os.path.join(args.out_dir, f"{args.stem}.transcript.txt"),
                         out_words, mark_whisper=bool(args.epub))

    print(f"[align] {args.stem}: {json.dumps(stats)}")
    big = [s for s in stretches if s["n_words"] >= STRETCH_MIN_WORDS
           or s["duration"] >= STRETCH_MIN_SEC]
    print(f"[align] notable whisper-only stretches: {len(big)} "
          f"({round(sum(s['duration'] for s in big),1)}s)")
    # emit a compact machine-readable summary for the driver/report
    summ = os.path.join(args.out_dir, f"{args.stem}.summary.json")
    json.dump({"stats": stats, "spans": spans,
               "notable_stretches": big, "book_total_words": len(book),
               "spine_meta": book_meta}, open(summ, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
