"""Fix regnal (papal) names in the e2a-normalized book text so they match what Owen
ACTUALLY read aloud — "Pius the Twelfth", not e2a's inconsistent "Pius twelve" /
"Pius XII" / "Pius eleven".

WHY: e2a's roman2number is inconsistent on regnal numerals — it sometimes converts
"XII"->"twelve" (cardinal, wrong: should be the ordinal "the Twelfth") and sometimes
leaves the roman numeral untouched. Owen read every one as an ordinal with "the". This
mismatch hurts (a) Whisper alignment (book text vs heard words) and (b) what Orpheus
learns. cogito's diff first surfaced these as red-flag spots; this pass corrects them.

Scope: only POPE/REGNAL given-names followed by a roman numeral OR an e2a-mangled
cardinal word are converted. Ambiguous common names (John, Paul) are converted ONLY in
their roman-numeral form (a roman after a name is unambiguously regnal), never the bare
cardinal form (avoids e.g. a scripture-style "John three" being mauled).

Usage (any python):
  python fix_regnal_names.py            # DRY RUN — print every proposed change
  python fix_regnal_names.py --apply    # write corrections in place (after a *.bak backup)
"""
import os
import re
import sys
import glob
import shutil

NORM_DIR = "/mnt/c/tmp/bf-voice-tests/booktext_norm"

# Pope/regnal given names. Unambiguous popes (safe for cardinal-word form too):
POPE_NAMES = ["Pius", "Leo", "Benedict", "Gregory", "Innocent", "Clement", "Urban",
              "Boniface", "Alexander", "Sixtus", "Honorius", "Celestine", "Nicholas",
              "Adrian", "Hadrian", "Callixtus", "Eugene"]
# Ambiguous given names — convert ONLY the roman-numeral form, never bare cardinal:
AMBIG_NAMES = ["John", "Paul", "Stephen"]

ROMAN = {"II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8, "IX": 9,
         "X": 10, "XI": 11, "XII": 12, "XIII": 13, "XIV": 14, "XV": 15, "XVI": 16,
         "XVII": 17, "XVIII": 18, "XIX": 19, "XX": 20, "XXI": 21, "XXII": 22,
         "XXIII": 23, "XXIV": 24, "XXV": 25}
CARD = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
        "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
        "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
        "nineteen": 19, "twenty": 20, "twenty-one": 21, "twenty-two": 22,
        "twenty-three": 23, "twenty-four": 24, "twenty-five": 25}
ORD = {1: "First", 2: "Second", 3: "Third", 4: "Fourth", 5: "Fifth", 6: "Sixth",
       7: "Seventh", 8: "Eighth", 9: "Ninth", 10: "Tenth", 11: "Eleventh",
       12: "Twelfth", 13: "Thirteenth", 14: "Fourteenth", 15: "Fifteenth",
       16: "Sixteenth", 17: "Seventeenth", 18: "Eighteenth", 19: "Nineteenth",
       20: "Twentieth", 21: "Twenty-First", 22: "Twenty-Second", 23: "Twenty-Third",
       24: "Twenty-Fourth", 25: "Twenty-Fifth"}

ROMAN_ALT = "|".join(sorted(ROMAN, key=len, reverse=True))
CARD_ALT = "|".join(sorted(CARD, key=len, reverse=True))

# pope name + (the)? + roman|cardinal   -> all forms convertible
RE_POPE = re.compile(r"\b(" + "|".join(POPE_NAMES) + r")\s+(?:the\s+)?(" +
                     ROMAN_ALT + "|" + CARD_ALT + r")\b")
# ambiguous name + (the)? + roman ONLY
RE_AMBIG = re.compile(r"\b(" + "|".join(AMBIG_NAMES) + r")\s+(?:the\s+)?(" +
                      ROMAN_ALT + r")\b")


def to_ordinal(token):
    if token in ROMAN:
        return ORD[ROMAN[token]]
    if token in CARD:
        return ORD[CARD[token]]
    return None


def fix_text(text):
    changes = []

    def repl(m):
        name, num = m.group(1), m.group(2)
        ordw = to_ordinal(num)
        if not ordw:
            return m.group(0)
        new = f"{name} the {ordw}"
        if new != m.group(0):
            changes.append((m.group(0), new))
        return new

    text = RE_POPE.sub(repl, text)
    text = RE_AMBIG.sub(repl, text)
    return text, changes


def main():
    apply = "--apply" in sys.argv
    files = sorted(glob.glob(os.path.join(NORM_DIR, "*.txt")))
    total = 0
    for p in files:
        raw = open(p, encoding="utf-8").read()
        fixed, changes = fix_text(raw)
        if not changes:
            continue
        total += len(changes)
        counts = {}
        for a, b in changes:
            counts[(a, b)] = counts.get((a, b), 0) + 1
        print(f"\n{os.path.basename(p)}  ({len(changes)} fixes)")
        for (a, b), n in sorted(counts.items(), key=lambda kv: -kv[1]):
            print(f"   {n:3d}x  {a!r:>22}  ->  {b!r}")
        if apply:
            shutil.copyfile(p, p + ".bak")
            open(p, "w", encoding="utf-8").write(fixed)
    print(f"\n{'APPLIED' if apply else 'DRY RUN'} — {total} total fixes across "
          f"{len(files)} files." + ("" if apply else "  Re-run with --apply to write."))


if __name__ == "__main__":
    main()
