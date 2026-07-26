#!/usr/bin/env python3
r"""session.py — one driver for an Orpheus voice-training SESSION.

A "session" is one voice whose training data ACCUMULATES over time: each batch of
contiguous source chunks is a PART-SET (pt01-04 today, pt05-08 next week). Every
part-set is merged, aligned, verified, and cut on its own; `stage` then unions all
part-sets' clips into ONE WSL dataset dir whose combined CSVs train together.

The manual runbook this automates is README.md ("THE proven recipe" +
"The pipeline (commands)"). This script is the tie-it-together layer; the real work
still lives in the proven single-file tools it drives:
  merge  -> ffmpeg concat  (the N source chunks -> one merged wav; duration-verified)
  align  -> bookforge-tts --generate-sentences   (epub-as-truth sentence VTT + coverage)
  verify -> drift GATE: coverage driftSelfCheck + verify_vtt.py (independent whisper) +
            waveform near-silence spot-check. Persists <vtt>.verify.json. `cut` refuses
            to run a part-set whose verify.json is missing/failed.
  cut    -> cut_audiobook.py (mix buckets, <=20s cap, trail-cap 0.1) + STATIC gain to
            target LUFS + validate_dataset.py           (per-part-set clip dir)
  stage  -> copy every part-set clip dir onto WSL ext4 + write COMBINED train/eval CSVs
  train  -> preflight-gated WSL launch: orpheus_owen.py build + train (v4 recipe) with
            cwd OUTSIDE any git repo (unsloth cache leak). Tees the log to Windows.
  status -> which stages are complete for each part-set + overall.
  audition -> print/run the bookforge-tts A/B render for the trained voice.

HARD RULES honored here (see CLAUDE.md / MEMORY):
  * NO FALLBACKS. A missing file, a duration mismatch, a failed subprocess, or a failed
    gate raises / exits non-zero with a clear message. Nothing is silently defaulted.
  * Idempotent/resumable: every stage detects its own completed, consistent outputs and
    reports "already done"; --force redoes.
  * verify is a GATE for cut; the full verify set is a preflight for train.
  * WSL gotchas: literal paths only inside bash -c (shell vars get eaten); a clean Linux
    PATH is exported before conda-adjacent work; no `2>NUL` anywhere (reserved name in
    Git Bash); PYTHONIOENCODING=utf-8 for e2a/bookforge tooling.

Deploy is deliberately NOT here — deploy_voice.sh exists and the ear-check is a human
step. `train` stops at the merged 16-bit model.

Usage:
  python session.py --config sessions/deathstalker_cod.json status
  python session.py --config sessions/deathstalker_cod.json merge  [--part-set pt01-04] [--force]
  python session.py --config sessions/deathstalker_cod.json align  [--part-set ...] [--force] [--print-only]
  python session.py --config sessions/deathstalker_cod.json verify [--part-set ...] [--force]
  python session.py --config sessions/deathstalker_cod.json cut    [--part-set ...] [--force]
  python session.py --config sessions/deathstalker_cod.json stage  [--force]
  python session.py --config sessions/deathstalker_cod.json train  [--dry-run-preflight]
  python session.py --config sessions/deathstalker_cod.json audition --text sample.txt [--run]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path, PureWindowsPath

# Windows consoles default to cp1252 — HF datasets/tqdm progress bars emit
# Unicode block glyphs that crash a naive sys.stdout.write mid-train. The
# driver must never die from cosmetic output.
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(errors="replace")

HERE = Path(__file__).resolve().parent
VERIFY_VTT = HERE / "verify_vtt.py"
CUT_AUDIOBOOK = HERE / "cut_audiobook.py"
ORPHEUS_OWEN = HERE / "orpheus_owen.py"
VALIDATE_DATASET = HERE / "validate_dataset.py"

# Verification gate thresholds (seconds). The independent whisper probe is the hard gate.
FAIL_MAXOFF_S = 1.5      # any independent cue off by more than this FAILS the part-set
WARN_MAXOFF_S = 0.75     # warn (but pass) above this
FAIL_DRIFT_MAX_S = 3.0   # coverage driftSelfCheck: align auto-corrects >3s, so >3s residual is a fail
# waveform near-silence spot-check: a cue boundary should sit within +/- this of a silence trough
WF_TOLERANCE_S = 0.20
WF_SILENCE_RMS = 0.02    # below this RMS counts as a silence trough (matches validate_dataset edge floor)
WF_MIN_NEAR_FRAC = 0.40  # fewer than this fraction near-silence => gross drift => FAIL
WF_SAMPLES = 24          # cue boundaries spot-checked, evenly spread

# Training preflight defaults (overridable in config.train)
DEFAULT_MIN_FREE_VRAM_MIB = 16000
DEFAULT_GPU_CONSUMER_MIB = 500   # a foreign compute app using more than this blocks training
# python cmdlines that mean "a heavy TTS/align/train job is already running"
BUSY_JOB_PATTERNS = ("orpheus_owen.py", "cut_audiobook.py", "bookforge-tts",
                     "transcribe_whisper", "batch_transcribe", "align_from_epub",
                     "verify_vtt.py", "whisperx")


# --------------------------------------------------------------------------- util
def die(msg: str) -> "NoReturn":
    """Fail loudly. NO FALLBACKS - every unrecoverable condition ends here."""
    print(f"\n[session] ERROR: {msg}", file=sys.stderr, flush=True)
    raise SystemExit(2)


def info(msg: str) -> None:
    print(f"[session] {msg}", flush=True)


def win_to_wsl(p: str) -> str:
    r"""C:\Users\x\y  ->  /mnt/c/Users/x/y  (lowercase drive letter, spaces preserved).
    Only for absolute Windows paths; a path already POSIX-style is returned unchanged."""
    p = str(p)
    if p.startswith("/"):
        return p
    pw = PureWindowsPath(p)
    if not pw.drive or not pw.is_absolute():
        die(f"win_to_wsl needs an absolute Windows path, got: {p!r}")
    drive = pw.drive[0].lower()
    rest = p[len(pw.drive):].replace("\\", "/").lstrip("/")
    return f"/mnt/{drive}/{rest}"


def ffprobe_duration(path: str) -> float:
    """Exact media duration in seconds (metadata only - no decode). Fails loud if absent."""
    if not Path(path).exists():
        die(f"file not found for ffprobe: {path}")
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    s = out.stdout.strip()
    if not s:
        die(f"ffprobe returned no duration for: {path}")
    return float(s)


def wsl_base_cmd(distro: str) -> list[str]:
    return ["wsl.exe", "-d", distro, "-e"]


def wsl_capture(distro: str, argv: list[str]) -> subprocess.CompletedProcess:
    """Run a Linux program in WSL WITHOUT a shell (no var/PATH pitfalls). Captured."""
    return subprocess.run(wsl_base_cmd(distro) + argv,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def wsl_bash(distro: str, script: str) -> subprocess.CompletedProcess:
    """Run a bash -c one-liner in WSL. Caller MUST use literal paths (no shell vars) and
    should export a clean PATH itself when conda is involved."""
    return subprocess.run(wsl_base_cmd(distro) + ["bash", "-c", script],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def stream(argv: list[str], logfile: Path | None = None, env: dict | None = None,
           cwd: str | None = None) -> int:
    """Run a command, echoing stdout live to the console AND (optionally) a log file.
    stderr is merged into stdout. Returns the exit code. No shell — no `2>NUL` traps."""
    if logfile is not None:
        logfile.parent.mkdir(parents=True, exist_ok=True)
    lf = open(logfile, "w", encoding="utf-8") if logfile else None
    try:
        proc = subprocess.Popen(
            argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
            env=env, cwd=cwd)
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            if lf:
                lf.write(line)
                lf.flush()
        return proc.wait()
    finally:
        if lf:
            lf.close()


# --------------------------------------------------------------------------- config
class PartSet:
    def __init__(self, cfg: "Session", d: dict):
        self.name = _req(d, "name", "part_sets[]")
        self.chunks_glob = _req(d, "chunks_glob", f"part_sets[{self.name}]")
        self.merged_wav = _req(d, "merged_wav", f"part_sets[{self.name}]")
        self.vtt = _req(d, "vtt", f"part_sets[{self.name}]")
        # manual exclusions: audio second-ranges never trained on (e.g. a cue whose
        # drift-audit correction was independently proven wrong). Each entry:
        # {"range": [start_s, end_s], "reason": "..."} — reason is REQUIRED so the
        # config documents itself.
        self.manual_excludes = []
        for entry in d.get("exclude_ranges_s", []):
            if not isinstance(entry, dict) or "range" not in entry or "reason" not in entry:
                die(f"[{self.name}] exclude_ranges_s entries need 'range' [a,b] and 'reason': {entry}")
            a, b = entry["range"]
            if not (isinstance(a, (int, float)) and isinstance(b, (int, float)) and b > a):
                die(f"[{self.name}] exclude_ranges_s bad range: {entry['range']}")
            self.manual_excludes.append((float(a), float(b)))
        # clip dir is derived unless explicitly overridden
        self.clip_dir = d.get("clip_dir") or str(
            Path(cfg.clips_root) / cfg.voice / self.name)

    @property
    def coverage_json(self) -> str:
        # bookforge-tts --report default: <out>.coverage.json REPLACES the .vtt suffix
        return str(Path(self.vtt).with_suffix(".coverage.json"))

    @property
    def verify_json(self) -> str:
        return self.vtt + ".verify.json"

    @property
    def merge_json(self) -> str:
        return self.merged_wav + ".merge.json"

    def wsl_subdir(self, cfg: "Session") -> str:
        return f"{cfg.wsl_dataset_dir}/{self.name}"

    def source_chunks(self) -> list[Path]:
        """Sorted, deduped list of source chunks. Fails loud if the glob matches nothing."""
        # split the glob into a base dir + pattern so we can glob a real directory
        gp = PureWindowsPath(self.chunks_glob)
        base = Path(str(gp.parent))
        pattern = gp.name
        if not base.is_dir():
            die(f"[{self.name}] chunk dir does not exist: {base}")
        hits = sorted(base.glob(pattern))
        if not hits:
            die(f"[{self.name}] chunks_glob matched no files: {self.chunks_glob}")
        return hits


def _req(d: dict, key: str, where: str):
    if key not in d or d[key] in (None, ""):
        die(f"config: missing required key '{key}' in {where}")
    return d[key]


class Session:
    def __init__(self, path: Path):
        if not path.exists():
            die(f"config not found: {path}")
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            die(f"config is not valid JSON: {path}: {e}")
        self.path = path
        self.voice = _req(d, "voice", "config")
        self.token = d.get("token", self.voice)   # prompt token; defaults to voice NAME
        self.epub = _req(d, "epub", "config")
        self.mix = _req(d, "mix", "config")
        self.max_clip = float(_req(d, "max_clip", "config"))
        if self.max_clip > 20.0:
            die(f"max_clip {self.max_clip} > 20s - PROVEN-BROKEN regime (EOS failure). "
                "Never train above 20s / max_seq_length 2048 (README recipe).")
        self.target_lufs = float(d.get("target_lufs", -20.0))
        self.target_minutes = float(d.get("target_minutes", 0.0))  # 0 = keep whole book
        self.clips_root = _req(d, "clips_root", "config")
        self.wsl_dataset_dir = _req(d, "wsl_dataset_dir", "config")
        self.out_base = _req(d, "out_base", "config")
        self.log_dir = d.get("log_dir", str(HERE / "logs"))
        # tool locations
        self.bookforge_repo = _req(d, "bookforge_repo", "config")
        self.windows_python = _req(d, "windows_python", "config")     # e2a python_env for cut
        self.whisperx_python = _req(d, "whisperx_python", "config")   # faster-whisper for verify
        self.align_python = d.get("align_python", "python")           # drives bookforge-tts
        # WSL
        # align_workers: opt-in wav2vec2 worker count for the align stage (each
        # budgets ~5GB; bookforge auto-sizing reserves 12GB for a concurrent WSL
        # vLLM lane and often picks 1). Set only for sessions where aligns run
        # while the GPU/WSL lane is idle; the pool self-shrinks under pressure.
        self.align_workers = d.get("align_workers")
        if self.align_workers is not None and (not isinstance(self.align_workers, int)
                                               or self.align_workers < 1):
            die(f"config: align_workers must be a positive integer, got {self.align_workers!r}")
        self.wsl_distro = d.get("wsl_distro", "Ubuntu")
        self.wsl_conda_prefix = d.get("wsl_conda_prefix", "/home/telltale/anaconda3")
        self.wsl_train_env = d.get("wsl_train_env", "orpheus_train")
        # preflight
        t = d.get("train", {})
        self.min_free_vram_mib = int(t.get("min_free_vram_mib", DEFAULT_MIN_FREE_VRAM_MIB))
        self.gpu_consumer_mib = int(t.get("gpu_consumer_mib", DEFAULT_GPU_CONSUMER_MIB))

        raw_ps = _req(d, "part_sets", "config")
        if not isinstance(raw_ps, list) or not raw_ps:
            die("config: 'part_sets' must be a non-empty list")
        self.part_sets = [PartSet(self, ps) for ps in raw_ps]
        names = [p.name for p in self.part_sets]
        if len(names) != len(set(names)):
            die(f"config: duplicate part-set names: {names}")

    @property
    def wsl_train_python(self) -> str:
        return f"{self.wsl_conda_prefix}/envs/{self.wsl_train_env}/bin/python"

    @property
    def wsl_ds_build_dir(self) -> str:
        return f"{self.wsl_dataset_dir}_ds"

    @property
    def wsl_lora_dir(self) -> str:
        return f"{self.out_base}/orpheus_{self.voice}_lora"

    @property
    def wsl_merged_dir(self) -> str:
        return f"{self.out_base}/orpheus_{self.voice}_merged"

    def select(self, name: str | None) -> list[PartSet]:
        if name is None:
            return self.part_sets
        for p in self.part_sets:
            if p.name == name:
                return [p]
        die(f"--part-set {name!r} not in config (have: {[p.name for p in self.part_sets]})")


# --------------------------------------------------------------------------- merge
def _sum_chunk_durations(ps: PartSet) -> tuple[float, list[tuple[str, float]]]:
    parts = []
    total = 0.0
    for c in ps.source_chunks():
        dur = ffprobe_duration(str(c))
        parts.append((str(c), dur))
        total += dur
    return total, parts


def merge_state(ps: PartSet, tol: float) -> tuple[bool, str]:
    """(done, detail). done == merged wav exists AND its duration == sum(parts) within tol."""
    if not Path(ps.merged_wav).exists():
        return False, "merged wav missing"
    total, _ = _sum_chunk_durations(ps)
    merged = ffprobe_duration(ps.merged_wav)
    delta = abs(merged - total)
    if delta > tol:
        return False, (f"DURATION MISMATCH: merged {merged:.6f}s vs sum {total:.6f}s "
                       f"(delta {delta*1000:.1f} ms > tol {tol*1000:.1f} ms)")
    return True, f"merged {merged:.3f}s == sum {total:.3f}s (delta {delta*1000:.2f} ms)"


def cmd_merge(sess: Session, args) -> int:
    tol = args.merge_tol
    for ps in sess.select(args.part_set):
        done, detail = merge_state(ps, tol)
        if done and not args.force:
            info(f"[{ps.name}] merge already done: {detail}")
            continue
        if Path(ps.merged_wav).exists() and not done and not args.force:
            # exists but INCONSISTENT — never silently overwrite/accept. Fail loud.
            die(f"[{ps.name}] merged wav exists but is inconsistent: {detail}. "
                f"Re-run with --force to rebuild, or investigate the sources.")
        total, parts = _sum_chunk_durations(ps)
        info(f"[{ps.name}] merging {len(parts)} chunks -> {ps.merged_wav} "
             f"(expected {total:.3f}s)")
        # ffmpeg concat demuxer, stream-copy (sample-accurate for PCM wav).
        listfd, listpath = tempfile.mkstemp(suffix=".txt", prefix="concat_")
        os.close(listfd)
        try:
            with open(listpath, "w", encoding="utf-8") as f:
                for p, _ in parts:
                    # concat demuxer: single-quote the path, forward slashes on Windows
                    f.write("file '" + str(p).replace("\\", "/").replace("'", "'\\''") + "'\n")
            rc = stream(["ffmpeg", "-nostdin", "-hide_banner", "-y", "-f", "concat",
                         "-safe", "0", "-i", listpath, "-c", "copy", ps.merged_wav])
            if rc != 0:
                die(f"[{ps.name}] ffmpeg concat failed (exit {rc})")
        finally:
            os.remove(listpath)
        done, detail = merge_state(ps, tol)
        if not done:
            die(f"[{ps.name}] post-merge verification FAILED: {detail}")
        Path(ps.merge_json).write_text(json.dumps({
            "part_set": ps.name, "merged_wav": ps.merged_wav,
            "expected_sum_s": round(total, 6),
            "merged_s": round(ffprobe_duration(ps.merged_wav), 6),
            "parts": [{"file": p, "duration_s": round(d, 6)} for p, d in parts],
            "created": _dt.datetime.now().isoformat(timespec="seconds"),
        }, indent=2), encoding="utf-8")
        info(f"[{ps.name}] merge OK: {detail}")
    return 0


# --------------------------------------------------------------------------- align
def align_state(ps: PartSet) -> tuple[bool, str]:
    v = Path(ps.vtt).exists()
    c = Path(ps.coverage_json).exists()
    if v and c:
        return True, "vtt + coverage present"
    if v and not c:
        return False, "vtt present but coverage json missing (align in progress or ran without --report)"
    return False, "vtt missing"


def cmd_align(sess: Session, args) -> int:
    for ps in sess.select(args.part_set):
        done, detail = align_state(ps)
        if done and not args.force:
            info(f"[{ps.name}] align already done: {detail}")
            continue
        # merge must be complete first
        m_done, m_detail = merge_state(ps, args.merge_tol)
        if not m_done:
            die(f"[{ps.name}] cannot align: merge not complete ({m_detail})")
        cmd = [sess.align_python,
               str(Path(sess.bookforge_repo) / "cli" / "bookforge-tts.py"),
               "--generate-sentences",
               "--audio", ps.merged_wav,
               "--epub", sess.epub,
               "--out", ps.vtt,
               "--report",
               "--rough-cache"]   # re-runs skip the ~13-min transcribe pass
        if sess.align_workers is not None:
            cmd += ["--align-workers", str(sess.align_workers)]
        info(f"[{ps.name}] align command:\n  " + " ".join(f'"{c}"' if " " in c else c for c in cmd))
        if args.print_only:
            continue
        if Path(ps.vtt).exists() and not args.force:
            die(f"[{ps.name}] {ps.vtt} exists (an align job may be writing it). "
                f"Refusing to clobber - pass --force to regenerate.")
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        log = Path(sess.log_dir) / f"{sess.voice}_{ps.name}_align.log"
        rc = stream(cmd, logfile=log, env=env)
        if rc != 0:
            die(f"[{ps.name}] alignment failed (exit {rc}); log: {log}")
        done, detail = align_state(ps)
        if not done:
            die(f"[{ps.name}] align ran but outputs incomplete: {detail}")
        info(f"[{ps.name}] align OK: {detail}")
    return 0


# --------------------------------------------------------------------------- verify
def parse_drift_self_check(coverage_json: str) -> dict:
    """Pull the driftSelfCheck cue-offset stats out of a bookforge coverage JSON."""
    if not Path(coverage_json).exists():
        die(f"coverage json missing (run align with --report): {coverage_json}")
    try:
        d = json.loads(Path(coverage_json).read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        die(f"coverage json not valid JSON: {coverage_json}: {e}")
    dsc = d.get("driftSelfCheck")
    if not isinstance(dsc, dict):
        die(f"coverage json has no driftSelfCheck section: {coverage_json}")
    # tolerate a few key spellings the report has used; require SOMETHING numeric for max
    def pick(*keys):
        for k in keys:
            if k in dsc and isinstance(dsc[k], (int, float)):
                return float(dsc[k])
        return None
    out = {"median": pick("median", "medianOffset", "medianOffsetS", "medianAbsSeconds"),
           "p95": pick("p95", "p95Offset", "p95OffsetS", "p95AbsSeconds"),
           "max": pick("max", "maxOffset", "maxOffsetS", "maxAbsSeconds"),
           "corrected_cues": pick("correctedCues") or 0.0,
           "correction_threshold": pick("correctionThresholdSeconds")}
    if out["max"] is None:
        die(f"driftSelfCheck has no numeric max offset: {coverage_json} -> {dsc}")
    # times the audit MOVED cues to — independently re-verify exactly there
    out["corrected_times_s"] = []
    for c in dsc.get("corrected", []):
        t = c.get("movedTo")
        if isinstance(t, str):
            p = t.split(":")
            out["corrected_times_s"].append(
                round(int(p[0]) * 3600 + int(p[1]) * 60 + float(p[2]), 1))
    return out


def audio_not_in_epub_ranges(coverage_json: str) -> list[tuple[float, float]]:
    """audioNotInEpub holes = narration with no book text (ads/foreword/branding),
    filled with whisper-fallback ASR cues. NEVER training material — cut excludes them."""
    d = json.loads(Path(coverage_json).read_text(encoding="utf-8"))
    holes = d.get("audioNotInEpub")
    if holes is None:
        die(f"coverage json has no audioNotInEpub section: {coverage_json}")
    if isinstance(holes, dict):
        holes = [holes]
    out = []
    for h in holes:
        a, b = h.get("audioStart"), h.get("audioEnd")
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            die(f"audioNotInEpub entry missing audioStart/audioEnd: {h}")
        out.append((float(a), float(b)))
    return out


def parse_verify_summary(text: str) -> dict:
    """Parse verify_vtt.py's summary line: USABLE=.. SKIPPED=.. MEDOFF=.. P95OFF=.. MAXOFF=.."""
    m = re.search(r"USABLE=(\d+)\s+SKIPPED=(\d+)\s+MEDOFF=([-\d.]+)\s+"
                  r"P95OFF=([-\d.]+)\s+MAXOFF=([-\d.]+)", text)
    if not m:
        die("could not find verify_vtt.py summary line (USABLE=... MAXOFF=...) in its output")
    return {"usable": int(m.group(1)), "skipped": int(m.group(2)),
            "medoff": float(m.group(3)), "p95off": float(m.group(4)),
            "maxoff": float(m.group(5))}


def chunk_join_seconds(ps: PartSet) -> list[float]:
    """Cumulative durations = the times in the merged wav where one source chunk joins
    the next. verify_vtt.py probes these explicitly (a join is where drift can appear)."""
    _, parts = _sum_chunk_durations(ps)
    joins, acc = [], 0.0
    for _, d in parts[:-1]:
        acc += d
        joins.append(round(acc, 1))
    return joins


def waveform_spotcheck(ps: PartSet) -> dict:
    """Cheap, whisper-free drift proxy: sample WF_SAMPLES cue boundaries evenly across
    the VTT and confirm each sits within WF_TOLERANCE_S of a silence trough. Gross drift
    parks a boundary deep in continuous speech -> no nearby low-RMS point."""
    try:
        import numpy as np
    except ModuleNotFoundError:
        die("verify's waveform spot-check needs numpy - run session.py under a python that "
            "has it (e.g. the e2a python_env in config's windows_python), not a bare python.")
    cues = _parse_vtt_starts(ps.vtt)
    if len(cues) < 4:
        die(f"[{ps.name}] VTT has too few cues to spot-check: {ps.vtt}")
    idxs = sorted(set(int(round(x)) for x in np.linspace(0, len(cues) - 1, WF_SAMPLES)))
    near = 0
    checked = 0
    worst = 0.0
    for i in idxs:
        st = cues[i]
        a = max(0.0, st - 0.30)
        pcm = subprocess.run(
            ["ffmpeg", "-nostdin", "-hide_banner", "-v", "error", "-y",
             "-ss", f"{a:.3f}", "-t", "0.60", "-i", ps.merged_wav,
             "-ac", "1", "-ar", "24000", "-f", "f32le", "-"],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout
        y = np.frombuffer(pcm, dtype=np.float32)
        if y.size < 24000 // 20:
            continue
        checked += 1
        sr = 24000
        win = int(0.020 * sr)
        # RMS over sliding 20ms windows; find the quietest and where it is
        n = len(y) - win
        best_rms = 1.0
        best_pos = 0.0
        for j in range(0, n, win // 2 or 1):
            seg = y[j:j + win]
            rms = float(np.sqrt(np.mean(seg * seg)))
            if rms < best_rms:
                best_rms = rms
                best_pos = a + (j + win / 2) / sr
        off = abs(best_pos - st)
        worst = max(worst, off)
        if best_rms <= WF_SILENCE_RMS and off <= WF_TOLERANCE_S:
            near += 1
    frac = near / checked if checked else 0.0
    return {"checked": checked, "near_silence": near,
            "near_frac": round(frac, 3), "worst_offset_s": round(worst, 3)}


def _parse_vtt_starts(vtt: str) -> list[float]:
    def ts(t):
        p = t.strip().split(":")
        return (int(p[0]) * 3600 + int(p[1]) * 60 + float(p[2])) if len(p) == 3 \
            else int(p[0]) * 60 + float(p[1])
    starts = []
    if not Path(vtt).exists():
        die(f"VTT not found: {vtt}")
    for line in Path(vtt).read_text(encoding="utf-8").splitlines():
        m = re.match(r"(\d[\d:.]+)\s*-->", line)
        if m:
            starts.append(ts(m.group(1)))
    if not starts:
        die(f"VTT has no cues: {vtt}")
    return starts


def _vtt_stat(vtt: str) -> dict:
    """Identity of the VTT a verification applies to. If the VTT is regenerated
    (align --force), any prior verify.json must stop counting as a pass."""
    st = Path(vtt).stat()
    return {"size": st.st_size, "mtime_ns": st.st_mtime_ns}


def verify_state(ps: PartSet) -> tuple[bool, str]:
    if not Path(ps.verify_json).exists():
        return False, "no verify.json"
    try:
        d = json.loads(Path(ps.verify_json).read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False, "verify.json unreadable"
    if d.get("passed") is not True:
        return False, "verify.json present but FAILED"
    if not Path(ps.vtt).exists():
        return False, "verify.json present but VTT missing"
    if d.get("vtt_stat") != _vtt_stat(ps.vtt):
        return False, "verify.json STALE (VTT changed since verification) - re-run verify"
    ip = d.get("independent_probe", {})
    return True, f"PASSED (maxoff {ip.get('maxoff')}s)"


def cmd_verify(sess: Session, args) -> int:
    for ps in sess.select(args.part_set):
        done, detail = verify_state(ps)
        if done and not args.force:
            info(f"[{ps.name}] verify already done: {detail}")
            continue
        a_done, a_detail = align_state(ps)
        if not a_done:
            die(f"[{ps.name}] cannot verify: align not complete ({a_detail})")

        drift = parse_drift_self_check(ps.coverage_json)
        info(f"[{ps.name}] driftSelfCheck: median={drift['median']} p95={drift['p95']} "
             f"max={drift['max']}")

        # dense independent whisper probe (verify_vtt.py). Extra targets: chunk joins
        # (where merge drift would appear) + every spot the drift audit CORRECTED
        # (the corrections themselves must be independently confirmed).
        joins = chunk_join_seconds(ps)
        extra = joins + drift["corrected_times_s"]
        cmd = [sess.whisperx_python, str(VERIFY_VTT), ps.vtt, ps.merged_wav] + \
              [f"{j:.1f}" for j in extra]
        # cues inside excluded ranges will never be cut — the gate verifies what
        # trains, so the probe must not pick them as candidates.
        if ps.manual_excludes:
            cmd.append("--exclude=" + ",".join(f"{a:.2f}-{b:.2f}"
                                               for a, b in ps.manual_excludes))
        info(f"[{ps.name}] independent probe (this is slow - whisper on CPU): "
             + " ".join(cmd))
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding="utf-8", errors="replace", env=env)
        sys.stdout.write(proc.stdout)
        probe = parse_verify_summary(proc.stdout)

        wf = waveform_spotcheck(ps)
        info(f"[{ps.name}] waveform spot-check: {wf['near_silence']}/{wf['checked']} "
             f"boundaries near silence (frac {wf['near_frac']}, worst {wf['worst_offset_s']}s)")

        warnings = []
        if probe["maxoff"] > WARN_MAXOFF_S:
            warnings.append(f"independent maxoff {probe['maxoff']}s > warn {WARN_MAXOFF_S}s")
        if drift["p95"] is not None and drift["p95"] > WARN_MAXOFF_S:
            warnings.append(f"drift p95 {drift['p95']}s > warn {WARN_MAXOFF_S}s")

        fails = []
        if probe["maxoff"] > FAIL_MAXOFF_S:
            fails.append(f"independent maxoff {probe['maxoff']}s > FAIL {FAIL_MAXOFF_S}s")
        # driftSelfCheck.max is measured BEFORE the audit's own corrections. If cues
        # were corrected, the residual is bounded by the correction threshold and the
        # corrected spots are independently probed above. Uncorrected >threshold = fail.
        if drift["max"] > FAIL_DRIFT_MAX_S and not drift["corrected_cues"]:
            fails.append(f"driftSelfCheck max {drift['max']}s > FAIL {FAIL_DRIFT_MAX_S}s "
                         f"and NOTHING was corrected")
        if drift["correction_threshold"] is not None \
                and drift["correction_threshold"] > FAIL_DRIFT_MAX_S:
            fails.append(f"drift correction threshold {drift['correction_threshold']}s "
                         f"looser than FAIL {FAIL_DRIFT_MAX_S}s - residual unbounded")
        if wf["checked"] and wf["near_frac"] < WF_MIN_NEAR_FRAC:
            fails.append(f"only {wf['near_frac']} of boundaries near silence "
                         f"< {WF_MIN_NEAR_FRAC} (gross drift)")

        passed = not fails
        record = {
            "part_set": ps.name, "vtt": ps.vtt, "vtt_stat": _vtt_stat(ps.vtt),
            "audio": ps.merged_wav,
            "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
            "drift_self_check": drift,
            "independent_probe": probe,
            "waveform_spotcheck": wf,
            "chunk_joins_s": joins,
            "thresholds": {"fail_maxoff_s": FAIL_MAXOFF_S, "warn_maxoff_s": WARN_MAXOFF_S,
                           "fail_drift_max_s": FAIL_DRIFT_MAX_S},
            "warnings": warnings, "failures": fails, "passed": passed,
        }
        Path(ps.verify_json).write_text(json.dumps(record, indent=2), encoding="utf-8")
        for w in warnings:
            info(f"[{ps.name}] WARN: {w}")
        if not passed:
            die(f"[{ps.name}] VERIFICATION FAILED: {'; '.join(fails)}. "
                f"Artifact: {ps.verify_json}")
        info(f"[{ps.name}] VERIFY PASSED -> {ps.verify_json}")
    return 0


# --------------------------------------------------------------------------- cut
def _measure_lufs(path: str) -> float:
    """Integrated loudness (LUFS) of the whole file via ffmpeg ebur128. Used to compute a
    STATIC gain to target LUFS — README rule #4: never per-clip loudnorm (dynamic gain
    bakes amplitude wobble into the training data)."""
    proc = subprocess.run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-i", path, "-af", "ebur128",
         "-f", "null", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        encoding="utf-8", errors="replace")
    # ebur128 prints a "Summary:" block ending with "    I:  -XX.X LUFS"
    matches = re.findall(r"I:\s*(-?\d+(?:\.\d+)?)\s*LUFS", proc.stdout)
    if not matches:
        die(f"could not measure integrated loudness of {path} (ffmpeg ebur128 gave no I:)")
    return float(matches[-1])


def cut_state(ps: PartSet, sess: Session) -> tuple[bool, str]:
    tr = Path(ps.clip_dir) / "metadata_train.csv"
    ev = Path(ps.clip_dir) / "metadata_eval.csv"
    wavs = Path(ps.clip_dir) / "wavs"
    if tr.exists() and ev.exists() and wavs.is_dir():
        n = len(list(wavs.glob("*.wav")))
        if n > 0:
            return True, f"{n} clips in {ps.clip_dir}"
        return False, "clip dir present but no wavs"
    return False, "no clip dir / CSVs"


def cmd_cut(sess: Session, args) -> int:
    for ps in sess.select(args.part_set):
        done, detail = cut_state(ps, sess)
        if done and not args.force:
            info(f"[{ps.name}] cut already done: {detail}")
            continue
        # GATE: verify must have passed for this part-set
        v_done, v_detail = verify_state(ps)
        if not v_done:
            die(f"[{ps.name}] cut BLOCKED - verification gate not passed: {v_detail}. "
                f"Run `verify` first (fix drift before cutting).")

        # static source gain to target LUFS (NOT per-clip loudnorm)
        measured = _measure_lufs(ps.merged_wav)
        gain_db = round(sess.target_lufs - measured, 2)
        info(f"[{ps.name}] measured {measured:.2f} LUFS -> static gain {gain_db:+.2f} dB "
             f"to reach {sess.target_lufs:.1f} LUFS")

        # --source-name = the PROMPT TOKEN baked into training (config `token`, not the
        # session/voice name — a mismatch is the trained-rohan/served-deathstalker bug).
        cmd = [sess.windows_python, str(CUT_AUDIOBOOK),
               "--vtt", ps.vtt, "--audio", ps.merged_wav, "--epub", sess.epub,
               "--out-dir", ps.clip_dir, "--source-name", sess.token,
               "--mix", sess.mix, "--max-clip", str(sess.max_clip),
               "--trail-cap", "0.1", "--target-minutes", str(sess.target_minutes),
               "--gain-db", str(gain_db)]
        # audioNotInEpub holes carry whisper-fallback ASR cues (not book truth) —
        # exclude them from cutting so ASR text never trains. Manual config
        # exclusions (proven-bad cues) are merged in.
        excl = audio_not_in_epub_ranges(ps.coverage_json) + ps.manual_excludes
        if excl:
            rng = ",".join(f"{a:.2f}-{b:.2f}" for a, b in excl)
            info(f"[{ps.name}] excluding audio ranges (ASR holes + manual): {rng}")
            cmd += ["--exclude-ranges", rng]
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        log = Path(sess.log_dir) / f"{sess.voice}_{ps.name}_cut.log"
        info(f"[{ps.name}] cutting -> {ps.clip_dir}")
        rc = stream(cmd, logfile=log, env=env)
        if rc != 0:
            die(f"[{ps.name}] cut_audiobook.py failed (exit {rc}); log: {log}")

        # validate the fresh dataset
        vcmd = [sess.windows_python, str(VALIDATE_DATASET), "--ds-dir", ps.clip_dir,
                "--sample-out", str(Path(ps.clip_dir) / "_sample")]
        rc = stream(vcmd, env=env)
        if rc != 0:
            die(f"[{ps.name}] validate_dataset.py failed (exit {rc})")

        done, detail = cut_state(ps, sess)
        if not done:
            die(f"[{ps.name}] cut ran but produced no dataset: {detail}")
        info(f"[{ps.name}] cut OK: {detail}")
    return 0


# --------------------------------------------------------------------------- stage
def _win_wav_count(clip_dir: str) -> int:
    w = Path(clip_dir) / "wavs"
    return len(list(w.glob("*.wav"))) if w.is_dir() else 0


def _wsl_wav_count(sess: Session, wsl_dir: str) -> int | None:
    proc = wsl_bash(sess.wsl_distro,
                    f"find '{wsl_dir}/wavs' -name '*.wav' 2>/dev/null | wc -l")
    if proc.returncode != 0:
        return None
    try:
        return int(proc.stdout.strip())
    except ValueError:
        return None


def stage_state(sess: Session) -> tuple[bool, str]:
    """Staged == combined CSVs exist on WSL AND each part-set's WSL wav count matches the
    Windows clip dir (staleness guard)."""
    tr = wsl_bash(sess.wsl_distro,
                  f"test -f '{sess.wsl_dataset_dir}/metadata_train.csv' && echo Y || echo N")
    out = tr.stdout.strip()
    if tr.returncode != 0 or out not in ("Y", "N"):
        return False, "WSL unreachable"
    if out != "Y":
        return False, "combined metadata_train.csv missing on WSL"
    stale = []
    for ps in sess.part_sets:
        win_n = _win_wav_count(ps.clip_dir)
        wsl_n = _wsl_wav_count(sess, ps.wsl_subdir(sess))
        if win_n == 0:
            return False, f"[{ps.name}] not cut yet (0 windows on Windows)"
        if wsl_n != win_n:
            stale.append(f"{ps.name} win={win_n} wsl={wsl_n}")
    if stale:
        return False, "STALE: " + "; ".join(stale)
    return True, "all part-sets staged, CSVs present, counts match"


def _combined_csv_text(sess: Session, which: str) -> str:
    """Concatenate per-part-set CSVs into one, rewriting rel paths to '<partset>/wavs/x'."""
    header = "audio_file|text|speaker_name"
    lines = [header]
    for ps in sess.part_sets:
        src = Path(ps.clip_dir) / which
        if not src.exists():
            die(f"[{ps.name}] {which} missing - cut the part-set first: {src}")
        rows = src.read_text(encoding="utf-8").splitlines()
        if not rows or rows[0].strip() != header:
            die(f"[{ps.name}] {which} has unexpected header: {rows[0] if rows else '<empty>'}")
        for r in rows[1:]:
            if not r.strip():
                continue
            rel, rest = r.split("|", 1)
            if not rel.startswith("wavs/"):
                die(f"[{ps.name}] {which} row has unexpected path {rel!r}")
            lines.append(f"{ps.name}/{rel}|{rest}")
    return "\n".join(lines) + "\n"


def cmd_stage(sess: Session, args) -> int:
    done, detail = stage_state(sess)
    if done and not args.force:
        info(f"stage already done: {detail}")
        return 0
    # all part-sets must be cut
    for ps in sess.part_sets:
        c_done, c_detail = cut_state(ps, sess)
        if not c_done:
            die(f"[{ps.name}] cannot stage: not cut ({c_detail})")

    mk = wsl_capture(sess.wsl_distro, ["mkdir", "-p", sess.wsl_dataset_dir])
    if mk.returncode != 0:
        die(f"mkdir -p {sess.wsl_dataset_dir} failed on WSL: {mk.stderr.strip()}")

    for ps in sess.part_sets:
        src_wsl = win_to_wsl(ps.clip_dir)
        dst = ps.wsl_subdir(sess)
        win_n = _win_wav_count(ps.clip_dir)
        wsl_n = _wsl_wav_count(sess, dst)
        if wsl_n == win_n and win_n > 0 and not args.force:
            info(f"[{ps.name}] already staged ({wsl_n} wavs) - skipping copy")
            continue
        info(f"[{ps.name}] staging {win_n} clips: {src_wsl} -> {dst}")
        # copy runs INSIDE WSL (ext4<-/mnt, fast); literal paths, no shell vars.
        rm = wsl_capture(sess.wsl_distro, ["rm", "-rf", dst])
        if rm.returncode != 0:
            die(f"[{ps.name}] rm -rf {dst} failed: {rm.stderr.strip()}")
        cp = wsl_capture(sess.wsl_distro, ["cp", "-r", src_wsl, dst])
        if cp.returncode != 0:
            die(f"[{ps.name}] cp failed: {cp.stderr.strip()}")
        after = _wsl_wav_count(sess, dst)
        if after != win_n:
            die(f"[{ps.name}] post-copy count mismatch: win={win_n} wsl={after}")

    # write combined CSVs (build content on Windows, then cp into WSL — tiny files)
    for which in ("metadata_train.csv", "metadata_eval.csv"):
        text = _combined_csv_text(sess, which)
        fd, tmp = tempfile.mkstemp(suffix=".csv", prefix="combined_")
        os.close(fd)
        try:
            Path(tmp).write_text(text, encoding="utf-8")
            cp = wsl_capture(sess.wsl_distro,
                             ["cp", win_to_wsl(tmp), f"{sess.wsl_dataset_dir}/{which}"])
            if cp.returncode != 0:
                die(f"failed to copy combined {which} into WSL: {cp.stderr.strip()}")
        finally:
            os.remove(tmp)
        n = text.count("\n") - 1
        info(f"staged combined {which}: {n} rows -> {sess.wsl_dataset_dir}/{which}")

    done, detail = stage_state(sess)
    if not done:
        die(f"stage completed but verification failed: {detail}")
    info(f"stage OK: {detail}")
    return 0


# --------------------------------------------------------------------------- train preflight
def _nvidia_free_and_apps() -> tuple[int, list[tuple[str, int]]]:
    """(free_mib, [(process_name, used_mib), ...]) from nvidia-smi on Windows."""
    q = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if q.returncode != 0:
        die(f"nvidia-smi failed - cannot verify GPU state: {q.stderr.strip()}")
    free = int(q.stdout.strip().splitlines()[0])
    a = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=process_name,used_memory",
         "--format=csv,noheader,nounits"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    apps = []
    for line in a.stdout.strip().splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[1].isdigit():
            apps.append((parts[0], int(parts[1])))
    return free, apps


def _busy_python_jobs(sess: Session) -> list[str]:
    """Windows + WSL python processes whose cmdline looks like a heavy TTS/align/train job."""
    hits = []
    # Windows: PowerShell CIM query for python cmdlines
    ps = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "Get-CimInstance Win32_Process -Filter \"Name='python.exe' OR Name='pythonw.exe'\""
         " | Select-Object -ExpandProperty CommandLine"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    for line in (ps.stdout or "").splitlines():
        low = line.lower()
        if any(pat in low for pat in BUSY_JOB_PATTERNS) and "session.py" not in low:
            hits.append("win: " + line.strip())
    # WSL: pgrep -af python
    wp = wsl_bash(sess.wsl_distro, "pgrep -af python || true")
    for line in (wp.stdout or "").splitlines():
        low = line.lower()
        if any(pat in low for pat in BUSY_JOB_PATTERNS):
            hits.append("wsl: " + line.strip())
    return hits


def train_preflight(sess: Session) -> None:
    # (c) verification gate passed for every included part-set
    for ps in sess.part_sets:
        v_done, v_detail = verify_state(ps)
        if not v_done:
            die(f"[{ps.name}] train BLOCKED - verify gate not passed: {v_detail}")
    # (d) WSL dataset present + not stale vs Windows clips
    s_done, s_detail = stage_state(sess)
    if not s_done:
        die(f"train BLOCKED - dataset not staged / stale: {s_detail}. Run `stage`.")
    # (a) GPU headroom
    free, apps = _nvidia_free_and_apps()
    foreign = [(n, m) for (n, m) in apps if m >= sess.gpu_consumer_mib]
    info(f"GPU: {free} MiB free; compute-apps: {apps if apps else 'none'}")
    if foreign:
        die(f"train BLOCKED - GPU already in use by: {foreign}. Free the GPU first.")
    if free < sess.min_free_vram_mib:
        die(f"train BLOCKED - only {free} MiB free < required {sess.min_free_vram_mib} MiB.")
    # (b) no other heavy python job running
    busy = _busy_python_jobs(sess)
    if busy:
        die("train BLOCKED - another training/align job appears to be running:\n  "
            + "\n  ".join(busy))
    info("preflight PASSED - GPU free, no competing jobs, dataset staged & verified.")


def cmd_train(sess: Session, args) -> int:
    train_preflight(sess)
    if args.dry_run_preflight:
        info("preflight only (--dry-run-preflight) - not launching training.")
        return 0

    py = sess.wsl_train_python
    oo = win_to_wsl(str(ORPHEUS_OWEN))
    clean_path = f"export PATH={sess.wsl_conda_prefix}/bin:/usr/bin:/bin; "
    # cwd MUST be outside any git repo (unsloth writes unsloth_compiled_cache/ into cwd).
    cwd = sess.out_base
    common = (f"{clean_path}export PYTHONIOENCODING=utf-8; cd '{cwd}'; ")

    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    logdir = Path(sess.log_dir)
    build_log = logdir / f"{sess.voice}_{ts}_build.log"
    train_log = logdir / f"{sess.voice}_{ts}_train.log"

    # 1. build: manifest + SNAC-rate measurement over the COMBINED dataset
    build_cmd = (f"{common}'{py}' '{oo}' --source-name '{sess.token}' "
                 f"--dataset-dir '{sess.wsl_dataset_dir}' "
                 f"--output-dir '{sess.wsl_ds_build_dir}' build")
    info(f"[train] build -> log {build_log}")
    rc = stream(wsl_base_cmd(sess.wsl_distro) + ["bash", "-c", build_cmd], logfile=build_log)
    if rc != 0:
        die(f"[train] build failed (exit {rc}); log: {build_log}")

    # 2. train: v4 recipe on the COMBINED CSVs (read from --recut-dir), merge to 16-bit
    train_cmd = (f"{common}'{py}' '{oo}' --source-name '{sess.token}' "
                 f"--recut-dir '{sess.wsl_dataset_dir}' --out-base '{sess.out_base}' "
                 f"--mask-prompt-loss --no-dedup --lr-schedule constant_with_warmup "
                 f"--merge train")
    info(f"[train] launching WSL training (env {sess.wsl_train_env}) -> log {train_log}")
    info(f"[train]   {train_cmd}")
    rc = stream(wsl_base_cmd(sess.wsl_distro) + ["bash", "-c", train_cmd], logfile=train_log)
    if rc != 0:
        die(f"[train] training failed (exit {rc}); log: {train_log}")

    _report_training(train_log, sess)
    return 0


def _report_training(train_log: Path, sess: Session) -> None:
    text = train_log.read_text(encoding="utf-8", errors="replace")
    # per-epoch eval_loss lines that HF Trainer prints, e.g. {'eval_loss': 3.63, 'epoch': 3.0}
    rows = []
    for m in re.finditer(r"\{[^{}]*'eval_loss':\s*([-\d.]+)[^{}]*'epoch':\s*([-\d.]+)[^{}]*\}",
                         text):
        rows.append((float(m.group(2)), float(m.group(1))))
    print("\n================ TRAINING SUMMARY ================")
    if rows:
        print(f"  {'epoch':>6}  {'eval_loss':>10}")
        best = min(rows, key=lambda r: r[1])
        for ep, el in rows:
            mark = "  <- best" if (ep, el) == best else ""
            print(f"  {ep:6.1f}  {el:10.4f}{mark}")
    else:
        print("  (no per-epoch eval_loss lines parsed - check the log)")
    mb = re.search(r"BEST epoch checkpoint = (\S+) \(eval_loss ([-\d.]+)\)", text)
    if mb:
        print(f"  best checkpoint : {mb.group(1)} (eval_loss {mb.group(2)})")
    print(f"  best LoRA dir   : {sess.wsl_lora_dir}")
    print(f"  merged 16-bit   : {sess.wsl_merged_dir}   (vLLM; voice '{sess.token}')")
    print(f"  full log        : {train_log}")
    print("  next: ear-check, then `bash deploy_voice.sh " + sess.voice +
          " \"<Display Name>\"`  (deploy is a human step)")
    print("==================================================")


# --------------------------------------------------------------------------- audition
def cmd_audition(sess: Session, args) -> int:
    if not args.text:
        die("audition needs --text <passage.txt> (the A/B sample to render)")
    out = args.out or str(Path(sess.log_dir) / f"{sess.voice}_audition.wav")
    cmd = [sess.align_python,
           str(Path(sess.bookforge_repo) / "cli" / "bookforge-tts.py"),
           "--tts", "--engine=orpheus", f"--voice={sess.voice}",
           "--input", args.text, "--out", out]
    if args.tier:
        cmd.append(f"--tier={args.tier}")
    info("audition render (requires the merged voice installed in BookForge's models dir):")
    print("  " + " ".join(f'"{c}"' if " " in c else c for c in cmd))
    if not args.run:
        info("(pass --run to execute; otherwise this just prints the command)")
        return 0
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    rc = stream(cmd, env=env)
    if rc != 0:
        die(f"audition render failed (exit {rc})")
    info(f"audition wav -> {out}")
    return 0


# --------------------------------------------------------------------------- status
def cmd_status(sess: Session, args) -> int:
    print(f"\nSession: {sess.voice} (token '{sess.token}')  config {sess.path}")
    print(f"  epub: {sess.epub}")
    print(f"  mix {sess.mix}  max_clip {sess.max_clip}s  target {sess.target_lufs} LUFS  "
          f"target_minutes {sess.target_minutes}")
    print(f"  WSL dataset: {sess.wsl_dataset_dir}   out_base: {sess.out_base}")
    print("\n  part-set   merge   align   verify   cut     staged")
    print("  " + "-" * 52)
    for ps in sess.part_sets:
        m = "yes" if merge_state(ps, args.merge_tol)[0] else "no"
        a = "yes" if align_state(ps)[0] else "no"
        v_done, v_detail = verify_state(ps)
        v = ("PASS" if v_done else ("FAIL" if "FAILED" in v_detail else "no"))
        c = "yes" if cut_state(ps, sess)[0] else "no"
        win_n = _win_wav_count(ps.clip_dir)
        wsl_n = _wsl_wav_count(sess, ps.wsl_subdir(sess))
        st = "?" if wsl_n is None else ("yes" if (win_n and wsl_n == win_n) else "no")
        print(f"  {ps.name:<10} {m:<7} {a:<7} {v:<8} {c:<7} {st}")
    print("  " + "-" * 52)
    s_done, s_detail = stage_state(sess)
    print(f"  stage(all): {'yes' if s_done else 'no'}  ({s_detail})")
    # trained?
    tr = wsl_bash(sess.wsl_distro, f"test -d '{sess.wsl_merged_dir}' && echo Y || echo N")
    trained = tr.stdout.strip() == "Y" if tr.returncode == 0 else None
    print(f"  trained   : {'yes' if trained else ('?' if trained is None else 'no')}  "
          f"({sess.wsl_merged_dir})")
    # per-part-set detail lines
    print()
    for ps in sess.part_sets:
        for label, fn in (("merge", lambda: merge_state(ps, args.merge_tol)),
                          ("align", lambda: align_state(ps)),
                          ("verify", lambda: verify_state(ps)),
                          ("cut", lambda: cut_state(ps, sess))):
            ok, detail = fn()
            print(f"  [{ps.name}] {label:<7} {detail}")
    return 0


# --------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description="Orpheus voice-training session driver")
    ap.add_argument("--config", required=True, help="session JSON config")
    ap.add_argument("--part-set", default=None, help="limit to one part-set (default: all)")
    ap.add_argument("--force", action="store_true", help="redo a stage even if complete")
    ap.add_argument("--merge-tol", type=float, default=0.010,
                    help="merge duration tolerance in seconds (default 0.010 = 10 ms)")
    ap.add_argument("--print-only", action="store_true", help="align: print the command, don't run")
    ap.add_argument("--dry-run-preflight", action="store_true",
                    help="train: run preflight checks then stop (don't launch)")
    ap.add_argument("--text", default=None, help="audition: passage text file")
    ap.add_argument("--out", default=None, help="audition: output wav")
    ap.add_argument("--tier", default=None, help="audition: BookForge tier (fast/moderate/...)")
    ap.add_argument("--run", action="store_true", help="audition: actually render")
    ap.add_argument("cmd", choices=["merge", "align", "verify", "cut", "stage",
                                    "train", "audition", "status"])
    args = ap.parse_args()

    sess = Session(Path(args.config).resolve())
    dispatch = {"merge": cmd_merge, "align": cmd_align, "verify": cmd_verify,
                "cut": cmd_cut, "stage": cmd_stage, "train": cmd_train,
                "audition": cmd_audition, "status": cmd_status}
    return dispatch[args.cmd](sess, args)


if __name__ == "__main__":
    raise SystemExit(main())
