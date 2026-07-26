"""run_metrics.py — durable per-run analytics for the Orpheus training pipeline.

Append-only JSONL at logs/run_metrics.jsonl. Every stage (cut, build, train)
appends a record so past runs are auditable: stage durations, per-epoch train
time + eval_loss, and GPU health sampled each epoch (temp / SM clock / power /
util). A throttling or dying GPU shows up as rising temp + sagging SM clock +
ballooning epoch seconds — exactly the signature of the dead-fan incident.

Import from any stage. Every function is best-effort and NEVER raises into the
caller — analytics must not break a training run.

Review later:  python run_metrics.py            (pretty summary of recent runs)
               python run_metrics.py --json      (raw records)
"""
import json, os, subprocess, sys, time
from pathlib import Path

METRICS = Path(__file__).resolve().parent / "logs" / "run_metrics.jsonl"
RUNS_DIR = Path(__file__).resolve().parent / "logs" / "runs"

def save_run(name, data):
    """Write a self-contained per-run JSON file (what ran + when + full detail) to
    logs/runs/<name>_<YYYYmmdd_HHMMSS>.json. Returns the path (or None). Never raises."""
    try:
        import datetime as dt
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        p = RUNS_DIR / f"{name}_{stamp}.json"
        p.write_text(json.dumps({"saved_at": dt.datetime.now().isoformat(timespec="seconds"),
                                 **data}, indent=2), encoding="utf-8")
        return str(p)
    except Exception:
        return None

def gpu_snapshot():
    """Cheap nvidia-smi sample -> dict (temp/sm-clock/power/util/mem). {} on failure."""
    try:
        q = "temperature.gpu,clocks.sm,power.draw,utilization.gpu,memory.used"
        out = subprocess.run(["nvidia-smi", f"--query-gpu={q}",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        t, clk, pw, util, mem = [x.strip() for x in out.splitlines()[0].split(",")]
        return {"temp_c": float(t), "sm_mhz": float(clk), "power_w": float(pw),
                "util_pct": float(util), "mem_mib": float(mem)}
    except Exception:
        return {}

def record(stage, data):
    """Append one metrics record ({ts, stage, ...data}). Never raises."""
    try:
        METRICS.parent.mkdir(parents=True, exist_ok=True)
        with open(METRICS, "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": time.time(), "stage": stage, **data}) + "\n")
    except Exception:
        pass

class Stage:
    """Context manager: times a stage and records it. `s.extra` merges into the record.
        with Stage('cut', source='owen', vtt=name) as s: ...; s.extra['clips']=n
    """
    def __init__(self, stage, **fields):
        self.stage = stage; self.fields = fields; self.extra = {}
    def __enter__(self):
        self.t0 = time.time(); return self
    def __exit__(self, *exc):
        record(self.stage, {**self.fields, **self.extra,
                            "elapsed_s": round(time.time() - self.t0, 1),
                            "ok": exc[0] is None})
        return False  # don't suppress exceptions

# --------------------------------------------------------------------------- review
def _fmt_dur(s):
    s = int(s); h, s = divmod(s, 3600); m, s = divmod(s, 60)
    return (f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s")

def _summary():
    if not METRICS.exists():
        print("no metrics yet:", METRICS); return
    recs = [json.loads(l) for l in open(METRICS, encoding="utf-8") if l.strip()]
    import datetime as dt
    for r in recs[-40:]:
        when = dt.datetime.fromtimestamp(r["ts"]).strftime("%m-%d %H:%M")
        st = r["stage"]
        if st == "train":
            eps = r.get("epochs", [])
            temps = [e["gpu_temp_c"] for e in eps if e.get("gpu_temp_c") is not None]
            clks = [e["gpu_sm_mhz"] for e in eps if e.get("gpu_sm_mhz") is not None]
            etimes = [e.get("epoch_s") for e in eps if e.get("epoch_s")]
            health = ""
            if temps and clks:
                health = f" | gpu {min(clks):.0f}-{max(clks):.0f}MHz {min(temps):.0f}-{max(temps):.0f}C"
                if max(temps) >= 83 or (clks and min(clks) < 1500):
                    health += " ⚠THROTTLE?"
            avg = f" ~{_fmt_dur(sum(etimes)/len(etimes))}/ep" if etimes else ""
            print(f"{when}  TRAIN {r.get('voice','?'):12} {r.get('n_clips','?')}clips "
                  f"{_fmt_dur(r.get('total_s',0))} {len(eps)}ep best@{r.get('best_epoch','?')} "
                  f"loss {r.get('best_eval_loss','?')}{avg}{health}")
        elif st == "cut":
            print(f"{when}  CUT   {r.get('source','?'):12} {r.get('clips','?')}clips "
                  f"{r.get('minutes','?')}min {_fmt_dur(r.get('elapsed_s',0))}")
        elif st == "build":
            print(f"{when}  BUILD {r.get('source','?'):12} {r.get('kept','?')}clips "
                  f"snac {r.get('snac_tok_s','?')}tok/s {_fmt_dur(r.get('elapsed_s',0))}")
        else:
            print(f"{when}  {st.upper()} {json.dumps({k:v for k,v in r.items() if k not in('ts','stage')})[:90]}")

if __name__ == "__main__":
    if "--json" in sys.argv:
        print(open(METRICS, encoding="utf-8").read() if METRICS.exists() else "(none)")
    else:
        _summary()
