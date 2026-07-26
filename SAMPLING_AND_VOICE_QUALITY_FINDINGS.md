# Sampling & Voice-Quality Findings — 2026-07-12 A/B campaign

One evening of controlled A/B renders on **deathstalker_v3** (EOS-safe ≤20s/2048 recipe)
through the real BookForge CLI pipeline (`bookforge-tts.py --tts`, WSL vLLM 0.7.3).
Test text: Deathstalker ch.1 excerpts (~650 and ~100 words). Every claim below was
rendered and ear-checked by Owen; objective metrics noted where used. This doc exists so
the next retrain can pick up exactly where inference tuning hit its ceiling.

## Final inference defaults (baked into e2a `bookforge` branch, lib/classes/tts_engines/orpheus.py)

| knob | value | why |
|---|---|---|
| packing | 350 chars, **no sentence cap** | prosody win; 450 = silent early-EOS truncation on every model |
| temperature | **0.85** | winner of 0.6→1.0 ladder; every arm 0 guard trips (EOS-safe voices don't need cool temps) |
| top_p | 0.8 | lowering to 0.7 did NOT reduce cracks — cuts probability mass, not ratio |
| min_p | **0.05** | confidence-scaled junk-tail cutoff (vLLM only; MLX has no min_p) |
| repetition_penalty | **1.1** | pause governor — Owen chose pauses-right over cracks-fewer (see below) |
| vLLM dtype | **bfloat16** | checkpoints are bf16; the old float16 cast audibly muffled ("significantly clearer" — despite only 0.6dB band-RMS delta; trust ears) |

All overridable per render: CLI `--temperature/--top-p/--min-p/--rep-penalty`, env
`ORPHEUS_VLLM_DTYPE`. Forwarded into the WSL worker via parallel-tts-bridge forwardKeys.

## Finding 1 — the muffle is TRAINING DATA, not the pipeline

- deathstalker renders: >8kHz energy **−26.5dB** relative to overall.
- owen-voice renders through the IDENTICAL pipeline: **−21.6dB** (~5dB brighter, clearly audible).
- Ruled out: batch path resampling (pure 24k PCM end-to-end), fp16 cast (0.6dB), SNAC
  ceiling (applies to both voices equally; 12kHz hard limit regardless).
- **Yardstick for the retrain**: the EQ-matched-source retrain should close most of the
  5dB gap. Measure:
  `ffmpeg -i render.wav -af "highpass=f=8000,astats=measure_overall=RMS_level" -f null -`
  vs full-band RMS.

## Finding 2 — repetition penalty is the PAUSE GOVERNOR (and a crack driver)

Audio silence tokens repeat, so the rep penalty is what stops a pause from sprawling.
Ladder (temp 0.85, same 100-word text):

| rep penalty | pauses | mid-word breathy "cracks" |
|---|---|---|
| 1.1 | right | occasional (e.g. "cruisers", "people") |
| 1.07 | still long per Owen's ear | fewer |
| 1.05 | audibly long | ~one per excerpt |
| 1.0 | **2× runtime (75s audio for 37s of speech), token-cap runaways** | still one (!) |

The penalty also chokes *legitimately* repeating codes mid-vowel — that's the crack
mechanism (spectrogram: harmonic stack thins, breath-noise fills in, blended with
formants — NOT a hard-edged codec/frame glitch).

**Key falsification: one crack still occurred at rep penalty 1.0.** So breathy frames are
IN the learned distribution — the model rates them highly at some steps. No sampler
setting removes high-probability learned behavior. That's the retrain's job.

## Finding 3 — what didn't work

- **top_p 0.7**: cracks unchanged, new odd sounds; keeps tail junk when the distribution
  is flat, squeezes legit variety when peaked. Wrong tool.
- **fp16→bf16 for cracks**: no change (it fixed muffle, not cracks).

## Prescriptions for the next deathstalker retrain (fold into the EQ-matched re-cut)

1. **EQ-matched sources** (already prepared: `E:\training\deathstalker\source\matched\`,
   see the memory/WHY_REJECTED notes there) — fixes the muffle. Verify with the 5dB yardstick.
2. **Cap internal pauses at ~0.7–0.8s in the re-cut** (tighten `_normalize_pauses` /
   cutter settings; v2 recuration used ≤2s "natural" — that's the pause prior the model
   learned, and why low rep penalty sprawls). A short pause prior makes rep penalty
   ~1.02–1.05 safe, which is where the cracks fade. Also matches the pipeline's fixed
   0.6s inter-chunk gap for consistency.
3. **Screen source clips for the narrator's REAL vocal cracks/fry** and exclude them —
   the model learned breathy devoicing as a feature of the voice. (Older narrator; the
   cracks are genuinely in the source.)
4. Keep the standing rule: **≤20s clips / max_seq_length 2048** (38s/4096 breaks EOS —
   see TRAINING_CLIP_LENGTH_RESEARCH.md and the 2026-07-12 conviction matrix).

## After that retrain, revisit

- Drop `ORPHEUS_REP_PENALTY` toward 1.02–1.05 (cracks) once the pause prior is short.
- Re-run the temperature ladder — a cleaner voice may prefer a different sweet spot.
- The chars/sec + token-cap guards in orpheus.py are the objective regression tally
  (guard-trip lines in the worker log); whisper-diff for completeness.

## 2026-07-17 — sample rate, source hiss, and the rep-penalty ceiling

### Orpheus is a 24 kHz system — training files MUST be 24 kHz mono
Orpheus emits **SNAC tokens** decoded by the **snac_24khz** codec. It trains on 24 kHz
mono and **outputs 24 kHz** (`orpheus_owen.TARGET_SAMPLE_RATE = 24000`; cut_audiobook
decodes to 24 k). Consequences:
- Prep clips at **24 kHz mono**. cut_audiobook already does this.
- **Denoise/clean at the SOURCE's native rate (e.g. 48 k) FIRST, then downsample to 24 k**
  with a high-quality resampler (`aresample=resampler=soxr:precision=28`). Never clean
  after downsampling — the denoiser has less to work with, and a poor resampler adds
  aliasing that reads as harshness.
- The 24 k ceiling loses everything >12 kHz ("air") — output is inherently a touch duller
  than 48 k. That's the rate, not a bug.

### The "scratchy" texture is mostly SOURCE HISS, not SNAC
A SNAC round-trip test (clean 24 k clip → SNAC encode/decode, no model) plus listening to
the raw source proved the audible hiss is **in the recording** — Audible's lossy compression
riding *on the voice* (present over speech, same locations, not in the silence). The model
faithfully learns it; SNAC just re-renders it as hiss. SNAC adds its own mild grain, but it's
NOT the dominant problem. Fix = clean the SOURCE before training:
- **audio-separator denoise roformer** `denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt`
  (via `run_audio_separator.py`, bookforge-urvc env) cleanly subtracts the hiss and keeps the
  real voice (the "(dry)" stem). ~75 s for 30 min on GPU. This is the tool of choice.
- **Do NOT use resemble-enhance** for this — it re-synthesizes and damages/smears detail
  (Owen-confirmed). Also viable: RVC re-render through a same-voice model (re-synthesizes
  clean), but that needs a trained RVC model and changes timbre slightly.
- Pipeline: denoise source (48 k) → cut_audiobook → 24 k clips → retrain. (Running this as a
  30-min/3-epoch test on mistborn 2026-07-17.)

### Repetition penalty: verified plumbing, and why it can't fix dialogue pauses
`ORPHEUS_REP_PENALTY` env → `orpheus.py:178 REP_PENALTY` → vLLM `SamplingParams(repetition_
penalty=...)` (whole-sequence). **MLX only applies it over the last 20 tokens**
(`make_logits_processors(None, REP_PENALTY, 20)`), so this test is ONLY meaningful on
vLLM/Windows. Default 1.1; deployed voices ~1.15. Raising 1.15→1.5 (verified applied) barely
moved the long dialogue pauses — because SNAC silence is a *varied* token run, not an
*exactly-repeated* token, so a repetition penalty doesn't catch it. **The long-pause fix is
the training-data pause prior (cap internal pauses ~0.7–0.8 s in the re-cut, see above), NOT
rep penalty and NOT post-hoc silence removal.**
