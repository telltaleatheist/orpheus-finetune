# RVC Training Pipeline — Draft Pending Sprint Validation

**Status: DRAFT (2026-07-20).** Unlike VOICE_TRAINING_PIPELINE.md (locked), this
run-book encodes (a) verified trainer ground truth, (b) multi-source community
consensus, and (c) the leading hypothesis from the July 20 blur investigation —
all awaiting our own quick-train sprint validation before locking.

## The July 20 blur mystery — state of the investigation

New-chain models (v2 Marked Man, v3 CoD) convert blurry/muffled with a bass
smear at EVERY epoch (50/75/100/137) and every setting; the older v1
(one-off run, ~ep101 snapshot) is clean on identical inputs.

**Excluded by measurement/A-B:** conversion settings (persists at index 0),
index file, seed reverb (all sets ~200 ms decay), roformer seed denoise (seeds
sound clean), pretrained bases (byte-identical), trainer code+env (no changes
between runs), epoch vintage, seed spectral tilt, source book (v2≠v3 sources,
both blurry), overtraining alone (ep50 also blurry).

**LEADING HYPOTHESIS (evidence-backed, untested by us):** sample-rate
mismatch. v1 trained 48k on NATIVE-48k seeds; v2/v3 trained 48k on seeds cut
at 44.1k → the vocoder learns against an interpolated-empty top octave.
Community precedent: kalomaze's A/B showed 40k beating 48k on 44.1k source —
strong enough that RVC rolled back its 48k default (RVC-Project issue #233).
Our sources measure: CoD 48k, AoL 48k, ThirdReich 48k (all native) — the seed
builder itself downsampled them to 44.1k. Marked Man is 44.1k native.
**Fix to validate: re-cut seeds at native 48k (CoD/AoL/TR); for 44.1k-native
sources (MM), train 40k instead.**

## Trainer ground truth (read from source, ultimate-rvc BookForge fork)

| Stage | What actually happens |
|---|---|
| Slicing (Automatic) | RMS slicer: threshold -42 dB, min clip 1.5 s, min silence 400 ms, keeps ≤500 ms silence per boundary side; then 3.0 s windows, 0.3 s overlap; tails kept (Simple mode DROPS tails — avoid) |
| Silence | Internal pauses <400 ms untouched; 400 ms–1 s largely SURVIVE (≤0.5 s each side); edge silence <400–500 ms survives verbatim. Noise floors above -42 dB are never treated as silence |
| Filtering | 5th-order Butterworth HP at 48 Hz, whole file. Nothing else. Hiss/hum above 48 Hz trains straight in |
| Normalization | Per-file peak alpha-mix (75% peak-norm to 0.675 + 25% raw). NO loudness norm — one loud transient ducks its whole file; per-file consistency is prep's job. Files peaking >2.5 dropped silently |
| Resampling | soxr_vhq everywhere (44.1→48 upsample is high quality but can't create the missing octave) |
| Training input | One random 0.36 s crop per slice per step; 900-frame (9 s) cap; bucket sampler SILENTLY DROPS clips >6.75 s under Skip mode; 2 mute samples appended per speaker |
| Features | contentvec on 16 kHz copies (source rate irrelevant to intelligibility); rmvpe hop 10 ms, f0 clamp 50–1100 Hz |

**Implications:** clip length/count/boundaries don't matter (total voiced
seconds does); breath-safe cutting is an Orpheus concern, NOT an RVC one;
silence capping and loudness consistency are prep responsibilities; noise
cleaning is entirely prep's job.

## Tooling note (2026-07-20, Owen's directive)

File-level cleaning passes (silence truncation, HP, gain, gating) should run
through the **ClipForge CLI** rather than ad-hoc ffmpeg:
`node cli/clipforge-process.js --input in.wav --recipe r.json --out out.wav
--keep-stages` (BookForge repo, `feat/clipforge`). Every pass writes a
`.provenance.json` (chain, settings, hashes, tool versions) — the v1-blur
"what did we do to this file?" archaeology must never recur. lowpass/resample
require explicit allow-flags in the recipe (this doc's rules, machine-
enforced). The seed BUILDER (`build_rvc_seeds.py`) remains canonical for
sampling/cutting until ClipForge's export phase ports it.

## Dataset-prep checklist (community consensus × trainer ground truth)

1. **Source triage:** lossless masters only; Spek/FFT the true bandwidth —
   training rate ≤ real bandwidth (44.1k source → train 40k; 48k source →
   48k). NEVER downsample a 48k master to 44.1k in the seed cutter (the v2/v3
   mistake). Reject reverberant sources (>1 dereverb pass needed = reject).
2. **Noise:** mild constant hiss is TOLERATED by RVC (pretrains were built on
   noisy data) — prefer leaving it, or ONE pass (RX Spectral Denoise or
   aufr33 mel-roformer denoise, normal not aggr). Then noise-GATE the silent
   gaps (~-40 dB, generous hold). Never stack cleaning passes.
3. **Manual pass:** remove SFX/other speakers/coughs/clicks and isolated or
   harsh breaths; KEEP natural voiced breaths attached to speech (majority
   doctrine; they prevent robotic output).
4. **No EQ curve, no compression, no limiting, no presence boost.** Optional
   de-ess only for genuinely harsh sibilance. HP 48 Hz happens in-trainer.
5. **Truncate silence:** internal gaps to 0.15–0.25 s (detect ~-42 dB);
   trim lead/tail similarly. (The trainer keeps up to ~1 s per pause if you
   don't.)
6. **Loudness:** files mutually consistent (-23 LUFS dual-mono if you want a
   number); louder-unclipped beats quiet ("mumbly"); de-click first so peak
   norm isn't hijacked.
7. **Export:** lossless WAV/FLAC, mono (downmix yourself), NATIVE rate.
   Files tens of seconds–minutes; nothing under ~3.5 s; 15–45 min total
   pure speech (45 min ceiling; more shifts overtraining earlier, adds risk).
8. **Preprocess:** split Automatic; normalization post; built-in noise
   filter/reduction OFF.

## Training + selection

- **Pretrained base:** swap stock for **TITAN 48k** (first choice for clean
  narration; converges faster) or KLM 4.9 HFG 48k; at 40k: Ov2Super or
  RIN_E3 / SnowieV3.1-X-RinE3. Hosted per links in the research reports
  (blaise-tk/TITAN etc. on HF).
- batch 8, rmvpe, contentvec (all confirmed correct).
- **Save every 10–25 epochs; build the index BEFORE training** so early
  checkpoints are testable.
- **Selection is by EAR on a fixed test-clip set** (include synthetic/Orpheus
  input — our actual inference case). g/total + rising mel-loss only bracket
  the window; "when in doubt, slightly undertrained." Expect keeper zone
  ~100 epochs at ~1 h data, EARLIER with custom pretrains and bigger sets.
- Inference recipes: real-voice restoration idx 0.5/prot 0.2; synthetic
  (TTS) input idx 0.3/prot 0.33. Blur that persists at idx 0 is a MODEL
  problem, not a settings problem (July 20 lesson).

## Staged and ready (2026-07-20 evening — sprint NOT yet run; RVC training on hold per Owen)

- `bookforge_train/build_rvc_seeds.py` rewritten: per-voice NATIVE rate with
  probe-and-abort on mismatch, internal-silence truncation (-42 dB, >0.25 s →
  0.20 s, lead/tail trimmed), 45-min even-spread cap, denoise=True hard-gated
  to 44.1k sources (roformer env constraint; 48k sources train with their mild
  hiss per consensus).
- Seed set `E:\training\rvc-seeds\deathstalker_cod48\` BUILT: 160 clips,
  44.1 min, all 48000 Hz, RMS spread 2.8 dB (p5–p95). The old 44.1k
  `deathstalker_cod` dir is UNTOUCHED — it is Arm C (control). Do not rebuild it.
- TITAN 48k pretrain pair at
  `C:\Users\tellt\Projects\bookforge\models\rvc\pretraineds\custom\TITAN_48k\`
  (G 249 MB, D 817 MB; dir name ends "48k" as the trainer's rate parser requires;
  train with pretrained_type Custom + custom_pretrained TITAN_48k).

## Validation sprint (next session's plan)

Fixed test input: `ds_2h_ep440_gap06.wav` (+ a real-Rohan clip). Arms, one
variable each, ear verdict per arm (Mac inference per MAC_INFERENCE.md while
the GPU trains):
- **A:** 48k-NATIVE CoD seeds (re-cut, no downsample), stock pretrain — tests
  the rate-mismatch hypothesis alone.
- **B:** A + TITAN 48k base — the expected production recipe.
- **C (control):** current 44.1k seeds + stock (= v3, known blurry).
- If A/B still blurry: 40k arm on the same seeds, then interrogate v1's
  surviving sliced_audios for what else differed.
Keeper by ear across the epoch ladder; then lock this doc.
