# Audible Book → Orpheus Training Dataset — Cleaning Pipeline

Proven end-to-end recipe, built on **Marked Man** (deathstalker narrator, Richard Rohan),
2026-07-18. Every stage was ear- and number-checked. Scripts referenced here live in this
repo (`cut_audiobook.py`, `autoeditor_silences.py`) and the working scratch (`denoise_chapters.py`,
`normalize_and_concat.py`, `subsample_8h.py`, `clip_check.py`).

---

## The pipeline (in order)

### FCPX prep (source side)
1. **Fresh Audible pull** — per-chapter. Keep the raw files; they're the clean reference.
2. **Fix EQ** (treble/presence), **strip music + ads** at chapter fronts/ends.
3. **Turn master volume DOWN for headroom** — peaks should land ≤ −3 dBFS.
4. **Export chapters as WAV (lossless), not AAC.** AAC is *lossy* — 111 kbps bakes in codec
   artifacts that denoise cannot remove.

### Per chapter, before joining
5. **Clipping check** — peak + mean per chapter (`ffmpeg volumedetect`; sample-level flat-top
   detection in `clip_check.py`). If any chapter hits 0.0 dBFS, compare against the raw Audible
   file to confirm it's an FCPX per-chapter gain issue (proven on MM: FCPX boosted some chapters
   +3 dB into clipping while the raw source was clean at −0.7…−2.9 dB). **Fix = re-export with
   headroom, one uniform gain.** Do NOT reconstruct with `adeclip` (it invents samples). If
   re-export is impossible, exclude clipped clips instead (only viable when you have surplus hours).
6. **Analysis + pre-clean QC** — measure:
   - 99%-energy spectral rolloff (brightness; a bright audiobook is ~7–9 kHz)
   - noise floor (10th-percentile frame RMS) and mains hum (strongest 40–130 Hz tone)
   - HF speech-vs-silence (confirms the voice has *real* high-frequency content, not codec-dead)
   - **empty spaces, chapter-intro announcements, and tonal artifacts** — check these *before*
     cleaning, so structural gaps and tones are known going in.
7. **Denoise** — audio-separator mel-band roformer (masking):
   `run_audio_separator.py <in> --model_filename denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt`,
   keep the `(dry)` stem. Per chapter: downmix dual-mono → mono (lossless), silence-chunk ~1200 s
   in its own process (memory-safe), stitch (`denoise_chapters.py`). Timing-preserving (±0.00 s),
   so alignment stays valid. Removes broadband hiss + hum; voice HF untouched.
   - Caveat: the roformer runs at 44.1 kHz/16-bit (resamples 48→44.1 k). Harmless for a <10 kHz
     voice; the final target is 24 k anyway.
   - **LIMITATION:** denoise is broadband masking. It does **not** remove **narrowband tones**
     (AAC "birdies", electronic tones, hum harmonics). Those survive and get learned into the
     voice. A tonal-QC / notch step is still TODO.
8. **Loudness-match per chapter** — measure integrated LUFS (`ffmpeg loudnorm print_format=json`),
   apply a single linear gain each to a common target (peak-safe ≤ −1 dBTP). Linear gain only —
   no compression or limiting (`normalize_and_concat.py`).

### Join + cut
9. **Concatenate to a contiguous book — AS FLAC, never WAV.**
   ⚠️ WAV has a 4 GB (uint32) size-field limit. An 11.89 h 24-bit/44.1 k mono book = 5.66 GB →
   header overflows → tools read only ~9 h → **silent truncation of ~3 h of training data.** FLAC
   has no such limit. One ffmpeg pass: per-input `volume=<gain>dB`, downmix mono, `concat`,
   `-c:a flac`. Record chapter offsets.
10. **epub-align** (book text = ground truth):
    `bookforge-tts.py --generate-sentences --audio book.flac --epub <epub> --out book.vtt --report`
    (whisperx, GPU by default). The coverage report lists `epubNotInAudio` (cut intros/outro) and
    `audioNotInEpub` (foreign audio). Chapter-number announcements are handled as structural
    boundaries. Can run in parallel with denoise (timelines match to the sample).
11. **Auto-editor silence map → cut:**
    - `autoeditor_silences.py book.flac silences.json --thr 0.012` → per-frame loudness →
      silence-interval JSON.
    - `cut_audiobook.py --vtt book.vtt --audio book.flac --epub <epub> --silence-map silences.json
      --out-dir <cut> --source-name <voice> --max-clip 20 --trail-cap 0.1 --dialogue-aware
      --snap-window 0.5`. Auto-editor drives boundary snapping AND the trail-cap trim (its absolute
      threshold beats librosa's relative one, which can read a quiet clip's trailing breath as
      speech). Outputs 24 k mono. Validate with a boundary-RMS check (every clip ends silent) + ear.
12. **Transcript-fidelity QC (MANDATORY before training):** whisper-transcribe a random
    30–60 clip sample (faster-whisper `small.en`) and diff against the metadata text.
    Healthy: mean WER < 0.10, and the clips' FIRST and LAST words present. **Red flags:
    clips missing their opening words / containing the next clip's words** = boundary
    drift; do NOT train on it.
    ⚠ Lesson (Marked Man, 2026-07-18): `autoeditor_silences.py` once computed
    `fps = frames/duration`; auto-editor's timebase is EXACTLY 30 fps but it drops the
    trailing partial chunk, so that division linearly stretched the silence map
    (+1.5 s drift by the end of an 11.9 h book). Cuts snapped into mid-sentence phantom
    silences → ~half the clips were missing opening words → the trained model produced
    fragmented, half-spoken words and dead-air runaways. The renders were butchered by
    the DATA, not the epoch choice. fps is now hardcoded 30.0 with a frame-count sanity
    check — but the QC step above is what actually catches this class of bug.
13. **Subsample to 8 h + train:**
    - `subsample_8h.py <cut> <8h_dir> 8 1.5` (even-spread, hardlinks).
    - Copy the dataset into WSL ext4 (fast — not the /mnt/e 9p mount).
    - `orpheus_owen.py --source-name <voice> --recut-dir <ds> --out-base /home/telltale/xtts_ft
      --mask-prompt-loss --no-dedup --lr-schedule constant_with_warmup --stop-overtrain
      --overtrain-patience 2 train`. Auto-stops at settle+1 (~5–6 epochs, ~2 h on a 3090 Ti).

---

## Key lessons
- **Always fidelity-QC the cut before training** (step 12). A subtle timeline bug in the
  silence map poisoned an entire training run; whisper-vs-transcript diffing on a 40-clip
  sample catches it in minutes.
- **Registered voice token must equal the trained speaker_name** (prep_voice.sh arg 6) —
  the token is the verbatim prompt prefix; a mismatched suffix gets spoken aloud.
- **AAC is lossy.** Export WAV chapters, rejoin as FLAC. Re-encoding lossy → lossless recovers
  nothing; you need a truly lossless/higher-bitrate *source* to gain fidelity.
- **Full-book audio must be FLAC** — WAV's 4 GB limit silently truncates long books.
- **Denoise ≠ de-tone.** It only removes broadband noise; narrowband tones survive and get learned
  into the voice. Add a tonal-QC/notch step for lossy sources.
- **Clipping** is usually inconsistent FCPX gain, not the source → re-export with headroom.
- **Repetition runaway** (a chunk repeats N×) is an *inference-side* problem, not a cleaning one:
  per-voice repetition penalty (~1.15) + max-chars/sec cap (~21.5) + sentence-length caps in
  packing + a runaway watchdog that re-rolls the stuck sentence.

## The EOS-margin findings (2026-07-19) — READ BEFORE EVERY TRAIN

The deathstalker v4 deploy failed at book scale: ~20-50% of long chunks generated
to the token cap without ever emitting end-of-audio, but ONLY on vLLM/CUDA — the
same md5-identical weights rendered clean on Mac/MLX with byte-identical prompts.
Root cause, proven causally by twin quick-trains and closed by a full retrain
(v5: 0 runaways across a 1,371-chunk book on the exact config that failed):

### 1. ROOM TONE IN TRAINING AUDIO IS LOAD-BEARING. Never train on digitally-dead silence.

Aggressive cleaning (dehum/denoise/silence-flooring) leaves pauses spectrally
dead. Dead pauses SNAC-encode into a concentrated handful of frames (broken MM
corpus: top-3 frames = 27% of all pause frames in full-clip encodes; healthy
novels corpus: 5%). The model then learns an overwhelming "silence-frame follows
silence-frame" transition; at the end of speech, p(one more silence frame) vs
p(EOS) becomes a razor-thin tie — and BACKEND ARITHMETIC decides it (CUDA/bf16
loops forever, Metal stops). Every runaway is the SAME single SNAC frame looping.

**Fix (validated at 120-clip and full scale): add a noise bed to the whole clip
at ~-65 dBFS.** Best = GENUINE room hiss harvested from the source's own quiet
regions (concatenate noise-floor windows, unit-RMS normalize, scale to -65 dB,
RANDOM offset per clip — a fixed loop start reintroduces periodicity). Genuine
hiss beat synthetic noise AND the healthy reference corpus on frame diversity
(1.4% vs 5-8%). Even a periodic mains hum works in practice (the SNAC frame grid
beats against the hum period), but random hiss is strictly safer. Scripts:
`build_hiss_corpus.py` pattern in this repo's history; harvest per-source.

Corpus-processing margin gradient (greedy probe on 20 long chunks):
clean recordings 0% · EQ-only 15% · neural-dehum ~70% · dehum+31%-silence-mass
70-90% (production-broken). Neural cleanup is a margin TAX — spend it knowingly.

### 2. eval_loss CANNOT see EOS margin — GREEDY-GATE every keeper before deploy.

The broken deploy was the eval-loss-settled epoch of its run; every epoch of that
run was broken (70-90% greedy runaway, flat curve). The 6-minute gate that catches
this class: `voice_diff.py <merged_dir> <token> <session-state.json>` — renders 20
long (320-352 char) chunks at temperature 0 AND sampled. **Ship only greedy ≈0.**
Greedy is the margin meter (is EOS ever argmax?); sampled is production behavior.
A voice can be sampled-clean with zero greedy margin (thirdreich) — one config
drift from breaking. Gate after ANY corpus/penalty/keeper-rule change.

### 3. QC a corpus in 12 minutes BEFORE burning a full train.

120-clip subset + the standard train flags settles in ~6 min and REPRODUCES the
full run's margin profile (broken corpus subset: 70% sampled runaway; fixed
subset: 0%). Quick-train + greedy gate = corpus validation. Also measure silence
mass (broken corpus was 31% vs healthy 21-24%) and pause-frame diversity from
FULL-CLIP SNAC encodes (isolated-segment encodes MISPREDICT — the instrument must
match training context).

### 4. Repetition penalty is MITIGATION and part of the system — tune it by data.

Penalty 1.0 = 100% greedy runaway even on a healthy corpus (upstream README
demands >=1.1; silence codes legitimately repeat and argmax follows them without
the penalty). But 1.15 audibly chokes mid-vowel code repeats ("wobble"/cracks).
On the v5 hiss corpus, 1.10 was the ear-validated sweet spot AND the best match
to the human narrator's measured pause distribution (median/p90/long-pause share
via 50ms-RMS pause stats — measure, don't guess: 1.15 flattened pause variety,
1.05 began to sprawl). Rate cap: derive maxCharsPerSec from the NEW voice's
measured rate (v5: natural tail <25, genuine truncations >26.8 → cap 25.0);
never carry the old voice's cap forward. KEEP the rate guard — it caught real
early-EOS truncations (~0.6%) that nothing else detects.

### 5. Epoch choice is a PROSODY choice once every epoch gates clean.

With a healthy corpus, all epochs pass the greedy gate from epoch 1 — so pick the
keeper BY EAR across epochs at the deployed penalty (late epochs = most
consolidated pauses; v5 keeper was the final epoch). Render the same sample text
per epoch; never compare epochs at different penalties (the wobble contaminates
the inflection judgment).
