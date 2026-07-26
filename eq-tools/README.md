# EQ matching tools (2026-07-11)

The `matched/` folder = the 5 original narrator files spectrally matched to
**deathstalker legacy** (Owen's pick for best sound) + level-matched to -19.7 LUFS.
Originals untouched. Method: speech-gated LTAS (27 x ~1/6-octave bands, 60Hz-11.5kHz)
diff vs legacy -> smoothed gain curve (clamped +/-6 dB) -> ffmpeg firequalizer
(linear-phase FIR) -> static volume trim. Iterated 4x from ORIGINALS each pass
(never double-processed).

Result: within +/-1 dB of legacy below 8 kHz; air band (8-11.5k) within ~+1.3-1.9 dB
(method noise floor - sibilance-gated measurement). Loudness exact.

- ltas.py <files...>       - print 8-band comparison table (first file = reference row)
- match_eq.py <ref> <targets...> - compute fresh gain plan -> eq_plan.json
- refine_eq.py <dir>       - residual iteration: re-measure matched/, compose into
                             eq_plan.json, re-render from originals
- eq_plan.json             - final composed gain curves that produced matched/
Needs: ffmpeg on PATH, python + numpy.

NOTE: deliberately NO neural enhancement (resemble/Adobe class tools) in this chain -
that's the artifact source being escaped. Deterministic EQ only.
