# Driving the Mac's RVC and Orpheus Inference From the PC

Verified 2026-07-20 (RVC path end-to-end that day; Orpheus path per the July
deploy sessions). Use the Mac whenever the PC GPU is owned by a training
sequencer — Mac inference is fast enough for samples and full-book RVC.

## Access

- `ssh mac` — alias in `~/.ssh/config` (MagicDNS `owens-mac-studio.hs.owenmorgan.com`,
  user `telltale`, key `~/.ssh/id_ed25519_mac`). Prefer the alias over the raw
  LAN IP: the LAN IP is DHCP-drifting and tailscale measured FASTER than raw IP.
- **Non-interactive ssh gets no PATH.** Use full binary paths: node is
  `/opt/homebrew/bin/node`, system python3 for helper scripts.
- Transfers run ~3.6 MB/s (both machines on Wi-Fi — every packet crosses the
  air twice). Fine for models/samples; for many small files use a tar pipe.
  Plugging the Mac Studio's en0 into ethernet would roughly double it.

## Key paths on the Mac

| What | Where |
|---|---|
| BookForge checkout (CLI lives here) | `/Volumes/Callisto/Projects/BookForgeApp` |
| Runtime root | `~/Library/Application Support/BookForge/runtime/` |
| Orpheus models + registry | `runtime/orpheus-models/` + `models.json` |
| RVC voice models | `runtime/rvc-models/rvc/voice_models/<Name>/` |
| e2a checkout (keep synced via git, NEVER cp) | `/Users/telltale/Projects/ebook2audiobook-latest` |

**After pulling BookForge on the Mac: run `npx tsc` or the CLI runs stale
dist.** MPS renders use 1 worker (vs 4 on CPU).

## RVC inference (verified end-to-end)

1. **Install a model** (three files in one dir named after the model):
```
ssh mac "mkdir -p ~/Library/'Application Support'/BookForge/runtime/rvc-models/rvc/voice_models/<NAME>"
scp <model>.pth  "mac:Library/Application\ Support/BookForge/runtime/rvc-models/rvc/voice_models/<NAME>/<NAME>.pth"
scp <model>.index "mac:Library/Application\ Support/BookForge/runtime/rvc-models/rvc/voice_models/<NAME>/<NAME>.index"
# voice.json: {"label": "...", "matches": "<substring>", "defaultIndexRate": 0.5}
```
   The `.index` is REQUIRED for index-rate > 0. If training hasn't finished,
   build one on the PC from the training dir (CPU-only, safe next to GPU work):
   `urvc-train\python.exe bookforge_train\build_index.py <trainRoot>\<model>`
   (trainRoot = `C:\Users\tellt\Projects\bookforge\models\rvc\training`).
   Mid-training weights: the trainer maintains a rolling `<model>_best.pth`
   plus `<model>_<epoch>.pth` every 25 epochs.

2. **Convert** (BookForge's bridge — handles MPS memory recycling and
   chunk/stitch for long inputs):
```
ssh mac "cd /Volumes/Callisto/Projects/BookForgeApp && /opt/homebrew/bin/node \
  --require ./cli/electron-stub.js cli/rvc-convert.js \
  --input /tmp/in.wav --out /tmp/out.wav --model <NAME> \
  --index-rate 0.5 --protect-rate 0.2 --f0-method rmvpe"
```
   Throughput: ~2.4x realtime on a 66 s clip (28 s). Settings: restoration
   recipe = idx 0.5 / prot 0.2; cleanup-of-synthetic-input grid winner was
   idx 0.3 / prot 0.33 — A/B per use case.
   **Never use RVC output as Orpheus TRAINING data** — passes spectral gates,
   trains up rough (training amplifies what listening forgives).

## Orpheus inference

Voices live in `runtime/orpheus-models/` with a `models.json` entry
(`token` MUST equal the trained speaker name or the mismatch is SPOKEN aloud;
`backends.vllm.repPenalty` 1.10 — 1.15 wobbles). Two ways to get a voice there:

- **Deployed voices**: `~/pull_orpheus_voice.sh <voice> "<Label>"` pulls from HF
  (the deploy_voice.sh flow pushes there from the PC).
- **Test checkpoints**: scp the raw LoRA adapter dir, merge ON the Mac
  (peft+transformers f16, ~5 min — pattern in `C:\tmp\mac_prep_ds_ep5.py`),
  save into `orpheus-models/<id>` and append the models.json entry. NOTE: the
  merge needs `peft`, which the e2a-env python
  (`runtime/e2a-env/bin/python`) does NOT have — check imports first, or ship
  the already-merged model dir instead (~6.2 GB, ~30 min over Wi-Fi; merging
  on the PC and shipping is usually SLOWER than merging on the Mac).

Render via the CLI (same interface as the PC):
```
ssh mac "cd /Volumes/Callisto/Projects/BookForgeApp && python3 cli/bookforge-tts.py \
  --tts --engine=orpheus --voice=<id> --input /tmp/text.txt \
  --out /tmp/sample.wav --sentence-gap 0.6"
```
Backend is MLX (fast, CUDA-graphs-class speed on Apple silicon). Caveat from
the EOS forensics: MLX and vLLM decode can differ on razor-thin token ties —
for pre-DEPLOY gating, render on the PC's vLLM (production backend); Mac
renders are for auditions and books consumed from the Mac.

## Deciding where to run

- PC GPU free → PC (production backend, no transfer).
- PC GPU owned by a sequencer → Mac for RVC always (model ships in ~30 s);
  Mac for Orpheus only if the voice is already there or HF-pullable — a 6.2 GB
  model transfer over Wi-Fi usually loses to just waiting for the GPU.
- Never slot inference onto the PC GPU while a chain owns it (host-OOM'd via
  vmmemWSL commit once already).
