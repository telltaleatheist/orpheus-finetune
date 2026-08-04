# Fine-tuning on the Mac Studio (Apple Silicon, MLX)

Second training environment for this repo, alongside the CUDA rig. Set up and
verified 2026-08-02 on the M1 Ultra Mac Studio (64 GB unified memory,
macOS 26.3, arm64). Nothing has been trained here yet — this document exists so
the first real run does not begin with an afternoon of environment archaeology.

**Nothing in the rig's path changed.** `requirements.txt`, `training_profiles.json`,
`text_sft.py`, `orpheus_owen.py` are untouched and keep working on Windows/WSL
exactly as before. The Mac additions are three files: this doc,
`requirements-mac.txt`, and `configs/mac/qwen3_titles_mlx.yaml`.

## The one thing to internalise first

**Unsloth does not run on macOS, and MLX is a different trainer — not a port.**

`text_sft.py` is built on Unsloth, which is built on Triton and CUDA. Neither has
an Apple Silicon backend, and there is no shim that makes them appear. Do not try
to `pip install unsloth` here; it will either fail to build or install something
that imports and then dies on the first kernel launch.

The Apple Silicon trainer is **`mlx-lm`**, from Apple's MLX project. It does LoRA,
QLoRA, DoRA and full fine-tuning natively against unified memory, and it is a
genuinely good trainer. But it is a *separate implementation*: different optimizer
code, different RNG, different LoRA parameterisation, different data pipeline.

Concretely, what does **not** transfer 1:1 from the rig:

| Rig (Unsloth/peft) | Mac (mlx-lm) | Note |
|---|---|---|
| `lora_alpha` | `lora_parameters.scale` | `scale = alpha / rank`. r=32/alpha=32 → scale 1.0 |
| `num_train_epochs` | `iters` | MLX counts **iterations**, not epochs. `iters = epochs × rows ÷ batch_size` — compute it, don't guess. **Not** ÷ `(batch_size × grad_accumulation_steps)`; see below |
| `gradient_accumulation_steps` | `grad_accumulation_steps` | same effective batch, but it does **not** change how many rows an iteration consumes |
| `dataset.assistant_only_loss` | `--mask-prompt` | same intent |
| `target_modules` (bare names) | `lora_parameters.keys` (dotted suffixes) | `q_proj` → `self_attn.q_proj` |
| all layers by default | **top 16 layers by default** | set `num_layers: -1` or you are running a different experiment than you think |
| `EarlyStoppingCallback` | *none* | MLX has no early stopping. Checkpoint often (`save_every`), watch val loss, pick the best checkpoint by hand |
| `seed: 3407` reproduces a run | reproduces *an* MLX run | same seed, different numbers. Never compare a Mac loss curve to a rig loss curve and conclude anything |

Everything the rig learned about *the data* still transfers — the chronological
split, assistant-only loss, "both text models peaked at epoch 1 on ~2,000
examples". Everything it learned about *the numbers* does not.

### An iteration is `batch_size` rows. Grad accumulation does not change that.

**This table was wrong until 2026-08-04 and the correction is worth its own
heading, because the failure is silent.** The original formula divided by the
*effective* batch, `batch_size × grad_accumulation_steps`. It does not work that
way. From `mlx_lm/tuner/trainer.py` (0.31.3):

```python
for it, batch in zip(
        range(1, args.iters + 1),
        iterate_batches(dataset=train_dataset, batch_size=args.batch_size, ...)):
    lvalue, toks, grad_accum = step(batch, grad_accum, it % grad_accum_steps == 0)
```

One iteration pulls **one batch of `batch_size` rows**. `grad_accumulation_steps`
only decides how often the optimizer applies what has accumulated — it changes
the *effective* batch, not the data consumed per step. So:

```
iters = epochs × rows ÷ batch_size
```

Getting this wrong divides your training by the accumulation factor. A config
built with the old formula at `batch_size 1, grad_accumulation_steps 16` and
"2 epochs" of a 7,829-row corpus asks for 978 iterations — which is 978 rows,
an eighth of one epoch. It runs cleanly, writes an adapter, reports a falling
loss, and the model is simply undertrained. Nothing announces the mistake.

The cheapest way to confirm it on any new mlx-lm version: run 5 iterations with
`--steps-per-report 1` and `mask_prompt: true`, and look at `Trained Tokens`. The
per-iteration delta is the assistant tokens of `batch_size` rows. If it looks
like one example's worth of output, an iteration is one example.

## Environment

Conda env named **`finetune`**, Python 3.12, living in the default conda envs
directory:

```
/opt/homebrew/Caskroom/miniconda/base/envs/finetune
```

Created with:

```bash
conda create -n finetune python=3.12 -y
conda activate finetune
pip install -r requirements-mac.txt
```

### The ExFAT trap — this is the one that bites

`/Volumes/Callisto` is **ExFAT**. ExFAT has no symlinks, no POSIX modes, no hard
links. A Python environment is made of symlinks, and pip writes console-script
shims that need the executable bit. Put a conda env or a venv on Callisto and it
will half-work — imports resolve, then something in the middle of a long run
fails on a path that was silently never created.

So:

| Lives on Callisto (fine) | Lives in `$HOME` (required) |
|---|---|
| this repo / code | the conda env — `~/…/miniconda/base/envs/finetune` |
| datasets (JSONL, corpora) | the HF cache — `~/.cache/huggingface` |
| training output: adapters, fused models, logs | pip's cache |

Adapters and fused weights on Callisto are fine — they are plain files.
Do **not** set `HF_HOME` to anywhere under `/Volumes/`.

(There is already a stale `/Volumes/Callisto/titles14b/env` conda env on the
machine. It is on ExFAT. Do not build on it.)

## Verified working

The commands below were run end to end on 2026-08-02. This is a *setup* check,
not a training run — the dataset and adapter were deleted afterwards, and the
downloaded base model was kept in the HF cache because it is reusable.

**1. MLX sees the GPU:**

```bash
conda activate finetune
python -c "import mlx.core as mx; print(mx.default_device())"
# Device(gpu, 0)
```

If that says `cpu`, stop — nothing below will be fast, and it means the
`mlx-metal` wheel did not install.

**2. Five iterations of LoRA on a 20-row throwaway dataset:**

`mlx_lm.lora` takes a *directory* containing `train.jsonl` and `valid.jsonl`
(`test.jsonl` optional) — not two separate file flags like the rig harness. The
row format is the same `{"messages": [{"role": ..., "content": ...}, ...]}`
chat JSONL that `build_titles_dataset.py` already emits, so a rig dataset works
unchanged; it just has to be laid out as those two files.

```bash
mlx_lm.lora \
  --model Qwen/Qwen3-0.6B \
  --train \
  --data /path/to/smoke-data \
  --fine-tune-type lora \
  --num-layers 4 \
  --batch-size 2 \
  --iters 5 \
  --steps-per-report 1 \
  --steps-per-eval 5 \
  --val-batches 2 \
  --max-seq-length 512 \
  --adapter-path /path/to/smoke-adapter \
  --seed 3407
```

Observed:

```
Trainable parameters: 0.121% (0.721M/596.050M)
Iter 1: Val loss 7.395
Iter 1: Train loss 7.932 ... Peak mem 1.441 GB
Iter 4: Train loss 5.881
Iter 5: Val loss 6.006
Iter 5: Train loss 6.115 ... Peak mem 1.494 GB
Saved final weights to .../adapters.safetensors
```

Finite loss, moving in the right direction, 56-tensor `adapters.safetensors`
written, ~34 s wall clock including the model download.

**3. The example config runs:**

```bash
mlx_lm.lora -c configs/mac/qwen3_titles_mlx.yaml \
    --model Qwen/Qwen3-0.6B --data /path/to/smoke-data --iters 2 ...
```

Reported `Trainable parameters: 3.386% (20.185M/596.050M)` against 0.121% for the
defaults above — which is the check that `num_layers: -1` and the seven
`lora_parameters.keys` actually took effect. If you edit that config, watch that
percentage: it is the fastest way to catch a LoRA that is quietly adapting two
projections in sixteen layers.

## Launching a real run

```bash
conda activate finetune
cd /Volumes/Callisto/Projects/orpheus-finetune
mlx_lm.lora -c configs/mac/qwen3_titles_mlx.yaml
```

Edit `data:`, `adapter_path:`, `model:` and `iters:` in the config, or override
any of them on the command line — CLI flags win over the file.

Resume from an interrupted run with `--resume-adapter-file <path>/adapters.safetensors`.
MLX writes `NNNNNNN_adapters.safetensors` every `save_every` iterations plus a
rolling `adapters.safetensors`, so a crash costs at most that interval.

Evaluate a candidate without fusing anything:

```bash
mlx_lm.generate --model Qwen/Qwen3-4B --adapter-path <adapters-dir> \
    --prompt "..." --max-tokens 40
```

## Getting weights back out

MLX adapters are **MLX layout** (`lora_a`/`lora_b`, dotted MLX module paths) in a
`.safetensors` file with a sibling `adapter_config.json`. They are not peft
adapters and `PeftModel.from_pretrained` will not read them. Nothing downstream
should consume the raw adapter — fuse first.

```bash
# adapter + base -> a normal HF-layout model directory
mlx_lm.fuse --model Qwen/Qwen3-4B \
            --adapter-path <adapters-dir> \
            --save-path /Volumes/Callisto/training/titles/fused

# straight to GGUF for llama.cpp / BookForge's bundled llama-server
mlx_lm.fuse --model Qwen/Qwen3-4B --adapter-path <adapters-dir> \
            --save-path <dir> --export-gguf --gguf-path model-f16.gguf
```

`--export-gguf` only supports a few architectures and only f16; if it refuses,
fuse to an HF directory and run llama.cpp's `convert_hf_to_gguf.py` +
`llama-quantize` as the rubric publishing flow already does. Add `--dequantize`
when the base was 4-bit and you want 16-bit weights out.

To train against a 4-bit base in the first place, quantize once and point at the
local directory:

```bash
mlx_lm.convert --hf-path Qwen/Qwen3-32B --mlx-path ~/mlx-models/Qwen3-32B-4bit -q --q-bits 4
```

## What actually fits in 64 GB — honestly

Unified memory means "VRAM" and "RAM" are the same pool, and macOS caps how much
of it the GPU may wire down — by default roughly 75% (~48 GB here;
`sysctl iogpu.wired_limit_mb` reads `0`, meaning system default). You can raise
it with `sudo sysctl iogpu.wired_limit_mb=<mb>`, which does not survive a reboot,
and which starves the rest of the OS if you get greedy.

| Model size | Approach | Verdict |
|---|---|---|
| ≤ 8B | bf16 LoRA | **Comfortable.** 8B bf16 ≈ 16 GB of weights; LoRA optimizer state is negligible. Room for batch size and 4k sequences |
| 14B – 32B | 4-bit QLoRA (`mlx_lm.convert -q`) | **Comfortable.** 14B@4bit ≈ 8 GB, 32B@4bit ≈ 18 GB of weights. This is the sweet spot for this machine — but see the measured 32B numbers below, the batch size is not free |
| 70B – 72B | 4-bit QLoRA | **Marginal.** Weights alone ≈ 40 GB against a ~48 GB wired ceiling. Needs `batch_size: 1`, short `max_seq_length`, `grad_checkpoint: true`, `num_layers` restricted, and patience. It will run; it will not be pleasant, and nothing else can run alongside it |
| anything, full fine-tune | — | Not on this machine. LoRA/QLoRA only |

### Measured: Qwen3-32B-4bit, rank-32 full-depth LoRA, 2048 ctx, grad checkpointing

Probed 2026-08-04 on an otherwise-cleared machine (~31 GB unused at the start),
`mlx-community/Qwen3-32B-4bit`, `num_layers: -1`, all seven projections,
`mask_prompt: true`. Trainable parameters 0.819% (268M/32,762M).

| `batch_size` | throughput | MLX peak mem | verdict |
|---|---|---|---|
| 1 | 10.5 s/row | 22.4 GB | works, wasteful |
| 2 | **7.6 s/row** | 25.4 GB | **best that fits** |
| 4 | — | — | **wall.** 8m40s elapsed at 1.6% CPU, ~4 GB unused system-wide — page-faulting instead of computing |

Batch 4 is the shape of the failure the next bullet describes: not an error, not
even an obvious slowdown at first, just a process that stops making progress
while looking alive. If a step time goes strange, check `%CPU` before you check
anything else — a training loop at 1.6% CPU is starved of memory, not slow.

Useful derived numbers at batch 2: **~4.5 s/row for a validation forward pass**
(so `val_batches: -1` over a 239-row set is ~18 min a time — budget it), and
**~1.07 GB per saved checkpoint** at rank 32 on 32B, which is what `save_every`
actually costs in disk.

Two further cautions specific to *this* Mac:

- **This machine is not idle.** It runs BookForge and Orpheus/XTTS TTS, and the
  Orpheus MLX engine alone holds ~15 GB steady and peaks ~22 GB at batch 96
  (measured; see BookForge's notes). A 32B QLoRA train plus a TTS render will put
  the system into swap and both will crawl. **Run long trains when the machine is
  otherwise idle**, and expect to be told to stop by whatever else you wanted the
  Mac to do that evening.
- **Memory pressure on macOS degrades rather than crashes.** You will not get a
  clean OOM like CUDA gives you; you get compressed memory, then swap, then a
  training loop that mysteriously runs 10× slower. Watch Activity Monitor's
  memory-pressure graph, not just the peak-mem number MLX prints.

## Things that will cost you a day if nobody tells you

- **`warmup` interacts badly with short smoke runs.** The example config has
  `warmup: 50`. A 5-iteration smoke check therefore reports `Learning Rate
  0.000e+00` and the loss barely moves. That is correct behaviour, not a broken
  setup — but it makes a smoke run useless as a "is it learning" signal. For
  smoke checks, override the schedule or just read "did it produce an adapter
  and is the loss finite".
- **MLX's default `num_layers` is 16, not all of them.** Silent, and it makes
  every hyperparameter you copied from the rig mean something different.
- **There is no early stopping.** The rig's most valuable finding — both text
  models peaked at epoch 1 and got worse — has no automatic safety net here.
  Set `save_every` low enough to have real choices and pick by val loss.
- **Do not put the env or the HF cache on Callisto.** See the ExFAT table above.
- **The HF token** for the OwenMorgan account lives at
  `~/.config/bookforge/hf-owenmorgan.token` (mode 600). Export it only when a
  push or a gated download needs it (`export HF_TOKEN=$(cat ...)`); never commit
  it, never copy it into a config, never bake it into a shell rc. Qwen base
  models are public and need no auth at all.

## Intended first use

The **YouTube titling model** — the `qwen3_titles` / `qwen3_titles_14b` line of
work. The dataset side (`build_titles_dataset.py`, `fetch_captions.py`,
`vtt_to_text.py`, `audit_summaries.py`) is platform-agnostic and runs here
unchanged; only the training step differs, and `configs/mac/qwen3_titles_mlx.yaml`
is the starting point for it.

The honest reason to train it here rather than on the rig: the rig's GPU is
frequently owned by a voice-training sequencer, and a 4B–14B title model is
exactly the size this Mac handles comfortably. The honest reason *not* to: the
rig's recipe is proven and its numbers are comparable across runs, and moving to
MLX resets that baseline. If a title run needs to be compared against the
existing ledger in `MODEL_LEDGER.md`, run it on the rig.
