# Setup — from a bare machine to a training run

Written for someone who has never run this repo. If you only want to *use* a model that
someone else trained, you do not need any of this; see "Inference only" at the bottom.

## What you need

- **An NVIDIA GPU with 24 GB VRAM** for the recipes as written. A 3090 / 3090 Ti / 4090
  is the reference. 16 GB works for the 4B text models if you drop batch size; the voice
  work at 2048 sequence length wants the full 24.
- **Linux, or Windows with WSL2.** Not optional for the voice pipeline: vLLM only captures
  CUDA graphs on Linux, and without them generation is roughly 6× slower. Training itself
  works on native Windows; inference is where it hurts.
- **Tens of GB of fast disk** for datasets and merged models. A merged 4B is ~8 GB, a 14B
  is ~28 GB, and you will keep several.

## 1. Environment

```bash
conda create -n orpheus_train python=3.11 -y
conda activate orpheus_train

# Torch FIRST, from the index matching your CUDA driver. Check yours with `nvidia-smi`.
pip install torch --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
pip install --no-deps trl==0.22.2   # --no-deps is required, see requirements.txt
```

If you are on WSL, keep the conda env and all training output on the **Linux**
filesystem. Putting them under `/mnt/c` works and is agonisingly slow — the 9p mount is
the single most common cause of "why is this taking all night".

## 2. Point it at your disks

```bash
cp .env.example .env
$EDITOR .env
```

Nothing has a default. If a root is unset, the script that needs it fails immediately and
tells you which variable to set and what it means — deliberately, because a silent default
sends an eight-hour training run somewhere you did not intend and you find out afterwards.

## 3. Verify before you commit a long run

```bash
python -c "from config import FT_ROOT; print(FT_ROOT)"     # resolves? good.
python orpheus_owen.py --profile qwen3_titles --max-steps 5 --limit 64 train \
    --train-data <train.jsonl> --eval-data <eval.jsonl> \
    --run-name smoke --out-base "$ORPHEUS_FT_ROOT"
```

Five steps on 64 examples. It proves the data loads, the template applies, the loss mask
is right and the GPU fits, in about a minute. Do this every time you change the dataset
shape. Note the **globals come before the subcommand** (`--profile X train`, not
`train --profile X`) — argparse will not tell you nicely.

## 4. Train

Drop `--max-steps/--limit`, add `--merge` to get 16-bit weights out at the end:

```bash
python orpheus_owen.py --profile <profile> train \
    --train-data <train.jsonl> --eval-data <eval.jsonl> \
    --run-name <name> --out-base "$ORPHEUS_FT_ROOT" --merge
```

Profiles live in `training_profiles.json`. `qwen3_titles` (4B) and `qwen3_titles_14b` are
text models; `orpheus` is the voice recipe. Start from the closest profile rather than
inventing hyperparameters.

## Text vs voice

Two pipelines share this repo:

- **Voice cloning** — `orpheus_owen.py`, `session.py`, `pipeline/`, `cut_audiobook.py` and
  the audio tooling. Start at `README.md`, then `VOICE_TRAINING_PIPELINE.md`.
- **Text fine-tuning** — `text_sft.py` plus a profile. Generic chat-JSONL SFT: give it
  `{"messages": [...]}` per line and it trains, with assistant-only loss and early
  stopping. The YouTube-title scripts (`build_titles_dataset.py`, `fetch_captions.py`,
  `vtt_to_text.py`, `audit_summaries.py`) are one worked example of building such a
  dataset; the harness itself does not care what the text is.

## Things that will cost you a day if nobody tells you

- **Set your GPU power limit before long runs** if you have had stability trouble:
  `nvidia-smi -pl 400` (needs an elevated/root shell, and resets every boot). A 14B run
  here died with `CUBLAS_STATUS_INTERNAL_ERROR` at stock 450 W.
- **Both text models peaked at epoch 1** on ~2,000 examples and got worse after. Early
  stopping is on by default for a reason; if your eval loss rises after the first epoch,
  the answer is more data, not more epochs and not a bigger model.
- **Merging a 14B to 16-bit needs ~28 GB of system RAM**, not VRAM. Under WSL that means
  raising the `.wslconfig` memory cap. You can skip merging entirely and run the adapter
  over the same 4-bit base it was trained against, which is the more faithful path anyway.
- **Windows Python defaults to cp1252.** Every file read/write in this repo passes
  `encoding="utf-8"` explicitly. Keep doing that or you will hit `UnicodeDecodeError` on
  the first non-ASCII character in someone's data.

## Inference only

You need `transformers`, `torch`, and the merged model directory. No training deps, no
unsloth, no vLLM:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype="bfloat16", device_map="cuda")
```

For the title models, apply the chat template with `enable_thinking=False` and sample with
`num_return_sequences=10` to get ten candidates from one call — about 3 seconds warm on a
3090 Ti for the 4B.
