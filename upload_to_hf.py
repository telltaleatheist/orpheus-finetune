"""Upload a merged 16-bit Orpheus voice model to HuggingFace, tagged for BookForge.

The card carries the `bookforge-orpheus-voice` tag + flat `orpheus_token`/`label`/
`sample_rate` keys, so the moment it's on HF it appears in BookForge's download catalog
(Settings → Orpheus voices) — see electron/orpheus-hf-catalog.ts.

Run in WSL (env orpheus_train, has huggingface_hub):
  python upload_to_hf.py --model-dir <dir> --repo <user/name> --voice-token <tok> --label <txt>

Defaults reproduce the Owen upload. Token read from the canonical file or HF_TOKEN.
Real-person voice clones default to PRIVATE; the owner's BookForge lists/installs private
repos with its configured token. Use with the subject's consent.
"""
import argparse
import os
import re


TOKEN_FILE = "/mnt/c/Users/tellt/Downloads/bookforge-hf-token.txt"

CARD_TMPL = """---
license: {license}
base_model: {base_model}
tags:
- orpheus
- tts
- text-to-speech
- voice-clone
- snac
- bookforge-orpheus-voice
orpheus_token: {voice_token}
label: {label}
sample_rate: {sample_rate}
---

# {label} — Orpheus-3B voice

A LoRA fine-tune of Orpheus-3B (merged to 16-bit) on a single narration voice, for
BookForge. Single speaker.

- **Voice / prompt format:** `{voice_token}: <text>`  (the source name is `{voice_token}`)
- **Audio codec:** SNAC 24 kHz (`hubertsiuzdak/snac_24khz`); 7 codes/frame, offset 128266.
- **Base:** `{base_model}`. **Best epoch** selected by held-out eval loss.
- Load in vLLM (CUDA graphs) or transformers; generate audio tokens, reverse the
  7-per-frame layout, `snac.decode` -> 24 kHz waveform.

Method + inference code: github.com/telltaleatheist/orpheus-voice-finetune
(`orpheus_owen.py speak`). Personal voice model — use with the subject's consent.
"""


def resolve_token() -> str:
    env = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip()
    if env:
        return env
    m = re.search(r"hf_[A-Za-z0-9_]{20,}", open(TOKEN_FILE).read())
    if not m:
        raise SystemExit("no hf_ token in env or token file")
    return m.group(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/home/telltale/xtts_ft/orpheus_owen_merged")
    ap.add_argument("--repo", default="owenmorgan/owen-morgan-orpheus-3b")
    ap.add_argument("--voice-token", default="owen", help="the prompt token (orpheus_token)")
    ap.add_argument("--label", default="Owen Morgan")
    ap.add_argument("--sample-rate", type=int, default=24000)
    ap.add_argument("--base-model", default="unsloth/orpheus-3b-0.1-ft")
    ap.add_argument("--license", default="apache-2.0")
    ap.add_argument("--public", action="store_true", help="default is PRIVATE")
    args = ap.parse_args()

    from huggingface_hub import HfApi

    token = resolve_token()
    api = HfApi(token=token)
    private = not args.public
    print(f"creating repo {args.repo} (private={private})")
    api.create_repo(args.repo, repo_type="model", private=private, exist_ok=True)

    card = CARD_TMPL.format(
        license=args.license, base_model=args.base_model, voice_token=args.voice_token,
        label=args.label, sample_rate=args.sample_rate,
    )
    api.upload_file(path_or_fileobj=card.encode("utf-8"), path_in_repo="README.md",
                    repo_id=args.repo, repo_type="model")
    print(f"uploading {args.model_dir} ...")
    api.upload_folder(folder_path=args.model_dir, repo_id=args.repo, repo_type="model",
                      ignore_patterns=[".cache/*", "*.lock"])
    print(f"DONE -> https://huggingface.co/{args.repo}  (token '{args.voice_token}')")


if __name__ == "__main__":
    main()
