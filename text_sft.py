"""Text-native chat JSONL LoRA training, separate from Orpheus/SNAC encoding."""

from __future__ import annotations

import json
from pathlib import Path

from training_profiles import TrainingProfile


def _read_chat_jsonl(path: Path, messages_field: str) -> list[dict]:
    if not path.is_file():
        raise SystemExit(f"[text-train] dataset does not exist: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not (line := line.strip()):
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise SystemExit(f"[text-train] {path}:{line_number}: invalid JSON: {error}") from error
            messages = record.get(messages_field)
            if not isinstance(messages, list) or not messages:
                raise SystemExit(f"[text-train] {path}:{line_number}: '{messages_field}' must be a non-empty list")
            for message in messages:
                if not isinstance(message, dict):
                    raise SystemExit(f"[text-train] {path}:{line_number}: each message must be an object")
                if not isinstance(message.get("role"), str) or not isinstance(message.get("content"), str):
                    raise SystemExit(f"[text-train] {path}:{line_number}: every message needs string role and content")
            if messages[-1]["role"] != "assistant":
                raise SystemExit(f"[text-train] {path}:{line_number}: final message must be the assistant target")
            rows.append({messages_field: messages})
    if not rows:
        raise SystemExit(f"[text-train] dataset has no usable records: {path}")
    return rows


def _encode_conversation(tokenizer, messages, max_seq_length, template_kwargs,
                         assistant_only, where) -> dict:
    """Tokenize one conversation, masking every non-assistant token when the profile
    asks for assistant-only loss.

    Chat templates render the WHOLE conversation at once, so per-message token spans
    are recovered by rendering growing prefixes and taking the delta. That is valid
    only while the template is prefix-monotonic (appending a message only appends
    tokens). The prefix is re-verified at every step and a violation raises, because
    a template that rewrites earlier tokens would mislabel the batch silently.

    Over-length conversations raise rather than truncate: a target clipped mid-answer
    teaches the model to stop early, which is exactly the failure mode an OCR-repair
    model must not learn.
    """
    input_ids: list[int] = []
    labels: list[int] = []
    previous: list[int] = []
    for index, message in enumerate(messages):
        rendered = tokenizer.apply_chat_template(
            messages[:index + 1], tokenize=False, add_generation_prompt=False,
            **template_kwargs)
        ids = tokenizer(rendered, add_special_tokens=False).input_ids
        if ids[:len(previous)] != previous:
            raise SystemExit(
                f"[text-train] {where}: chat template is not prefix-monotonic "
                f"(message {index} rewrote earlier tokens); assistant-only loss cannot "
                f"be derived from prefix deltas for this tokenizer")
        span = ids[len(previous):]
        input_ids.extend(span)
        labels.extend(span if (message["role"] == "assistant" or not assistant_only)
                      else [-100] * len(span))
        previous = ids

    if len(input_ids) > max_seq_length:
        raise SystemExit(
            f"[text-train] {where}: conversation is {len(input_ids)} tokens but "
            f"max_seq_length is {max_seq_length}; split the record during dataset prep "
            f"or raise max_seq_length — truncating would corrupt the target")
    if all(label == -100 for label in labels):
        raise SystemExit(f"[text-train] {where}: no assistant tokens to learn from")
    return {"input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels}


def train_text_sft(args, profile: TrainingProfile) -> int:
    """Train a text model from explicitly supplied chronological train/eval JSONL."""
    if profile.kind != "text_sft":
        raise SystemExit(f"profile '{profile.name}' is not a text SFT profile")
    if profile.dataset_format != "chat_jsonl" or not profile.messages_field:
        raise SystemExit(f"profile '{profile.name}' must declare chat_jsonl dataset settings")
    if not args.train_data or not args.eval_data or not args.run_name or not args.out_base:
        raise SystemExit("[text-train] --train-data, --eval-data, --run-name, and --out-base are required")
    if args.stop_overtrain:
        raise SystemExit("[text-train] --stop-overtrain is Orpheus-specific; use normal early stopping")
    if args.mask_prompt_loss:
        raise SystemExit("[text-train] --mask-prompt-loss is Orpheus-specific; text profiles "
                         "declare dataset.assistant_only_loss instead")

    import torch
    from datasets import Dataset
    from transformers import DataCollatorForSeq2Seq, EarlyStoppingCallback, Trainer, TrainingArguments
    from unsloth import FastLanguageModel

    field = profile.messages_field
    train_rows = _read_chat_jsonl(Path(args.train_data), field)
    eval_rows = _read_chat_jsonl(Path(args.eval_data), field)
    if args.limit:
        train_rows = train_rows[:args.limit]
        eval_rows = eval_rows[:max(1, args.limit // 8)]

    max_seq_length = args.max_seq_length or profile.max_seq_length
    epochs = args.epochs or profile.max_epochs
    lr_schedule = args.lr_schedule or profile.lr_scheduler_type
    lora_dir, merged_dir = profile.output_dirs(Path(args.out_base), run_name=args.run_name)
    base_model = args.train_from or profile.model_id
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model, max_seq_length=max_seq_length, dtype=None,
        load_in_4bit=profile.load_in_4bit)
    model = FastLanguageModel.get_peft_model(
        model, r=profile.lora_r, target_modules=list(profile.target_modules),
        lora_alpha=profile.lora_alpha, lora_dropout=profile.lora_dropout, bias="none",
        use_gradient_checkpointing="unsloth", random_state=profile.seed)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def encode(split):
        def _encode(record, index):
            return _encode_conversation(
                tokenizer, record[field], max_seq_length, profile.template_kwargs,
                profile.assistant_only_loss, f"{split} row {index}")
        return _encode

    train_ds = Dataset.from_list(train_rows).map(
        encode("train"), with_indices=True, remove_columns=[field])
    eval_ds = Dataset.from_list(eval_rows).map(
        encode("eval"), with_indices=True, remove_columns=[field])
    supervised = sum(sum(1 for label in row if label != -100) for row in train_ds["labels"])
    total = sum(len(row) for row in train_ds["labels"])

    smoke = bool(args.max_steps)
    load_best = not smoke and not args.no_early_stop
    training_args = TrainingArguments(
        per_device_train_batch_size=profile.per_device_train_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=profile.gradient_accumulation_steps,
        warmup_ratio=profile.warmup_ratio,
        num_train_epochs=(1 if smoke else epochs),
        max_steps=(args.max_steps if smoke else -1),
        learning_rate=profile.learning_rate, logging_steps=10, optim=profile.optimizer,
        weight_decay=profile.weight_decay, lr_scheduler_type=lr_schedule,
        seed=profile.seed, output_dir=str(lora_dir), report_to="none",
        save_strategy=("no" if smoke else "epoch"),
        eval_strategy=("no" if smoke else "epoch"),
        save_total_limit=(profile.save_total_limit if load_best else None),
        load_best_model_at_end=load_best,
        metric_for_best_model=("eval_loss" if load_best else None),
        greater_is_better=False,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    callbacks = ([EarlyStoppingCallback(early_stopping_patience=profile.early_stopping_patience)]
                 if load_best else [])
    trainer = Trainer(
        model=model, args=training_args, train_dataset=train_ds,
        eval_dataset=(eval_ds if not smoke else None),
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True,
                                             label_pad_token_id=-100),
        callbacks=callbacks)
    resume = args.resume_from_checkpoint
    if resume == "latest":
        checkpoints = sorted(lora_dir.glob("checkpoint-*"), key=lambda item: int(item.name.split("-")[1]))
        if not checkpoints:
            raise SystemExit(f"[text-train] no checkpoint-* in {lora_dir}")
        resume = str(checkpoints[-1])
    elif resume and not Path(resume).is_dir():
        raise SystemExit(f"[text-train] --resume-from-checkpoint is not a directory: {resume}")

    print(f"[text-train] profile={profile.name} base={base_model} train={len(train_ds)} "
          f"eval={len(eval_ds)} r={profile.lora_r} lr={profile.learning_rate} "
          f"seq={max_seq_length} qlora={profile.load_in_4bit}")
    print(f"[text-train] assistant_only_loss={profile.assistant_only_loss}: "
          f"{supervised}/{total} train tokens carry loss "
          f"({supervised / total * 100:.1f}%)")
    trainer.train(resume_from_checkpoint=resume)
    model.save_pretrained(str(lora_dir))
    tokenizer.save_pretrained(str(lora_dir))
    print(f"[text-train] saved LoRA adapters -> {lora_dir}")
    if args.merge:
        model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")
        print(f"[text-train] merged 16-bit model -> {merged_dir}")
    return 0
