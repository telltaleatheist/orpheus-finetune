"""Strict loader for the versioned training-profile settings file.

Profiles are deliberately data-free: model and optimization settings live here,
while every dataset path remains an explicit CLI argument. This prevents a model
run from silently using another project's data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class ProfileError(ValueError):
    """A training profile is absent or structurally incomplete."""


def _required(mapping: dict[str, Any], key: str, context: str) -> Any:
    try:
        return mapping[key]
    except KeyError as error:
        raise ProfileError(f"missing required setting '{context}.{key}'") from error


@dataclass(frozen=True)
class TrainingProfile:
    name: str
    kind: str
    description: str
    model_id: str
    max_seq_length: int
    load_in_4bit: bool
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: tuple[str, ...]
    learning_rate: float
    optimizer: str
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    max_epochs: int
    early_stopping_patience: int
    save_total_limit: int
    warmup_ratio: float
    weight_decay: float
    lr_scheduler_type: str
    seed: int
    lora_dir_template: str
    merged_dir_template: str
    default_out_base: str | None = None
    dataset_format: str | None = None
    messages_field: str | None = None
    assistant_only_loss: bool | None = None
    template_kwargs: dict[str, Any] = field(default_factory=dict)

    def output_dirs(self, out_base: Path, **values: str) -> tuple[Path, Path]:
        try:
            lora_dir = self.lora_dir_template.format(**values)
            merged_dir = self.merged_dir_template.format(**values)
        except KeyError as error:
            raise ProfileError(
                f"profile '{self.name}' output template requires '{error.args[0]}'"
            ) from error
        return out_base / lora_dir, out_base / merged_dir


def load_training_profile(path: Path, name: str) -> TrainingProfile:
    if not path.is_file():
        raise ProfileError(f"training profile file does not exist: {path}")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ProfileError(f"invalid JSON in training profile file {path}: {error}") from error

    profiles = _required(document, "profiles", "root")
    if not isinstance(profiles, dict):
        raise ProfileError("'profiles' must be an object")
    try:
        raw = profiles[name]
    except KeyError as error:
        available = ", ".join(sorted(profiles))
        raise ProfileError(f"unknown training profile '{name}'; available: {available}") from error
    if not isinstance(raw, dict):
        raise ProfileError(f"profile '{name}' must be an object")

    model = _required(raw, "model", f"profiles.{name}")
    lora = _required(raw, "lora", f"profiles.{name}")
    training = _required(raw, "training", f"profiles.{name}")
    output = _required(raw, "output", f"profiles.{name}")
    for section_name, section in (("model", model), ("lora", lora),
                                  ("training", training), ("output", output)):
        if not isinstance(section, dict):
            raise ProfileError(f"profiles.{name}.{section_name} must be an object")

    dataset = raw.get("dataset")
    if dataset is not None and not isinstance(dataset, dict):
        raise ProfileError(f"profiles.{name}.dataset must be an object")
    template_kwargs: dict[str, Any] = {}
    if dataset is not None:
        template_kwargs = _required(dataset, "template_kwargs", f"profiles.{name}.dataset")
        if not isinstance(template_kwargs, dict):
            raise ProfileError(f"profiles.{name}.dataset.template_kwargs must be an object")

    try:
        return TrainingProfile(
            name=name,
            kind=str(_required(raw, "kind", f"profiles.{name}")),
            description=str(_required(raw, "description", f"profiles.{name}")),
            model_id=str(_required(model, "id", f"profiles.{name}.model")),
            max_seq_length=int(_required(model, "max_seq_length", f"profiles.{name}.model")),
            load_in_4bit=bool(_required(model, "load_in_4bit", f"profiles.{name}.model")),
            lora_r=int(_required(lora, "r", f"profiles.{name}.lora")),
            lora_alpha=int(_required(lora, "alpha", f"profiles.{name}.lora")),
            lora_dropout=float(_required(lora, "dropout", f"profiles.{name}.lora")),
            target_modules=tuple(str(item) for item in _required(lora, "target_modules", f"profiles.{name}.lora")),
            learning_rate=float(_required(training, "learning_rate", f"profiles.{name}.training")),
            optimizer=str(_required(training, "optimizer", f"profiles.{name}.training")),
            per_device_train_batch_size=int(_required(training, "per_device_train_batch_size", f"profiles.{name}.training")),
            gradient_accumulation_steps=int(_required(training, "gradient_accumulation_steps", f"profiles.{name}.training")),
            max_epochs=int(_required(training, "max_epochs", f"profiles.{name}.training")),
            early_stopping_patience=int(_required(training, "early_stopping_patience", f"profiles.{name}.training")),
            save_total_limit=int(_required(training, "save_total_limit", f"profiles.{name}.training")),
            warmup_ratio=float(_required(training, "warmup_ratio", f"profiles.{name}.training")),
            weight_decay=float(_required(training, "weight_decay", f"profiles.{name}.training")),
            lr_scheduler_type=str(_required(training, "lr_scheduler_type", f"profiles.{name}.training")),
            seed=int(_required(training, "seed", f"profiles.{name}.training")),
            lora_dir_template=str(_required(output, "lora_dir_template", f"profiles.{name}.output")),
            merged_dir_template=str(_required(output, "merged_dir_template", f"profiles.{name}.output")),
            default_out_base=(str(output["default_out_base"]) if "default_out_base" in output else None),
            dataset_format=(str(_required(dataset, "format", f"profiles.{name}.dataset")) if dataset else None),
            messages_field=(str(_required(dataset, "messages_field", f"profiles.{name}.dataset")) if dataset else None),
            assistant_only_loss=(bool(_required(dataset, "assistant_only_loss", f"profiles.{name}.dataset")) if dataset else None),
            template_kwargs=template_kwargs,
        )
    except (TypeError, ValueError) as error:
        raise ProfileError(f"invalid value in training profile '{name}': {error}") from error
