import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from time import perf_counter, time
from typing import Any

import numpy as np
import torch
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)
import typer

from taed2_safeagent.data.common import LABELS, load_params, write_json, write_jsonl
from taed2_safeagent.data.inputs import build_input
from taed2_safeagent.modeling.common import code_identity, dataset_identity, load_examples
from taed2_safeagent.modeling.energy import measure_training
from taed2_safeagent.modeling.evaluate import compute_metrics
from taed2_safeagent.modeling.tracking import dvc_command

app = typer.Typer(pretty_exceptions_show_locals=False)


def encode_examples(rows, mode, tokenizer, maximum):
    texts = [build_input(row, mode) for row in rows]
    lengths = tokenizer(texts, truncation=False, return_length=True)["length"]
    encoded = tokenizer(texts, truncation=True, max_length=maximum, padding=False)
    examples = [
        {
            **{key: values[index] for key, values in encoded.items()},
            "labels": LABELS.index(row["label"]),
        }
        for index, row in enumerate(rows)
    ]
    stats = {
        "rows": len(rows),
        "truncated_rows": sum(length > maximum for length in lengths),
        "maximum_raw_tokens": max(lengths),
        "max_length": maximum,
    }
    return examples, stats


def predicted_labels(logits):
    if isinstance(logits, tuple):
        logits = logits[0]
    if logits.ndim != 2 or logits.shape[1] != len(LABELS) or not np.isfinite(logits).all():
        raise ValueError("Invalid validation scores")
    return [LABELS[index] for index in np.argmax(logits, axis=-1).tolist()]


def validation_metrics(prediction):
    expected = [LABELS[index] for index in prediction.label_ids.tolist()]
    predicted = predicted_labels(prediction.predictions)
    return {"macro_f1": compute_metrics(expected, predicted)["macro_f1"]}


def validate_settings(settings):
    fields = ("epochs", "learning_rate", "batch_size", "gradient_accumulation_steps", "max_length")
    if any(settings[field] <= 0 for field in fields):
        raise ValueError("Training settings must be positive")
    if not 0 <= settings["warmup_ratio"] <= 1 or settings["weight_decay"] < 0:
        raise ValueError("Invalid warmup or weight decay")
    if "class_weights" in settings and settings["class_weights"] is not None:
        weights = settings["class_weights"]
        if not isinstance(weights, list) or len(weights) != len(LABELS):
            raise ValueError(f"class_weights must be a list of length {len(LABELS)}")
        if any(w <= 0 for w in weights):
            raise ValueError("All class weights must be positive")


def training_arguments(settings, directory, cpu):
    return TrainingArguments(
        output_dir=directory,
        num_train_epochs=settings["epochs"],
        learning_rate=settings["learning_rate"],
        per_device_train_batch_size=settings["batch_size"],
        per_device_eval_batch_size=settings["batch_size"],
        gradient_accumulation_steps=settings["gradient_accumulation_steps"],
        warmup_ratio=settings["warmup_ratio"],
        weight_decay=settings["weight_decay"],
        gradient_checkpointing=settings["gradient_checkpointing"],
        lr_scheduler_type=settings.get("lr_scheduler_type", "linear"),
        seed=settings["seed"],
        data_seed=settings["seed"],
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        fp16=not cpu,
        use_cpu=cpu,
        report_to=[],
        logging_steps=100,
        dataloader_num_workers=0,
        disable_tqdm=True,
    )


class WeightedTrainer(Trainer):
    """Trainer that applies class weights to CrossEntropyLoss."""

    def __init__(self, *args, class_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | None = None,
    ):
        if self.class_weights is None:
            return super().compute_loss(
                model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch
            )
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits") if isinstance(outputs, dict) else outputs[1]
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        if self.args.average_tokens_across_devices and num_items_in_batch is not None:
            loss *= self.accelerator.num_processes if self.args.n_gpu <= 1 else self.args.n_gpu
        return (loss, outputs) if return_outputs else loss


def prepared_inputs(params, root):
    source = code_identity()
    if not source["commit"] or source["dirty"] is not False:
        raise ValueError("Commit source and settings before encoder training")
    if json.loads(dvc_command(root, "status", "--json", "audit", "prepare", "validate")):
        raise ValueError("Prepare and validate the data before encoder training")
    quality = json.loads(
        (params["paths"]["reports"] / "validation.json").read_text(encoding="utf-8")
    )
    if quality["success"] is not True:
        raise ValueError("Data quality checks did not pass")
    train_rows = load_examples(params, "train")
    validation_rows = load_examples(params, "validation")
    for field in ("id", "group_id"):
        if {row[field] for row in train_rows} & {row[field] for row in validation_rows}:
            raise ValueError(f"Training and validation overlap: {field}")
    return source, train_rows, validation_rows


def load_backbone(backbone):
    if not re.fullmatch(r"[0-9a-f]{40}", backbone["revision"]):
        raise ValueError("Pin the backbone revision before training")
    options = {
        "revision": backbone["revision"],
        "trust_remote_code": False,
        "num_labels": len(LABELS),
        "id2label": dict(enumerate(LABELS)),
        "label2id": {label: index for index, label in enumerate(LABELS)},
    }
    config = AutoConfig.from_pretrained(backbone["name"], **options)
    if config.model_type == "modernbert":
        config.reference_compile = False
    model = AutoModelForSequenceClassification.from_pretrained(
        backbone["name"],
        revision=backbone["revision"],
        config=config,
        trust_remote_code=False,
        attn_implementation="sdpa",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        backbone["name"],
        revision=backbone["revision"],
        trust_remote_code=False,
    )
    return model, tokenizer


def save_validation(trainer, examples, rows, output):
    result = trainer.predict(examples)
    predicted = predicted_labels(result.predictions)
    metrics = {
        "split": "validation",
        "rows": len(rows),
        **compute_metrics([row["label"] for row in rows], predicted),
    }
    write_json(output / "metrics.json", metrics)
    write_jsonl(
        output / "predictions.jsonl",
        [
            {"id": row["id"], "true_label": row["label"], "predicted_label": guess}
            for row, guess in zip(rows, predicted)
        ],
    )
    return metrics


# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals,too-many-statements
def fit_encoder(params_path, name, mode, output, variant=None, allow_cpu=False):
    if mode not in {"command", "command-context"}:
        raise ValueError("Use command or command with context inputs")
    params_path = Path(params_path).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Use a new output directory to preserve existing results")
    params = load_params(params_path)
    backbone = params["encoders"]["backbones"][name]
    if variant:
        if "variants" not in params["encoders"] or variant not in params["encoders"]["variants"]:
            raise ValueError(f"Variant '{variant}' not found in approved variants")
        settings = params["encoders"]["variants"][variant]
    else:
        settings = params["encoders"]["training"]
    validate_settings(settings)
    cpu = not torch.cuda.is_available()
    if cpu and not allow_cpu:
        raise ValueError("A GPU is required for the full encoder run")
    source, training, validation = prepared_inputs(params, params_path.parent)
    identity = dataset_identity(params)
    set_seed(settings["seed"])
    model, tokenizer = load_backbone(backbone)
    if settings["max_length"] > tokenizer.model_max_length:
        raise ValueError("Token limit exceeds the backbone limit")
    training, train_stats = encode_examples(training, mode, tokenizer, settings["max_length"])
    examples, validation_stats = encode_examples(
        validation, mode, tokenizer, settings["max_length"]
    )
    output.mkdir(parents=True)
    input_lock = (params_path.parent / "dvc.lock").read_bytes()
    (output / "input_dvc.lock").write_bytes(input_lock)
    (output / "params.yaml").write_bytes(params_path.read_bytes())
    metadata = {
        "model": name,
        "variant": variant,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "backbone": backbone,
        "input_mode": mode,
        "labels": list(LABELS),
        "settings": settings,
        "code": source,
        "dataset": identity,
        "input_dvc_lock_hash": hashlib.sha256(input_lock).hexdigest(),
        "tokenization": {"train": train_stats, "validation": validation_stats},
        "hardware": {
            "device": "cpu" if cpu else torch.cuda.get_device_name(0),
            "visible_gpus": torch.cuda.device_count(),
        },
        "versions": {
            package: version(package) for package in ("torch", "transformers", "accelerate")
        },
    }
    weights = None
    if settings.get("class_weights") is not None:
        weights = torch.tensor(settings["class_weights"], dtype=torch.float32)

    with TemporaryDirectory(prefix="safeagent-checkpoints-") as checkpoints:
        trainer = WeightedTrainer(
            model=model,
            args=training_arguments(settings, checkpoints, cpu),
            train_dataset=training,
            eval_dataset=examples,
            processing_class=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8),
            compute_metrics=validation_metrics,
            class_weights=weights,
        )
        metadata["start_time_ms"] = int(time() * 1000)
        started = perf_counter()
        with measure_training(params["energy"], output / "energy.json"):
            trainer.train()
        metadata["training_seconds"] = perf_counter() - started
        metadata["end_time_ms"] = int(time() * 1000)
        metrics = save_validation(trainer, examples, validation, output)
        trainer.model.save_pretrained(output / "model", safe_serialization=True)
        tokenizer.save_pretrained(output / "model")
        metadata["model_bytes"] = sum(
            path.stat().st_size for path in (output / "model").rglob("*") if path.is_file()
        )
        metadata["best_validation_macro_f1"] = trainer.state.best_metric
    if code_identity() != source:
        raise ValueError("Source changed during encoder training")
    metadata["status"] = "FINISHED"
    write_json(output / "metadata.json", metadata)
    return metrics


# pylint: disable=too-many-arguments,too-many-positional-arguments
@app.command()
def main(
    model: str,
    output: Path,
    mode: str = "command",
    variant: str | None = None,
    params: Path = Path("params.yaml"),
    allow_cpu: bool = False,
):
    try:
        metrics = fit_encoder(params, model, mode, output, variant, allow_cpu)
    except (ValueError, TypeError, KeyError, OSError, RuntimeError) as error:
        typer.echo(f"Encoder training failed: {error}", err=True)
        raise typer.Exit(1) from None
    typer.echo(f"Validation macro F1: {metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    app()
