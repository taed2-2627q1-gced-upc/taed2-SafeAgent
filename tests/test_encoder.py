from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
import torch
from transformers import (
    AutoModelForSequenceClassification,
    ModernBertConfig,
    PreTrainedTokenizerFast,
    RobertaConfig,
)
import yaml

from taed2_safeagent.data.common import LABELS, write_json
from taed2_safeagent.data.validation import validate
from taed2_safeagent.modeling import encoder, energy
from taed2_safeagent.modeling.common import load_examples


@dataclass
class Measurement:
    """Small test measurement."""

    duration: float = 1.0
    energy_consumed: float = 0.01
    emissions: float = 0.001


@pytest.fixture(name="tokenizer")
def small_tokenizer():
    vocabulary = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "printf": 4}
    core = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    core.pre_tokenizer = Whitespace()
    result = PreTrainedTokenizerFast(
        tokenizer_object=core,
        pad_token="[PAD]",
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        model_max_length=64,
    )
    result.model_input_names = ["input_ids", "attention_mask"]
    return result


@pytest.fixture(name="tracker")
def recorded_tracker(monkeypatch):
    instance = Mock()
    instance.final_emissions_data = Measurement()
    monkeypatch.setattr(energy, "OfflineEmissionsTracker", Mock(return_value=instance))
    return instance


def test_energy_stops_and_reports_failed_training(params, tmp_path, tracker):
    report = tmp_path / "energy.json"
    with pytest.raises(ValueError), energy.measure_training(params["energy"], report):
        raise ValueError("Training failed")
    tracker.stop.assert_called_once()
    result = json.loads(report.read_text(encoding="utf-8"))
    assert result["status"] == "recorded"
    assert result["training_succeeded"] is False
    assert result["measurement"]["energy_consumed"] == 0.01


def test_missing_energy_monitor_does_not_invent_values(params, tmp_path, monkeypatch):
    monkeypatch.setattr(energy, "OfflineEmissionsTracker", Mock(side_effect=RuntimeError()))
    report = tmp_path / "energy.json"
    with energy.measure_training(params["energy"], report):
        pass
    result = json.loads(report.read_text(encoding="utf-8"))
    assert result["status"] == "unavailable"
    assert result["training_succeeded"] is True
    assert "measurement" not in result


def test_input_modes_keep_labels_and_annotation_fields_out(bundle, tokenizer):
    rows = load_examples(bundle, "train")
    encoded, stats = encoder.encode_examples(rows, "command", tokenizer, 32)
    assert len(encoded) == len(rows)
    assert {row["labels"] for row in encoded} == {0, 1, 2}
    assert stats["truncated_rows"] == 0
    contextual, stats = encoder.encode_examples(rows, "command-context", tokenizer, 8)
    assert stats["truncated_rows"] > 0
    assert all(len(row["input_ids"]) <= 8 for row in contextual)
    assert all(set(row) == {"input_ids", "attention_mask", "labels"} for row in contextual)


def test_validation_metrics_support_tuple_model_outputs():
    result = encoder.validation_metrics(
        SimpleNamespace(
            predictions=(np.eye(3), np.zeros(3)),
            label_ids=np.arange(3),
        )
    )
    assert result == {"macro_f1": 1.0}


@pytest.mark.parametrize("values", [np.full((2, 3), np.nan), np.zeros((2, 2))])
def test_invalid_model_scores_are_rejected(values):
    with pytest.raises(ValueError, match="Invalid validation scores"):
        encoder.predicted_labels(values)


@pytest.mark.parametrize(
    "field", ["epochs", "learning_rate", "batch_size", "gradient_accumulation_steps", "max_length"]
)
def test_nonpositive_training_settings_are_rejected(params, field):
    settings = params["encoders"]["training"].copy()
    settings[field] = 0
    with pytest.raises(ValueError, match="must be positive"):
        encoder.validate_settings(settings)


@pytest.mark.parametrize("name", ["codebert", "modernbert"])
def test_tiny_backbones_fit_and_save_aligned_validation(
    bundle,
    tokenizer,
    tracker,
    monkeypatch,
    name,
):
    root = bundle["paths"]["raw"].parent
    torch.set_num_threads(1)
    validate(bundle)
    settings = bundle["encoders"]["training"]
    settings.update(epochs=1, batch_size=64, max_length=32, gradient_accumulation_steps=1)
    source = {"commit": "a" * 40, "dirty": False}
    monkeypatch.setattr(encoder, "code_identity", lambda: source)
    monkeypatch.setattr(encoder, "dvc_command", lambda *_arguments: "{}")
    common = {
        "vocab_size": len(tokenizer),
        "hidden_size": 16,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "intermediate_size": 32,
        "num_labels": 3,
        "max_position_embeddings": 64,
        "pad_token_id": 0,
        "bos_token_id": 2,
        "eos_token_id": 3,
        "id2label": dict(enumerate(LABELS)),
        "label2id": {label: index for index, label in enumerate(LABELS)},
    }
    config = (
        RobertaConfig(**common)
        if name == "codebert"
        else ModernBertConfig(
            **common,
            reference_compile=False,
        )
    )
    model = AutoModelForSequenceClassification.from_config(config, attn_implementation="sdpa")
    monkeypatch.setattr(encoder, "load_backbone", lambda _settings: (model, tokenizer))
    params = root / "params.yaml"
    params.write_text(
        yaml.safe_dump(json.loads(json.dumps(bundle, default=str))), encoding="utf-8"
    )
    (root / "dvc.lock").write_bytes(b"stages: {}\n")
    output = root / "encoder-output"
    metrics = encoder.fit_encoder(params, name, "command-context", output, allow_cpu=True)
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "FINISHED"
    assert metadata["input_mode"] == "command-context"
    assert metadata["code"] == source
    assert metadata["tokenization"]["validation"]["rows"] == len(
        load_examples(bundle, "validation")
    )
    assert metrics["split"] == "validation"
    saved = AutoModelForSequenceClassification.from_pretrained(output / "model")
    assert saved.config.id2label == dict(enumerate(LABELS))
    predictions = [
        json.loads(line) for line in (output / "predictions.jsonl").read_text().splitlines()
    ]
    assert [row["id"] for row in predictions] == [
        row["id"] for row in load_examples(bundle, "validation")
    ]
    tracker.stop.assert_called_once()


def test_dirty_inputs_fail_before_loading_backbone(bundle, tmp_path, monkeypatch):
    monkeypatch.setattr(encoder, "code_identity", lambda: {"commit": "a" * 40, "dirty": True})
    with pytest.raises(ValueError, match="Commit source"):
        encoder.prepared_inputs(bundle, tmp_path)


def test_changed_data_quality_inputs_are_rejected(bundle, tmp_path, monkeypatch):
    monkeypatch.setattr(encoder, "code_identity", lambda: {"commit": "a" * 40, "dirty": False})
    monkeypatch.setattr(encoder, "dvc_command", lambda *_args: '{"validate": ["changed"]}')
    with pytest.raises(ValueError, match="Prepare and validate"):
        encoder.prepared_inputs(bundle, tmp_path)


def test_existing_output_is_preserved(tmp_path):
    marker = tmp_path / "existing"
    write_json(marker, {"keep": True})
    with pytest.raises(ValueError, match="preserve existing"):
        encoder.fit_encoder(Path("params.yaml"), "codebert", "command", tmp_path)
    assert json.loads(marker.read_text()) == {"keep": True}
