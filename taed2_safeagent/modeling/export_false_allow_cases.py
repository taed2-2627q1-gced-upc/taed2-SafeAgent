from pathlib import Path

import pandas as pd

from taed2_safeagent.config import MODEL_NAMES, PROJ_ROOT


def export_false_allows_for_model(model_name: str, model_dir: Path) -> int:
    """Filtra casos con True DENY predichos como ALLOW y los guarda en JSONL."""
    joined_path = model_dir / "predictions_joined_with_metadata.jsonl"
    if not joined_path.exists():
        print(f"Skipping {model_name}: {joined_path} not found.")
        return 0

    df = pd.read_json(joined_path, lines=True)

    # Filtrar falsos ALLOWs (Real DENY -> Predicho ALLOW)
    false_allows_df = df[
        (df["true_label"] == "DENY") & (df["predicted_label"] == "ALLOW")
    ][["id", "command", "true_label", "predicted_label"]]

    output_path = model_dir / "false_allow_cases.jsonl"
    false_allows_df.to_json(output_path, orient="records", lines=True)

    count = len(false_allows_df)
    print(f"[{model_name}] Exportados {count} casos 'False ALLOW' en: {output_path}")
    return count


def main() -> None:
    # 1. Encoders
    for model_name in MODEL_NAMES:
        model_dir = PROJ_ROOT / "models" / "encoders" / model_name
        export_false_allows_for_model(model_name, model_dir)

    # 2. Baseline (Scikit-Learn)
    baseline_dir = PROJ_ROOT / "models" / "tfidf_char_linear_svm"
    if baseline_dir.exists():
        export_false_allows_for_model("tfidf_char_linear_svm", baseline_dir)


if __name__ == "__main__":
    main()