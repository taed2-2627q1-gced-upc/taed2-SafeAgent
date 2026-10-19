import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from taed2_safeagent.config import MODEL_NAMES


def compute_subset_metrics(subset):
    """Recieves a subset of the dataset and computes its metrics: Element
    count, accuracy, F1-score, precision, recall and critical_cases (cases
    where the model predicted "ALLOW" and the real label was "DENY")"""

    y_true = subset["true_label"]
    y_pred = subset["predicted_label"]

    count = len(subset)
    critical_cases = int(((y_true == "DENY") & (y_pred == "ALLOW")).sum())
    accuracy = (y_true == y_pred).mean()

    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=["DENY", "ALLOW"], zero_division=0
    )

    return {
        "count": int(count),
        "accuracy": float(accuracy),
        "f1_deny": float(f1[0]),
        "f1_allow": float(f1[1]),
        "precision_deny": float(p[0]),
        "recall_deny": float(r[0]),
        "critical_cases": critical_cases,
        "critical_cases_ratio": float(critical_cases / count) if count > 0 else 0.0,
    }


def compute_model_results(predictions_and_metadata_df):

    model_results = {"shell": {}, "category": {}, "command_length": {}}

    # Group by shell
    for group_name, group_df in predictions_and_metadata_df.groupby("shell", observed=False):
        model_results["shell"][str(group_name)] = compute_subset_metrics(group_df)

    # Group by category
    for group_name, group_df in predictions_and_metadata_df.groupby("category", observed=False):
        model_results["category"][str(group_name)] = compute_subset_metrics(group_df)

    # Create distinct classes for each command_length
    predictions_and_metadata_df["command_length_slice"] = pd.qcut(
        predictions_and_metadata_df["command_length"],
        q=4,
        labels=["Q1_short", "Q2_medium_short", "Q3_medium_long", "Q4_long"],
        duplicates="drop",  # Evita errores si hay muchos comandos con la misma longitud exactas
    )

    # Group by command_length
    for group_name, group_df in predictions_and_metadata_df.groupby(
        "command_length_slice", observed=False
    ):
        model_results["command_length"][str(group_name)] = compute_subset_metrics(group_df)

    return model_results


def main():

    project_root = Path(__file__).resolve().parents[2]
    print(f"Project root: {project_root}")

    for model_name in MODEL_NAMES:
        predictions_and_metadata_path = project_root / Path(
            f"models/encoders/{model_name}/predictions_joined_with_metadata.jsonl"
        )
        predictions_and_metadata_df = pd.read_json(predictions_and_metadata_path, lines=True)

        # Add command_length column
        predictions_and_metadata_df["command_length"] = (
            predictions_and_metadata_df["command"].astype(str).str.len()
        )
        model_results = compute_model_results(predictions_and_metadata_df)

        output_metrics_path = project_root / f"models/encoders/{model_name}/slice_metrics.json"
        with open(output_metrics_path, "w", encoding="utf-8") as f:
            json.dump(model_results, f, indent=4)

        print(f"Results from model {model_name} were saved on ")


if __name__ == "__main__":
    main()



